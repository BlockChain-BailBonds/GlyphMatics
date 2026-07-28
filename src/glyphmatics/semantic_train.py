"""Training loop for the multilingual semantic glyph language model."""

from __future__ import annotations

import json
import math
from pathlib import Path
import random
import time
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, random_split

from .semantic_codec import BOS_ID, EOS_ID, SemanticVocabulary
from .semantic_lm import ModelConfig, SemanticGlyphLM, save_checkpoint


class TokenBlocks(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, token_ids: list[int], block_size: int, *, stride: int | None = None):
        self.token_ids = torch.tensor(token_ids, dtype=torch.long)
        self.block_size = block_size
        self.stride = stride or block_size
        self.starts = list(range(0, max(0, len(token_ids) - block_size), self.stride))
        if not self.starts:
            raise ValueError("corpus is too small for the requested context length")

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = self.starts[index]
        block = self.token_ids[start:start + self.block_size + 1]
        return block[:-1], block[1:]


def load_corpus_ids(path: str | Path, *, seed: int = 918) -> list[int]:
    records: list[list[int]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        ids = [int(item) for item in row.get("ids", [])]
        if ids:
            records.append([BOS_ID, *ids, EOS_ID])
    random.Random(seed).shuffle(records)
    return [token for record in records for token in record]


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def train(
    *,
    vocabulary_path: str | Path,
    corpus_path: str | Path,
    checkpoint_path: str | Path,
    steps: int = 500,
    batch_size: int = 16,
    context_length: int = 128,
    dimension: int = 192,
    layers: int = 4,
    heads: int = 6,
    learning_rate: float = 3e-4,
    device_name: str = "auto",
    seed: int = 918,
    log_every: int = 10,
    validation_fraction: float = 0.05,
) -> dict[str, Any]:
    if steps < 1:
        raise ValueError("steps must be positive")
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    vocabulary = SemanticVocabulary.load(vocabulary_path)
    token_ids = load_corpus_ids(corpus_path, seed=seed)
    dataset = TokenBlocks(token_ids, context_length)
    if not 0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be between 0 and 0.5")
    validation_size = max(1, round(len(dataset) * validation_fraction))
    training_size = len(dataset) - validation_size
    if training_size < 1:
        raise ValueError("corpus does not contain enough blocks for validation")
    generator = torch.Generator().manual_seed(seed)
    training_dataset, validation_dataset = random_split(
        dataset,
        [training_size, validation_size],
        generator=generator,
    )
    loader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=len(training_dataset) >= batch_size,
        generator=generator,
    )
    device = resolve_device(device_name)
    config = ModelConfig(
        vocab_size=vocabulary.model_vocab_size,
        context_length=context_length,
        dimension=dimension,
        layers=layers,
        heads=heads,
    )
    model = SemanticGlyphLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    iterator = iter(loader)
    start_time = time.monotonic()
    losses: list[float] = []
    model.train()

    for step in range(1, steps + 1):
        try:
            inputs, targets = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            inputs, targets = next(iterator)
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
            enabled=use_amp,
        ):
            _, loss = model(inputs, targets=targets)
            assert loss is not None
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        loss_value = float(loss.detach().cpu())
        losses.append(loss_value)
        if step == 1 or step % log_every == 0 or step == steps:
            elapsed = time.monotonic() - start_time
            print(
                json.dumps(
                    {
                        "step": step,
                        "steps": steps,
                        "loss": round(loss_value, 6),
                        "perplexity": round(math.exp(min(loss_value, 20)), 3),
                        "tokens_per_second": round(
                            step * batch_size * context_length / max(elapsed, 1e-9), 1
                        ),
                        "device": str(device),
                    }
                ),
                flush=True,
            )

    model.eval()
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    validation_losses: list[float] = []
    with torch.inference_mode():
        for inputs, targets in validation_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
                enabled=use_amp,
            ):
                _, validation_loss = model(inputs, targets=targets)
            assert validation_loss is not None
            validation_losses.append(float(validation_loss.cpu()))
    mean_validation_loss = sum(validation_losses) / len(validation_losses)

    training = {
        "steps": steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "device": str(device),
        "elapsed_seconds": round(time.monotonic() - start_time, 3),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "minimum_loss": min(losses),
        "validation_loss": mean_validation_loss,
        "validation_perplexity": math.exp(min(mean_validation_loss, 20)),
        "parameter_count": model.parameter_count(),
        "training_tokens": len(token_ids),
        "dataset_blocks": len(dataset),
        "training_blocks": len(training_dataset),
        "validation_blocks": len(validation_dataset),
    }
    save_checkpoint(
        checkpoint_path,
        model,
        vocabulary_sha256=vocabulary.digest,
        training=training,
    )
    return training
