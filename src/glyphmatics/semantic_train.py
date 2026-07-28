"""Training, evaluation, and corpus-packing loop for the semantic glyph LM."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import glob
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler, RandomSampler, random_split

from .semantic_codec import BOS_ID, EOS_ID, SemanticVocabulary
from .semantic_lm import ModelConfig, SemanticGlyphLM, load_checkpoint, save_checkpoint


PACKED_CORPUS_FORMAT = "glyphmatics-packed-corpus-v1"

MODEL_PRESETS: dict[str, dict[str, int]] = {
    "reference": {"dimension": 192, "layers": 4, "heads": 6, "context_length": 128},
    "small": {"dimension": 256, "layers": 6, "heads": 8, "context_length": 256},
    "base": {"dimension": 384, "layers": 10, "heads": 12, "context_length": 512},
    "large": {"dimension": 512, "layers": 12, "heads": 16, "context_length": 1024},
    "xl": {"dimension": 768, "layers": 18, "heads": 16, "context_length": 2048},
}


@dataclass(frozen=True)
class TrainConfig:
    vocabulary_path: str | Path
    corpus_path: str | Path
    checkpoint_path: str | Path
    steps: int = 500
    batch_size: int = 16
    context_length: int | None = None
    dimension: int | None = None
    layers: int | None = None
    heads: int | None = None
    learning_rate: float = 3e-4
    min_learning_rate_ratio: float = 0.1
    warmup_steps: int = 50
    gradient_accumulation_steps: int = 1
    weight_decay: float = 0.1
    device_name: str = "auto"
    seed: int = 918
    log_every: int = 10
    validation_fraction: float = 0.05
    stride: int | None = None
    preset: str = "reference"
    resume_from: str | Path | None = None
    checkpoint_every: int = 0
    compile_model: bool = False
    num_workers: int = 0

    def validate(self) -> None:
        if self.steps < 1:
            raise ValueError("steps must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive")
        if not 0 < self.validation_fraction < 0.5:
            raise ValueError("validation_fraction must be between 0 and 0.5")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if not 0 < self.min_learning_rate_ratio <= 1:
            raise ValueError("min_learning_rate_ratio must be between 0 and 1")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.checkpoint_every < 0:
            raise ValueError("checkpoint_every must be non-negative")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")


@dataclass(frozen=True)
class DistributedState:
    enabled: bool
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    backend: str | None = None
    initialized_here: bool = False

    @property
    def is_main(self) -> bool:
        return self.rank == 0


@dataclass(frozen=True)
class CorpusTokens:
    token_array: np.ndarray
    token_count: int
    record_count: int
    source_kind: str

    def slice(self, start: int, end: int) -> np.ndarray:
        return self.token_array[start:end]


class TokenBlocks(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, corpus: CorpusTokens, block_size: int, *, stride: int | None = None):
        self.corpus = corpus
        self.block_size = block_size
        self.stride = stride or block_size
        self.starts = list(range(0, max(0, corpus.token_count - block_size), self.stride))
        if not self.starts:
            raise ValueError("corpus is too small for the requested context length")

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = self.starts[index]
        block = np.asarray(self.corpus.slice(start, start + self.block_size + 1), dtype=np.int64)
        tensor = torch.from_numpy(block.copy())
        return tensor[:-1], tensor[1:]


def _resolve_corpus_inputs(path: str | Path) -> list[Path]:
    raw = str(path)
    candidate = Path(path)
    if any(symbol in raw for symbol in "*?[]"):
        matches = sorted(Path(item) for item in glob.glob(raw))
        if not matches:
            raise FileNotFoundError(f"no corpus files matched pattern: {raw}")
        return matches
    if candidate.is_dir():
        manifest = candidate / "manifest.json"
        token_file = candidate / "tokens.u32"
        if manifest.exists() and token_file.exists():
            return [candidate]
        matches = sorted(candidate.glob("*.jsonl"))
        if not matches:
            raise FileNotFoundError(f"no JSONL corpus shards found in directory: {candidate}")
        return matches
    return [candidate]


def _load_raw_records(paths: list[Path], *, seed: int = 918) -> list[list[int]]:
    records: list[list[int]] = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            ids = [int(item) for item in row.get("ids", [])]
            if ids:
                records.append([BOS_ID, *ids, EOS_ID])
    random.Random(seed).shuffle(records)
    return records


def load_corpus(path: str | Path, *, seed: int = 918) -> CorpusTokens:
    inputs = _resolve_corpus_inputs(path)
    if len(inputs) == 1 and inputs[0].is_dir():
        directory = inputs[0]
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("format") != PACKED_CORPUS_FORMAT:
            raise ValueError(f"unsupported packed corpus format: {manifest.get('format')!r}")
        token_count = int(manifest["token_count"])
        record_count = int(manifest["record_count"])
        token_array = np.memmap(directory / "tokens.u32", dtype=np.uint32, mode="r", shape=(token_count,))
        return CorpusTokens(
            token_array=token_array,
            token_count=token_count,
            record_count=record_count,
            source_kind="packed",
        )
    records = _load_raw_records(inputs, seed=seed)
    flat = [token for record in records for token in record]
    token_array = np.asarray(flat, dtype=np.uint32)
    return CorpusTokens(
        token_array=token_array,
        token_count=int(token_array.shape[0]),
        record_count=len(records),
        source_kind="jsonl",
    )


def pack_corpus(
    *,
    corpus_path: str | Path,
    output_dir: str | Path,
    seed: int = 918,
) -> dict[str, Any]:
    corpus = load_corpus(corpus_path, seed=seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    token_file = output / "tokens.u32"
    manifest_file = output / "manifest.json"
    np.asarray(corpus.token_array, dtype=np.uint32).tofile(token_file)
    manifest = {
        "format": PACKED_CORPUS_FORMAT,
        "token_count": corpus.token_count,
        "record_count": corpus.record_count,
        "dtype": "uint32",
        "seed": seed,
        "source_kind": corpus.source_kind,
        "token_file": token_file.name,
    }
    manifest_file.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "output_dir": str(output),
        "token_file": str(token_file),
        "manifest": str(manifest_file),
        "token_count": corpus.token_count,
        "record_count": corpus.record_count,
    }


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def init_distributed(requested_device: str) -> tuple[DistributedState, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    initialized_here = False
    if world_size <= 1:
        return DistributedState(enabled=False), resolve_device(requested_device)
    backend = "nccl" if torch.cuda.is_available() and requested_device != "cpu" else "gloo"
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
        initialized_here = True
    if backend == "nccl":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return (
        DistributedState(
            enabled=True,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            backend=backend,
            initialized_here=initialized_here,
        ),
        device,
    )


def cleanup_distributed(state: DistributedState) -> None:
    if state.enabled and state.initialized_here and dist.is_initialized():
        dist.destroy_process_group()


def resolve_model_config(
    *,
    vocabulary: SemanticVocabulary,
    preset: str,
    context_length: int | None = None,
    dimension: int | None = None,
    layers: int | None = None,
    heads: int | None = None,
) -> ModelConfig:
    if preset not in MODEL_PRESETS:
        supported = ", ".join(sorted(MODEL_PRESETS))
        raise ValueError(f"unsupported model preset {preset!r}; choose: {supported}")
    defaults = MODEL_PRESETS[preset]
    return ModelConfig(
        vocab_size=vocabulary.model_vocab_size,
        context_length=context_length or defaults["context_length"],
        dimension=dimension or defaults["dimension"],
        layers=layers or defaults["layers"],
        heads=heads or defaults["heads"],
    )


def _scheduler_lambda(
    step: int,
    *,
    total_steps: int,
    warmup_steps: int,
    min_learning_rate_ratio: float,
) -> float:
    if total_steps <= 1:
        return 1.0
    if warmup_steps > 0 and step < warmup_steps:
        return max(step + 1, 1) / warmup_steps
    progress_steps = max(total_steps - warmup_steps, 1)
    progress = min(max((step - warmup_steps) / progress_steps, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_learning_rate_ratio + (1.0 - min_learning_rate_ratio) * cosine


def _unwrap_model(model: SemanticGlyphLM | DistributedDataParallel) -> SemanticGlyphLM:
    return model.module if isinstance(model, DistributedDataParallel) else model


def _parameter_count(model: SemanticGlyphLM | DistributedDataParallel) -> int:
    return _unwrap_model(model).parameter_count()


def _evaluate(
    model: SemanticGlyphLM | DistributedDataParallel,
    dataset: Dataset[tuple[torch.Tensor, torch.Tensor]],
    *,
    batch_size: int,
    device: torch.device,
    use_amp: bool,
    num_workers: int = 0,
) -> dict[str, float]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    losses: list[float] = []
    model.eval()
    with torch.inference_mode():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
                enabled=use_amp,
            ):
                _, validation_loss = model(inputs, targets=targets)
            assert validation_loss is not None
            losses.append(float(validation_loss.detach().cpu()))
    mean_loss = sum(losses) / len(losses)
    return {"loss": mean_loss, "perplexity": math.exp(min(mean_loss, 20))}


def _make_run_state(
    *,
    train_config: TrainConfig,
    model_config: ModelConfig,
    step: int,
    optimizer_steps: int,
    corpus: CorpusTokens,
    dataset: TokenBlocks,
    training_size: int,
    validation_size: int,
    losses: list[float],
    validation_metrics: dict[str, float],
    parameter_count: int,
    elapsed_seconds: float,
    device: torch.device,
    distributed_state: DistributedState,
) -> dict[str, Any]:
    return {
        "preset": train_config.preset,
        "steps": train_config.steps,
        "batch_size": train_config.batch_size,
        "learning_rate": train_config.learning_rate,
        "min_learning_rate_ratio": train_config.min_learning_rate_ratio,
        "warmup_steps": train_config.warmup_steps,
        "gradient_accumulation_steps": train_config.gradient_accumulation_steps,
        "weight_decay": train_config.weight_decay,
        "seed": train_config.seed,
        "device": str(device),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "minimum_loss": min(losses),
        "validation_loss": validation_metrics["loss"],
        "validation_perplexity": validation_metrics["perplexity"],
        "parameter_count": parameter_count,
        "training_tokens": corpus.token_count,
        "dataset_blocks": len(dataset),
        "training_blocks": training_size,
        "validation_blocks": validation_size,
        "record_count": corpus.record_count,
        "corpus_source_kind": corpus.source_kind,
        "completed_steps": step,
        "optimizer_steps": optimizer_steps,
        "model_config": asdict(model_config),
        "distributed": {
            "enabled": distributed_state.enabled,
            "world_size": distributed_state.world_size,
            "backend": distributed_state.backend,
        },
    }


def _maybe_broadcast_metrics(metrics: dict[str, float], distributed_state: DistributedState) -> dict[str, float]:
    if not distributed_state.enabled:
        return metrics
    payloads = [metrics if distributed_state.is_main else None]
    dist.broadcast_object_list(payloads, src=0)
    return payloads[0]


def train(
    *,
    vocabulary_path: str | Path,
    corpus_path: str | Path,
    checkpoint_path: str | Path,
    steps: int = 500,
    batch_size: int = 16,
    context_length: int | None = None,
    dimension: int | None = None,
    layers: int | None = None,
    heads: int | None = None,
    learning_rate: float = 3e-4,
    min_learning_rate_ratio: float = 0.1,
    warmup_steps: int = 50,
    gradient_accumulation_steps: int = 1,
    weight_decay: float = 0.1,
    device_name: str = "auto",
    seed: int = 918,
    log_every: int = 10,
    validation_fraction: float = 0.05,
    stride: int | None = None,
    preset: str = "reference",
    resume_from: str | Path | None = None,
    checkpoint_every: int = 0,
    compile_model: bool = False,
    num_workers: int = 0,
) -> dict[str, Any]:
    train_config = TrainConfig(
        vocabulary_path=vocabulary_path,
        corpus_path=corpus_path,
        checkpoint_path=checkpoint_path,
        steps=steps,
        batch_size=batch_size,
        context_length=context_length,
        dimension=dimension,
        layers=layers,
        heads=heads,
        learning_rate=learning_rate,
        min_learning_rate_ratio=min_learning_rate_ratio,
        warmup_steps=warmup_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        weight_decay=weight_decay,
        device_name=device_name,
        seed=seed,
        log_every=log_every,
        validation_fraction=validation_fraction,
        stride=stride,
        preset=preset,
        resume_from=resume_from,
        checkpoint_every=checkpoint_every,
        compile_model=compile_model,
        num_workers=num_workers,
    )
    train_config.validate()

    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    distributed_state, device = init_distributed(device_name)
    try:
        vocabulary = SemanticVocabulary.load(vocabulary_path)
        model_config = resolve_model_config(
            vocabulary=vocabulary,
            preset=preset,
            context_length=context_length,
            dimension=dimension,
            layers=layers,
            heads=heads,
        )
        corpus = load_corpus(corpus_path, seed=seed)
        dataset = TokenBlocks(corpus, model_config.context_length, stride=stride)
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

        training_sampler = None
        if distributed_state.enabled:
            training_sampler = DistributedSampler(
                training_dataset,
                num_replicas=distributed_state.world_size,
                rank=distributed_state.rank,
                shuffle=True,
                seed=seed,
                drop_last=len(training_dataset) >= batch_size,
            )
        loader = DataLoader(
            training_dataset,
            batch_size=batch_size,
            shuffle=training_sampler is None,
            sampler=training_sampler,
            drop_last=len(training_dataset) >= batch_size and training_sampler is None,
            generator=generator if training_sampler is None else None,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
        )

        model: SemanticGlyphLM | DistributedDataParallel = SemanticGlyphLM(model_config).to(device)
        if compile_model and hasattr(torch, "compile") and not distributed_state.enabled:
            model = torch.compile(model)  # type: ignore[assignment]
        optimizer = torch.optim.AdamW(
            _unwrap_model(model).parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: _scheduler_lambda(
                step,
                total_steps=steps,
                warmup_steps=warmup_steps,
                min_learning_rate_ratio=min_learning_rate_ratio,
            ),
        )
        use_amp = device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        resume_path = Path(resume_from) if resume_from else None
        start_step = 0
        optimizer_steps = 0
        resumed_elapsed = 0.0
        if resume_path is not None:
            loaded_model, payload = load_checkpoint(
                resume_path,
                device=device,
                expected_vocabulary_sha256=vocabulary.digest,
            )
            _unwrap_model(model).load_state_dict(loaded_model.state_dict())
            if payload.get("config") != asdict(model_config):
                raise ValueError("resume checkpoint model config does not match requested training config")
            if "optimizer_state_dict" in payload:
                optimizer.load_state_dict(payload["optimizer_state_dict"])
            if "scheduler_state_dict" in payload:
                scheduler.load_state_dict(payload["scheduler_state_dict"])
            if "scaler_state_dict" in payload and use_amp:
                scaler.load_state_dict(payload["scaler_state_dict"])
            training_state = payload.get("training_state", {})
            start_step = int(training_state.get("step", payload.get("training", {}).get("completed_steps", 0)))
            optimizer_steps = int(training_state.get("optimizer_steps", payload.get("training", {}).get("optimizer_steps", 0)))
            resumed_elapsed = float(payload.get("training", {}).get("elapsed_seconds", 0.0))
            if start_step >= steps:
                raise ValueError("resume checkpoint has already completed the requested number of steps")

        if distributed_state.enabled:
            model = DistributedDataParallel(
                _unwrap_model(model),
                device_ids=[device.index] if device.type == "cuda" else None,
                output_device=device.index if device.type == "cuda" else None,
            )

        iterator = iter(loader)
        losses: list[float] = []
        start_time = time.monotonic()
        epoch_index = 0
        model.train()

        for step in range(start_step + 1, steps + 1):
            optimizer.zero_grad(set_to_none=True)
            micro_losses: list[float] = []
            for _ in range(gradient_accumulation_steps):
                try:
                    inputs, targets = next(iterator)
                except StopIteration:
                    epoch_index += 1
                    if training_sampler is not None:
                        training_sampler.set_epoch(seed + epoch_index)
                    iterator = iter(loader)
                    inputs, targets = next(iterator)
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
                    enabled=use_amp,
                ):
                    _, loss = model(inputs, targets=targets)
                    assert loss is not None
                    scaled_loss = loss / gradient_accumulation_steps
                scaler.scale(scaled_loss).backward()
                micro_losses.append(float(loss.detach().cpu()))
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(_unwrap_model(model).parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer_steps += 1
            loss_value = sum(micro_losses) / len(micro_losses)
            losses.append(loss_value)
            if distributed_state.is_main and (step == start_step + 1 or step % log_every == 0 or step == steps):
                elapsed = resumed_elapsed + (time.monotonic() - start_time)
                tokens_per_second = (
                    max(step - start_step, 1)
                    * batch_size
                    * model_config.context_length
                    * gradient_accumulation_steps
                    * max(distributed_state.world_size, 1)
                    / max(time.monotonic() - start_time, 1e-9)
                )
                print(
                    json.dumps(
                        {
                            "step": step,
                            "steps": steps,
                            "loss": round(loss_value, 6),
                            "perplexity": round(math.exp(min(loss_value, 20)), 3),
                            "tokens_per_second": round(tokens_per_second, 1),
                            "learning_rate": round(float(optimizer.param_groups[0]["lr"]), 10),
                            "optimizer_steps": optimizer_steps,
                            "device": str(device),
                            "elapsed_seconds": round(elapsed, 3),
                            "world_size": distributed_state.world_size,
                        }
                    ),
                    flush=True,
                )
            if checkpoint_every and step < steps and step % checkpoint_every == 0:
                if distributed_state.enabled:
                    dist.barrier()
                validation_metrics = _evaluate(
                    model,
                    validation_dataset,
                    batch_size=batch_size,
                    device=device,
                    use_amp=use_amp,
                    num_workers=num_workers,
                ) if distributed_state.is_main else {"loss": 0.0, "perplexity": 0.0}
                validation_metrics = _maybe_broadcast_metrics(validation_metrics, distributed_state)
                if distributed_state.is_main:
                    training = _make_run_state(
                        train_config=train_config,
                        model_config=model_config,
                        step=step,
                        optimizer_steps=optimizer_steps,
                        corpus=corpus,
                        dataset=dataset,
                        training_size=training_size,
                        validation_size=validation_size,
                        losses=losses,
                        validation_metrics=validation_metrics,
                        parameter_count=_parameter_count(model),
                        elapsed_seconds=resumed_elapsed + (time.monotonic() - start_time),
                        device=device,
                        distributed_state=distributed_state,
                    )
                    step_checkpoint = Path(checkpoint_path).with_name(
                        f"{Path(checkpoint_path).stem}.step{step}{Path(checkpoint_path).suffix}"
                    )
                    save_checkpoint(
                        step_checkpoint,
                        _unwrap_model(model),
                        vocabulary_sha256=vocabulary.digest,
                        training=training,
                        optimizer_state_dict=optimizer.state_dict(),
                        scheduler_state_dict=scheduler.state_dict(),
                        scaler_state_dict=scaler.state_dict() if use_amp else None,
                        training_state={"step": step, "optimizer_steps": optimizer_steps},
                    )

        if distributed_state.enabled:
            dist.barrier()
        validation_metrics = _evaluate(
            model,
            validation_dataset,
            batch_size=batch_size,
            device=device,
            use_amp=use_amp,
            num_workers=num_workers,
        ) if distributed_state.is_main else {"loss": 0.0, "perplexity": 0.0}
        validation_metrics = _maybe_broadcast_metrics(validation_metrics, distributed_state)
        training = _make_run_state(
            train_config=train_config,
            model_config=model_config,
            step=steps,
            optimizer_steps=optimizer_steps,
            corpus=corpus,
            dataset=dataset,
            training_size=training_size,
            validation_size=validation_size,
            losses=losses,
            validation_metrics=validation_metrics,
            parameter_count=_parameter_count(model),
            elapsed_seconds=resumed_elapsed + (time.monotonic() - start_time),
            device=device,
            distributed_state=distributed_state,
        )
        if distributed_state.is_main:
            save_checkpoint(
                checkpoint_path,
                _unwrap_model(model),
                vocabulary_sha256=vocabulary.digest,
                training=training,
                optimizer_state_dict=optimizer.state_dict(),
                scheduler_state_dict=scheduler.state_dict(),
                scaler_state_dict=scaler.state_dict() if use_amp else None,
                training_state={"step": steps, "optimizer_steps": optimizer_steps},
            )
        return training
    finally:
        cleanup_distributed(distributed_state)


def evaluate_checkpoint(
    *,
    vocabulary_path: str | Path,
    corpus_path: str | Path,
    checkpoint_path: str | Path,
    batch_size: int = 16,
    validation_fraction: float = 0.05,
    stride: int | None = None,
    device_name: str = "auto",
    seed: int = 918,
    num_workers: int = 0,
) -> dict[str, Any]:
    if not 0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be between 0 and 0.5")
    distributed_state, device = init_distributed(device_name)
    try:
        vocabulary = SemanticVocabulary.load(vocabulary_path)
        model, payload = load_checkpoint(
            checkpoint_path,
            device=device,
            expected_vocabulary_sha256=vocabulary.digest,
        )
        corpus = load_corpus(corpus_path, seed=seed)
        dataset = TokenBlocks(corpus, model.config.context_length, stride=stride)
        validation_size = max(1, round(len(dataset) * validation_fraction))
        training_size = len(dataset) - validation_size
        if training_size < 1:
            raise ValueError("corpus does not contain enough blocks for validation")
        generator = torch.Generator().manual_seed(seed)
        _, validation_dataset = random_split(
            dataset,
            [training_size, validation_size],
            generator=generator,
        )
        use_amp = device.type == "cuda"
        metrics = _evaluate(
            model,
            validation_dataset,
            batch_size=batch_size,
            device=device,
            use_amp=use_amp,
            num_workers=num_workers,
        )
        return {
            "checkpoint": str(checkpoint_path),
            "device": str(device),
            "validation_loss": metrics["loss"],
            "validation_perplexity": metrics["perplexity"],
            "parameter_count": model.parameter_count(),
            "context_length": model.config.context_length,
            "validation_blocks": len(validation_dataset),
            "dataset_blocks": len(dataset),
            "record_count": corpus.record_count,
            "corpus_source_kind": corpus.source_kind,
            "checkpoint_training": payload.get("training", {}),
        }
    finally:
        cleanup_distributed(distributed_state)
