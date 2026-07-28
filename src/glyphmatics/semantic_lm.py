"""Small decoder-only Transformer for semantic glyph sequences.

This is a trainable reference model, not a claim of frontier-scale pretraining.
Its vocabulary is the exact multilingual glyph vocabulary produced by
``semantic_data``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    context_length: int = 256
    dimension: int = 256
    layers: int = 6
    heads: int = 8
    feed_forward_multiplier: int = 4
    dropout: float = 0.1

    def validate(self) -> None:
        if self.vocab_size < 5:
            raise ValueError("vocab_size must include semantic and special tokens")
        if self.dimension % self.heads:
            raise ValueError("dimension must be divisible by heads")
        if self.context_length < 2 or self.layers < 1:
            raise ValueError("invalid context length or layer count")


class SemanticGlyphLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.dimension)
        self.position_embedding = nn.Embedding(config.context_length, config.dimension)
        layer = nn.TransformerEncoderLayer(
            d_model=config.dimension,
            nhead=config.heads,
            dim_feedforward=config.dimension * config.feed_forward_multiplier,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=config.layers,
            norm=nn.LayerNorm(config.dimension),
            enable_nested_tensor=False,
        )
        self.output = nn.Linear(config.dimension, config.vocab_size, bias=False)
        self.output.weight = self.token_embedding.weight
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        _, sequence_length = input_ids.shape
        if sequence_length > self.config.context_length:
            raise ValueError("sequence exceeds configured context length")
        positions = torch.arange(sequence_length, device=input_ids.device)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        causal_mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                dtype=torch.bool,
                device=input_ids.device,
            ),
            diagonal=1,
        )
        hidden = self.transformer(hidden, mask=causal_mask, is_causal=True)
        logits = self.output(hidden)
        loss = None
        if targets is not None:
            if targets.shape != input_ids.shape:
                raise ValueError("targets must have the same shape as input_ids")
            loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                targets.reshape(-1),
            )
        return logits, loss

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        eos_id: int = 2,
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> torch.Tensor:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.eval()
        output = input_ids
        for _ in range(max_new_tokens):
            context = output[:, -self.config.context_length:]
            logits, _ = self(context)
            next_logits = logits[:, -1, :] / temperature
            if 0 < top_k < next_logits.shape[-1]:
                threshold = torch.topk(next_logits, top_k).values[:, -1, None]
                next_logits = next_logits.masked_fill(next_logits < threshold, float("-inf"))
            next_token = torch.multinomial(F.softmax(next_logits, dim=-1), 1)
            output = torch.cat((output, next_token), dim=1)
            if torch.all(next_token == eos_id):
                break
        return output

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def save_checkpoint(
    path: str | Path,
    model: SemanticGlyphLM,
    *,
    vocabulary_sha256: str,
    training: dict[str, Any],
    optimizer_state_dict: dict[str, Any] | None = None,
    scheduler_state_dict: dict[str, Any] | None = None,
    scaler_state_dict: dict[str, Any] | None = None,
    training_state: dict[str, Any] | None = None,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "glyphmatics-semantic-lm-v2",
        "config": asdict(model.config),
        "vocabulary_sha256": vocabulary_sha256,
        "training": training,
        "state_dict": model.state_dict(),
    }
    if optimizer_state_dict is not None:
        payload["optimizer_state_dict"] = optimizer_state_dict
    if scheduler_state_dict is not None:
        payload["scheduler_state_dict"] = scheduler_state_dict
    if scaler_state_dict is not None:
        payload["scaler_state_dict"] = scaler_state_dict
    if training_state is not None:
        payload["training_state"] = training_state
    torch.save(payload, target)


def load_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
    expected_vocabulary_sha256: str | None = None,
) -> tuple[SemanticGlyphLM, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if payload.get("format") not in {"glyphmatics-semantic-lm-v1", "glyphmatics-semantic-lm-v2"}:
        raise ValueError("unsupported semantic model checkpoint")
    digest = payload.get("vocabulary_sha256")
    if expected_vocabulary_sha256 and digest != expected_vocabulary_sha256:
        raise ValueError("checkpoint vocabulary does not match codec vocabulary")
    model = SemanticGlyphLM(ModelConfig(**payload["config"]))
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    return model, payload
