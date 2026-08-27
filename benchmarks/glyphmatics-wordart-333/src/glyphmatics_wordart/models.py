# Copyright 2026 918 Technologies
# SPDX-License-Identifier: Apache-2.0
"""Immutable public data contracts for the bridge."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .canon333 import CANON333


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _unit_interval(name: str, value: float) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be finite and in [0,1], got {value!r}")
    return number


@dataclass(frozen=True, slots=True)
class GlyphEdge:
    source: int
    target: int
    relation: str

    def __post_init__(self) -> None:
        if self.source < 0 or self.target < 0:
            raise ValueError("edge indexes must be non-negative")
        canonical = CANON333.by_token(self.relation).token
        object.__setattr__(self, "relation", canonical)

    @classmethod
    def from_value(cls, value: Any) -> "GlyphEdge":
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            return cls(int(value["source"]), int(value["target"]), str(value["relation"]))
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 3:
            return cls(int(value[0]), int(value[1]), str(value[2]))
        raise TypeError("edge must be GlyphEdge, mapping, or [source,target,relation]")

    def to_list(self) -> list[Any]:
        return [self.source, self.target, self.relation]


@dataclass(frozen=True, slots=True)
class GlyphProgram:
    tokens: tuple[str, ...]
    edges: tuple[GlyphEdge, ...] = ()
    version: str = CANON333.version

    def __post_init__(self) -> None:
        if self.version != CANON333.version:
            raise ValueError(f"unsupported canon version {self.version!r}")
        if not self.tokens:
            raise ValueError("glyph program requires at least one token")
        if len(self.tokens) > 333:
            raise ValueError("glyph program cannot exceed 333 token occurrences")
        canonical_tokens = tuple(CANON333.by_token(token).token for token in self.tokens)
        canonical_edges = tuple(GlyphEdge.from_value(edge) for edge in self.edges)
        for edge in canonical_edges:
            if edge.source >= len(canonical_tokens) or edge.target >= len(canonical_tokens):
                raise ValueError(
                    f"edge ({edge.source},{edge.target}) is outside token range 0..{len(canonical_tokens)-1}"
                )
        object.__setattr__(self, "tokens", canonical_tokens)
        object.__setattr__(self, "edges", canonical_edges)

    @property
    def glyph_ids(self) -> tuple[int, ...]:
        return tuple(CANON333.by_token(token).glyph_id for token in self.tokens)

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def canonical_bytes(self) -> bytes:
        return stable_json(self.to_mapping()).encode("utf-8")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "edges": [edge.to_list() for edge in self.edges],
            "tokens": list(self.tokens),
            "version": self.version,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "GlyphProgram":
        allowed = {"version", "tokens", "edges"}
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown glyph program keys: {sorted(unknown)}")
        raw_tokens = value.get("tokens")
        if not isinstance(raw_tokens, list):
            raise TypeError("glyph program tokens must be a JSON list")
        raw_edges = value.get("edges", [])
        if not isinstance(raw_edges, list):
            raise TypeError("glyph program edges must be a JSON list")
        return cls(
            version=str(value.get("version", CANON333.version)),
            tokens=tuple(str(token) for token in raw_tokens),
            edges=tuple(GlyphEdge.from_value(edge) for edge in raw_edges),
        )


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    case_id: str
    word: str
    expected_tokens: tuple[str, ...]
    accepted_answers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        case_id = str(self.case_id).strip()
        word = str(self.word).strip().lower()
        if not case_id or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", case_id):
            raise ValueError(f"invalid case_id {self.case_id!r}")
        if not word or not re.fullmatch(r"[a-z]+", word):
            raise ValueError(f"word must contain only ASCII letters: {self.word!r}")
        expected = tuple(dict.fromkeys(CANON333.by_token(token).token for token in self.expected_tokens))
        if not expected:
            raise ValueError("expected_tokens cannot be empty")
        answers = tuple(
            dict.fromkeys(
                answer.strip().lower()
                for answer in (word, *self.accepted_answers)
                if str(answer).strip()
            )
        )
        if any(not re.fullmatch(r"[a-z]+", answer) for answer in answers):
            raise ValueError("accepted answers must be single ASCII words")
        object.__setattr__(self, "case_id", case_id)
        object.__setattr__(self, "word", word)
        object.__setattr__(self, "expected_tokens", expected)
        object.__setattr__(self, "accepted_answers", answers)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "BenchmarkCase":
        return cls(
            case_id=str(value["case_id"]),
            word=str(value["word"]),
            expected_tokens=tuple(str(x) for x in value["expected_tokens"]),
            accepted_answers=tuple(str(x) for x in value.get("accepted_answers", ())),
        )


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    max_artist_attempts: int = 2
    max_attempts: int = 3
    guess_points: tuple[float, ...] = (2.0, 1.5, 1.0)
    max_art_chars: int = 4000
    semantic_weight: float = 0.30
    graph_weight: float = 0.20
    robustness_weight: float = 0.15
    round_trip_weight: float = 0.15
    compression_weight: float = 0.10
    execution_weight: float = 0.10
    max_collision_penalty: float = 0.10

    def __post_init__(self) -> None:
        if not 1 <= int(self.max_artist_attempts) <= 5:
            raise ValueError("max_artist_attempts must be in [1,5]")
        if not 1 <= int(self.max_attempts) <= 10:
            raise ValueError("max_attempts must be in [1,10]")
        if len(self.guess_points) != self.max_attempts:
            raise ValueError("guess_points length must equal max_attempts")
        if any(not math.isfinite(float(point)) or float(point) < 0 for point in self.guess_points):
            raise ValueError("guess_points must be finite and non-negative")
        if self.max_art_chars < 32:
            raise ValueError("max_art_chars must be at least 32")
        weights = (
            self.semantic_weight,
            self.graph_weight,
            self.robustness_weight,
            self.round_trip_weight,
            self.compression_weight,
            self.execution_weight,
        )
        if any(not math.isfinite(float(weight)) or float(weight) < 0 for weight in weights):
            raise ValueError("component weights must be finite and non-negative")
        if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"component weights must sum to 1.0, got {sum(weights)}")
        _unit_interval("max_collision_penalty", self.max_collision_penalty)


@dataclass(frozen=True, slots=True)
class ScoreBreakdown:
    semantic_fidelity: float
    graph_fidelity: float
    robustness: float
    round_trip: float
    compression: float
    execution: float
    collision_penalty: float
    word_art_points: float
    correct_attempt: int | None

    def __post_init__(self) -> None:
        for field_name in (
            "semantic_fidelity", "graph_fidelity", "robustness", "round_trip",
            "compression", "execution", "collision_penalty",
        ):
            object.__setattr__(self, field_name, _unit_interval(field_name, getattr(self, field_name)))
        if not math.isfinite(float(self.word_art_points)) or self.word_art_points < 0:
            raise ValueError("word_art_points must be finite and non-negative")
        if self.correct_attempt is not None and self.correct_attempt < 1:
            raise ValueError("correct_attempt must be positive or None")

    def composite(self, config: BenchmarkConfig | None = None) -> float:
        cfg = config or BenchmarkConfig()
        weighted = (
            cfg.semantic_weight * self.semantic_fidelity
            + cfg.graph_weight * self.graph_fidelity
            + cfg.robustness_weight * self.robustness
            + cfg.round_trip_weight * self.round_trip
            + cfg.compression_weight * self.compression
            + cfg.execution_weight * self.execution
        )
        penalty = cfg.max_collision_penalty * self.collision_penalty
        return round(100.0 * max(0.0, min(1.0, weighted - penalty)), 6)

    def to_mapping(self, config: BenchmarkConfig | None = None) -> dict[str, Any]:
        return {
            "collision_penalty": self.collision_penalty,
            "composite": self.composite(config),
            "compression": self.compression,
            "correct_attempt": self.correct_attempt,
            "execution": self.execution,
            "graph_fidelity": self.graph_fidelity,
            "robustness": self.robustness,
            "round_trip": self.round_trip,
            "semantic_fidelity": self.semantic_fidelity,
            "word_art_points": self.word_art_points,
        }


@dataclass(frozen=True, slots=True)
class CaseResult:
    case_id: str
    target_hash: str
    art: str
    guesses: tuple[str, ...]
    glyph_program: GlyphProgram | None
    score: ScoreBreakdown
    failures: tuple[str, ...] = ()
    art_digest: str = ""
    transport: str = ""

    def to_mapping(self, config: BenchmarkConfig | None = None) -> dict[str, Any]:
        return {
            "art": self.art,
            "art_digest": self.art_digest,
            "case_id": self.case_id,
            "failures": list(self.failures),
            "glyph_program": self.glyph_program.to_mapping() if self.glyph_program else None,
            "guesses": list(self.guesses),
            "score": self.score.to_mapping(config),
            "target_hash": self.target_hash,
            "transport": self.transport,
        }


@dataclass(frozen=True, slots=True)
class CompositeResult:
    cases: tuple[CaseResult, ...]
    mean_composite: float
    total_word_art_points: float
    max_word_art_points: float
    canon_digest: str = field(default=CANON333.digest)
    schema: str = "glyphmatics.wordart.report.v1"

    def to_mapping(self, config: BenchmarkConfig | None = None) -> dict[str, Any]:
        return {
            "canon_digest": self.canon_digest,
            "case_count": len(self.cases),
            "cases": [case.to_mapping(config) for case in self.cases],
            "max_word_art_points": self.max_word_art_points,
            "mean_composite": self.mean_composite,
            "schema": self.schema,
            "total_word_art_points": self.total_word_art_points,
        }
