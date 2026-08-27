# Copyright 2026 918 Technologies
# SPDX-License-Identifier: Apache-2.0
"""Transparent scoring for Word Art and GlyphMatics-native properties."""

from __future__ import annotations

from collections import Counter

from .art_rules import answer_matches, art_digest, normalize_art, robustness_variants
from .codec import GlyphTransportCodec
from .executor import ExecutionResult, GlyphGraphExecutor
from .models import BenchmarkCase, BenchmarkConfig, GlyphProgram, ScoreBreakdown


def _set_f1(expected: tuple[str, ...], observed: tuple[str, ...]) -> float:
    expected_set = set(expected)
    observed_set = set(observed)
    if not expected_set and not observed_set:
        return 1.0
    if not expected_set or not observed_set:
        return 0.0
    true_positive = len(expected_set.intersection(observed_set))
    precision = true_positive / len(observed_set)
    recall = true_positive / len(expected_set)
    return 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)


class CompositeScorer:
    def __init__(
        self,
        config: BenchmarkConfig | None = None,
        codec: GlyphTransportCodec | None = None,
        executor: GlyphGraphExecutor | None = None,
    ) -> None:
        self.config = config or BenchmarkConfig()
        self.codec = codec or GlyphTransportCodec()
        self.executor = executor or GlyphGraphExecutor(self.codec)

    def correct_attempt(self, case: BenchmarkCase, guesses: tuple[str, ...]) -> int | None:
        for index, guess in enumerate(guesses[: self.config.max_attempts], start=1):
            if answer_matches(guess, case.accepted_answers):
                return index
        return None

    def word_art_points(self, correct_attempt: int | None) -> float:
        if correct_attempt is None or correct_attempt > len(self.config.guess_points):
            return 0.0
        return float(self.config.guess_points[correct_attempt - 1])

    @staticmethod
    def compression_score(program: GlyphProgram, transport: str) -> float:
        baseline_characters = max(1, len(program.canonical_bytes().decode("utf-8")))
        logical_transport_units = len(transport) + 3 * len(program.edges)
        return round(max(0.0, min(1.0, 1.0 - logical_transport_units / baseline_characters)), 6)

    @staticmethod
    def robustness_score(art: str) -> float:
        variants = robustness_variants(art)
        reference = normalize_art(art)
        invariant = sum(normalize_art(variant) == reference for variant in variants) / max(1, len(variants))
        nonempty_lines = [line for line in reference.splitlines() if line.strip()]
        density_ok = bool(nonempty_lines) and max(len(line) for line in nonempty_lines) >= 3
        return round(0.8 * invariant + 0.2 * float(density_ok), 6)

    def score_case(
        self,
        case: BenchmarkCase,
        art: str,
        guesses: tuple[str, ...],
        program: GlyphProgram | None,
        collision_count: int = 1,
        execution: ExecutionResult | None = None,
    ) -> ScoreBreakdown:
        attempt = self.correct_attempt(case, guesses)
        points = self.word_art_points(attempt)
        semantic = points / max(self.config.guess_points) if self.config.guess_points else 0.0
        if program is None:
            graph = 0.0
            round_trip = 0.0
            compression = 0.0
            execution_score = 0.0
        else:
            transport = self.codec.encode_program(program)
            runtime = execution or self.executor.execute(program, transport)
            graph = _set_f1(case.expected_tokens, program.tokens)
            round_trip = float(runtime.round_trip)
            compression = self.compression_score(program, transport)
            execution_score = runtime.execution_score
        collision = 0.0 if collision_count <= 1 else min(1.0, (collision_count - 1) / 2.0)
        return ScoreBreakdown(
            semantic_fidelity=semantic,
            graph_fidelity=round(graph, 6),
            robustness=self.robustness_score(art),
            round_trip=round_trip,
            compression=compression,
            execution=execution_score,
            collision_penalty=collision,
            word_art_points=points,
            correct_attempt=attempt,
        )

    def collision_counts(self, arts: tuple[str, ...]) -> dict[str, int]:
        return dict(Counter(art_digest(art) for art in arts))

