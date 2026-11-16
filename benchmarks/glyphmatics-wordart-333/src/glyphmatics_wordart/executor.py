# Copyright 2026 918 Technologies
# SPDX-License-Identifier: Apache-2.0
"""Deterministic graph executor and runtime validation for glyph programs."""

from __future__ import annotations

from dataclasses import dataclass

from .codec import GlyphTransportCodec
from .models import GlyphProgram


_ROLE_TOKENS = frozenset({"OBJECT", "AGENT", "TARGET", "OBSTACLE", "COMPONENT"})
_RELATION_TOKENS = frozenset(
    {"ADJACENT", "CONNECTED", "CONTAIN", "OVERLAP", "CAUSE", "EFFECT", "BEFORE", "AFTER", "SEQUENCE"}
)


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    valid: bool
    round_trip: bool
    acyclic: bool
    role_binding: bool
    coverage: float
    execution_score: float
    trace: tuple[str, ...]
    errors: tuple[str, ...]


class GlyphGraphExecutor:
    """Execute the declarative portion of a GlyphMatics program.

    Execution means canonical decoding, edge validation, dependency traversal,
    and semantic-role binding. It does not claim to execute arbitrary external
    capabilities; unsupported or cyclic graphs fail closed.
    """

    def __init__(self, codec: GlyphTransportCodec | None = None) -> None:
        self.codec = codec or GlyphTransportCodec()

    @staticmethod
    def _topological_order(program: GlyphProgram) -> tuple[tuple[int, ...], bool]:
        adjacency: dict[int, list[int]] = {index: [] for index in range(len(program.tokens))}
        indegree = [0] * len(program.tokens)
        for edge in program.edges:
            adjacency[edge.source].append(edge.target)
            indegree[edge.target] += 1
        ready = sorted(index for index, degree in enumerate(indegree) if degree == 0)
        order: list[int] = []
        while ready:
            node = ready.pop(0)
            order.append(node)
            for target in sorted(adjacency[node]):
                indegree[target] -= 1
                if indegree[target] == 0:
                    ready.append(target)
                    ready.sort()
        return tuple(order), len(order) == len(program.tokens)

    def execute(self, program: GlyphProgram, transport: str | None = None) -> ExecutionResult:
        errors: list[str] = []
        trace: list[str] = []
        payload = transport if transport is not None else self.codec.encode_program(program)
        try:
            round_trip = self.codec.verify_program(program, payload)
        except (KeyError, TypeError, ValueError) as exc:
            round_trip = False
            errors.append(f"transport:{exc}")
        trace.append(f"decode={'PASS' if round_trip else 'FAIL'}")

        order, acyclic = self._topological_order(program)
        has_cycle_marker = bool({"LOOP", "CYCLE"}.intersection(program.tokens))
        if not acyclic and not has_cycle_marker:
            errors.append("graph:cycle_without_LOOP_or_CYCLE")
        effective_acyclic = acyclic or has_cycle_marker
        trace.append("order=" + ",".join(str(index) for index in order))

        role_indexes = {index for index, token in enumerate(program.tokens) if token in _ROLE_TOKENS}
        binding_edges = {
            index
            for edge in program.edges
            if edge.relation in _RELATION_TOKENS
            for index in (edge.source, edge.target)
        }
        role_binding = bool(role_indexes and role_indexes.intersection(binding_edges))
        if not role_binding:
            errors.append("semantics:no_bound_role_node")
        trace.append(f"role_binding={'PASS' if role_binding else 'FAIL'}")

        referenced = {index for edge in program.edges for index in (edge.source, edge.target)}
        structural = {
            index
            for index, token in enumerate(program.tokens)
            if token not in {"BEGIN", "END", "COMMIT", "RETURN", "HALT"}
        }
        coverage = len(referenced.intersection(structural)) / max(1, len(structural))
        schema_valid = bool(program.tokens) and all(0 <= edge.source < len(program.tokens) for edge in program.edges)
        score = (
            0.25 * float(schema_valid)
            + 0.25 * float(round_trip)
            + 0.20 * float(effective_acyclic)
            + 0.20 * float(role_binding)
            + 0.10 * coverage
        )
        valid = schema_valid and round_trip and effective_acyclic and role_binding
        trace.append(f"coverage={coverage:.6f}")
        trace.append(f"commit={'PASS' if valid else 'REJECT'}")
        return ExecutionResult(
            valid=valid,
            round_trip=round_trip,
            acyclic=effective_acyclic,
            role_binding=role_binding,
            coverage=round(coverage, 6),
            execution_score=round(score, 6),
            trace=tuple(trace),
            errors=tuple(errors),
        )

