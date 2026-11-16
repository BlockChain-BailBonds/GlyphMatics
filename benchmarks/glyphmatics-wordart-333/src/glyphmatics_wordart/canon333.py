# Copyright 2026 918 Technologies
# SPDX-License-Identifier: Apache-2.0
"""Exact immutable GlyphMatics 333-token canon.

The ordering and token spellings intentionally match the production
``ARC3_PRODUCTION_RDL_ADL_GHOSTBRIDGE_GLYPH333_QWEN38`` artifact.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass


_OBSERVATION = (
    "STATE", "GRID", "CELL", "COLOR", "OBJECT", "AGENT", "TARGET", "OBSTACLE",
    "SCORE", "REWARD", "LEVEL", "TERMINAL", "LEGAL_ACTIONS", "POSITION", "COUNTER", "EVENT",
    "RNG_STATE", "METADATA", "FRAME", "MOTION", "CHANGE", "NO_CHANGE", "PROGRESS", "WIN",
    "LOSS", "UNKNOWN", "NOVELTY", "HISTORY", "MEMORY", "HYPOTHESIS", "CONFIDENCE", "PROVENANCE",
)

_TOPOLOGY = (
    "BOUNDARY", "NORTH", "SOUTH", "EAST", "WEST", "NORTHEAST", "NORTHWEST", "SOUTHEAST",
    "SOUTHWEST", "CENTER", "INSIDE", "OUTSIDE", "ADJACENT", "CONNECTED", "DISCONNECTED", "WRAP",
    "CLAMP", "REFLECT", "BLOCK", "PORTAL", "ONE_WAY", "TOROIDAL", "EDGE", "CORNER",
    "ROW", "COLUMN", "REGION", "COMPONENT", "PATH", "FRONTIER", "PREDECESSOR", "SUCCESSOR",
    "INTERSECTION", "OVERLAP", "CONTAIN", "HOLE", "LINE", "RECTANGLE", "SYMMETRY", "ROTATE90",
    "ROTATE180", "ROTATE270", "FLIP_H", "FLIP_V", "TRANSLATE", "SCALE", "DISTANCE", "REACHABLE",
)

_ACTION = (
    "MOVE", "MOVE_NORTH", "MOVE_SOUTH", "MOVE_EAST", "MOVE_WEST", "CLICK", "SELECT", "TOGGLE",
    "PUSH", "PULL", "PICKUP", "DROP", "PAINT", "ERASE", "ROTATE", "SWAP",
    "COPY", "MERGE", "SPLIT", "WAIT", "RESET", "CONFIRM", "CANCEL", "OPEN",
    "CLOSE", "ACTIVATE", "DEACTIVATE", "INCREMENT", "DECREMENT", "SET", "CLEAR", "PROBE",
    "VERIFY", "RETRY", "AVOID", "EXPLORE", "EXPLOIT", "BRANCH", "BACKTRACK", "COMMIT",
    "RESTORE", "CLONE", "SERIALIZE", "DESERIALIZE", "OBSERVE", "PREDICT", "PLAN", "EXECUTE",
)

_CAUSAL_TEMPORAL = (
    "CAUSE", "EFFECT", "PRECONDITION", "POSTCONDITION", "NECESSARY", "SUFFICIENT", "CORRELATED", "CAUSAL",
    "CONTRADICTED", "FALSIFIED", "SUPPORTED", "DELAYED", "IMMEDIATE", "BEFORE", "AFTER", "DURING",
    "ORDERED", "UNORDERED", "PHASE", "TICK", "LOOP", "STALL", "CYCLE", "TRANSITION",
    "TRAJECTORY", "SEQUENCE", "FIRST", "LAST", "REPEAT", "ONCE", "MULTI_STEP", "BRIDGE",
    "MISSING_BRIDGE", "PHANTOM_BRIDGE", "REVERSE_PATH", "FORWARD_PATH", "TWIN", "COUNTERFACTUAL", "ABLATION", "INTERVENTION",
    "CONTROL", "VARIANT", "BASELINE", "DELTA", "OUTCOME", "UTILITY", "RISK", "COST",
)

_RUNTIME = (
    "RDL_DIVISION", "RDL_ROUNDING", "RDL_NEGATIVE_INDEXING", "RDL_SLICE_BOUNDARIES",
    "RDL_ITERATOR_SEMANTICS", "RDL_ORDERING", "RDL_COPY_SEMANTICS", "RDL_NUMERIC_DTYPE",
    "RDL_OVERFLOW", "RDL_BOOL_INT_COERCION", "RDL_STRING_BYTES", "RDL_RNG",
    "RDL_SERIALIZATION", "RDL_MUTATION_TIMING", "RDL_EXCEPTIONS_DEFAULTS", "RDL_LIBRARY_BEHAVIOR",
    "FLOOR_DIV", "TRUE_DIV", "TRUNC_DIV", "ROUND_FLOOR", "ROUND_CEIL", "ROUND_NEAREST", "ROUND_TRUNC", "NEGATIVE_INDEX",
    "SLICE_INCLUSIVE", "SLICE_EXCLUSIVE", "ITERATOR_LAZY", "ITERATOR_MATERIALIZED", "ITERATOR_REUSED", "ORDER_INSERTION", "ORDER_SORTED", "ORDER_REVERSE",
    "COPY_REFERENCE", "COPY_SHALLOW", "COPY_DEEP", "DTYPE_SIGNED", "DTYPE_UNSIGNED", "DTYPE_NARROW", "DTYPE_WIDE", "OVERFLOW_WRAP",
    "OVERFLOW_SATURATE", "OVERFLOW_ERROR", "BOOL_STRICT", "INT_COERCE", "ENUM_VALUE", "TEXT_UTF8", "RAW_BYTES", "RNG_SEED",
    "RNG_CONSUME", "RNG_STREAM", "SERIALIZE_STRICT", "SERIALIZE_DEFAULT", "FIELD_MISSING", "MUTATE_BEFORE", "MUTATE_AFTER", "MUTATE_STAGED",
    "EXCEPTION_RAISE", "DEFAULT_VALUE", "CONTINUE_ON_ERROR", "LIBRARY_COMPAT", "LIBRARY_STRICT", "SEMANTIC_PROFILE", "RUNTIME_CONTEXT", "SEMANTIC_DISTANCE",
)

_PLANNING = (
    "ADL", "DIFFERENCE_LEARNING", "WINNING_DIFFERENCE", "LOSING_DIFFERENCE", "CAPABILITY_DIFF", "RUNTIME_DIFF", "TRANSITION_DIFF", "OUTCOME_DIFF",
    "GHOSTBRIDGE", "NEGATIVE_SPACE", "CAPABILITY_ABSENCE", "MINIMUM_BRIDGE", "MECHANIC", "INVARIANT", "GENERALIZE", "TRANSFER",
    "GLYPH", "GLYPH333", "ORACLE", "MULTIVERSE", "WORLD", "WORLD_BRANCH", "POSTERIOR", "ENTROPY",
    "INFORMATION_GAIN", "PROBE_VALUE", "EXPECTED_VALUE", "WORST_CASE", "ROBUST_SCORE", "ACTION_BUDGET", "SOFT_STALL", "HARD_STALL",
    "NO_IMPACT", "SUCCESS_PROTECT", "TRIAGE", "DWE", "WEIGHT", "DECAY", "PRIOR", "BELIEF",
    "BAYES_UPDATE", "CONFIRM_MECHANIC", "INVALIDATE", "UNLEARN", "RETRIEVE", "COMPILE", "POLICY", "EXECUTOR",
)

_VALIDATION = (
    "FAIL_CLOSED", "EXACTLY_ONCE", "PRE_RECORD", "POST_RECORD", "ACTION_DEBT", "ZERO_DEBT", "SANITIZED", "PUBLIC_ONLY",
    "OBSERVATIONAL_ONLY", "SOURCE_ALLOWED", "SOURCE_FORBIDDEN", "PRIVATE_FORBIDDEN", "NETWORK_FORBIDDEN", "DETERMINISTIC", "SAME_SEED", "SAME_STATE",
    "SAME_ACTION", "REPRODUCIBLE", "REACHABILITY_TEST", "CAUSAL_TEST", "SCORE_TEST", "SCHEMA_TEST", "HASH_CHECK", "VERSION_CHECK",
    "MODEL_CHECK", "RUNTIME_CHECK", "INPUT_CHECK", "LEAKAGE_ZERO", "AUDIT", "PASS", "REJECT", "ERROR",
)

_META = ("BEGIN", "END", "TRUE", "FALSE", "NONE", "AND", "OR", "NOT", "IF", "THEN", "ELSE", "RETURN", "HALT")

_TOKENS = _OBSERVATION + _TOPOLOGY + _ACTION + _CAUSAL_TEMPORAL + _RUNTIME + _PLANNING + _VALIDATION + _META
if len(_TOKENS) != 333:
    raise RuntimeError(f"GlyphMatics canon must contain exactly 333 tokens, found {len(_TOKENS)}")
if len(set(_TOKENS)) != 333:
    raise RuntimeError("GlyphMatics canon contains duplicate token names")


@dataclass(frozen=True, slots=True)
class GlyphDefinition:
    glyph_id: int
    token: str
    category: str


class Canon333:
    """Fixed, deterministic 333-token mechanic vocabulary."""

    version = "glyphmatics-333-v1"

    def __init__(self) -> None:
        categories = (
            ("observation", _OBSERVATION),
            ("topology", _TOPOLOGY),
            ("action", _ACTION),
            ("causal_temporal", _CAUSAL_TEMPORAL),
            ("runtime", _RUNTIME),
            ("planning_learning", _PLANNING),
            ("validation_control", _VALIDATION),
            ("meta", _META),
        )
        definitions: list[GlyphDefinition] = []
        next_id = 1
        for category, tokens in categories:
            for token in tokens:
                definitions.append(GlyphDefinition(next_id, token, category))
                next_id += 1
        self._by_id = {definition.glyph_id: definition for definition in definitions}
        self._by_token = {definition.token: definition for definition in definitions}
        payload = [(item.glyph_id, item.token, item.category) for item in definitions]
        self._digest = hashlib.sha256(
            json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        ).hexdigest()

    @property
    def digest(self) -> str:
        return self._digest

    def by_id(self, glyph_id: int) -> GlyphDefinition:
        try:
            return self._by_id[int(glyph_id)]
        except (KeyError, ValueError, TypeError) as exc:
            raise KeyError(f"unknown GlyphMatics glyph id {glyph_id!r}") from exc

    def by_token(self, token: str) -> GlyphDefinition:
        key = str(token).strip().upper()
        try:
            return self._by_token[key]
        except KeyError as exc:
            raise KeyError(f"unknown GlyphMatics token {token!r}") from exc

    def definitions(self) -> tuple[GlyphDefinition, ...]:
        return tuple(self._by_id[index] for index in range(1, 334))

    def search(self, *keywords: str) -> tuple[GlyphDefinition, ...]:
        needles = tuple(str(keyword).upper() for keyword in keywords if str(keyword).strip())
        if not needles:
            return self.definitions()
        return tuple(
            definition
            for definition in self.definitions()
            if all(needle in definition.token for needle in needles)
        )


CANON333 = Canon333()

