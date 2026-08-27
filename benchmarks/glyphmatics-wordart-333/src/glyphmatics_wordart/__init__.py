# Copyright 2026 918 Technologies
# SPDX-License-Identifier: Apache-2.0
"""GlyphMatics 333 bridge for Kaggle Word Art."""

from .art_rules import ArtVerdict, check_art, extract_art, extract_guess
from .benchmark import BenchmarkRunner, PromptModel, load_cases
from .canon333 import CANON333, Canon333, GlyphDefinition
from .codec import GlyphTransportCodec
from .executor import ExecutionResult, GlyphGraphExecutor
from .learning import ADLEngine, FailureLedger, GhostBridge
from .models import (
    BenchmarkCase,
    BenchmarkConfig,
    CaseResult,
    CompositeResult,
    GlyphEdge,
    GlyphProgram,
    ScoreBreakdown,
)
from .scoring import CompositeScorer

__all__ = [
    "ADLEngine",
    "ArtVerdict",
    "BenchmarkCase",
    "BenchmarkConfig",
    "BenchmarkRunner",
    "CANON333",
    "Canon333",
    "CaseResult",
    "CompositeResult",
    "CompositeScorer",
    "ExecutionResult",
    "FailureLedger",
    "GhostBridge",
    "GlyphDefinition",
    "GlyphEdge",
    "GlyphGraphExecutor",
    "GlyphProgram",
    "GlyphTransportCodec",
    "PromptModel",
    "ScoreBreakdown",
    "check_art",
    "extract_art",
    "extract_guess",
    "load_cases",
]

__version__ = "1.0.0"

