# Copyright 2026 918 Technologies
# SPDX-License-Identifier: Apache-2.0
# %% [markdown]
# # GlyphMatics Word Art 333
#
# Kaggle Benchmarks task: isolated artist/guesser conversations, exact 333
# canon enforcement, Word Art scoring, and GlyphMatics-native diagnostics.

# %%
from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path

import kaggle_benchmarks as kbench

TASK_ROOT = Path.cwd()
LOCAL_SRC = TASK_ROOT / "src"
if LOCAL_SRC.is_dir() and str(LOCAL_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAL_SRC))

from glyphmatics_wordart import ADLEngine, BenchmarkRunner, CANON333, FailureLedger, load_cases


class _KaggleSession:
    def __init__(self, llm) -> None:
        self.llm = llm

    def prompt(self, prompt: str) -> str:
        return str(self.llm.prompt(prompt, reasoning="high", temperature=0.2))


class _KagglePromptModel:
    def __init__(self, llm) -> None:
        self.llm = llm

    @contextmanager
    def session(self, name: str):
        with kbench.chats.new(name, orphan=True):
            yield _KaggleSession(self.llm)


# %%
@kbench.task(
    name="glyphmatics-wordart-333",
    description="Visual-semantic round-trip benchmark using the immutable GlyphMatics 333 canon",
    version=1,
)
def glyphmatics_wordart_333(llm) -> dict:
    cases = load_cases()
    adl = ADLEngine()
    ledger = FailureLedger(TASK_ROOT / "GLYPHMATICS_WORDART_FAILURES.jsonl")
    runner = BenchmarkRunner(
        _KagglePromptModel(llm),
        cases=cases,
        ledger=ledger,
        adl=adl,
    )
    result = runner.run_all()
    report = result.to_mapping(runner.config)
    report["adl_snapshot"] = adl.snapshot()
    report["failure_ledger_records"] = len(ledger.read_all())
    kbench.assertions.assert_true(
        len(CANON333.definitions()) == 333,
        expectation="The immutable GlyphMatics canon contains exactly 333 unique definitions",
    )
    kbench.assertions.assert_true(
        report["case_count"] == len(cases),
        expectation="Every configured Word Art case produced a result",
    )
    return report


# %%
glyphmatics_wordart_333.run(kbench.llm)

