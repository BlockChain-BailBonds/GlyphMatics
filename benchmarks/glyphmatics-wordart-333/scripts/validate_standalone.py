# Copyright 2026 918 Technologies
# SPDX-License-Identifier: Apache-2.0
"""Execute the generated task against a local Kaggle SDK contract stub."""

from __future__ import annotations

import contextlib
import runpy
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class _Chats:
    def __init__(self) -> None:
        self.stack: list[str] = []

    @contextlib.contextmanager
    def new(self, name: str, orphan: bool = False):  # noqa: ARG002 - SDK-compatible argument
        self.stack.append(name)
        try:
            yield self
        finally:
            popped = self.stack.pop()
            if popped != name:
                raise RuntimeError("conversation stack corruption")


class _LLM:
    def __init__(self, chats: _Chats) -> None:
        self.chats = chats

    def prompt(self, prompt: str, **kwargs) -> str:  # noqa: ARG002 - SDK-compatible options
        from glyphmatics_wordart.benchmark import DeterministicDemoModel

        if not self.chats.stack:
            raise RuntimeError("prompt executed without an isolated conversation")
        return DeterministicDemoModel()._prompt(self.chats.stack[-1], prompt)


class _Assertions:
    @staticmethod
    def assert_true(value, expectation: str = "") -> None:
        if not value:
            raise AssertionError(expectation or "assert_true failed")


class _Task:
    def __init__(self, function, module: types.ModuleType) -> None:
        self.function = function
        self.module = module

    def run(self, llm):
        result = self.function(llm)
        self.module._last_result = result
        return types.SimpleNamespace(passed=True, result=result, assertion_results=[])


def validate() -> dict:
    stub = types.ModuleType("kaggle_benchmarks")
    chats = _Chats()
    stub.chats = chats
    stub.llm = _LLM(chats)
    stub.assertions = _Assertions()
    stub._last_result = None

    def task(**metadata):  # noqa: ARG001 - metadata is accepted by the real decorator
        def decorate(function):
            return _Task(function, stub)
        return decorate

    stub.task = task
    sys.modules["kaggle_benchmarks"] = stub
    runpy.run_path(str(ROOT / "GLYPHMATICS_WORDART_TASK.py"), run_name="__main__")
    result = stub._last_result
    if not isinstance(result, dict):
        raise RuntimeError("standalone task produced no dictionary result")
    if result.get("case_count") != 10:
        raise RuntimeError(f"standalone task case count mismatch: {result.get('case_count')}")
    if result.get("total_word_art_points") != result.get("max_word_art_points"):
        raise RuntimeError("standalone task deterministic score mismatch")
    return result


if __name__ == "__main__":
    report = validate()
    print(
        "STANDALONE_TASK_PASS "
        f"cases={report['case_count']} composite={report['mean_composite']} "
        f"points={report['total_word_art_points']}/{report['max_word_art_points']}"
    )

