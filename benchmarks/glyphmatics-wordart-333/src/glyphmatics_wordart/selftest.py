# Copyright 2026 918 Technologies
# SPDX-License-Identifier: Apache-2.0
"""Dependency-free smoke and integration tests."""

from __future__ import annotations

import tempfile
from pathlib import Path

from .art_rules import check_art, extract_art, extract_guess, extract_program
from .benchmark import BenchmarkRunner, DeterministicDemoModel, load_cases
from .canon333 import CANON333
from .codec import GlyphTransportCodec
from .learning import FailureLedger, GhostBridge
from .models import GlyphEdge, GlyphProgram


def run() -> None:
    assert len(CANON333.definitions()) == 333
    assert CANON333.by_id(1).token == "STATE"
    assert CANON333.by_token("TOROIDAL").glyph_id > 1

    program = GlyphProgram(
        tokens=("BEGIN", "OBJECT", "COMPONENT", "END"),
        edges=(GlyphEdge(1, 2, "CONTAIN"),),
    )
    codec = GlyphTransportCodec()
    transport = codec.encode_program(program)
    assert codec.decode_tokens(transport) == program.tokens
    assert codec.verify_program(program, transport)

    response = (
        '<glyphmatics>{"version":"glyphmatics-333-v1","tokens":["BEGIN","OBJECT","END"],'
        '"edges":[[1,2,"CONTAIN"]]}</glyphmatics><art>\n /\\\n/__\\\n</art>'
    )
    assert extract_program(response).tokens == ("BEGIN", "OBJECT", "END")
    assert "/__\\" in extract_art(response)
    assert extract_guess('reasoning {"guess":"CAT"}') == "cat"
    assert check_art(" /\\\n/__\\", "cat").accepted
    assert not check_art("C-A-T", "cat").accepted

    cases = load_cases()
    with tempfile.TemporaryDirectory() as temporary:
        ledger = FailureLedger(Path(temporary) / "failures.jsonl")
        runner = BenchmarkRunner(DeterministicDemoModel(cases), cases=cases, ledger=ledger)
        report = runner.run_all()
        assert len(report.cases) == len(cases)
        assert report.total_word_art_points == report.max_word_art_points
        assert report.mean_composite > 80.0
        events = ledger.read_all()
        assert all(event.sequence == index for index, event in enumerate(events, start=1))
        assert GhostBridge.pre_move(events)
    print(
        f"SELFTEST PASS canon=333 cases={len(cases)} "
        f"composite={report.mean_composite:.6f} points={report.total_word_art_points:.1f}"
    )


if __name__ == "__main__":
    run()

