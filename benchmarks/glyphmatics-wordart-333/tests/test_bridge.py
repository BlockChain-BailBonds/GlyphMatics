# Copyright 2026 918 Technologies
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from glyphmatics_wordart.art_rules import (
    answer_matches,
    check_art,
    extract_art,
    extract_guess,
    extract_program,
    normalize_art,
)
from glyphmatics_wordart.benchmark import BenchmarkRunner, DeterministicDemoModel, load_cases
from glyphmatics_wordart.canon333 import CANON333
from glyphmatics_wordart.codec import GlyphTransportCodec
from glyphmatics_wordart.executor import GlyphGraphExecutor
from glyphmatics_wordart.learning import ADLEngine, FailureLedger, GhostBridge
from glyphmatics_wordart.models import BenchmarkConfig, GlyphEdge, GlyphProgram, ScoreBreakdown


def test_exact_canon_identity_and_uniqueness() -> None:
    definitions = CANON333.definitions()
    assert len(definitions) == 333
    assert len({item.token for item in definitions}) == 333
    assert [item.glyph_id for item in definitions] == list(range(1, 334))
    assert CANON333.by_id(1).token == "STATE"
    assert CANON333.by_id(333).token == "HALT"
    assert CANON333.by_token("TOROIDAL").token == "TOROIDAL"


def test_canon_is_strict() -> None:
    with pytest.raises(KeyError):
        CANON333.by_token("CAT")
    with pytest.raises(KeyError):
        CANON333.by_id(334)


def test_codec_round_trip_all_333_ids() -> None:
    codec = GlyphTransportCodec()
    ids = tuple(range(1, 334))
    transport = codec.encode_ids(ids)
    assert len(transport) == 666
    assert codec.decode_ids(transport) == ids
    assert all(0x2800 <= ord(char) <= 0x28FF for char in transport)


def test_codec_rejects_corruption() -> None:
    codec = GlyphTransportCodec()
    pair = codec.encode_id(42)
    corrupt = pair[0] + chr(0x2800 + ((ord(pair[1]) - 0x2800 + 4) % 256))
    with pytest.raises(ValueError, match="checksum"):
        codec.decode_id(corrupt)


def test_program_validation_and_immutability() -> None:
    program = GlyphProgram(("begin", "object", "component", "end"), (GlyphEdge(1, 2, "contain"),))
    assert program.tokens == ("BEGIN", "OBJECT", "COMPONENT", "END")
    with pytest.raises(FrozenInstanceError):
        program.version = "other"  # type: ignore[misc]
    with pytest.raises(ValueError):
        GlyphProgram(("BEGIN", "OBJECT"), (GlyphEdge(1, 4, "CONTAIN"),))


def test_response_parsers_last_marker_wins() -> None:
    response = (
        '<glyphmatics>{"tokens":["OBJECT"],"edges":[]}</glyphmatics>'
        '<glyphmatics>{"tokens":["BEGIN","OBJECT","END"],"edges":[[1,2,"CONTAIN"]]}</glyphmatics>'
        '<art>old</art><art>\n /\\\n/__\\\n</art>'
    )
    assert extract_program(response).tokens == ("BEGIN", "OBJECT", "END")
    assert normalize_art(extract_art(response)) == "/\\\n/__\\"
    assert extract_guess('{"guess":"dog"} then {"guess":"CAT"}') == "cat"


def test_art_checks_target_and_text_runs() -> None:
    assert check_art(" /\\\n/__\\", "cat").accepted
    assert not check_art("C-A-T", "cat").accepted
    assert not check_art("dog", "cat").accepted
    assert check_art("OOO", "cat").accepted


def test_answer_matching_plural_and_irregular() -> None:
    assert answer_matches("cats", ("cat",))
    assert answer_matches("children", ("child",))
    assert not answer_matches("kitten", ("cat",))


def test_executor_valid_and_fail_closed_cycle() -> None:
    executor = GlyphGraphExecutor()
    valid = GlyphProgram(
        ("BEGIN", "OBJECT", "COMPONENT", "END"),
        (GlyphEdge(1, 2, "CONTAIN"),),
    )
    result = executor.execute(valid)
    assert result.valid
    assert result.round_trip
    cyclic = GlyphProgram(
        ("BEGIN", "OBJECT", "COMPONENT", "END"),
        (GlyphEdge(1, 2, "CONTAIN"), GlyphEdge(2, 1, "CONNECTED")),
    )
    rejected = executor.execute(cyclic)
    assert not rejected.valid
    assert any("cycle" in error for error in rejected.errors)


def test_score_formula() -> None:
    score = ScoreBreakdown(1, 1, 1, 1, 1, 1, 0, 2, 1)
    assert score.composite(BenchmarkConfig()) == 100.0
    collided = ScoreBreakdown(1, 1, 1, 1, 1, 1, 1, 2, 1)
    assert collided.composite(BenchmarkConfig()) == 90.0


def test_adl_records_only_evidence_backed_wins() -> None:
    adl = ADLEngine()
    adl.post_move(["remove-labels"], 0.2, 0.8)
    assert adl.winning_differences() == ()
    adl.post_move(["remove-labels"], 0.3, 0.9)
    assert adl.winning_differences() == ("remove-labels",)


def test_failure_ledger_chain_dedupe_and_corruption(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    ledger = FailureLedger(path)
    case = load_cases()[0]
    gap = GhostBridge.infer(
        case,
        ("semantic_decode_failure",),
        None,
        ScoreBreakdown(0, 0, 0, 0, 0, 0, 0, 0, None),
    )
    first = ledger.append_gap(gap, {"score": 0}, {"tokens": list(case.expected_tokens)})
    duplicate = ledger.append_gap(gap, {"score": 0}, {"tokens": list(case.expected_tokens)})
    assert duplicate.event_id == first.event_id
    assert len(ledger.read_all()) == 1
    record = json.loads(path.read_text().strip())
    record["observed"]["score"] = 1
    path.write_text(json.dumps(record) + "\n")
    with pytest.raises(RuntimeError, match="content hash"):
        ledger.read_all()


def test_end_to_end_demo(tmp_path: Path) -> None:
    cases = load_cases()
    ledger = FailureLedger(tmp_path / "failures.jsonl")
    runner = BenchmarkRunner(DeterministicDemoModel(cases), cases=cases, ledger=ledger)
    report = runner.run_all()
    assert len(report.cases) == 10
    assert report.total_word_art_points == 20.0
    assert report.mean_composite > 80.0
    assert all(case.score.correct_attempt == 1 for case in report.cases)
    assert all(case.glyph_program is not None for case in report.cases)

