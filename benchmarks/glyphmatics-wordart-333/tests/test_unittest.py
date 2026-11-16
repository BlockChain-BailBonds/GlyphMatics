# Copyright 2026 918 Technologies
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from glyphmatics_wordart.art_rules import answer_matches, check_art, extract_guess
from glyphmatics_wordart.benchmark import BenchmarkRunner, DeterministicDemoModel, load_cases
from glyphmatics_wordart.canon333 import CANON333
from glyphmatics_wordart.codec import GlyphTransportCodec
from glyphmatics_wordart.executor import GlyphGraphExecutor
from glyphmatics_wordart.learning import ADLEngine, FailureLedger, GhostBridge
from glyphmatics_wordart.models import BenchmarkConfig, GlyphEdge, GlyphProgram, ScoreBreakdown


class BridgeTests(unittest.TestCase):
    def test_exact_canon(self) -> None:
        definitions = CANON333.definitions()
        self.assertEqual(len(definitions), 333)
        self.assertEqual(len({item.token for item in definitions}), 333)
        self.assertEqual(definitions[0].token, "STATE")
        self.assertEqual(definitions[-1].token, "HALT")

    def test_all_ids_round_trip(self) -> None:
        codec = GlyphTransportCodec()
        expected = tuple(range(1, 334))
        encoded = codec.encode_ids(expected)
        self.assertEqual(len(encoded), 666)
        self.assertEqual(codec.decode_ids(encoded), expected)

    def test_corruption_is_rejected(self) -> None:
        codec = GlyphTransportCodec()
        pair = codec.encode_id(42)
        corrupt = pair[0] + chr(0x2800 + ((ord(pair[1]) - 0x2800 + 4) % 256))
        with self.assertRaisesRegex(ValueError, "checksum"):
            codec.decode_id(corrupt)

    def test_art_and_answer_rules(self) -> None:
        self.assertTrue(check_art(" /\\\n/__\\", "cat").accepted)
        self.assertFalse(check_art("C-A-T", "cat").accepted)
        self.assertFalse(check_art("dog", "cat").accepted)
        self.assertTrue(answer_matches("cats", ("cat",)))
        self.assertTrue(answer_matches("children", ("child",)))
        self.assertEqual(extract_guess('x {"guess":"dog"} y {"guess":"CAT"}'), "cat")

    def test_executor(self) -> None:
        valid = GlyphProgram(
            ("BEGIN", "OBJECT", "COMPONENT", "END"),
            (GlyphEdge(1, 2, "CONTAIN"),),
        )
        self.assertTrue(GlyphGraphExecutor().execute(valid).valid)
        cyclic = GlyphProgram(
            ("BEGIN", "OBJECT", "COMPONENT", "END"),
            (GlyphEdge(1, 2, "CONTAIN"), GlyphEdge(2, 1, "CONNECTED")),
        )
        self.assertFalse(GlyphGraphExecutor().execute(cyclic).valid)

    def test_composite_formula(self) -> None:
        perfect = ScoreBreakdown(1, 1, 1, 1, 1, 1, 0, 2, 1)
        self.assertEqual(perfect.composite(BenchmarkConfig()), 100.0)
        collision = ScoreBreakdown(1, 1, 1, 1, 1, 1, 1, 2, 1)
        self.assertEqual(collision.composite(BenchmarkConfig()), 90.0)

    def test_adl_evidence_gate(self) -> None:
        adl = ADLEngine()
        adl.post_move(["repair"], 0.1, 0.8)
        self.assertEqual(adl.winning_differences(), ())
        adl.post_move(["repair"], 0.2, 0.9)
        self.assertEqual(adl.winning_differences(), ("repair",))

    def test_ledger_dedupe_and_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "events.jsonl"
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
            self.assertEqual(first.event_id, duplicate.event_id)
            self.assertEqual(len(ledger.read_all()), 1)
            value = json.loads(path.read_text())
            value["observed"]["score"] = 1
            path.write_text(json.dumps(value) + "\n")
            with self.assertRaisesRegex(RuntimeError, "content hash"):
                ledger.read_all()

    def test_full_demo(self) -> None:
        cases = load_cases()
        with tempfile.TemporaryDirectory() as temporary:
            runner = BenchmarkRunner(
                DeterministicDemoModel(cases),
                cases=cases,
                ledger=FailureLedger(Path(temporary) / "failures.jsonl"),
            )
            report = runner.run_all()
        self.assertEqual(len(report.cases), 10)
        self.assertEqual(report.total_word_art_points, 20.0)
        self.assertGreater(report.mean_composite, 80.0)


if __name__ == "__main__":
    unittest.main()

