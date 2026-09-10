# Copyright 2026 918 Technologies
# SPDX-License-Identifier: Apache-2.0
"""Command-line entry point for deterministic validation and reporting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .benchmark import BenchmarkRunner, DeterministicDemoModel, load_cases
from .learning import FailureLedger


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="glyphmatics-wordart")
    subparsers = parser.add_subparsers(dest="command", required=True)
    demo = subparsers.add_parser("demo", help="run the deterministic offline contract fixture")
    demo.add_argument("--cases", type=Path, default=None)
    demo.add_argument("--output", type=Path, default=Path("GLYPHMATICS_WORDART_REPORT.json"))
    demo.add_argument("--ledger", type=Path, default=Path("GLYPHMATICS_WORDART_FAILURES.jsonl"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "demo":
        cases = load_cases(args.cases)
        ledger = FailureLedger(args.ledger)
        runner = BenchmarkRunner(DeterministicDemoModel(cases), cases=cases, ledger=ledger)
        report = runner.run_all().to_mapping(runner.config)
        report["adl_snapshot"] = runner.adl.snapshot()
        report["failure_ledger_records"] = len(ledger.read_all())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({
            "cases": report["case_count"],
            "mean_composite": report["mean_composite"],
            "output": str(args.output),
            "word_art_points": f"{report['total_word_art_points']}/{report['max_word_art_points']}",
        }, sort_keys=True))
        return 0
    raise RuntimeError(f"unhandled command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())

