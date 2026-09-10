# Copyright 2026 918 Technologies
# SPDX-License-Identifier: Apache-2.0
"""ADL difference learning and GhostBridge causal-gap records."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .models import BenchmarkCase, GlyphProgram, ScoreBreakdown, stable_json

try:
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - Windows path
    _fcntl = None


ZERO_HASH = "0" * 64


@dataclass(frozen=True, slots=True)
class BridgeGap:
    case_id: str
    missing_tokens: tuple[str, ...]
    failure_causes: tuple[str, ...]
    repair_prompt: str
    confidence: float


@dataclass(frozen=True, slots=True)
class FailureEvent:
    sequence: int
    event_id: str
    created_ns: int
    case_id: str
    phase: str
    failure_codes: tuple[str, ...]
    gap_tokens: tuple[str, ...]
    observed: Mapping[str, Any]
    expected: Mapping[str, Any]
    previous_hash: str
    content_hash: str
    schema: str = "glyphmatics.wordart.failure.v1"

    def hash_payload(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "created_ns": self.created_ns,
            "event_id": self.event_id,
            "expected": dict(self.expected),
            "failure_codes": list(self.failure_codes),
            "gap_tokens": list(self.gap_tokens),
            "observed": dict(self.observed),
            "phase": self.phase,
            "previous_hash": self.previous_hash,
            "schema": self.schema,
            "sequence": self.sequence,
        }

    def to_mapping(self) -> dict[str, Any]:
        return {**self.hash_payload(), "content_hash": self.content_hash}

    @classmethod
    def build(
        cls,
        *,
        sequence: int,
        case_id: str,
        phase: str,
        failure_codes: Iterable[str],
        gap_tokens: Iterable[str],
        observed: Mapping[str, Any],
        expected: Mapping[str, Any],
        previous_hash: str,
        created_ns: int | None = None,
    ) -> "FailureEvent":
        failures = tuple(sorted(set(map(str, failure_codes))))
        gaps = tuple(sorted(set(map(str, gap_tokens))))
        dedupe_payload = {
            "case_id": case_id,
            "expected": dict(expected),
            "failure_codes": list(failures),
            "gap_tokens": list(gaps),
            "observed": dict(observed),
            "phase": phase,
        }
        event_id = hashlib.sha256(stable_json(dedupe_payload).encode("utf-8")).hexdigest()
        partial = cls(
            sequence=sequence,
            event_id=event_id,
            created_ns=int(created_ns if created_ns is not None else time.time_ns()),
            case_id=str(case_id),
            phase=str(phase),
            failure_codes=failures,
            gap_tokens=gaps,
            observed=json.loads(stable_json(dict(observed))),
            expected=json.loads(stable_json(dict(expected))),
            previous_hash=previous_hash,
            content_hash="",
        )
        digest = hashlib.sha256(stable_json(partial.hash_payload()).encode("utf-8")).hexdigest()
        return cls(**{
            "sequence": partial.sequence,
            "event_id": partial.event_id,
            "created_ns": partial.created_ns,
            "case_id": partial.case_id,
            "phase": partial.phase,
            "failure_codes": partial.failure_codes,
            "gap_tokens": partial.gap_tokens,
            "observed": partial.observed,
            "expected": partial.expected,
            "previous_hash": partial.previous_hash,
            "content_hash": digest,
            "schema": partial.schema,
        })

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FailureEvent":
        return cls(
            sequence=int(value["sequence"]),
            event_id=str(value["event_id"]),
            created_ns=int(value["created_ns"]),
            case_id=str(value["case_id"]),
            phase=str(value["phase"]),
            failure_codes=tuple(map(str, value["failure_codes"])),
            gap_tokens=tuple(map(str, value["gap_tokens"])),
            observed=dict(value["observed"]),
            expected=dict(value["expected"]),
            previous_hash=str(value["previous_hash"]),
            content_hash=str(value["content_hash"]),
            schema=str(value.get("schema", "glyphmatics.wordart.failure.v1")),
        )


class FailureLedger:
    """Append-only JSONL ledger with sequence, chain, hash, and dedupe checks."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._thread_lock = threading.RLock()

    @staticmethod
    def _lock(fd: int, exclusive: bool) -> None:
        if _fcntl is not None:
            _fcntl.flock(fd, _fcntl.LOCK_EX if exclusive else _fcntl.LOCK_SH)

    @staticmethod
    def _unlock(fd: int) -> None:
        if _fcntl is not None:
            _fcntl.flock(fd, _fcntl.LOCK_UN)

    @staticmethod
    def _read_locked(fd: int) -> list[FailureEvent]:
        os.lseek(fd, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 65536)
            if not chunk:
                break
            chunks.append(chunk)
        payload = b"".join(chunks)
        if payload and not payload.endswith(b"\n"):
            raise RuntimeError("failure ledger ends with an incomplete record")
        events: list[FailureEvent] = []
        previous_hash = ZERO_HASH
        event_ids: set[str] = set()
        for line_number, raw_line in enumerate(payload.splitlines(), start=1):
            if not raw_line.strip():
                raise RuntimeError(f"failure ledger has blank record at line {line_number}")
            try:
                value = json.loads(raw_line)
                event = FailureEvent.from_mapping(value)
            except Exception as exc:
                raise RuntimeError(f"invalid failure ledger record at line {line_number}: {exc}") from exc
            if event.sequence != line_number:
                raise RuntimeError(f"failure ledger sequence mismatch at line {line_number}")
            if event.previous_hash != previous_hash:
                raise RuntimeError(f"failure ledger chain mismatch at line {line_number}")
            calculated = hashlib.sha256(stable_json(event.hash_payload()).encode("utf-8")).hexdigest()
            if event.content_hash != calculated:
                raise RuntimeError(f"failure ledger content hash mismatch at line {line_number}")
            if event.event_id in event_ids:
                raise RuntimeError(f"failure ledger duplicate event_id at line {line_number}")
            event_ids.add(event.event_id)
            previous_hash = event.content_hash
            events.append(event)
        return events

    def read_all(self) -> tuple[FailureEvent, ...]:
        with self._thread_lock:
            fd = os.open(self.path, os.O_RDONLY | os.O_CREAT, 0o600)
            try:
                self._lock(fd, exclusive=False)
                return tuple(self._read_locked(fd))
            finally:
                self._unlock(fd)
                os.close(fd)

    def append_gap(
        self,
        gap: BridgeGap,
        observed: Mapping[str, Any],
        expected: Mapping[str, Any],
        phase: str = "POST_MOVE",
    ) -> FailureEvent:
        with self._thread_lock:
            fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
            try:
                self._lock(fd, exclusive=True)
                events = self._read_locked(fd)
                candidate = FailureEvent.build(
                    sequence=len(events) + 1,
                    case_id=gap.case_id,
                    phase=phase,
                    failure_codes=gap.failure_causes,
                    gap_tokens=gap.missing_tokens,
                    observed=observed,
                    expected=expected,
                    previous_hash=events[-1].content_hash if events else ZERO_HASH,
                )
                existing = next((event for event in events if event.event_id == candidate.event_id), None)
                if existing is not None:
                    return existing
                record = (stable_json(candidate.to_mapping()) + "\n").encode("utf-8")
                os.lseek(fd, 0, os.SEEK_END)
                offset = 0
                while offset < len(record):
                    written = os.write(fd, record[offset:])
                    if written <= 0:
                        raise OSError("short write while appending failure ledger")
                    offset += written
                os.fsync(fd)
                return candidate
            finally:
                self._unlock(fd)
                os.close(fd)


@dataclass(slots=True)
class _DifferenceStat:
    wins: int = 0
    losses: int = 0
    total_delta: float = 0.0

    @property
    def evidence(self) -> int:
        return self.wins + self.losses

    @property
    def mean_delta(self) -> float:
        return self.total_delta / self.evidence if self.evidence else 0.0


class ADLEngine:
    """Learn which explicit prompt/encoding differences improve score."""

    def __init__(self) -> None:
        self._stats: dict[str, _DifferenceStat] = {}

    def post_move(self, changed_features: Iterable[str], before: float, after: float) -> dict[str, float]:
        delta = float(after) - float(before)
        result: dict[str, float] = {}
        for feature in sorted(set(map(str, changed_features))):
            stat = self._stats.setdefault(feature, _DifferenceStat())
            if delta > 0:
                stat.wins += 1
            else:
                stat.losses += 1
            stat.total_delta += delta
            result[feature] = stat.mean_delta
        return result

    def winning_differences(self, minimum_evidence: int = 2) -> tuple[str, ...]:
        ranked = [
            (feature, stat.mean_delta, stat.evidence)
            for feature, stat in self._stats.items()
            if stat.evidence >= minimum_evidence and stat.mean_delta > 0
        ]
        ranked.sort(key=lambda item: (item[1], item[2], item[0]), reverse=True)
        return tuple(feature for feature, _, _ in ranked)

    def snapshot(self) -> dict[str, dict[str, float | int]]:
        return {
            feature: {
                "evidence": stat.evidence,
                "losses": stat.losses,
                "mean_delta": round(stat.mean_delta, 6),
                "wins": stat.wins,
            }
            for feature, stat in sorted(self._stats.items())
        }


class GhostBridge:
    """Infer the smallest explicit bridge after a failed round."""

    @staticmethod
    def pre_move(prior_events: Iterable[FailureEvent]) -> str:
        counts: dict[str, int] = {}
        for event in prior_events:
            for token in event.gap_tokens:
                counts[token] = counts.get(token, 0) + 1
        if not counts:
            return "No prior causal gap is established; use a minimal valid graph and clean non-text art."
        ranked = sorted(counts, key=lambda token: (-counts[token], token))[:6]
        return "Prior failures most often lacked: " + ", ".join(ranked) + ". Add only those that fit this concept."

    @staticmethod
    def infer(
        case: BenchmarkCase,
        failures: Iterable[str],
        program: GlyphProgram | None,
        score: ScoreBreakdown,
    ) -> BridgeGap:
        observed = set(program.tokens if program else ())
        missing = tuple(token for token in case.expected_tokens if token not in observed)
        causes = tuple(sorted(set(map(str, failures))))
        if score.semantic_fidelity == 0:
            causes = tuple(sorted(set((*causes, "semantic_decode_failure"))))
        if score.graph_fidelity < 1:
            causes = tuple(sorted(set((*causes, "graph_fidelity_gap"))))
        if score.execution < 1:
            causes = tuple(sorted(set((*causes, "execution_contract_gap"))))
        repairs: list[str] = []
        if missing:
            repairs.append("add canonical roles/relations " + ", ".join(missing))
        if "art_disqualified" in causes:
            repairs.append("remove all target/text runs from the art")
        if "semantic_decode_failure" in causes:
            repairs.append("increase silhouette and distinctive visual cues without labels")
        if "execution_contract_gap" in causes:
            repairs.append("bind a role node through a canonical relation edge")
        if not repairs:
            repairs.append("retain the current minimal bridge")
        evidence = len(causes) + len(missing)
        confidence = min(1.0, 0.35 + 0.1 * evidence)
        return BridgeGap(
            case_id=case.case_id,
            missing_tokens=missing,
            failure_causes=causes,
            repair_prompt="; ".join(repairs) + ".",
            confidence=round(confidence, 6),
        )
