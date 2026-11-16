# Copyright 2026 918 Technologies
# SPDX-License-Identifier: Apache-2.0
"""End-to-end artist/guesser benchmark orchestration."""

from __future__ import annotations

import hashlib
import json
import re
from contextlib import contextmanager
from importlib.resources import files
from pathlib import Path
from typing import ContextManager, Iterator, Protocol

from .art_rules import (
    answer_matches,
    art_digest,
    check_art,
    extract_art,
    extract_guess,
    extract_program,
    normalize_art,
)
from .canon333 import CANON333
from .codec import GlyphTransportCodec
from .executor import GlyphGraphExecutor
from .learning import ADLEngine, FailureLedger, GhostBridge
from .models import (
    BenchmarkCase,
    BenchmarkConfig,
    CaseResult,
    CompositeResult,
    GlyphEdge,
    GlyphProgram,
)
from .scoring import CompositeScorer


class PromptSession(Protocol):
    def prompt(self, prompt: str) -> str: ...


class PromptModel(Protocol):
    def session(self, name: str) -> ContextManager[PromptSession]: ...


def load_cases(path: str | Path | None = None) -> tuple[BenchmarkCase, ...]:
    if path is None:
        payload = files("glyphmatics_wordart").joinpath("data/cases.json").read_text(encoding="utf-8")
    else:
        payload = Path(path).read_text(encoding="utf-8")
    value = json.loads(payload)
    if not isinstance(value, list) or not value:
        raise ValueError("cases file must contain a non-empty JSON list")
    cases = tuple(BenchmarkCase.from_mapping(item) for item in value)
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("case IDs must be unique")
    return cases


def _canon_prompt_manifest() -> str:
    grouped: dict[str, list[str]] = {}
    for definition in CANON333.definitions():
        grouped.setdefault(definition.category, []).append(definition.token)
    return "\n".join(f"{category}: {', '.join(tokens)}" for category, tokens in grouped.items())


_CANON_MANIFEST = _canon_prompt_manifest()


class BenchmarkRunner:
    def __init__(
        self,
        model: PromptModel,
        *,
        cases: tuple[BenchmarkCase, ...] | None = None,
        config: BenchmarkConfig | None = None,
        ledger: FailureLedger | None = None,
        adl: ADLEngine | None = None,
        ghostbridge: GhostBridge | None = None,
    ) -> None:
        if len(CANON333.definitions()) != 333:
            raise RuntimeError("exact-333 canon startup contract failed")
        self.model = model
        self.cases = cases or load_cases()
        self.config = config or BenchmarkConfig()
        self.ledger = ledger
        self.adl = adl or ADLEngine()
        self.ghostbridge = ghostbridge or GhostBridge()
        self.codec = GlyphTransportCodec()
        self.executor = GlyphGraphExecutor(self.codec)
        self.scorer = CompositeScorer(self.config, self.codec, self.executor)

    def _artist_prompt(self, case: BenchmarkCase, pre_move_hint: str) -> str:
        return f"""You are the ARTIST in GlyphMatics Word Art.

SECRET WORD: {case.word}

Create a structural drawing that makes a blind teammate guess the secret word.
The drawing must be at most {self.config.max_art_chars} characters. Never place
the target, a synonym, a caption, or any run of 3+ mixed ASCII letters inside
the art. Use punctuation, box drawing, blocks, arrows, or Braille for shape.

Before drawing, compile a small semantic graph using only the exact immutable
GlyphMatics canon below. Use 4-12 token occurrences, normally including BEGIN,
END, OBJECT or AGENT, visual topology tokens, and at least one relation edge.
Edges are [source_index,target_index,RELATION_TOKEN]. Indexes are zero-based.
Do not place the target word or arbitrary strings in the glyph program.

Canon version: {CANON333.version}
Canon SHA-256: {CANON333.digest}
{_CANON_MANIFEST}

GhostBridge PRE_MOVE brief: {pre_move_hint}

Return exactly one strict JSON glyph block followed by one art block. The art
block must be last. Example format:
<glyphmatics>{{"version":"{CANON333.version}","tokens":["BEGIN","OBJECT","COMPONENT","END"],"edges":[[1,2,"CONTAIN"]]}}</glyphmatics>
<art>
structural drawing here
</art>
"""

    @staticmethod
    def _artist_repair_prompt(failures: tuple[str, ...], detail: str) -> str:
        return f"""The previous candidate was rejected before commit.
Failure codes: {', '.join(failures)}
Detail: {detail}
Repair only the failed constraints. Return a complete replacement
<glyphmatics> JSON block and a non-empty <art> block, with the art block last.
"""

    def _guesser_prompt(self, art: str, guesses: tuple[str, ...], attempt: int) -> str:
        prior = ", ".join(guesses) if guesses else "none"
        points = self.config.guess_points[attempt - 1]
        return f"""You are the GUESSER in Word Art. You never see the secret word or
the artist's reasoning. Infer the single concept depicted by the structural art.
This is attempt {attempt} of {self.config.max_attempts}; a correct answer earns
{points} points. Previous guesses, all wrong: {prior}.

ART:
{art}

End with exactly one JSON object: {{"guess":"singleword"}}
"""

    def _prior_events(self):
        return self.ledger.read_all() if self.ledger is not None else ()

    def run_case(self, case: BenchmarkCase) -> CaseResult:
        failures: list[str] = []
        pre_move_hint = self.ghostbridge.pre_move(self._prior_events())
        program: GlyphProgram | None = None
        art = ""
        final_response = ""
        repaired_codes: set[str] = set()

        with self.model.session(f"artist-{case.case_id}") as artist:
            prompt = self._artist_prompt(case, pre_move_hint)
            for artist_attempt in range(1, self.config.max_artist_attempts + 1):
                final_response = str(artist.prompt(prompt))
                attempt_failures: list[str] = []
                detail: list[str] = []
                try:
                    candidate_program = extract_program(final_response)
                except (KeyError, TypeError, ValueError) as exc:
                    candidate_program = None
                    attempt_failures.append("program_parse")
                    detail.append(str(exc))
                try:
                    candidate_art = extract_art(final_response)
                except ValueError as exc:
                    candidate_art = ""
                    attempt_failures.append("art_parse")
                    detail.append(str(exc))
                if candidate_art:
                    verdict = check_art(candidate_art, case.word, self.config.max_art_chars)
                    if not verdict.accepted:
                        attempt_failures.append("art_disqualified")
                        detail.append(f"{verdict.reason}:{verdict.detail}")
                if not attempt_failures:
                    program = candidate_program
                    art = candidate_art
                    if repaired_codes:
                        self.adl.post_move((f"repair:{code}" for code in repaired_codes), 0.0, 1.0)
                    break
                repaired_codes.update(attempt_failures)
                failures.extend(f"artist_attempt_{artist_attempt}:{code}" for code in attempt_failures)
                if artist_attempt < self.config.max_artist_attempts:
                    prompt = self._artist_repair_prompt(tuple(attempt_failures), "; ".join(detail))
            else:
                if repaired_codes:
                    self.adl.post_move((f"repair:{code}" for code in repaired_codes), 0.0, 0.0)

        visible_art = art if art else "(drawing rejected by constraint checks)"
        guesses: list[str] = []
        with self.model.session(f"guesser-{case.case_id}") as guesser:
            for attempt in range(1, self.config.max_attempts + 1):
                response = str(guesser.prompt(self._guesser_prompt(visible_art, tuple(guesses), attempt)))
                try:
                    guess = extract_guess(response)
                except ValueError:
                    guess = "invalid"
                    failures.append(f"guesser_attempt_{attempt}:guess_parse")
                guesses.append(guess)
                if answer_matches(guess, case.accepted_answers):
                    break

        execution = self.executor.execute(program) if program is not None else None
        if execution is not None:
            failures.extend(f"executor:{error}" for error in execution.errors)
        score = self.scorer.score_case(case, art, tuple(guesses), program, execution=execution)
        if score.correct_attempt is None:
            failures.append("semantic_decode_failure")

        transport = self.codec.encode_program(program) if program is not None else ""
        result = CaseResult(
            case_id=case.case_id,
            target_hash=hashlib.sha256(case.word.encode("utf-8")).hexdigest(),
            art=art,
            guesses=tuple(guesses),
            glyph_program=program,
            score=score,
            failures=tuple(dict.fromkeys(failures)),
            art_digest=art_digest(art),
            transport=transport,
        )
        if self.ledger is not None and (result.failures or score.composite(self.config) < 100.0):
            gap = self.ghostbridge.infer(case, result.failures, program, score)
            self.ledger.append_gap(
                gap,
                observed={
                    "art_digest": result.art_digest,
                    "guesses": list(result.guesses),
                    "program_digest": program.digest if program else None,
                    "score": score.to_mapping(self.config),
                },
                expected={
                    "expected_tokens": list(case.expected_tokens),
                    "target_hash": result.target_hash,
                },
                phase="POST_MOVE",
            )
        return result

    def run_all(self) -> CompositeResult:
        initial = tuple(self.run_case(case) for case in self.cases)
        counts = self.scorer.collision_counts(tuple(result.art for result in initial))
        final: list[CaseResult] = []
        for case, result in zip(self.cases, initial):
            collision_count = counts[result.art_digest]
            execution = self.executor.execute(result.glyph_program) if result.glyph_program else None
            rescored = self.scorer.score_case(
                case,
                result.art,
                result.guesses,
                result.glyph_program,
                collision_count=collision_count,
                execution=execution,
            )
            final.append(
                CaseResult(
                    case_id=result.case_id,
                    target_hash=result.target_hash,
                    art=result.art,
                    guesses=result.guesses,
                    glyph_program=result.glyph_program,
                    score=rescored,
                    failures=result.failures + (("cross_concept_collision",) if collision_count > 1 else ()),
                    art_digest=result.art_digest,
                    transport=result.transport,
                )
            )
        cases = tuple(final)
        mean = sum(case.score.composite(self.config) for case in cases) / max(1, len(cases))
        total_points = sum(case.score.word_art_points for case in cases)
        maximum = len(cases) * max(self.config.guess_points)
        return CompositeResult(
            cases=cases,
            mean_composite=round(mean, 6),
            total_word_art_points=round(total_points, 6),
            max_word_art_points=round(maximum, 6),
        )


_DEMO_ART = {
    "cat": " /\\_/\\\n( o.o )\n > ^ <",
    "umbrella": "   .-^-.\n .'     '.\n'-._____.-'\n    |\n    |\n   _|_",
    "bicycle": "   __o\n _ \\<,_\n(_)/ (_)",
    "clock": " .-------.\n|    ^    |\n|    |--> |\n|    o    |\n '-------'",
    "bridge": "|\\          /|\n| \\________/ |\n|_____________|\n~~~~~    ~~~~~",
    "fish": "><(((°>",
    "tree": "   /\\\n  /**\\\n /****\\\n   ||\n   ||",
    "snowflake": "\\  |  /\n --+--\n/  |  \\",
    "airplane": "     __|__\n--o--(_)--o--\n      |",
    "key": "o====[]__/\\_",
}


class _DemoSession:
    def __init__(self, model: "DeterministicDemoModel", name: str) -> None:
        self.model = model
        self.name = name

    def prompt(self, prompt: str) -> str:
        return self.model._prompt(self.name, prompt)


class DeterministicDemoModel:
    """Offline contract fixture. It is not presented as a learned baseline."""

    def __init__(self, cases: tuple[BenchmarkCase, ...] | None = None) -> None:
        self.cases = cases or load_cases()
        self.by_word = {case.word: case for case in self.cases}
        self.by_art = {normalize_art(_DEMO_ART[case.word]): case.word for case in self.cases}

    @contextmanager
    def session(self, name: str) -> Iterator[_DemoSession]:
        yield _DemoSession(self, name)

    def _prompt(self, session_name: str, prompt: str) -> str:
        if session_name.startswith("artist-"):
            match = re.search(r"SECRET WORD:\s*([a-z]+)", prompt)
            if not match:
                raise RuntimeError("demo artist prompt omitted the secret word")
            word = match.group(1)
            case = self.by_word[word]
            edges = [GlyphEdge(i, i + 1, "CONNECTED") for i in range(1, len(case.expected_tokens) - 2)]
            if not edges and len(case.expected_tokens) >= 3:
                edges = [GlyphEdge(1, 2, "CONTAIN")]
            program = GlyphProgram(tokens=case.expected_tokens, edges=tuple(edges))
            return (
                "<glyphmatics>" + json.dumps(program.to_mapping(), separators=(",", ":"))
                + "</glyphmatics>\n<art>\n" + _DEMO_ART[word] + "\n</art>"
            )
        if session_name.startswith("guesser-"):
            art_match = re.search(r"ART:\n(.*?)\n\nEnd with", prompt, re.DOTALL)
            art = normalize_art(art_match.group(1) if art_match else "")
            guess = self.by_art.get(art, "unknown")
            return json.dumps({"guess": guess}, separators=(",", ":"))
        raise RuntimeError(f"unknown demo session {session_name!r}")

