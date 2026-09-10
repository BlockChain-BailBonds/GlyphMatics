# Copyright 2026 918 Technologies
# SPDX-License-Identifier: Apache-2.0
"""Word Art parsing, validation, normalization, and answer matching."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Iterator

from .models import GlyphProgram


_ART_RE = re.compile(
    r"<\s*art\s*>((?:(?!<\s*art\s*>).)*?)<\s*/\s*art\s*>",
    re.IGNORECASE | re.DOTALL,
)
_PROGRAM_RE = re.compile(
    r"<\s*glyphmatics\s*>((?:(?!<\s*glyphmatics\s*>).)*?)<\s*/\s*glyphmatics\s*>",
    re.IGNORECASE | re.DOTALL,
)
_ASCII_LETTER_RUN = re.compile(r"[A-Za-z]{3,}")
_IRREGULAR_EQUIVALENTS = {
    "child": "children",
    "children": "child",
    "person": "people",
    "people": "person",
    "mouse": "mice",
    "mice": "mouse",
    "goose": "geese",
    "geese": "goose",
    "tooth": "teeth",
    "teeth": "tooth",
    "foot": "feet",
    "feet": "foot",
}


@dataclass(frozen=True, slots=True)
class ArtVerdict:
    accepted: bool
    reason: str | None = None
    detail: str | None = None
    checked_by: str = "fallback"


def extract_art(response: str) -> str:
    matches = list(_ART_RE.finditer(str(response)))
    if not matches:
        raise ValueError("response does not contain an <art>...</art> block")
    art = matches[-1].group(1)
    if not art.strip():
        raise ValueError("the final <art> block is empty")
    return art


def extract_program(response: str) -> GlyphProgram:
    matches = list(_PROGRAM_RE.finditer(str(response)))
    if not matches:
        raise ValueError("response does not contain a <glyphmatics>...</glyphmatics> block")
    raw = matches[-1].group(1).strip()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"glyph program is not valid JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise TypeError("glyph program JSON must be an object")
    return GlyphProgram.from_mapping(value)


def _json_objects(text: str) -> Iterator[dict[str, Any]]:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            yield value


def extract_guess(response: str) -> str:
    objects = [value for value in _json_objects(str(response)) if "guess" in value]
    if not objects:
        raise ValueError('response contains no parseable JSON object with a "guess" key')
    guess = objects[-1]["guess"]
    if not isinstance(guess, str) or not guess.strip():
        raise ValueError("guess must be a non-empty string")
    normalized = guess.strip().lower()
    if not re.fullmatch(r"[a-z]+", normalized):
        raise ValueError("guess must be a single ASCII word")
    return normalized


def _has_bad_letter_run(text: str) -> str | None:
    for run in _ASCII_LETTER_RUN.findall(text):
        if len(set(run.lower())) >= 2:
            return run
    return None


def _fallback_check_art(art: str, target_word: str, max_art_chars: int) -> ArtVerdict:
    if not isinstance(art, str) or not art.strip():
        return ArtVerdict(False, "empty", "drawing is empty")
    if len(art) > max_art_chars:
        return ArtVerdict(False, "too_long", f"drawing has {len(art)} characters; maximum is {max_art_chars}")

    target = "".join(char.lower() for char in target_word if char.isascii() and char.isalnum())
    compact = "".join(char.lower() for char in art if char.isascii() and char.isalnum())
    if target and (target in compact or target[::-1] in compact):
        return ArtVerdict(False, "target_word", "drawing contains the target word forwards or reversed")

    lines = art.splitlines() or [art]
    for row_index, line in enumerate(lines):
        bad = _has_bad_letter_run(line)
        if bad is not None:
            return ArtVerdict(False, "contains_words", f"row {row_index + 1} contains letter run {bad!r}")

    width = max((len(line) for line in lines), default=0)
    for column in range(width):
        column_text = "".join(line[column] if column < len(line) else " " for line in lines)
        bad = _has_bad_letter_run(column_text)
        if bad is not None:
            return ArtVerdict(False, "contains_words", f"column {column + 1} contains letter run {bad!r}")
    return ArtVerdict(True)


def check_art(art: str, target_word: str, max_art_chars: int = 4000) -> ArtVerdict:
    """Apply the official Kaggle check when installed, otherwise a local mirror.

    The fallback deliberately implements the public target-word and row/column
    letter-run rules. Reports expose ``checked_by`` so official and fallback
    results cannot be confused.
    """

    try:
        from kaggle_environments.envs.word_art.word_art import check_art as official_check_art
    except (ImportError, ModuleNotFoundError):
        return _fallback_check_art(art, target_word, max_art_chars)
    verdict = official_check_art(art, target_word, max_art_chars)
    if verdict is None:
        return ArtVerdict(True, checked_by="kaggle")
    if isinstance(verdict, (list, tuple)):
        reason = str(verdict[0]) if verdict else "rejected"
        detail = str(verdict[1]) if len(verdict) > 1 else reason
    else:
        reason = str(verdict)
        detail = reason
    return ArtVerdict(False, reason, detail, checked_by="kaggle")


def normalize_art(art: str) -> str:
    text = str(art).replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    nonempty = [line for line in lines if line.strip()]
    common_indent = min((len(line) - len(line.lstrip(" ")) for line in nonempty), default=0)
    return "\n".join(line[common_indent:] if line else "" for line in lines)


def art_digest(art: str) -> str:
    return hashlib.sha256(normalize_art(art).encode("utf-8")).hexdigest()


def robustness_variants(art: str) -> tuple[str, ...]:
    normalized = normalize_art(art)
    lines = normalized.splitlines()
    padded = "\n".join(f"  {line}   " for line in lines)
    crlf = normalized.replace("\n", "\r\n")
    framed = "\n" + normalized + "\n"
    return tuple(dict.fromkeys((normalized, padded, crlf, framed)))


def _singular_plural_forms(word: str) -> set[str]:
    forms = {word}
    irregular = _IRREGULAR_EQUIVALENTS.get(word)
    if irregular:
        forms.add(irregular)
    if word.endswith("ies") and len(word) > 3:
        forms.add(word[:-3] + "y")
    elif word.endswith("es") and len(word) > 2:
        forms.add(word[:-2])
        forms.add(word[:-1])
    elif word.endswith("s") and len(word) > 1:
        forms.add(word[:-1])
    else:
        forms.add(word + "s")
        if word.endswith(("s", "x", "z", "ch", "sh")):
            forms.add(word + "es")
        if word.endswith("y") and len(word) > 1 and word[-2] not in "aeiou":
            forms.add(word[:-1] + "ies")
    return forms


def answer_matches(guess: str, accepted_answers: tuple[str, ...]) -> bool:
    normalized = str(guess).strip().lower()
    return any(normalized in _singular_plural_forms(answer.lower()) for answer in accepted_answers)

