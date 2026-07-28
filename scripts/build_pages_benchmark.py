#!/usr/bin/env python3
"""Generate GitHub Pages benchmark assets from local GlyphMatics artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

from glyphmatics.programming_syntax import (
    EXECUTABLE_SYSTEMS,
    canonicalize_program,
    encode_program_lossless,
    tokenize_program,
)
from glyphmatics.semantic_codec import SemanticVocabulary


ROOT = Path(__file__).resolve().parents[1]
DOCS_BENCHMARK = ROOT / "docs" / "benchmark"
ASSETS_DIR = DOCS_BENCHMARK / "assets"

SEMANTIC_V2_DIR = ROOT / "artifacts" / "semantic-programming-v2"
SYSTEMS_V3_DIR = ROOT / "artifacts" / "semantic-programming-systems-v3"

SEMANTIC_V2_VOCAB = SEMANTIC_V2_DIR / "semantic_vocab.json"
SEMANTIC_V2_CORPUS = SEMANTIC_V2_DIR / "semantic_corpus.jsonl"
SEMANTIC_V2_MANIFEST = SEMANTIC_V2_DIR / "manifest.json"

SYSTEMS_V3_VOCAB = SYSTEMS_V3_DIR / "semantic_vocab.json"
SYSTEMS_V3_MANIFEST = SYSTEMS_V3_DIR / "manifest.json"

TARGET_SAMPLE_LANGUAGES = (
    ("en", "English"),
    ("zh-Hans", "Chinese"),
    ("es", "Spanish"),
    ("hi", "Hindi"),
    ("ar", "Arabic"),
    ("code:python", "Python"),
    ("code:javascript", "JavaScript"),
)


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _copy_minified_json(source: Path, destination: Path) -> None:
    payload = json.loads(source.read_text(encoding="utf-8"))
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )


def _pick_semantic_samples() -> list[dict[str, str]]:
    selected: dict[str, dict[str, str]] = {}
    fallback: dict[str, dict[str, str]] = {}
    minimum_lengths = {
        "en": 18,
        "zh-Hans": 4,
        "es": 18,
        "hi": 12,
        "ar": 12,
        "code:python": 6,
        "code:javascript": 6,
    }
    with SEMANTIC_V2_CORPUS.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            language = str(row["language"])
            if language not in {item[0] for item in TARGET_SAMPLE_LANGUAGES}:
                continue
            text = str(row["text"])
            fallback.setdefault(
                language,
                {
                    "language": language,
                    "text": text,
                    "source": str(row.get("source", "")),
                },
            )
            if language in selected:
                continue
            if len(text) < minimum_lengths.get(language, 1):
                continue
            selected[language] = {
                "language": language,
                "text": text,
                "source": str(row.get("source", "")),
            }
            if len(selected) == len(TARGET_SAMPLE_LANGUAGES):
                break
    samples: list[dict[str, str]] = []
    for language, label in TARGET_SAMPLE_LANGUAGES:
        row = selected.get(language) or fallback.get(language)
        if row is None:
            raise RuntimeError(f"missing benchmark sample for language {language!r}")
        samples.append(
            {
                "id": language.replace(":", "-"),
                "label": label,
                **row,
            }
        )
    return samples


def _preview(text: str, *, limit: int = 120) -> dict[str, object]:
    return {
        "text": text[:limit],
        "truncated": len(text) > limit,
    }


def _build_semantic_sample_rows(vocabulary: SemanticVocabulary) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for sample in _pick_semantic_samples():
        text = sample["text"]
        glyphs = vocabulary.encode_glyphs(text)
        binary = vocabulary.encode_binary(text)
        decoded = vocabulary.decode_glyphs(glyphs)
        if decoded != text:
            raise RuntimeError(f"semantic glyph round-trip failed for {sample['language']}")
        rows.append(
            {
                **sample,
                "stats": vocabulary.compression_stats(text).as_dict(),
                "glyph_preview": _preview(glyphs),
                "binary_preview_hex": binary[:32].hex(),
                "lossless_roundtrip": True,
            }
        )
    return rows


def _build_system_rows(vocabulary: SemanticVocabulary) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for system in EXECUTABLE_SYSTEMS:
        lossless_glyphs = encode_program_lossless(system.source, "python")
        canonical_glyphs = canonicalize_program(system.source, "python")
        semantic_stats = vocabulary.compression_stats(system.source).as_dict()
        rows.append(
            {
                "system_id": system.system_id,
                "title": system.title,
                "description": system.description,
                "glyph": system.glyph,
                "source": system.source,
                "expected_stdout": system.expected_stdout,
                "program_token_count": len(tokenize_program(system.source, "python")),
                "source_characters": len(system.source),
                "source_utf8_bytes": len(system.source.encode("utf-8")),
                "lossless_program_glyphs": _preview(lossless_glyphs),
                "lossless_program_glyph_characters": len(lossless_glyphs),
                "lossless_program_utf8_bytes": len(lossless_glyphs.encode("utf-8")),
                "canonical_glyphs": _preview(canonical_glyphs),
                "canonical_glyph_characters": len(canonical_glyphs),
                "system_glyph_characters": 1,
                "system_char_ratio": round(len(system.source) / 1, 6),
                "lossless_program_char_ratio": round(
                    len(system.source) / max(1, len(lossless_glyphs)),
                    6,
                ),
                "semantic_stats_v3": semantic_stats,
            }
        )
    return rows


def build() -> Path:
    for path in (SEMANTIC_V2_VOCAB, SEMANTIC_V2_CORPUS, SEMANTIC_V2_MANIFEST, SYSTEMS_V3_VOCAB, SYSTEMS_V3_MANIFEST):
        if not path.exists():
            raise FileNotFoundError(path)

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    semantic_v2_vocab = SemanticVocabulary.load(SEMANTIC_V2_VOCAB)
    systems_v3_vocab = SemanticVocabulary.load(SYSTEMS_V3_VOCAB)

    semantic_samples = _build_semantic_sample_rows(semantic_v2_vocab)
    system_rows = _build_system_rows(systems_v3_vocab)

    benchmark = {
        "generated_from": {
            "semantic_v2_manifest": _load_json(SEMANTIC_V2_MANIFEST),
            "systems_v3_manifest": _load_json(SYSTEMS_V3_MANIFEST),
        },
        "vocabularies": {
            "semantic_v2": {
                "sha256": semantic_v2_vocab.digest,
                "records": len(semantic_v2_vocab.records),
                "model_vocabulary_size": semantic_v2_vocab.model_vocab_size,
                "asset": "assets/semantic-vocab-v2.json",
            },
            "systems_v3": {
                "sha256": systems_v3_vocab.digest,
                "records": len(systems_v3_vocab.records),
                "model_vocabulary_size": systems_v3_vocab.model_vocab_size,
            },
        },
        "summary": {
            "semantic_sample_count": len(semantic_samples),
            "executable_system_count": len(system_rows),
            "mean_semantic_binary_ratio": round(
                mean(row["stats"]["binary_ratio"] for row in semantic_samples),
                6,
            ),
            "mean_semantic_visual_character_ratio": round(
                mean(row["stats"]["visual_character_ratio"] for row in semantic_samples),
                6,
            ),
            "mean_lossless_program_char_ratio": round(
                mean(row["lossless_program_char_ratio"] for row in system_rows),
                6,
            ),
            "mean_system_char_ratio": round(
                mean(row["system_char_ratio"] for row in system_rows),
                6,
            ),
            "max_system_char_ratio": max(row["system_char_ratio"] for row in system_rows),
        },
        "semantic_samples": semantic_samples,
        "executable_systems": system_rows,
    }

    (ASSETS_DIR / "benchmark-data.json").write_text(
        json.dumps(benchmark, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _copy_minified_json(SEMANTIC_V2_VOCAB, ASSETS_DIR / "semantic-vocab-v2.json")
    return ASSETS_DIR / "benchmark-data.json"


if __name__ == "__main__":
    path = build()
    print(path)
