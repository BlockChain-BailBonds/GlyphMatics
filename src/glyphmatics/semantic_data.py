"""Build multilingual semantic glyph vocabularies and language-model corpora."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
import json
from pathlib import Path
from typing import Any

from .programming_syntax import (
    EXECUTABLE_SYSTEMS,
    PROGRAM_EXAMPLES,
    all_fixed_program_glyphs,
    iter_executable_system_rows,
    iter_program_training_rows,
)
from .semantic_codec import SemanticVocabulary, iter_glyphs, tokenize_lossless


LANGUAGES = ("en", "zh-Hans", "es", "hi", "ar")
SEMANTIC_MARKER = "⟪SEM⟫"

PROGRAM_SUMMARY_TEMPLATES: dict[str, str] = {
    "en": "{code_language} program for {intent}. Canonical glyphs {glyphs}.",
    "zh-Hans": "{code_language} 程序用于{intent}。规范字形 {glyphs}。",
    "es": "Programa de {code_language} para {intent}. Glifos canónicos {glyphs}.",
    "hi": "{code_language} प्रोग्राम {intent} के लिए। मानक ग्लिफ {glyphs}।",
    "ar": "برنامج {code_language} لغرض {intent}. الرموز المعيارية {glyphs}.",
}

SYSTEM_SUMMARY_TEMPLATES: dict[str, str] = {
    "en": "{title}: {description} Expected output {stdout}.",
    "zh-Hans": "{title}：{description} 预期输出 {stdout}。",
    "es": "{title}: {description} Salida esperada {stdout}.",
    "hi": "{title}: {description} अपेक्षित आउटपुट {stdout}।",
    "ar": "{title}: {description} والمخرجات المتوقعة {stdout}.",
}


def language_token(language: str) -> str:
    return f"⟪LANG:{language}⟫"


def concept_token(concept: str) -> str:
    return f"⟪CONCEPT:{concept}⟫"


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return ()
    return (
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


class SemanticCorpusBuilder:
    def __init__(self) -> None:
        self.counts: Counter[str] = Counter()
        self.meanings: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._meaning_keys: dict[str, set[str]] = defaultdict(set)
        self.fixed_glyphs: dict[str, str] = {}
        self.corpus: list[dict[str, Any]] = []
        self.languages: set[str] = set()

    def add_text(
        self,
        text: str,
        *,
        language: str = "und",
        source: str,
        concepts: Iterable[str] = (),
        weight: int = 1,
    ) -> None:
        if not text:
            return
        if language not in {"und", "mul"}:
            self.languages.add(language)
        tokens = tokenize_lossless(text)
        self.counts.update({token: weight for token in tokens})
        self.counts[language_token(language)] += weight
        concept_list = [str(concept) for concept in concepts if str(concept)]
        if concept_list:
            self.counts[SEMANTIC_MARKER] += weight
            for concept in concept_list:
                canonical = concept_token(concept)
                self.counts[canonical] += weight
                self.add_meaning(
                    canonical,
                    {
                        "language": None,
                        "concept_id": concept,
                        "kind": "canonical-concept-token",
                    },
                    weight=weight,
                )
        self.corpus.append(
            {
                "text": text,
                "language": language,
                "source": source,
                "concept_sequence": concept_list,
            }
        )

    def add_token_sequence(
        self,
        tokens: Iterable[str],
        *,
        text: str,
        language: str,
        source: str,
        weight: int = 1,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        sequence = [str(token) for token in tokens if str(token)]
        if not sequence:
            return
        self.languages.add(language)
        self.counts.update({token: weight for token in sequence})
        self.counts[language_token(language)] += weight
        self.corpus.append(
            {
                "text": text,
                "language": language,
                "source": source,
                "tokens": sequence,
                **(metadata or {}),
            }
        )

    def add_fixed_glyph(
        self,
        token: str,
        glyph: str,
        meaning: dict[str, Any],
        *,
        weight: int = 100_000,
    ) -> None:
        if len(glyph) != 1:
            raise ValueError(f"fixed glyph must be one code point: {glyph!r}")
        existing = self.fixed_glyphs.get(token)
        if existing is not None and existing != glyph:
            raise ValueError(f"token {token!r} has conflicting fixed glyphs")
        if glyph in self.fixed_glyphs.values() and existing != glyph:
            raise ValueError(f"fixed glyph is already assigned: {glyph!r}")
        self.fixed_glyphs[token] = glyph
        self.counts[token] += weight
        key = json.dumps(meaning, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if key not in self._meaning_keys[token]:
            self.meanings[token].append(meaning)
            self._meaning_keys[token].add(key)

    def ingest_programming_syntax(self, *, repeats: int = 64) -> None:
        for token, glyph, meaning in all_fixed_program_glyphs():
            self.add_fixed_glyph(token, glyph, meaning)
        for row in iter_program_training_rows(repeats=repeats):
            metadata = {
                key: value
                for key, value in row.items()
                if key not in {"text", "language", "source", "tokens"}
            }
            self.add_token_sequence(
                row["tokens"],
                text=str(row["text"]),
                language=str(row["language"]),
                source=str(row["source"]),
                weight=16,
                metadata=metadata,
            )
        for row in iter_executable_system_rows(repeats=max(1, repeats // 2)):
            metadata = {
                key: value
                for key, value in row.items()
                if key not in {"text", "language", "source", "tokens"}
            }
            self.add_token_sequence(
                row["tokens"],
                text=str(row["text"]),
                language=str(row["language"]),
                source=str(row["source"]),
                weight=24,
                metadata=metadata,
            )

    def ingest_programming_semantics(self) -> None:
        for example in PROGRAM_EXAMPLES:
            intent = example.intent.replace("-", " ")
            for code_language in sorted(example.sources):
                for language, template in PROGRAM_SUMMARY_TEMPLATES.items():
                    self.add_text(
                        template.format(
                            code_language=code_language,
                            intent=intent,
                            glyphs=example.canonical_glyphs,
                        ),
                        language=language,
                        source="glyphmatics/programming_semantics",
                        concepts=[f"PROGRAM.{example.intent}", f"LANGUAGE.{code_language}"],
                        weight=6,
                    )

        for system in EXECUTABLE_SYSTEMS:
            stdout = system.expected_stdout.rstrip("\n") or "∅"
            for language, template in SYSTEM_SUMMARY_TEMPLATES.items():
                self.add_text(
                    template.format(
                        title=system.title,
                        description=system.description,
                        stdout=stdout,
                    ),
                    language=language,
                    source="glyphmatics/executable_semantics",
                    concepts=[f"SYSTEM.{system.system_id}"],
                    weight=8,
                )

    def add_meaning(self, token: str, meaning: dict[str, Any], *, weight: int = 8) -> None:
        if not token:
            return
        tokens = tokenize_lossless(token)
        if len(tokens) != 1:
            return
        token = tokens[0]
        language = meaning.get("language")
        if language and language not in {"und", "mul"}:
            self.languages.add(str(language))
        self.counts[token] += weight
        key = json.dumps(meaning, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if key not in self._meaning_keys[token]:
            self.meanings[token].append(meaning)
            self._meaning_keys[token].add(key)

    def ingest_maestro(self, root: Path) -> None:
        semantic = root / "semantic_language_v1"
        datasets = root / "datasets"
        concepts = {
            row["concept_id"]: row
            for row in read_jsonl(semantic / "concepts.jsonl")
            if row.get("concept_id")
        }

        for row in read_jsonl(semantic / "word_forms.jsonl"):
            surface = str(row.get("surface", ""))
            concept = concepts.get(row.get("concept_id"), {})
            meaning = {
                "language": row.get("language"),
                "concept_id": row.get("concept_id"),
                "sense_id": row.get("sense_id"),
                "part_of_speech": row.get("part_of_speech"),
                "gloss": row.get("gloss") or concept.get("definition"),
                "relations": concept.get("relations", []),
            }
            self.add_meaning(surface, meaning, weight=32)
            self.add_text(
                surface,
                language=str(row.get("language", "und")),
                source="semantic_language_v1/word_forms.jsonl",
                concepts=[str(row.get("concept_id", ""))],
                weight=8,
            )

        for row in read_jsonl(semantic / "parallel_sentences.jsonl"):
            concepts_in_sentence = row.get("concept_sequence", [])
            for language, text in row.get("translations", {}).items():
                self.add_text(
                    str(text),
                    language=str(language),
                    source="semantic_language_v1/parallel_sentences.jsonl",
                    concepts=concepts_in_sentence,
                    weight=16,
                )

        for filename in ("language_lexicon_seed.jsonl", "language_lexicon_10000.jsonl"):
            for row in read_jsonl(datasets / filename):
                surface = str(row.get("word") or row.get("surface") or "")
                language = str(row.get("language", "und"))
                meaning = {
                    "language": language,
                    "concept_id": row.get("concept_id"),
                    "sense_id": row.get("sense_id"),
                    "gloss": row.get("meaning"),
                    "context": row.get("context"),
                    "glyph_sequence": row.get("glyph_sequence", []),
                }
                self.add_meaning(surface, meaning)
                context = str(row.get("context") or "")
                if context:
                    self.add_text(
                        context,
                        language=language,
                        source=f"datasets/{filename}:context",
                        weight=1,
                    )

        for row in read_jsonl(datasets / "semantic_llm_training.jsonl"):
            languages = row.get("languages") or ["und"]
            language = str(languages[0]) if len(languages) == 1 else "mul"
            text = str(row.get("text") or "")
            self.add_text(
                text,
                language=language,
                source="datasets/semantic_llm_training.jsonl",
                concepts=row.get("glyph_sequence", []),
                weight=1,
            )

        for filename in ("general_concept_training.jsonl", "programming_language_training.jsonl"):
            for row in read_jsonl(datasets / filename):
                forms = row.get("forms")
                if isinstance(forms, dict):
                    for language, surface in forms.items():
                        self.add_meaning(
                            str(surface),
                            {
                                "language": language,
                                "concept_id": row.get("concept_id"),
                                "gloss": row.get("target"),
                                "glyph_sequence": row.get("glyph_sequence", []),
                            },
                        )
                surface = row.get("surface")
                if surface:
                    self.add_meaning(
                        str(surface),
                        {
                            "language": "en",
                            "concept_id": row.get("term_id"),
                            "gloss": row.get("target"),
                            "domain": row.get("domain"),
                            "glyph_sequence": row.get("glyph_sequence", []),
                        },
                    )

    def build_vocabulary(self, *, max_tokens: int = 60000, min_frequency: int = 1) -> SemanticVocabulary:
        # Make exact layout tokens cheap and stable.
        self.counts[" "] += 1_000_000
        self.counts["\n"] += 500_000
        for token in ".,!?;:()[]{}\"'=-_/":
            self.counts[token] += 100_000

        candidates = [
            (token, count)
            for token, count in self.counts.items()
            if count >= min_frequency and token not in {"\ue000", "\ue001"}
        ]
        candidates.sort(key=lambda item: (-item[1], item[0]))
        candidates = candidates[:max_tokens]
        reserved_glyphs = set(self.fixed_glyphs.values())
        glyph_source = (
            glyph for glyph in iter_glyphs() if glyph not in reserved_glyphs
        )
        records = []
        for token_id, (token, count) in enumerate(candidates):
            glyph = self.fixed_glyphs.get(token)
            records.append(
                {
                    "id": token_id,
                    "token": token,
                    "glyph": glyph if glyph is not None else next(glyph_source),
                    "frequency": count,
                    "languages": sorted(
                        {
                            item["language"]
                            for item in self.meanings.get(token, [])
                            if item.get("language")
                        }
                    ),
                    "semantics": self.meanings.get(token, []),
                }
            )
        return SemanticVocabulary(
            records,
            metadata={
                "languages": sorted(self.languages or LANGUAGES),
                "ordering": "frequency-descending-then-codepoint",
                "lossless": True,
                "unknown_policy": "utf8-braille-literal",
                "corpus_records": len(self.corpus),
            },
        )

    def write_corpus(self, path: Path, vocabulary: SemanticVocabulary) -> dict[str, Any]:
        path.parent.mkdir(parents=True, exist_ok=True)
        language_counts: Counter[str] = Counter()
        token_count = 0
        unknown_count = 0
        with path.open("w", encoding="utf-8") as handle:
            for row in self.corpus:
                tokens = row.get("tokens") or tokenize_lossless(row["text"])
                ids = [vocabulary.model_id(language_token(row["language"]))]
                ids.extend(vocabulary.model_id(token) for token in tokens)
                concepts = row.get("concept_sequence", [])
                if concepts:
                    ids.append(vocabulary.model_id(SEMANTIC_MARKER))
                    ids.extend(vocabulary.model_id(concept_token(item)) for item in concepts)
                token_count += len(ids)
                unknown_count += sum(model_id == 3 for model_id in ids)
                language_counts[row["language"]] += 1
                output = {**row, "ids": ids}
                handle.write(json.dumps(output, ensure_ascii=False, separators=(",", ":")) + "\n")
        return {
            "records": len(self.corpus),
            "tokens": token_count,
            "unknown_tokens": unknown_count,
            "languages": dict(sorted(language_counts.items())),
        }


def build_artifacts(
    semantic_root: str | Path,
    output_dir: str | Path,
    *,
    max_tokens: int = 60000,
    min_frequency: int = 1,
) -> dict[str, Any]:
    builder = SemanticCorpusBuilder()
    builder.ingest_maestro(Path(semantic_root))
    builder.ingest_programming_syntax(repeats=128)
    builder.ingest_programming_semantics()
    vocabulary = builder.build_vocabulary(max_tokens=max_tokens, min_frequency=min_frequency)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    vocabulary_path = output / "semantic_vocab.json"
    corpus_path = output / "semantic_corpus.jsonl"
    vocabulary.save(vocabulary_path)
    corpus_stats = builder.write_corpus(corpus_path, vocabulary)
    manifest = {
        "format": "glyphmatics-semantic-build-v1",
        "vocabulary": str(vocabulary_path),
        "corpus": str(corpus_path),
        "vocabulary_size": len(vocabulary.records),
        "model_vocabulary_size": vocabulary.model_vocab_size,
        "vocabulary_sha256": vocabulary.digest,
        "corpus": corpus_stats,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
