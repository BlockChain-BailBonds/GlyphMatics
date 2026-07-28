# VIL Multilingual Semantic Glyph LM

This implementation turns multilingual surface text into a deterministic
semantic glyph stream and trains a causal Transformer directly on those glyph
IDs.

## Contracts

There are three deliberately separate representations:

1. **Semantic model IDs** are compact integer tokens used by the Transformer.
2. **Visible glyphs** render every known token as one Unicode private-use
   character. Unknown text is carried by an explicit UTF-8/Braille escape.
3. **Glyph-LLM3 pages** wrap arbitrary artifacts in nested zlib + CRC32 frames
   and render the framed bytes as Braille.

The separation matters. One Braille character represents one byte and is a
transport encoding, not compression. A single semantic glyph can represent a
whole known word, but its meaning depends on a shared, checksum-identified
vocabulary.

No normalization is performed. Capitalization, whitespace, punctuation,
combining marks, and unknown Unicode round-trip byte-for-byte.

## Build the multilingual corpus

From the repository root:

```bash
PYTHONPATH=src:. python -m glyphmatics.semantic_cli build \
  --semantic-root /home/nine1eight/maestro_codex_template \
  --output artifacts/semantic-v1
```

The concept-first seed covers English, Simplified Chinese, Spanish, Hindi, and
Modern Standard Arabic. The extended local lexicon currently expands the build
to 28 language codes. Every exact surface form has its own reversible token;
concept and sense metadata connect forms across languages.

Training rows prepend a language-control glyph and append canonical concept
glyphs after a semantic boundary glyph. The causal model therefore learns both
surface continuation and sentence-to-concept structure.

## Encode, decode, and inspect meaning

```bash
VOCAB=artifacts/semantic-v1/semantic_vocab.json

PYTHONPATH=src:. python -m glyphmatics.semantic_cli encode \
  --vocabulary "$VOCAB" "I want water."

PYTHONPATH=src:. python -m glyphmatics.semantic_cli inspect \
  --vocabulary "$VOCAB" --language en water

PYTHONPATH=src:. python -m glyphmatics.semantic_cli stats \
  --vocabulary "$VOCAB" "I want water."
```

Use `--binary` on `encode` and `decode` for the storage representation. It
returns or accepts base64 at the terminal boundary.

## Train the language model

```bash
PYTHONPATH=src:. /home/nine1eight/vil/glyph_env/bin/python \
  -m glyphmatics.semantic_cli train \
  --vocabulary artifacts/semantic-v1/semantic_vocab.json \
  --corpus artifacts/semantic-v1/semantic_corpus.jsonl \
  --checkpoint checkpoints/semantic-glyph-lm.pt \
  --device cuda --steps 500
```

The default model is a small reference Transformer intended to prove the full
pipeline on local hardware. Model size, layers, heads, context, batch size, and
training duration are configurable. A production multilingual LLM still needs
substantially more licensed text, held-out evaluation, and longer training.

## Glyph-LLM3 artifact transport

```bash
PYTHONPATH=src:. python -m glyphmatics.semantic_cli pack \
  --title "Semantic Glyph LM" --name semantic_model \
  --input checkpoints/semantic-glyph-lm.pt \
  --output checkpoints/semantic-glyph-lm.glyphpage

PYTHONPATH=src:. python -m glyphmatics.semantic_cli unpack \
  --name semantic_model \
  --input checkpoints/semantic-glyph-lm.glyphpage \
  --output checkpoints/recovered.pt
```

CRC failures, malformed lengths, wrong model vocabulary hashes, and unknown
unescaped glyphs are rejected.

## Source architecture

- <https://github.com/xxNine1Eightxx/GlyphMatics>
- <https://github.com/GOD-IAM/SigilAGI>
- <https://github.com/Nine1Eight/Glyph-LLM3>
- <https://github.com/Nine1Eight/ApexAgentSigilagiGlyphNotes/blob/main/GlyphNotesCodex>
- <https://github.com/sigilagi918/-GylphMatics-Encoder->
- <https://huggingface.co/datasets/Nine1Eight/vil-canonical-glyph-system>

## Compression claims

“One word becomes one glyph” is true for tokens present in the shared
vocabulary. It reduces model sequence length and can reduce binary storage when
frequent tokens receive small IDs.

It does not imply that every visible Unicode glyph occupies one byte in UTF-8,
and it cannot compress arbitrary unknown or already-compressed data to one
character without external information. The `stats` command reports source
bytes, visible-glyph bytes, binary bytes, coverage, and verified round-trip
status separately.
