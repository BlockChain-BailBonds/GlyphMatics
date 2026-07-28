# GlyphMatics

GlyphMatics is a deterministic glyph-compression and training pipeline for:

- multilingual semantic text encoding
- cross-language programming syntax encoding
- executable-system glyph IDs for complete runnable code systems
- small reference language models trained directly on glyph-token sequences

The repo now includes a GitHub Pages benchmark surface, a reversible semantic
codec, a 15-language programming glyph layer, and a 50-system executable glyph
inventory trained into the latest reference checkpoint.

## What is in this repo

- Semantic codec: exact tokenization, visible glyph encoding, binary encoding,
  and round-trip verification in [semantic_codec.py](src/glyphmatics/semantic_codec.py)
- Programming glyphs: canonical syntax glyphs for Python, JavaScript,
  TypeScript, Rust, Go, Java, C, C++, C#, Ruby, PHP, Swift, Kotlin, Bash, and
  SQL in [programming_syntax.py](src/glyphmatics/programming_syntax.py)
- Executable system glyphs: 50 complete Python reference programs, each mapped
  to one fixed glyph token in [programming_syntax.py](src/glyphmatics/programming_syntax.py)
- Corpus + vocabulary builder in [semantic_data.py](src/glyphmatics/semantic_data.py)
- Reference Transformer training loop in [semantic_train.py](src/glyphmatics/semantic_train.py)
- GitHub Pages benchmark site in [docs](docs)

## GitHub Pages benchmark

The repo ships a static benchmark site intended for GitHub Pages deployment:

- Pages entrypoint: [docs/index.html](docs/index.html)
- Benchmark page: [docs/benchmark/index.html](docs/benchmark/index.html)
- Generator: [scripts/build_pages_benchmark.py](scripts/build_pages_benchmark.py)
- Workflow: [.github/workflows/pages.yml](.github/workflows/pages.yml)

The benchmark combines two local artifacts:

- `semantic-programming-v2` for multilingual semantic-text compression
- `semantic-programming-systems-v3` for the 50 executable-system glyphs

It includes:

- browser-side semantic benchmark logic replaying the repo codec rules
- precomputed tokenizer baselines using real BPE encodings
- generated multilingual sample benchmarks
- a dedicated decomposed-Unicode combining-diacritic benchmark sample
- generated tables for all 50 executable systems

Rebuild the Pages benchmark assets from the repo root:

```bash
PYTHONPATH=src:. python3 scripts/build_pages_benchmark.py
```

## Core commands

Build a semantic artifact:

```bash
PYTHONPATH=src:. python -m glyphmatics.semantic_cli build \
  --semantic-root /path/to/semantic-root \
  --output artifacts/semantic-v1
```

Encode and inspect text:

```bash
VOCAB=artifacts/semantic-programming-v2/semantic_vocab.json

PYTHONPATH=src:. python -m glyphmatics.semantic_cli encode \
  --vocabulary "$VOCAB" "The person eats food."

PYTHONPATH=src:. python -m glyphmatics.semantic_cli stats \
  --vocabulary "$VOCAB" "The person eats food."
```

Encode exact program syntax and canonical IR:

```bash
PYTHONPATH=src:. python -m glyphmatics.semantic_cli code-encode \
  --language python 'if score > 80: celebrate()'
```

Train the reference glyph LM:

```bash
PYTHONPATH=src:. /home/nine1eight/vil/glyph_env/bin/python \
  -m glyphmatics.semantic_cli train \
  --vocabulary artifacts/semantic-programming-systems-v3/semantic_vocab.json \
  --corpus artifacts/semantic-programming-systems-v3/semantic_corpus.jsonl \
  --checkpoint checkpoints/semantic-programming-glyph-lm-v3.pt \
  --device auto \
  --steps 2400
```

## Current local artifacts

Multilingual + programming semantic artifact:

- [artifacts/semantic-programming-v2/manifest.json](artifacts/semantic-programming-v2/manifest.json)
- vocabulary size: `22,659`
- model vocabulary size: `22,663`
- corpus records: `29,875`

Executable-system artifact:

- [artifacts/semantic-programming-systems-v3/manifest.json](artifacts/semantic-programming-systems-v3/manifest.json)
- vocabulary size: `547`
- model vocabulary size: `551`
- corpus records: `7,360`
- executable systems: `50`

Latest trained checkpoint:

- [checkpoints/semantic-programming-glyph-lm-v3.pt](checkpoints/semantic-programming-glyph-lm-v3.pt)
- parameters: `1,910,208`
- final loss: `0.16729924`
- validation loss: `0.16266710`
- validation perplexity: `1.1766449`
- post-train executable-system prediction check: `50/50`

## Benchmark interpretation

GlyphMatics measures several different things. They are not interchangeable.

- Visible glyph compression measures sequence length in the rendered glyph
  stream.
- Tokenizer baseline comparisons measure the same samples against real BPE
  token counts using `gpt2` and `cl100k_base`.
- Binary compression measures the repo’s storage-oriented codec, including a
  vocabulary checksum header.
- Combining diacritics are preserved in decomposed form; the codec does not
  normalize away Unicode marks before tokenization or round-trip checks.
- Programming lossless glyphs preserve exact code source with syntax glyphs and
  escaped literals.
- Executable-system glyphs are fixed learned identifiers for complete reference
  programs, not arbitrary source-code compressors for unseen software.

The benchmark page reports these surfaces separately so the claims stay tied to
the actual implementation.

## Verification status

Focused checks run locally before this state:

- `89 passed in 0.93s`
- `50/50` executable-system unit executions matched expected stdout
- `50/50` post-train executable-system token predictions matched exactly under
  the model’s real `128`-token context window

## Technical notes

Further details live in [docs/VIL_SEMANTIC_LM.md](docs/VIL_SEMANTIC_LM.md).
