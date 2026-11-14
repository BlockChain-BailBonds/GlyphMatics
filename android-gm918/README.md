# 918 Technologies Offline AI

This Samsung Galaxy A14 5G runs Qwen3 1.7B Q4_K_M from a page-aligned
GlyphMatics PNG. The patched ARM64 runtime memory-maps model pages directly
from the PNG and never writes an unpacked GGUF. Network support is disabled.

## Commands

Start an interactive local conversation:

```sh
offline-codex
```

Run one prompt without the slower reasoning mode:

```sh
offline-codex -p "Write a Python JSON parser" -n 256
```

Verify the PNG structure, chunk CRCs, alignment, and complete model SHA-256:

```sh
python ~/glyphmatics/sigilvm4/verify_mmap_png.py \
  ~/glyphmatics/vault-fast/918_Qwen3_1.7B.fast.png \
  --sha256 d2387ca2dbfee2ffabce7120d3770dadca0b293052bc2f0e138fdc940d9bc7b5
```

Use the 16.31% smaller Q3 candidate only when memory/storage matters more than
maximum quality (its broad quality suite is not yet complete):

```sh
offline-codex-lowmem
```

## Execution format

The canonical execution file is a valid PNG containing a `GM918MMAP4` header
and a page-aligned `gmDT` model stream. Its `gmSI` index divides weights into
313,096 sprites of 64×64 grayscale bytes—one 4 KiB Android page per sprite.
The model stream includes architecture,
tokenizer, quantized tensors, and generation metadata. The runtime seeks to its
virtual offset and lets Android demand-page weights like application assets.
"Decode" means resolving those offsets and reading quantized blocks directly;
there is no whole-model decompression or reconstructed model file.

## Measured baseline

- Model bytes: 1,282,439,264
- SHA-256: `d2387ca2dbfee2ffabce7120d3770dadca0b293052bc2f0e138fdc940d9bc7b5`
- Context: 1,024 tokens
- Threads: 2, pinned to Cortex-A76 CPUs 6 and 7
- Best short controlled Q4 result: 24.6 prompt tokens/s, 9.4 generation tokens/s
- Warm PNG/GGUF comparison: 26.9/7.2 vs 27.1/7.5 tokens/s (near parity)

The current launcher provides local chat and generation. Faster-than-GGUF is
not yet claimed: direct PNG execution has reached near parity while preserving
the exact Q4 weights. Further speed work must optimize layout/prefetch/kernels,
not lower model quality without passing the fixed evaluation suite.
