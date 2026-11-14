# GlyphMatics Compression Report

## Qwen3-1.7B Q4_K_M

| Representation | Bytes | Result |
|---|---:|---|
| Original pretrained checkpoint | unavailable locally | BLOCKED |
| Quantized GGUF Q4_K_M | 1,282,439,264 | VERIFIED |
| Compressed pixel PNG v4 vault | 1,254,888,627 | VERIFIED |
| PNG saving vs Q4 | 27,550,637 (2.15%) | VERIFIED |
| Fast mmap PNG | 1,282,443,376 | VERIFIED |
| Fast PNG overhead vs Q4 | 4,112 | VERIFIED |

Q4/model SHA-256: `d2387ca2dbfee2ffabce7120d3770dadca0b293052bc2f0e138fdc940d9bc7b5`.

The compressed vault contains 306 independent 4 MiB grayscale PNG model tiles plus one PNG manifest. Full native decode, per-tile SHA-256, PNG CRC, offsets, tile count, and whole-model SHA-256 passed. Full verification took 15.49 seconds.

The fast PNG stores the identical GGUF byte stream in a page-aligned `gmDT` PNG ancillary chunk. It is directly mmap-able but intentionally uncompressed. Archive and execution modes are separate because compressed PNG pixels cannot simultaneously be ordinary reclaimable mmap pages.

## Q3_K_M candidate

- GGUF: 1,073,242,528 bytes
- Fast PNG: 1,073,246,640 bytes
- Reduction from Q4: 16.31%
- SHA-256: `d544d30dbb7b2f608775d69c939a9b8d89c45df2be1a5ad1390ddc1bd7f8d6f5`
- Arithmetic output: correct (`391`)
- Warm performance: 12.1 prompt tok/s, 7.0 generation tok/s
- Status: **UNVERIFIED** for broad quality; retained only as low-memory option.

No claim is made that glyph appearance inherently compresses weights. No pretrained F16 checkpoint was downloaded, so total reduction versus original pretrained bytes is **BLOCKED**.
