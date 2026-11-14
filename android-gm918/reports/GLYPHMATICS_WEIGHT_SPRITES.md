# GlyphMatics Weight Sprites

Status: **VERIFIED** on the Samsung SM-A146U.

The production Qwen3-1.7B Q4_K_M image now carries a `gmSI` weight-sprite
index. Each sprite is a 64×64 8-bit grayscale byte plane: exactly one 4 KiB
Android memory page. Sprite `n` maps to virtual model bytes
`n * 4096 .. min((n + 1) * 4096, model_bytes)`.

- Sprite geometry: 64×64×8-bit
- Bytes per complete sprite: 4,096
- Sprite count: 313,096
- Final sprite bytes: 2,144
- Model payload bytes: 1,282,439,264
- Complete PNG bytes: 1,282,443,376
- PNG overhead: 4,112 bytes
- Model SHA-256: `d2387ca2dbfee2ffabce7120d3770dadca0b293052bc2f0e138fdc940d9bc7b5`
- Sprite PNG SHA-256: `754e42e56396876139de62eac5bb1e9259f192c5ec673cd03bfd60a254381b94`

The `gmDT` payload begins at byte 4096. The patched loader maps it directly;
the kernel page cache supplies individual weight sprites on demand. There is
no reconstructed GGUF and no whole-model decode buffer. Quantized matrix
kernels consume the bytes at their mapped addresses.

Strict verification checked PNG CRCs, sprite geometry/count, page alignment,
payload size, and the complete payload SHA-256. Real transformer inference from
the sprite image produced 23.4 prompt tok/s and 9.5 generation tok/s in the
production smoke test. Evidence: `evidence/sprite-production-smoke.txt`.

The previous non-indexed fast PNG was replaced only after byte comparison,
strict verification, and inference succeeded. Its exact payload remains
recoverable from the preserved source model and deterministic encoder.
