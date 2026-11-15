# GlyphMatics Native Loader Security

Status: **VERIFIED** for implemented metadata checks.

The patched `gguf_init_from_file` now validates before parsing model bytes:

- PNG signature and bounded pre-model chunk lengths
- PNG CRC for every metadata chunk before `gmDT`
- exactly one valid `IHDR`, `gmHD`, and `gmSI`
- `GM918MMAP4` header and declared payload size
- 64×64×8-bit, 4096-byte sprite geometry
- sprite count and final-sprite length against payload bytes
- 4096-byte payload alignment
- required chunk ordering and `gmDT` length

The native regression suite verified controlled rejection of metadata CRC
corruption, a missing sprite index, invalid sprite geometry, and disagreement
between declared and actual payload size. Evidence:
`evidence/hardened-loader-regression.txt`.

Q4 and Q3 real inference passed after the hardened rebuild. The old non-sprite
Q3 image was rejected before conversion, demonstrating that the loader does not
silently fall back.

The large `gmDT` payload SHA-256 remains a provisioning-time verification step
using `verify_mmap_png.py`; hashing 1.28 GB at every startup would defeat fast
demand paging. Cryptographic signatures and a per-sprite Merkle tree remain
future work. PNG CRC is corruption detection, not publisher authentication.
