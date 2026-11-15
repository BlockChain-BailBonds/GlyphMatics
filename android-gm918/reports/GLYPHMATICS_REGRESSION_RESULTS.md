# GlyphMatics Regression Results

| Test | Status | Evidence |
|---|---|---|
| Synthetic random-data round trip | VERIFIED | 2,097,152 bytes; SHA-256 exact |
| Full Q4 compressed-vault reconstruction | VERIFIED | 1,282,439,264 bytes; SHA-256 exact |
| Missing PNG tile rejection | VERIFIED | `manifest tile count mismatch` |
| Truncated PNG rejection | VERIFIED | `truncated PNG chunk` |
| Modified PNG data rejection | VERIFIED | `PNG CRC mismatch` |
| Invalid tensor/block offset rejection | VERIFIED | `tile index/offset mismatch` |
| Unsupported version rejection | VERIFIED | controlled version error |
| Corrupted manifest rejection | VERIFIED | inflate/integrity error |
| Compressed PNG real inference | VERIFIED | neural tokens generated from `/proc/self/fd/3` |
| Compressed PNG sustained safety | FAILED | available RAM reached ~364 MB; stopped |
| Fast PNG real inference | VERIFIED | correct arithmetic, direct PNG pathname |
| Fast PNG CRC/alignment/payload SHA | VERIFIED | strict streaming verifier; no extraction |
| 4 KiB weight-sprite index | VERIFIED | 313,096 sprites; exact full-model SHA; real inference |
| Native metadata CRC rejection | VERIFIED | corrupted metadata rejected before model parsing |
| Missing native sprite index | VERIFIED | old Q3 rejected; no fallback |
| Native sprite geometry validation | VERIFIED | malformed 32×64 index rejected |
| Native payload length validation | VERIFIED | gmHD/gmDT mismatch rejected |
| Q3 sprite conversion | VERIFIED | 262,023 sprites; payload SHA preserved; real inference |
| Strict A76 scheduling | VERIFIED | mean +5.9% prompt, +9.6% generation over three pairs; priority denied |
| N-gram speculation candidate | FAILED | 9.3 to 4.9 gen tok/s; output changed; rejected |
| Fast PNG vs GGUF warm speed | VERIFIED | near parity; 26.9/7.2 vs 27.1/7.5 tok/s |
| Q3 size reduction | VERIFIED | 16.31% smaller than Q4 |
| Q3 broad quality | UNVERIFIED | arithmetic only |
| Offline radios-disabled restart | UNVERIFIED | requires operator Android controls |
| Battery/thermal metrics | BLOCKED | Android permission denied |
| Incorrect tokenizer metadata | UNVERIFIED | authenticated mutation fixture not yet implemented |
| Low-RAM controlled rejection | VERIFIED | benchmark stopped at safety threshold |

Raw results are under `evidence/`.
