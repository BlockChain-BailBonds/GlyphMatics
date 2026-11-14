# GlyphMatics Scheduler Benchmark

Date: 2026-11-14. Device: Samsung SM-A146U. Model: byte-identical
Qwen3-1.7B Q4_K_M GlyphMatics direct-mapped PNG. Context: 1024. Generation
threads: 2. Batch threads: 2. Seed: 918. Temperature: 0. Reasoning: off.

The candidate adds strict placement on Cortex-A76 CPUs 6–7. Both configurations
were already constrained with `taskset`. Android rejected the attempted medium
worker priority with `Operation not permitted`, so priority did not contribute
to the result and has been removed from the production launchers.

| Pair | Baseline prompt/gen tok/s | Strict prompt/gen tok/s |
|---|---:|---:|
| 1, 48 generated tokens | 23.0 / 7.9 | 25.1 / 9.2 |
| 2, 32 generated tokens | 21.1 / 8.6 | 24.9 / 9.9 |
| 3, 32 generated tokens | 22.5 / 8.5 | 20.5 / 8.3 |
| Arithmetic mean | 22.2 / 8.3 | 23.5 / 9.1 |

The deterministic output matched within every pair. Mean improvement was about
5.9% for prompt processing and 9.6% for generation. One pair reversed, showing
normal mobile thermal/scheduler variance. Status: **VERIFIED** as a measured
mean improvement, not a guaranteed per-run speedup.

Evidence: `evidence/scheduler-baseline*.txt` and
`evidence/scheduler-strict*.txt`.

## Rejected optimization

`ngram-simple` speculative decoding was tested on deterministic 128-token
Python generation. Baseline generation was 9.3 tok/s; n-gram generation was
4.9 tok/s, and output hashes differed. It is not enabled. Evidence:
`evidence/spec-baseline-code.txt` and `evidence/spec-ngram-code.txt`.
