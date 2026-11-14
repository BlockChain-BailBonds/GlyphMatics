# GlyphMatics Offline Inference Report

## Verified local inference

The Qwen3-1.7B Q4 model generated real autoregressive tokens from both PNG implementations. The fast path reports the model pathname as `vault-fast/918_Qwen3_1.7B.fast.png`; no reconstructed GGUF pathname is used.

Arithmetic prompt, fixed seed 918, temperature 0, context 1024:

> What is 17 multiplied by 23?

Result: `391` (**VERIFIED**).

The conventional GGUF suite also completed reasoning, factorial code, unique-identifier retention, and 256-token sustained generation. Those results establish the reference model but not PNG acceptance.

Compressed-PNG anonymous-memory inference generated real tokens but reduced `MemAvailable` to about 364 MB during a second test. The suite was terminated according to the safety rule. Status: **FAILED** as the production execution path.

Fast mmap PNG inference completed cleanly and produced the same arithmetic response as GGUF. Status: **VERIFIED** for local PNG-backed inference.

## Offline proof boundary

The runtime has no network client dependency and `LLAMA_CURL=OFF` was used in the source build. However, Wi-Fi/mobile data were not disabled through Android controls during this session. End-to-end operator-observed airplane/offline restart testing is therefore **UNVERIFIED**, not claimed.
