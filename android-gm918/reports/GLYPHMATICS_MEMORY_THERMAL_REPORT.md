# GlyphMatics Memory and Thermal Report

## Conventional GGUF reference

- Peak sampled RSS across long cases: approximately 1,421,380 kB
- 256-token case elapsed: 96.75 s using the older four-thread packaged runtime
- No crash or Android process termination occurred

## Compressed PNG bridge

- Full model inflation into `memfd` caused `MemAvailable` to fall to approximately 364 MB.
- Test was terminated; no Android safety mechanism was disabled.
- Status: **FAILED** for production memory requirements.

## Fast mmap PNG

- File-backed pages remain reclaimable like an Android application model mapping.
- Warm Q4 result with source-built runtime: 26.9 prompt tok/s, 7.2 generation tok/s at four threads.
- Two-thread tuning reached 24.6 prompt tok/s and 9.4 generation tok/s in the short controlled test.
- Production launcher pins two threads to Cortex-A76 cores 6–7.
- Three paired scheduler tests averaged 22.2/8.3 prompt/generation tok/s for
  affinity alone and 23.5/9.1 with strict placement plus medium priority.
  Individual results varied; see `GLYPHMATICS_SCHEDULER_BENCHMARK.md`.

Thermal temperature and battery impact are **BLOCKED** because Android denied the Termux UID access to thermal and battery services. Frequency changes and performance variance indicate scheduling/thermal sensitivity, but no temperature is inferred.
