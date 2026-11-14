# GlyphMatics ARM64 Build Report

Status: **VERIFIED**.

Pinned source: llama.cpp `5676dfbe01e7ebe0387913680eee0ed39bf4a363` (`0.9.0-dev`).

Build configuration:

```sh
cmake -S . -B build-gm918 \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_OPENMP=OFF \
  -DGGML_VULKAN=OFF \
  -DLLAMA_CURL=OFF
cmake --build build-gm918 --target llama-cli -j2
```

Detected optimized features: ARM dot product and FP16 vector arithmetic. i8mm, SVE and SME were correctly unavailable. The native loader patch is in `third_party/llama.cpp/ggml/src/gguf.cpp`.

Two loaders were built:

1. `sigilvm4/gm918-png-runtime`: validates compressed PNG tiles and can bridge them through anonymous memory. Correct but memory-unsafe for sustained 1.7B use on this device.
2. Patched `build-gm918/bin/llama-cli`: finds a page-aligned `gmDT` payload inside a valid PNG and mmaps tensor pages directly. This is the production path.

The built binary must use matching libraries from `build-gm918/bin`; the production launcher sets `LD_LIBRARY_PATH` explicitly.
