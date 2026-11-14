# GlyphMatics Correction Queue

Smallest/highest-value first:

1. Cache the completed authenticated fast-PNG verification by size/mtime so it need not run on every launch.
2. Add an incorrect-tokenizer-metadata fixture with recomputed container hashes and confirm llama.cpp rejects it.
3. Complete Q3 reasoning, code, context, and sustained quality tests before considering promotion.
4. Add tensor-order tuning; current llama.cpp already uses whole-map `MAP_POPULATE`
   and `MADV_WILLNEED`, so do not add redundant prefetch. Accept only measured
   warm/cold gains. N-gram speculation was measured and rejected as slower.
5. Add PNG-encoded persistent prompt/KV cache with strict model/config binding.
6. Run operator-controlled radios-off inference, restart, and socket inspection.
7. Obtain battery/thermal evidence through authorized Termux:API or ADB `DUMP` permission.
8. Package training/fine-tuning provenance and adapter checkpoints as authenticated PNG assets. Full 1.7B base training is out of scope for 3.5 GiB RAM; adapter training or external training is required.
9. Evaluate a model distilled specifically for coding at the same or smaller parameter count; accept only if the fixed quality suite improves.
