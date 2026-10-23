# PrismaCoder: GlyphMatics glyph and compute-sigil renderer

## Responsibility

PrismaCoder is the designated AI-assisted **visual renderer** for GlyphMatics glyphs and compute sigils. Existing VIL, BrailleByte, canonical glyph registry and evidence ledger remain authoritative. Generated SVG is never a substitute for encoding, decoding, role preservation, uncertainty tracking or verification of meaning.

The optional PrismaCoder-9B model generates SVG through an explicitly configured OpenAI-compatible chat-completions endpoint. The validator checks IDs, exact labels and equation annotations, viewport dimensions and SVG active-content safety. Failures use a deterministic fallback unless the caller sets fail-closed behavior.

Each output includes an explicit rendering source (prismacoder or deterministic_fallback), SHA-256 hashes for canonical input and SVG output, and any fallback reason. PrismaCoder weights are neither bundled nor silently downloaded.

## Input schema

The version 1 JSON rendering specification includes a canvas size in integer pixels (240 through 4096), a list of glyph entries with unique identifiers from G005 through G332 and exact labels, and optional compute sigils with stable IDs, equations and 2-1024 finite waveform samples.

See [sample specification](../examples/sigil_render_spec.json).

**Callers must resolve glyph identities against the canonical registry before generating this rendering request.** The renderer does not invent meanings or replace the registry.

## Offline demonstration

From the repository root:

    PYTHONPATH=src python -m glyphmatics.visualization.prismacoder examples/sigil_render_spec.json sigil.svg

The command produces sigil.svg and sigil.svg.json (a provenance manifest). Without an endpoint the source field is deterministic_fallback.

## Hosted PrismaCoder demonstration

Serve a separately acquired PrismaCoder model on an operator-controlled OpenAI-compatible endpoint, then configure it:

    export PRISMACODER_ENDPOINT=http://127.0.0.1:8000/v1/chat/completions
    export PRISMACODER_MODEL=PrismaCoder-9B
    PYTHONPATH=src python -m glyphmatics.visualization.prismacoder examples/sigil_render_spec.json sigil.svg --fail-closed

For authenticated endpoints, set PRISMACODER_API_KEY. Do not send private canonical records to an untrusted endpoint.

## Safety and acceptance

1. Preserve all GlyphMatics/VIL/BrailleByte canonical semantics outside the visual renderer.
2. Reject unsafe active SVG content, unknown attributes, unexpected elements, mismatched IDs, altered labels and equations, and wrong dimensions.
3. Distinguish actual model generation from deterministic fallback in every manifest and evidence record.
4. SVG checks are structural; they do **not** prove a model-drawn curve matches every numerical sample. Use the deterministic renderer or independent numerical comparisons for high-integrity sigils.
5. Rasterize validated SVG using a trusted host tool before deploying a 480×320 M3rMa1d asset; do not attempt to run the model on ESP32-S3 firmware.
6. Keep model inference opt-in, and keep CI tests independent of GPU and network dependencies.

Research: https://huggingface.co/papers/2610.20538

Source: https://github.com/QiushiSun/PrismaCoder

Model: https://huggingface.co/QiushiSun/PrismaCoder-9B
