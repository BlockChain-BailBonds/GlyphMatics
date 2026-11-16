"""Opt-in SVG contract against a deployed PrismaCoder endpoint (never offline CI)."""

import hashlib
import json
import os
from pathlib import Path

import pytest

from glyphmatics.visualization.prismacoder import PrismaCoderRenderer


SAMPLE_SPEC = Path(__file__).resolve().parents[2] / "examples" / "sigil_render_spec.json"


def test_deployed_prismacoder_svg_contract():
    if os.getenv("PRISMACODER_LIVE_TEST") != "1":
        pytest.skip("live model access is opt-in: PRISMACODER_LIVE_TEST=1")

    endpoint = os.getenv("PRISMACODER_ENDPOINT")
    model = os.getenv("PRISMACODER_MODEL")
    if not endpoint or not model:
        pytest.fail("live test requires PRISMACODER_ENDPOINT and PRISMACODER_MODEL")

    spec = json.loads(SAMPLE_SPEC.read_text(encoding="utf-8"))
    result = PrismaCoderRenderer(
        endpoint=endpoint,
        model=model,
        api_key=os.getenv("PRISMACODER_API_KEY"),
        fallback=False,
    ).render(spec)
    manifest = result.manifest()

    # render() verifies SVG structure, canonical IDs/text, canvas dimensions,
    # and active-content safety; it does not prove waveform pixel fidelity.
    assert result.svg
    assert manifest["source"] == "prismacoder"
    assert manifest["fallback_reason"] is None
    assert "svg" not in manifest
    assert manifest["svg_sha256"] == hashlib.sha256(result.svg.encode("utf-8")).hexdigest()
    canonical_spec = json.dumps(
        spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    assert manifest["specification_sha256"] == hashlib.sha256(canonical_spec).hexdigest()
    print(json.dumps({"requested_model": model, "manifest": manifest}, sort_keys=True))
