import hashlib
import json
import pytest
from glyphmatics.visualization.prismacoder import PrismaCoderRenderer


@pytest.fixture
def spec():
    return {
        "version": 1, "canvas": {"width": 480, "height": 320},
        "glyphs": [{"id": "G005", "label": "cause → effect"},
                   {"id": "G332", "label": "⠁ BrailleByte"}],
        "sigils": [{"id": "S001", "equation": "sin(2πt)",
                    "samples": [0.0, 1.0, 0.0, -1.0, 0.0]}],
    }


def test_deterministic_and_manifest(spec):
    a = PrismaCoderRenderer().render(spec)
    b = PrismaCoderRenderer().render(spec)
    assert a == b
    assert a.source == "deterministic_fallback"
    assert a.fallback_reason is None
    assert 'data-glyph-id="G005"' in a.svg
    assert 'data-sigil-id="S001"' in a.svg
    assert "sin(2πt)" in a.svg
    assert a.svg_sha256 == hashlib.sha256(a.svg.encode()).hexdigest()
    assert "svg" not in a.manifest()


def test_valid_prismacoder_output(spec):
    candidate = PrismaCoderRenderer().render(spec).svg
    result = PrismaCoderRenderer(generate=lambda _: candidate).render(spec)
    assert result.source == "prismacoder"


@pytest.mark.parametrize("payload", [
    '<script>alert(1)</script>', '<image href="https://example.com/track"/>',
    '<foreignObject/>', '<g onclick="alert(1)"/>',
    '<style>@import url(https://example.com/style.css)</style>',
    '<!DOCTYPE svg [<!ENTITY x "bad">]>',
])
def test_model_svg_security_fallback(spec, payload):
    valid = PrismaCoderRenderer().render(spec).svg
    hostile = valid.replace("</svg>", payload + "</svg>")
    fallback = PrismaCoderRenderer(generate=lambda _: hostile).render(spec)
    assert fallback.source == "deterministic_fallback"
    assert fallback.fallback_reason
    with pytest.raises(ValueError, match="rejected"):
        PrismaCoderRenderer(generate=lambda _: hostile, fallback=False).render(spec)


@pytest.mark.parametrize("old,new", [
    ('data-glyph-id="G005"', 'data-glyph-id="G006"'),
    ('data-sigil-id="S001"', 'data-sigil-id="S002"'),
    ("cause → effect", "cause ≠ effect"),
    ("sin(2πt)", "cos(2πt)"),
    ('viewBox="0 0 480 320"', 'viewBox="0 0 500 320"'),
])
def test_model_canonical_mismatch_falls_back(spec, old, new):
    valid = PrismaCoderRenderer().render(spec).svg
    result = PrismaCoderRenderer(generate=lambda _: valid.replace(old, new)).render(spec)
    assert result.source == "deterministic_fallback"
    assert result.fallback_reason


@pytest.mark.parametrize("edit", [
    lambda s: s.update(version=2),
    lambda s: s["glyphs"].append({"id": "G005", "label": "duplicated"}),
    lambda s: s["glyphs"][0].update(id="G001"),
    lambda s: s["sigils"][0].update(samples=[0.0, float("nan")]),
    lambda s: s["canvas"].update(width=0),
    lambda s: s.update(unknown="field"),
    lambda s: s["sigils"][0].update(samples=[0.0]),
])
def test_invalid_canonical_spec_rejected(spec, edit):
    edit(spec)
    with pytest.raises(ValueError):
        PrismaCoderRenderer().render(spec)


def test_does_not_mutate_input(spec):
    before = json.dumps(spec, ensure_ascii=False)
    PrismaCoderRenderer().render(spec)
    assert before == json.dumps(spec, ensure_ascii=False)


def test_escapes_labels(spec):
    spec["glyphs"][0]["label"] = "<script>& joke"
    result = PrismaCoderRenderer().render(spec)
    assert "&lt;script&gt;" in result.svg


def test_generation_errors_fall_back_or_fail_closed(spec):
    def raises(_):
        raise OSError("endpoint offline")
    assert PrismaCoderRenderer(generate=raises).render(spec).source == "deterministic_fallback"
    with pytest.raises(ValueError, match="endpoint offline"):
        PrismaCoderRenderer(generate=raises, fallback=False).render(spec)


def test_cannot_mix_endpoint_and_callback():
    with pytest.raises(ValueError):
        PrismaCoderRenderer(endpoint="http://127.0.0.1:8000/v1/chat/completions",
                            generate=lambda _: "")
