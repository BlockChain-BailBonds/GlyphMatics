import json

import pytest

from glyphmatics.semantic_codec import SemanticVocabulary, iter_glyphs, tokenize_lossless


def vocabulary():
    glyphs = iter_glyphs()
    tokens = [" ", ".", "Hello", "world", "我", "水", "Quiero", "agua", "घर", "هنا"]
    records = []
    for token_id, token in enumerate(tokens):
        records.append(
            {
                "id": token_id,
                "token": token,
                "glyph": next(glyphs),
                "frequency": len(tokens) - token_id,
                "semantics": (
                    [{"language": "zh-Hans", "concept_id": "WATER", "gloss": "water"}]
                    if token == "水"
                    else []
                ),
            }
        )
    return SemanticVocabulary(records, metadata={"languages": ["en", "zh-Hans", "es", "hi", "ar"]})


@pytest.mark.parametrize(
    "text",
    [
        "Hello world.",
        "我 水",
        "Quiero agua.",
        "घर هنا.",
        "Unknown 🧠 text\npreserves\tbytes.",
        "\ue100 is input, not trusted encoded data",
    ],
)
def test_visible_and_binary_round_trip(text):
    codec = vocabulary()
    glyphs = codec.encode_glyphs(text)
    binary = codec.encode_binary(text)
    assert codec.decode_glyphs(glyphs) == text
    assert codec.decode_binary(binary) == text


def test_known_word_is_exactly_one_glyph():
    codec = vocabulary()
    assert len(codec.encode_glyphs("Hello")) == 1
    assert codec.encode_glyphs("Hello") == codec.token_to_glyph["Hello"]


def test_multilingual_semantics_are_retained():
    codec = vocabulary()
    assert codec.semantics("水", language="zh-Hans")[0]["concept_id"] == "WATER"
    assert codec.semantics("水", language="en") == []


def test_combining_scripts_and_layout_are_lossless():
    text = "मैं घर में हूँ।\nالعربيةُ هنا"
    assert "".join(tokenize_lossless(text)) == text


def test_vocabulary_checksum_rejects_mutation(tmp_path):
    codec = vocabulary()
    path = tmp_path / "vocab.json"
    codec.save(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["records"][0]["token"] = "changed"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        SemanticVocabulary.load(path)


def test_binary_requires_matching_vocabulary():
    first = vocabulary()
    records = [dict(record) for record in first.records]
    records[0]["frequency"] += 1
    second = SemanticVocabulary(records, metadata=first.metadata)
    with pytest.raises(ValueError, match="digest"):
        second.decode_binary(first.encode_binary("Hello"))
