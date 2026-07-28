import pytest

from glyphmatics.glyph_container import decode_page, pack_payload, unpack_payload


def test_glyph_llm3_payload_round_trip():
    payload = "English 中文 Español हिन्दी العربية 🧠".encode("utf-8")
    page = pack_payload("Semantic Model", "corpus", payload, meta={"version": 1})
    meta3, meta2, blocks = decode_page(page)
    assert meta3["inner_version"] == 2
    assert meta2["version"] == 1
    assert blocks[0]["name"] == "corpus"
    assert unpack_payload(page, expected_name="corpus") == payload


def test_corrupted_page_fails_crc():
    page = pack_payload("Test", "data", b"bit exact")
    payload_start = page.index("•") + 1
    original = ord(page[payload_start])
    replacement = chr(0x2800 + ((original - 0x2800 + 1) % 256))
    damaged = page[:payload_start] + replacement + page[payload_start + 1:]
    with pytest.raises(ValueError, match="CRC"):
        decode_page(damaged)


def test_wrong_block_name_is_rejected():
    page = pack_payload("Test", "right", b"value")
    with pytest.raises(ValueError, match="exactly one"):
        unpack_payload(page, expected_name="wrong")
