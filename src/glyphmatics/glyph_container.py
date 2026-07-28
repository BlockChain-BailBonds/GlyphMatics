"""Glyph-LLM3 compatible v2/v3 framing and Braille transport."""

from __future__ import annotations

import binascii
import json
import struct
import zlib
from typing import Any


MAGIC = b"GLYPHLLM"
BRAILLE_BASE = 0x2800
DEFAULT_MAX_BYTES = 256 * 1024 * 1024


def _json_bytes(value: dict[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _crc32(data: bytes) -> int:
    return binascii.crc32(data) & 0xFFFFFFFF


def _frame(raw: bytes) -> bytes:
    compressed = zlib.compress(raw, 9)
    return compressed + struct.pack("<I", _crc32(compressed))


def _unframe(framed: bytes, *, layer: str, max_bytes: int) -> bytes:
    if len(framed) < 5 or len(framed) > max_bytes:
        raise ValueError(f"{layer} frame has an invalid size")
    compressed, expected_raw = framed[:-4], framed[-4:]
    expected = struct.unpack("<I", expected_raw)[0]
    actual = _crc32(compressed)
    if expected != actual:
        raise ValueError(f"{layer} CRC mismatch")
    decompressor = zlib.decompressobj()
    raw = decompressor.decompress(compressed, max_bytes + 1)
    if len(raw) > max_bytes or decompressor.unconsumed_tail:
        raise ValueError(f"{layer} decompressed payload exceeds limit")
    raw += decompressor.flush(max(1, max_bytes + 1 - len(raw)))
    if len(raw) > max_bytes:
        raise ValueError(f"{layer} decompressed payload exceeds limit")
    return raw


def bytes_to_braille(data: bytes) -> str:
    return "".join(chr(BRAILLE_BASE + byte) for byte in data)


def braille_to_bytes(text: str) -> bytes:
    return bytes(
        ord(char) - BRAILLE_BASE
        for char in text
        if BRAILLE_BASE <= ord(char) <= BRAILLE_BASE + 255
    )


def encode_leaf(meta: dict[str, Any], blocks: list[tuple[str, bytes]]) -> bytes:
    metadata = _json_bytes(meta)
    if len(metadata) > 0xFFFF:
        raise ValueError("v2 metadata is too large")
    body = bytearray(MAGIC)
    body.extend(struct.pack("<BH", 2, len(metadata)))
    body.extend(metadata)
    for name, payload in blocks:
        encoded_name = name.encode("ascii")
        if not encoded_name or len(encoded_name) > 255:
            raise ValueError("block name must be 1-255 ASCII bytes")
        body.extend(struct.pack("<BB", 1, len(encoded_name)))
        body.extend(encoded_name)
        body.extend(struct.pack("<HffI", 0, 0.0, 0.0, len(payload)))
        body.extend(payload)
    body.append(0)
    return _frame(bytes(body))


def encode_page(
    title: str,
    leaf: bytes,
    *,
    page_index: int = 1,
    page_count: int = 1,
    extra_meta: dict[str, Any] | None = None,
) -> str:
    meta = {
        "comp": "book_page",
        "title": title,
        "page_index": page_index,
        "page_count": page_count,
        "inner_version": 2,
        "scheme": "single-leaf",
    }
    if extra_meta:
        meta.update(extra_meta)
    metadata = _json_bytes(meta)
    if len(metadata) > 0xFFFF:
        raise ValueError("v3 metadata is too large")
    raw = (
        MAGIC
        + struct.pack("<BH", 3, len(metadata))
        + metadata
        + struct.pack("<I", len(leaf))
        + leaf
    )
    framed = _frame(raw)
    return (
        f"⧈ΩϞ⧉ BOOK:{title} PAGE:{page_index}/{page_count} •"
        + bytes_to_braille(framed)
        + "• ⧉ϞΩ⧈"
    )


def pack_payload(title: str, name: str, payload: bytes, *, meta: dict[str, Any] | None = None) -> str:
    leaf_meta = {"comp": "semantic_payload", "title": title}
    if meta:
        leaf_meta.update(meta)
    return encode_page(title, encode_leaf(leaf_meta, [(name, payload)]))


def _take(raw: bytes, offset: int, size: int, *, layer: str) -> tuple[bytes, int]:
    if size < 0 or offset + size > len(raw):
        raise ValueError(f"{layer} contains a truncated field")
    return raw[offset:offset + size], offset + size


def _parse_header(raw: bytes, version: int, *, layer: str) -> tuple[dict[str, Any], int]:
    if len(raw) < 11 or raw[:8] != MAGIC or raw[8] != version:
        raise ValueError(f"{layer} MAGIC or version mismatch")
    metadata_length = struct.unpack_from("<H", raw, 9)[0]
    metadata_raw, offset = _take(raw, 11, metadata_length, layer=layer)
    return json.loads(metadata_raw.decode("utf-8")), offset


def decode_page(text: str, *, max_bytes: int = DEFAULT_MAX_BYTES) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    first_marker = text.find("•")
    last_marker = text.rfind("•")
    payload_text = (
        text[first_marker + 1:last_marker]
        if 0 <= first_marker < last_marker
        else text
    )
    framed_v3 = braille_to_bytes(payload_text)
    raw_v3 = _unframe(framed_v3, layer="v3", max_bytes=max_bytes)
    meta_v3, offset = _parse_header(raw_v3, 3, layer="v3")
    length_raw, offset = _take(raw_v3, offset, 4, layer="v3")
    inner_length = struct.unpack("<I", length_raw)[0]
    inner, offset = _take(raw_v3, offset, inner_length, layer="v3")
    if offset != len(raw_v3):
        raise ValueError("v3 has trailing bytes")

    raw_v2 = _unframe(inner, layer="v2", max_bytes=max_bytes)
    meta_v2, offset = _parse_header(raw_v2, 2, layer="v2")
    blocks: list[dict[str, Any]] = []
    while offset < len(raw_v2):
        block_type = raw_v2[offset]
        offset += 1
        if block_type == 0:
            if offset != len(raw_v2):
                raise ValueError("v2 has trailing bytes after terminator")
            break
        if block_type != 1:
            raise ValueError(f"unsupported v2 block type: {block_type}")
        name_length_raw, offset = _take(raw_v2, offset, 1, layer="v2")
        name_raw, offset = _take(raw_v2, offset, name_length_raw[0], layer="v2")
        header, offset = _take(raw_v2, offset, 14, layer="v2")
        ndim, vmin, vmax, payload_length = struct.unpack("<HffI", header)
        payload, offset = _take(raw_v2, offset, payload_length, layer="v2")
        blocks.append(
            {
                "type": block_type,
                "name": name_raw.decode("ascii"),
                "ndim": ndim,
                "vmin": vmin,
                "vmax": vmax,
                "payload": payload,
            }
        )
    return meta_v3, meta_v2, blocks


def unpack_payload(text: str, *, expected_name: str | None = None, max_bytes: int = DEFAULT_MAX_BYTES) -> bytes:
    _, _, blocks = decode_page(text, max_bytes=max_bytes)
    matches = [block for block in blocks if expected_name is None or block["name"] == expected_name]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one matching payload block, found {len(matches)}")
    return matches[0]["payload"]
