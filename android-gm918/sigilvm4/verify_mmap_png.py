#!/data/data/com.termux/files/usr/bin/python
"""Strictly verify a GM918 direct-mapped PNG without extracting its payload."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import struct
import zlib

PNG = b"\x89PNG\r\n\x1a\n"
SPRITE_MAGIC = b"GM918SPRITE1"


def verify(path: Path, expected_sha256: str | None, require_sprites: bool) -> None:
    total = path.stat().st_size
    payload_size = None
    payload_hash = hashlib.sha256()
    found_header = found_payload = found_end = False
    sprite_info = None

    with path.open("rb") as stream:
        if stream.read(8) != PNG:
            raise ValueError("invalid PNG signature")
        while stream.tell() < total:
            raw_length = stream.read(4)
            if len(raw_length) != 4:
                raise ValueError("truncated PNG chunk length")
            length = struct.unpack(">I", raw_length)[0]
            kind = stream.read(4)
            if len(kind) != 4:
                raise ValueError("truncated PNG chunk type")
            crc = zlib.crc32(kind)
            chunk_start = stream.tell()
            remaining = length
            while remaining:
                block = stream.read(min(4 * 1024 * 1024, remaining))
                if not block:
                    raise ValueError(f"truncated {kind!r} payload")
                crc = zlib.crc32(block, crc)
                if kind == b"gmHD":
                    if length != 18 or block[:10] != b"GM918MMAP4":
                        raise ValueError("invalid GM918 mmap header")
                    payload_size = struct.unpack(">Q", block[10:18])[0]
                    found_header = True
                elif kind == b"gmSI":
                    if length != 36 or block[:12] != SPRITE_MAGIC:
                        raise ValueError("invalid GM918 sprite index")
                    sprite_info = struct.unpack(">IIIIQ", block[12:36])
                elif kind == b"gmDT":
                    payload_hash.update(block)
                remaining -= len(block)
            stored_crc_bytes = stream.read(4)
            if len(stored_crc_bytes) != 4:
                raise ValueError(f"missing {kind!r} CRC")
            stored_crc = struct.unpack(">I", stored_crc_bytes)[0]
            if stored_crc != crc & 0xFFFFFFFF:
                raise ValueError(f"CRC mismatch in {kind.decode('ascii', 'replace')}")
            if kind == b"gmDT":
                if found_payload or not found_header:
                    raise ValueError("invalid gmDT ordering or duplicate")
                if chunk_start % 4096:
                    raise ValueError("gmDT payload is not page aligned")
                if payload_size != length:
                    raise ValueError("gmDT size disagrees with gmHD")
                found_payload = True
            elif kind == b"IEND":
                if length != 0:
                    raise ValueError("invalid IEND")
                found_end = True
                if stream.tell() != total:
                    raise ValueError("trailing bytes after IEND")
                break

    if not (found_header and found_payload and found_end):
        raise ValueError("missing required GM918 PNG chunks")
    if require_sprites and sprite_info is None:
        raise ValueError("missing GM918 weight-sprite index")
    if sprite_info is not None:
        width, height, sprite_bytes, last_bytes, count = sprite_info
        if width * height != sprite_bytes or sprite_bytes != 4096:
            raise ValueError("unsupported weight-sprite geometry")
        expected_count = (payload_size + sprite_bytes - 1) // sprite_bytes
        expected_last = payload_size - (expected_count - 1) * sprite_bytes
        if count != expected_count or last_bytes != expected_last:
            raise ValueError("weight-sprite index disagrees with payload")
    digest = payload_hash.hexdigest()
    if expected_sha256 and digest.lower() != expected_sha256.lower():
        raise ValueError(f"SHA-256 mismatch: {digest}")
    print(f"VERIFIED path={path}")
    print(f"payload_bytes={payload_size}")
    print(f"payload_sha256={digest}")
    if sprite_info is not None:
        width, height, sprite_bytes, last_bytes, count = sprite_info
        print(f"weight_sprites={count}")
        print(f"sprite_geometry={width}x{height}x8bit")
        print(f"last_sprite_bytes={last_bytes}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("png", type=Path)
    parser.add_argument("--sha256")
    parser.add_argument("--require-sprites", action="store_true")
    args = parser.parse_args()
    verify(args.png.resolve(), args.sha256, args.require_sprites)


if __name__ == "__main__":
    main()
