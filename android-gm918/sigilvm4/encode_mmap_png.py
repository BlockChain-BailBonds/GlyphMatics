#!/data/data/com.termux/files/usr/bin/python
"""Wrap GGUF bytes in a page-aligned, directly mmap-able valid PNG container."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import struct
import zlib

PNG = b"\x89PNG\r\n\x1a\n"
SPRITE_MAGIC = b"GM918SPRITE1"
SPRITE_WIDTH = 64
SPRITE_HEIGHT = 64
SPRITE_BYTES = SPRITE_WIDTH * SPRITE_HEIGHT

def chunk_bytes(kind, payload):
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xffffffff)

def encode(source: Path, output: Path):
    if output.exists(): raise SystemExit(f"refusing to overwrite {output}")
    partial = output.with_suffix(output.suffix + ".partial")
    model_size = source.stat().st_size
    if model_size >= 0x80000000: raise SystemExit("PNG chunk exceeds 31-bit portable limit")
    source_hash = hashlib.sha256()
    with partial.open("wb+") as out, source.open("rb") as src:
        out.write(PNG)
        out.write(chunk_bytes(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0)))
        out.write(chunk_bytes(b"IDAT", zlib.compress(b"\0\0", 9)))
        meta = b"GM918MMAP4" + struct.pack(">Q", model_size)
        out.write(chunk_bytes(b"gmHD", meta))
        sprite_count = (model_size + SPRITE_BYTES - 1) // SPRITE_BYTES
        last_sprite_bytes = model_size - (sprite_count - 1) * SPRITE_BYTES
        sprite_meta = SPRITE_MAGIC + struct.pack(
            ">IIIIQ", SPRITE_WIDTH, SPRITE_HEIGHT, SPRITE_BYTES,
            last_sprite_bytes, sprite_count,
        )
        out.write(chunk_bytes(b"gmSI", sprite_meta))
        padding = (-out.tell() - 20) % 4096
        out.write(chunk_bytes(b"gmPD", bytes(padding)))
        out.write(struct.pack(">I", model_size)); out.write(b"gmDT")
        payload_offset = out.tell()
        assert payload_offset % 4096 == 0
        crc = zlib.crc32(b"gmDT")
        while block := src.read(4 * 1024 * 1024):
            source_hash.update(block); crc = zlib.crc32(block, crc); out.write(block)
        out.write(struct.pack(">I", crc & 0xffffffff))
        out.write(chunk_bytes(b"IEND", b""))
        out.flush(); os.fsync(out.fileno())
    os.replace(partial, output)
    print(f"output={output}")
    print(f"payload_offset={payload_offset}")
    print(f"model_bytes={model_size}")
    print(f"weight_sprites={sprite_count}")
    print(f"sprite_geometry={SPRITE_WIDTH}x{SPRITE_HEIGHT}x8bit")
    print(f"png_bytes={output.stat().st_size}")
    print(f"sha256={source_hash.hexdigest()}")

def main():
    p=argparse.ArgumentParser(); p.add_argument("source"); p.add_argument("output")
    a=p.parse_args(); encode(Path(a.source).resolve(), Path(a.output).resolve())
if __name__ == "__main__": main()
