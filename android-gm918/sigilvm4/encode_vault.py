#!/data/data/com.termux/files/usr/bin/python
"""Encode an arbitrary model byte stream as canonical GM918 Sigil Image VM v4 PNG tiles."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import time
import zlib


PNG = b"\x89PNG\r\n\x1a\n"
MAGIC = b"GM918P4\0"
VERSION = 4
KIND_MODEL = 1
KIND_MANIFEST = 2
HEADER = struct.Struct(">8sIIIQII32s")


def png_chunk(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)


def make_png(payload: bytes, *, kind: int, index: int, offset: int, total: int, width: int) -> bytes:
    digest = hashlib.sha256(payload).digest()
    height = (len(payload) + width - 1) // width
    padded = payload + bytes(width * height - len(payload))
    scanlines = b"".join(b"\0" + padded[row * width:(row + 1) * width] for row in range(height))
    header = HEADER.pack(MAGIC, VERSION, kind, index, offset, len(payload), total, digest)
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    return PNG + png_chunk(b"IHDR", ihdr) + png_chunk(b"gmHD", header) + png_chunk(b"IDAT", zlib.compress(scanlines, 9)) + png_chunk(b"IEND", b"")


def atomic_write(path: Path, payload: bytes):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def encode(source: Path, vault: Path, chunk_size: int, width: int):
    if vault.exists():
        raise SystemExit(f"refusing to overwrite existing vault: {vault}")
    staging = vault.with_name(vault.name + ".partial")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    total_bytes = source.stat().st_size
    total_tiles = (total_bytes + chunk_size - 1) // chunk_size
    model_hash = hashlib.sha256()
    records = []
    started = time.monotonic()
    try:
        with source.open("rb") as stream:
            for index in range(total_tiles):
                offset = index * chunk_size
                payload = stream.read(chunk_size)
                model_hash.update(payload)
                digest = hashlib.sha256(payload).hexdigest()
                name = f"tile-{index:06d}.png"
                image = make_png(payload, kind=KIND_MODEL, index=index, offset=offset, total=total_tiles, width=width)
                atomic_write(staging / name, image)
                records.append({
                    "index": index, "file": name, "offset": offset,
                    "payload_bytes": len(payload), "png_bytes": len(image), "sha256": digest,
                })
                if index % 16 == 0 or index + 1 == total_tiles:
                    print(f"encoded {index + 1}/{total_tiles}", flush=True)
        manifest = {
            "format": "GM918 Sigil Image VM",
            "version": VERSION,
            "owner": "918 Technologies",
            "payload_format": "GGUF",
            "source_name": source.name,
            "model_bytes": total_bytes,
            "model_sha256": model_hash.hexdigest(),
            "chunk_size": chunk_size,
            "image_width": width,
            "tile_count": total_tiles,
            "tiles": records,
            "created_unix": int(time.time()),
        }
        manifest_payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        manifest_png = make_png(manifest_payload, kind=KIND_MANIFEST, index=0, offset=0, total=1, width=width)
        atomic_write(staging / "manifest.png", manifest_png)
        os.replace(staging, vault)
        png_bytes = sum(item["png_bytes"] for item in records) + len(manifest_png)
        print(json.dumps({
            "vault": str(vault), "model_bytes": total_bytes, "png_bytes": png_bytes,
            "png_overhead_bytes": png_bytes - total_bytes,
            "lossless_ratio_original_over_png": total_bytes / png_bytes,
            "model_sha256": model_hash.hexdigest(), "seconds": time.monotonic() - started,
        }, indent=2))
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("vault")
    parser.add_argument("--chunk-size", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--width", type=int, default=2048)
    args = parser.parse_args()
    encode(Path(args.source).resolve(), Path(args.vault).resolve(), args.chunk_size, args.width)


if __name__ == "__main__":
    main()
