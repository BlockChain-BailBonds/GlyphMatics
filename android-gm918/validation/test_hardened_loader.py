#!/data/data/com.termux/files/usr/bin/python
"""Exercise native GM918 metadata rejection without copying a full model."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import zlib

ROOT = Path(__file__).resolve().parents[1]
ENCODER = ROOT / "sigilvm4" / "encode_mmap_png.py"
LLAMA = ROOT / "third_party" / "llama.cpp" / "build-gm918" / "bin" / "llama-cli"
LIBS = LLAMA.parent
SOURCE = ROOT / "third_party" / "llama.cpp" / "models" / "ggml-vocab-qwen2.gguf"


def chunk(data: bytearray, kind: bytes) -> tuple[int, int, int]:
    pos = 8
    while pos + 12 <= len(data):
        length = struct.unpack_from(">I", data, pos)[0]
        if data[pos + 4:pos + 8] == kind:
            return pos + 4, pos + 8, length
        pos += 12 + length
    raise ValueError(f"missing {kind!r}")


def rewrite_crc(data: bytearray, type_pos: int, payload_pos: int, length: int) -> None:
    value = zlib.crc32(data[type_pos:type_pos + 4] + data[payload_pos:payload_pos + length]) & 0xFFFFFFFF
    struct.pack_into(">I", data, payload_pos + length, value)


def run(path: Path) -> str:
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = str(LIBS)
    proc = subprocess.run(
        [str(LLAMA), "-m", str(path), "--no-warmup", "-n", "1", "-p", "x"],
        text=True, capture_output=True, env=env,
    )
    return proc.stdout + proc.stderr


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="gm918-loader-") as tmp_name:
        tmp = Path(tmp_name)
        valid = tmp / "valid.png"
        subprocess.run([str(ENCODER), str(SOURCE), str(valid)], check=True, capture_output=True, text=True)

        valid_output = run(valid)
        if "invalid GM918 PNG" in valid_output:
            raise AssertionError("valid sprite container rejected by native loader")

        cases = []

        data = bytearray(valid.read_bytes())
        _, payload, _ = chunk(data, b"gmSI")
        data[payload] ^= 1
        cases.append(("metadata-crc", data, "PNG metadata CRC mismatch"))

        data = bytearray(valid.read_bytes())
        type_pos, payload, length = chunk(data, b"gmSI")
        data[type_pos:type_pos + 4] = b"zzSI"
        rewrite_crc(data, type_pos, payload, length)
        cases.append(("missing-sprite-index", data, "gmDT precedes required"))

        data = bytearray(valid.read_bytes())
        type_pos, payload, length = chunk(data, b"gmSI")
        struct.pack_into(">I", data, payload + 12, 32)
        rewrite_crc(data, type_pos, payload, length)
        cases.append(("bad-sprite-geometry", data, "unsupported gmSI sprite geometry"))

        data = bytearray(valid.read_bytes())
        type_pos, payload, length = chunk(data, b"gmHD")
        size = struct.unpack_from(">Q", data, payload + 10)[0]
        struct.pack_into(">Q", data, payload + 10, size + 1)
        rewrite_crc(data, type_pos, payload, length)
        cases.append(("payload-size-mismatch", data, "gmDT length disagrees with gmHD"))

        for name, mutated, expected in cases:
            path = tmp / f"{name}.png"
            path.write_bytes(mutated)
            output = run(path)
            if expected not in output:
                raise AssertionError(f"{name}: expected {expected!r}\n{output}")
            print(f"VERIFIED {name}: {expected}")


if __name__ == "__main__":
    main()
