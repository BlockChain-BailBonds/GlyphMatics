#!/data/data/com.termux/files/usr/bin/python
"""Nondestructive failure tests for GM918 PNG vaults using disposable copies."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import struct
import subprocess
import zlib


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "evidence" / "regression" / "synthetic-vault"
WORK = ROOT / "evidence" / "regression" / "failure-cases"
RUNTIME = ROOT / "sigilvm4" / "gm918-png-runtime"


def run(name, mutate, expected_fragment):
    target = WORK / name
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(SOURCE, target)
    mutate(target)
    result = subprocess.run([str(RUNTIME), "verify", str(target)], capture_output=True, text=True)
    combined = result.stdout + result.stderr
    passed = result.returncode != 0 and expected_fragment in combined
    return {
        "name": name,
        "status": "VERIFIED" if passed else "FAILED",
        "expected": expected_fragment,
        "exit_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def remove_tile(vault):
    (vault / "tile-000001.png").unlink()


def truncate_tile(vault):
    path = vault / "tile-000000.png"
    data = path.read_bytes()
    path.write_bytes(data[:-97])


def modify_idat(vault):
    path = vault / "tile-000000.png"
    data = bytearray(path.read_bytes())
    marker = data.index(b"IDAT") + 8
    data[marker] ^= 0x40
    path.write_bytes(data)


def mutate_chunk(path, chunk_type, change):
    data = bytearray(path.read_bytes())
    pos = 8
    while pos + 12 <= len(data):
        length = struct.unpack_from(">I", data, pos)[0]
        kind_pos = pos + 4
        payload_pos = pos + 8
        if data[kind_pos:kind_pos + 4] == chunk_type:
            payload = bytearray(data[payload_pos:payload_pos + length])
            change(payload)
            data[payload_pos:payload_pos + length] = payload
            crc = zlib.crc32(bytes(chunk_type) + payload) & 0xFFFFFFFF
            struct.pack_into(">I", data, payload_pos + length, crc)
            path.write_bytes(data)
            return
        pos = payload_pos + length + 4
    raise RuntimeError(f"chunk not found: {chunk_type!r}")


def invalid_offset(vault):
    mutate_chunk(vault / "tile-000001.png", b"gmHD", lambda p: struct.pack_into(">Q", p, 20, 123))


def unsupported_version(vault):
    mutate_chunk(vault / "tile-000000.png", b"gmHD", lambda p: struct.pack_into(">I", p, 8, 99))


def incorrect_manifest(vault):
    def change(payload):
        # Preserve PNG validity and internal tile SHA, but make decoded manifest invalid.
        payload[-1] ^= 1
    mutate_chunk(vault / "manifest.png", b"IDAT", change)


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    tests = [
        run("missing-tile", remove_tile, "manifest tile count mismatch"),
        run("truncated-tile", truncate_tile, "truncated PNG chunk"),
        run("modified-pixel", modify_idat, "PNG CRC mismatch"),
        run("invalid-offset", invalid_offset, "tile index/offset mismatch"),
        run("unsupported-version", unsupported_version, "incomplete/unsupported GM918 PNG"),
        run("incorrect-manifest", incorrect_manifest, "PNG inflate failed"),
    ]
    output = {"suite": "GM918 PNG failure validation", "tests": tests}
    (WORK / "results.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))
    raise SystemExit(0 if all(t["status"] == "VERIFIED" for t in tests) else 1)


if __name__ == "__main__":
    main()
