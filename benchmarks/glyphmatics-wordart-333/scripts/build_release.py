# Copyright 2026 918 Technologies
# SPDX-License-Identifier: Apache-2.0
"""Build a self-contained Kaggle task, notebook, manifest, and release zip."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VERSION = "1.0.0"
TASK_NAME = "glyphmatics-wordart-333"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_files() -> tuple[Path, ...]:
    selected = [
        ROOT / "README.md",
        ROOT / "NOTICE.md",
        ROOT / "VALIDATION.json",
        ROOT / "pyproject.toml",
        ROOT / "kaggle_task.py",
    ]
    selected.extend(sorted((ROOT / "src").rglob("*.py")))
    selected.extend(sorted((ROOT / "src").rglob("*.json")))
    selected.extend(sorted((ROOT / "scripts").glob("*.py")))
    selected.extend(sorted((ROOT / "tests").glob("*.py")))
    return tuple(path for path in selected if path.is_file())


def runtime_zip_bytes() -> bytes:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted((ROOT / "src").rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                archive.write(path, path.relative_to(ROOT).as_posix())
    return payload.getvalue()


def bootstrap_code(encoded: str) -> str:
    return f'''# Copyright 2026 918 Technologies
# SPDX-License-Identifier: Apache-2.0
import base64
import io
import hashlib
import os
import sys
import zipfile
from pathlib import Path

_PAYLOAD_SHA256 = "{hashlib.sha256(base64.b64decode(encoded)).hexdigest()}"
_PAYLOAD_B64 = {json.dumps(encoded)}
_payload = base64.b64decode(_PAYLOAD_B64)
if hashlib.sha256(_payload).hexdigest() != _PAYLOAD_SHA256:
    raise RuntimeError("embedded GlyphMatics runtime hash mismatch")
_working = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path.cwd()
_runtime = _working / "glyphmatics_wordart_333_runtime"
_runtime.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(_payload)) as _archive:
    for _member in _archive.infolist():
        _parts = Path(_member.filename).parts
        if Path(_member.filename).is_absolute() or ".." in _parts:
            raise RuntimeError(f"unsafe embedded member: {{_member.filename}}")
    _archive.extractall(_runtime)
_src = _runtime / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))
'''


TASK_BODY = '''
from contextlib import contextmanager
from pathlib import Path
import kaggle_benchmarks as kbench
from glyphmatics_wordart import ADLEngine, BenchmarkRunner, CANON333, FailureLedger, load_cases

class _KaggleSession:
    def __init__(self, llm):
        self.llm = llm
    def prompt(self, prompt: str) -> str:
        return str(self.llm.prompt(prompt, reasoning="high", temperature=0.2))

class _KagglePromptModel:
    def __init__(self, llm):
        self.llm = llm
    @contextmanager
    def session(self, name: str):
        with kbench.chats.new(name, orphan=True):
            yield _KaggleSession(self.llm)

@kbench.task(
    name="glyphmatics-wordart-333",
    description="Visual-semantic round-trip benchmark using the immutable GlyphMatics 333 canon",
    version=1,
)
def glyphmatics_wordart_333(llm) -> dict:
    cases = load_cases()
    adl = ADLEngine()
    ledger = FailureLedger(_working / "GLYPHMATICS_WORDART_FAILURES.jsonl")
    runner = BenchmarkRunner(_KagglePromptModel(llm), cases=cases, ledger=ledger, adl=adl)
    result = runner.run_all()
    report = result.to_mapping(runner.config)
    report["adl_snapshot"] = adl.snapshot()
    report["failure_ledger_records"] = len(ledger.read_all())
    kbench.assertions.assert_true(
        len(CANON333.definitions()) == 333,
        expectation="The immutable GlyphMatics canon contains exactly 333 unique definitions",
    )
    kbench.assertions.assert_true(
        report["case_count"] == len(cases),
        expectation="Every configured Word Art case produced a result",
    )
    return report

glyphmatics_wordart_333.run(kbench.llm)
'''


def build_standalone(encoded: str) -> Path:
    output = ROOT / "GLYPHMATICS_WORDART_TASK.py"
    text = bootstrap_code(encoded)
    output.write_text(text + "\n# %%\n" + TASK_BODY.lstrip(), encoding="utf-8")
    return output


def code_cell(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)}


def markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def build_notebook(encoded: str) -> Path:
    bootstrap = bootstrap_code(encoded)
    smoke = '''from glyphmatics_wordart import CANON333, GlyphTransportCodec\n\nassert len(CANON333.definitions()) == 333\n_codec = GlyphTransportCodec()\n_all_ids = tuple(range(1, 334))\nassert _codec.decode_ids(_codec.encode_ids(_all_ids)) == _all_ids\nprint(f"GlyphMatics startup PASS: canon=333 digest={CANON333.digest}")\n'''
    notebook = {
        "cells": [
            markdown_cell(
                "# GlyphMatics Word Art 333 — Kaggle Benchmark\n\n"
                "Self-contained release v1.0.0. It embeds the exact 333 canon, runs a startup round-trip gate, "
                "then executes isolated artist/guesser conversations through Kaggle Benchmarks.\n"
            ),
            code_cell(bootstrap),
            code_cell(smoke),
            code_cell(TASK_BODY.lstrip()),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    output = ROOT / "GLYPHMATICS_WORDART_KAGGLE.ipynb"
    output.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return output


def build_manifest(generated: tuple[Path, ...]) -> Path:
    files_to_hash = tuple(sorted(set((*source_files(), *generated))))
    manifest = {
        "artifact": "glyphmatics-wordart-333",
        "canon_count": 333,
        "files": [
            {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path), "size_bytes": path.stat().st_size}
            for path in files_to_hash
        ],
        "schema": "glyphmatics.wordart.release.v1",
        "version": VERSION,
    }
    output = ROOT / "MANIFEST.json"
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def build_release() -> Path:
    encoded = base64.b64encode(runtime_zip_bytes()).decode("ascii")
    standalone = build_standalone(encoded)
    notebook = build_notebook(encoded)
    manifest = build_manifest((standalone, notebook))
    release = ROOT / f"glyphmatics-wordart-333-v{VERSION}.zip"
    with zipfile.ZipFile(release, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in (*source_files(), standalone, notebook, manifest):
            archive.write(path, path.relative_to(ROOT).as_posix())
    return release


if __name__ == "__main__":
    release_path = build_release()
    print(json.dumps({"release": str(release_path), "sha256": sha256(release_path), "size_bytes": release_path.stat().st_size}, sort_keys=True))
