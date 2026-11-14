#!/data/data/com.termux/files/usr/bin/python
"""Run deterministic conventional-GGUF reference prompts with device evidence."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evidence" / "reference_inference"
MODEL = ROOT / "archives" / "918_Qwen3_1.7B.gm918" / "model.gguf"

TESTS = [
    ("A_arithmetic", "What is 17 multiplied by 23?", 96),
    ("B_reasoning", "Explain why a metal spoon feels colder than a wooden spoon at the same room temperature.", 160),
    ("C_code", "Write a Python function that returns the factorial of a nonnegative integer.", 192),
    ("D_context", "Remember this unique identifier exactly: GM918-A14-7F3C. After briefly explaining what deterministic inference means, state the identifier I asked you to remember.", 192),
    ("E_sustained", "Write a self-contained technical explanation of how transformer inference works on a resource-constrained ARM64 phone. Discuss tokenization, quantized weights, attention, KV cache, sampling, memory limits, and thermal constraints.", 256),
]


def meminfo():
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        values[key] = value.strip()
    return values


def cpu_freqs():
    values = {}
    base = Path("/sys/devices/system/cpu/cpufreq")
    for policy in sorted(base.glob("policy*")):
        path = policy / "scaling_cur_freq"
        try:
            values[policy.name] = int(path.read_text().strip())
        except (OSError, ValueError):
            values[policy.name] = None
    return values


def rss_kb(pid):
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except OSError:
        pass
    return 0


def run_one(name, prompt, tokens):
    command = [
        "llama-cli", "-m", str(MODEL), "-c", "1024", "-t", "4",
        "--seed", "918", "--temp", "0", "--reasoning", "off",
        "--simple-io", "--single-turn", "--prompt", prompt,
        "--predict", str(tokens),
    ]
    before = meminfo()
    freq_before = cpu_freqs()
    started_wall = time.time()
    started = time.monotonic()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    peak_rss = 0
    lowest_available_kb = None
    while process.poll() is None:
        peak_rss = max(peak_rss, rss_kb(process.pid))
        available = int(meminfo()["MemAvailable"].split()[0])
        lowest_available_kb = available if lowest_available_kb is None else min(lowest_available_kb, available)
        time.sleep(0.1)
    stdout, stderr = process.communicate()
    elapsed = time.monotonic() - started
    after = meminfo()
    record = {
        "name": name,
        "status": "VERIFIED" if process.returncode == 0 and stdout.strip() else "FAILED",
        "scope": "conventional GGUF reference only; not PNG-backed inference",
        "timestamp_unix": started_wall,
        "command": command,
        "prompt": prompt,
        "requested_tokens": tokens,
        "exit_code": process.returncode,
        "elapsed_seconds": elapsed,
        "peak_rss_kb_sampled": peak_rss,
        "lowest_mem_available_kb_sampled": lowest_available_kb,
        "mem_before": before,
        "mem_after": after,
        "cpu_freq_before_khz": freq_before,
        "cpu_freq_after_khz": cpu_freqs(),
        "stdout_file": f"{name}.stdout.txt",
        "stderr_file": f"{name}.stderr.txt",
    }
    (OUT / record["stdout_file"]).write_text(stdout, encoding="utf-8")
    (OUT / record["stderr_file"]).write_text(stderr, encoding="utf-8")
    return record


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if not MODEL.is_file():
        raise SystemExit(f"missing reference model: {MODEL}")
    records = []
    for test in TESTS:
        available_kb = int(meminfo()["MemAvailable"].split()[0])
        if available_kb < 500_000:
            records.append({
                "name": test[0], "status": "BLOCKED",
                "reason": f"MemAvailable {available_kb} kB below 500000 kB safety floor",
            })
            break
        records.append(run_one(*test))
    report = {
        "format": "GM918 conventional reference benchmark v1",
        "model": str(MODEL),
        "configuration": {"context": 1024, "threads": 4, "seed": 918, "temperature": 0, "reasoning": "off"},
        "tests": records,
    }
    (OUT / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
