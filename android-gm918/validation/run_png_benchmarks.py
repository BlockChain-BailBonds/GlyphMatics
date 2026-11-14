#!/data/data/com.termux/files/usr/bin/python
"""Run deterministic inference from the GM918 v4 PNG vault and capture evidence."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evidence" / "png_inference"
VAULT = ROOT / "vault" / "918_Qwen3_1.7B.gm918v4"
RUNTIME = ROOT / "sigilvm4" / "gm918-png-runtime"
TESTS = [
    ("A_arithmetic", "What is 17 multiplied by 23?", 160),
    ("B_reasoning", "Explain why a metal spoon feels colder than a wooden spoon at the same room temperature.", 160),
    ("C_code", "Write a Python function that returns the factorial of a nonnegative integer.", 192),
    ("D_context", "Remember this unique identifier exactly: GM918-A14-7F3C. After briefly explaining what deterministic inference means, state the identifier I asked you to remember.", 192),
    ("E_sustained", "Write a self-contained technical explanation of how transformer inference works on a resource-constrained ARM64 phone. Discuss tokenization, quantized weights, attention, KV cache, sampling, memory limits, and thermal constraints.", 256),
]

def mem_available():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1])
    return 0

def rss_kb(pid):
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except OSError:
        pass
    return 0

def run_test(name, prompt, tokens):
    command = [str(RUNTIME), "run", str(VAULT), "-c", "1024", "-t", "4", "--seed", "918", "--temp", "0",
               "--reasoning", "off", "--simple-io", "--single-turn", "--prompt", prompt, "--predict", str(tokens)]
    start = time.monotonic(); process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    peak = 0; minimum = mem_available()
    while process.poll() is None:
        peak = max(peak, rss_kb(process.pid)); minimum = min(minimum, mem_available()); time.sleep(0.1)
    stdout, stderr = process.communicate(); elapsed = time.monotonic() - start
    (OUT / f"{name}.stdout.txt").write_text(stdout)
    (OUT / f"{name}.stderr.txt").write_text(stderr)
    generated = "GM918_PNG_VERIFIED" in stderr and "[ Prompt:" in stdout
    return {"name": name, "status": "VERIFIED" if process.returncode == 0 and generated else "FAILED",
            "prompt": prompt, "requested_tokens": tokens, "exit_code": process.returncode,
            "elapsed_seconds": elapsed, "peak_rss_kb_sampled": peak,
            "lowest_mem_available_kb_sampled": minimum, "command": command,
            "stdout_file": f"{name}.stdout.txt", "stderr_file": f"{name}.stderr.txt"}

def main():
    OUT.mkdir(parents=True, exist_ok=True); results = []
    for test in TESTS:
        available = mem_available()
        if available < 500_000:
            results.append({"name": test[0], "status": "BLOCKED", "reason": f"MemAvailable {available} kB below safety floor"})
            continue
        results.append(run_test(*test))
    report = {"format": "GM918 PNG inference benchmark v1", "vault": str(VAULT),
              "configuration": {"context": 1024, "threads": 4, "seed": 918, "temperature": 0, "reasoning": "off"},
              "tests": results}
    (OUT / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))

if __name__ == "__main__": main()
