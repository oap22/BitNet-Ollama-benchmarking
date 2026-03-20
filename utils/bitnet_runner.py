"""Run BitNet speed benchmarks via bitnet.cpp's e2e_benchmark.py."""

import re
import subprocess
import time
from typing import Optional

from utils.ram_monitor import RAMMonitor


def run_bitnet_benchmark(
    model_path: str,
    prompt_tokens: int,
    gen_tokens: int,
    threads: int,
    benchmark_script: str,
) -> dict:
    """Run a single BitNet benchmark trial.

    Returns dict with keys: prefill_toks, decode_toks, peak_ram_mb, wall_time_s.
    """
    cmd = [
        "python", benchmark_script,
        "-m", model_path,
        "-p", str(prompt_tokens),
        "-n", str(gen_tokens),
        "-t", str(threads),
    ]

    start = time.perf_counter()
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )

    monitor = RAMMonitor(proc.pid)
    monitor.start()

    stdout, stderr = proc.communicate(timeout=600)
    wall_time = time.perf_counter() - start
    peak_ram = monitor.stop()

    if proc.returncode != 0:
        raise RuntimeError(
            f"BitNet benchmark failed (rc={proc.returncode}):\n{stderr}"
        )

    result = parse_bitnet_output(stdout)
    result["peak_ram_mb"] = round(peak_ram, 1)
    result["wall_time_s"] = round(wall_time, 3)
    return result


def parse_bitnet_output(stdout: str) -> dict:
    """Parse e2e_benchmark.py stdout for timing metrics.

    Expected output lines like:
        prompt eval time: ... (X tokens per second)
        eval time: ... (X tokens per second)
    """
    result = {"prefill_toks": 0.0, "decode_toks": 0.0}

    # Match patterns like "X tokens per second" or "X tok/s"
    prefill_match = re.search(
        r"prompt eval.*?(\d+\.?\d*)\s*tokens?\s*per\s*second", stdout, re.IGNORECASE
    )
    if not prefill_match:
        prefill_match = re.search(
            r"prompt eval.*?(\d+\.?\d*)\s*tok/s", stdout, re.IGNORECASE
        )

    decode_match = re.search(
        r"(?<!prompt\s)eval.*?(\d+\.?\d*)\s*tokens?\s*per\s*second", stdout, re.IGNORECASE
    )
    if not decode_match:
        decode_match = re.search(
            r"(?<!prompt\s)eval.*?(\d+\.?\d*)\s*tok/s", stdout, re.IGNORECASE
        )

    if prefill_match:
        result["prefill_toks"] = float(prefill_match.group(1))
    if decode_match:
        result["decode_toks"] = float(decode_match.group(1))

    return result
