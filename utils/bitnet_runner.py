"""Run BitNet speed benchmarks via bitnet.cpp's e2e_benchmark.py."""

import os
import re
import subprocess
import time

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
    # e2e_benchmark.py expects to be run from the BitNet repo root
    # (it references build/bin/ with relative paths)
    bitnet_repo = os.path.dirname(os.path.dirname(benchmark_script))

    cmd = [
        "python3", benchmark_script,
        "-m", model_path,
        "-p", str(prompt_tokens),
        "-n", str(gen_tokens),
        "-t", str(threads),
    ]

    start = time.perf_counter()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=bitnet_repo,
    )

    monitor = RAMMonitor(proc.pid)
    monitor.start()

    stdout, stderr = proc.communicate(timeout=600)
    wall_time = time.perf_counter() - start
    peak_ram = monitor.stop()

    # e2e_benchmark.py outputs to stderr and may exit with code 1
    # even on success — check for actual data in output
    output = stderr + stdout
    result = parse_bitnet_output(output)

    if result["prefill_toks"] == 0.0 and result["decode_toks"] == 0.0:
        raise RuntimeError(
            f"BitNet benchmark produced no timing data (rc={proc.returncode}):\n{output}"
        )

    result["peak_ram_mb"] = round(peak_ram, 1)
    result["wall_time_s"] = round(wall_time, 3)
    return result


def parse_bitnet_output(output: str) -> dict:
    """Parse e2e_benchmark.py output for timing metrics.

    Output is a markdown table like:
        | model | size | params | backend | ngl | threads | n_batch | test | t/s |
        | ... | ... | ... | ... | ... | ... | ... | pp64 | 51.51 ± 0.28 |
        | ... | ... | ... | ... | ... | ... | ... | tg64 | 50.87 ± 0.67 |
    """
    result = {"prefill_toks": 0.0, "decode_toks": 0.0}

    # Match rows: | ... | ppN | VALUE ± STDDEV |
    prefill_match = re.search(
        r"\|\s*pp\d+\s*\|\s*(\d+\.?\d*)\s*±", output
    )
    # Match rows: | ... | tgN | VALUE ± STDDEV |
    decode_match = re.search(
        r"\|\s*tg\d+\s*\|\s*(\d+\.?\d*)\s*±", output
    )

    if prefill_match:
        result["prefill_toks"] = float(prefill_match.group(1))
    if decode_match:
        result["decode_toks"] = float(decode_match.group(1))

    return result
