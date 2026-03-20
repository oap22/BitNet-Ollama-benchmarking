"""Run Ollama speed benchmarks via the REST API."""

import time

import psutil
import requests

from utils.ram_monitor import get_process_ram_mb

OLLAMA_API = "http://localhost:11434/api/generate"


def find_ollama_pid() -> int | None:
    """Find the PID of the running ollama server process."""
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            name = proc.info["name"] or ""
            cmdline = " ".join(proc.info["cmdline"] or [])
            if "ollama" in name.lower() and "serve" in cmdline.lower():
                return proc.info["pid"]
            # On macOS, ollama may run as just "ollama" without "serve" in cmdline
            if name.lower() == "ollama" and "serve" not in cmdline.lower():
                continue
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    # Fallback: find any ollama process
    for proc in psutil.process_iter(["pid", "name"]):
        try:
            if proc.info["name"] and "ollama" in proc.info["name"].lower():
                return proc.info["pid"]
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None


def generate_prompt_text(num_tokens: int) -> str:
    """Generate a prompt string of approximately `num_tokens` tokens.

    Uses a repeating pattern of common English words (~1 token each).
    """
    words = (
        "The quick brown fox jumps over the lazy dog and then runs across "
        "the wide open field under a bright blue sky while birds sing softly "
        "in the tall green trees nearby "
    )
    word_list = words.split()
    # Rough approximation: 1 word ≈ 1.3 tokens, so use ~0.77 words per token
    target_words = max(1, int(num_tokens * 0.77))
    repeated = []
    for i in range(target_words):
        repeated.append(word_list[i % len(word_list)])
    return " ".join(repeated)


def run_ollama_benchmark(
    model_name: str,
    prompt_tokens: int,
    gen_tokens: int,
) -> dict:
    """Run a single Ollama benchmark trial.

    Returns dict with keys: prefill_toks, decode_toks, peak_ram_mb, wall_time_s,
    actual_prompt_tokens, actual_gen_tokens.
    """
    prompt_text = generate_prompt_text(prompt_tokens)

    payload = {
        "model": model_name,
        "prompt": prompt_text,
        "stream": False,
        "options": {
            "num_predict": gen_tokens,
        },
    }

    # Measure RAM before
    ollama_pid = find_ollama_pid()
    idle_ram = get_process_ram_mb(ollama_pid) if ollama_pid else 0.0

    start = time.perf_counter()
    resp = requests.post(OLLAMA_API, json=payload, timeout=600)
    wall_time = time.perf_counter() - start
    resp.raise_for_status()
    data = resp.json()

    # Measure RAM after (model is loaded)
    peak_ram = get_process_ram_mb(ollama_pid) if ollama_pid else 0.0

    # Extract timing from response
    prompt_eval_count = data.get("prompt_eval_count", 0)
    prompt_eval_duration = data.get("prompt_eval_duration", 0)  # nanoseconds
    eval_count = data.get("eval_count", 0)
    eval_duration = data.get("eval_duration", 0)  # nanoseconds

    prefill_toks = (
        prompt_eval_count * 1e9 / prompt_eval_duration
        if prompt_eval_duration > 0 else 0.0
    )
    decode_toks = (
        eval_count * 1e9 / eval_duration
        if eval_duration > 0 else 0.0
    )

    return {
        "prefill_toks": round(prefill_toks, 2),
        "decode_toks": round(decode_toks, 2),
        "peak_ram_mb": round(peak_ram, 1),
        "idle_ram_mb": round(idle_ram, 1),
        "wall_time_s": round(wall_time, 3),
        "actual_prompt_tokens": prompt_eval_count,
        "actual_gen_tokens": eval_count,
    }


def preload_model(model_name: str) -> None:
    """Send a short request to ensure the model is loaded in memory."""
    payload = {
        "model": model_name,
        "prompt": "Hello",
        "stream": False,
        "options": {"num_predict": 1},
    }
    try:
        resp = requests.post(OLLAMA_API, json=payload, timeout=120)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"Warning: failed to preload {model_name}: {e}")
