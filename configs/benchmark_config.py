"""Centralized benchmark configuration."""

import os

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
BITNET_REPO = os.path.join(os.path.dirname(PROJECT_ROOT), "BitNet")
BITNET_BENCHMARK_SCRIPT = os.path.join(BITNET_REPO, "utils", "e2e_benchmark.py")
BITNET_INFERENCE_SCRIPT = os.path.join(BITNET_REPO, "run_inference.py")

# ── BitNet Models ──────────────────────────────────────────────────────────
BITNET_MODELS = {
    "BitNet-b1.58-2B-4T": {
        "path": os.path.join(BITNET_REPO, "models", "BitNet-b1.58-2B-4T", "ggml-model-i2_s.gguf"),
        "param_count": "2.4B",
        "quant_method": "native_1.58bit_i2_s",
        "comparison": "comp1",
    },
    "Llama3-8B-1.58": {
        "path": os.path.join(BITNET_REPO, "models", "Llama3-8B-1.58-100B-tokens", "ggml-model-i2_s.gguf"),
        "param_count": "8.0B",
        "quant_method": "post_training_1.58bit_i2_s",
        "comparison": "comp2",
    },
}

# ── Ollama Models ──────────────────────────────────────────────────────────
OLLAMA_MODELS = {
    "qwen2.5:1.5b": {
        "param_count": "1.5B",
        "quant_method": "Q4_K_M",  # verify with: ollama show qwen2.5:1.5b --modelfile
        "comparison": "comp1",
    },
    "llama3.2:3b": {
        "param_count": "3.2B",
        "quant_method": "Q4_K_M",
        "comparison": "comp1",
    },
    "gemma2:2b": {
        "param_count": "2.6B",
        "quant_method": "Q4_0",
        "comparison": "comp1",
    },
}

# ── Prompt Configurations ─────────────────────────────────────────────────
PROMPT_CONFIGS = {
    "short-short":   {"prompt_tokens": 64,  "gen_tokens": 64},
    "short-long":    {"prompt_tokens": 64,  "gen_tokens": 256},
    "medium-medium": {"prompt_tokens": 256, "gen_tokens": 128},
    "long-short":    {"prompt_tokens": 512, "gen_tokens": 64},
    "long-long":     {"prompt_tokens": 512, "gen_tokens": 256},
}

# ── Thread Configurations ─────────────────────────────────────────────────
PHYSICAL_CORES = 14  # update for your machine
THREAD_COUNTS = [1, 2, 4, PHYSICAL_CORES]

# ── Benchmark Protocol ─────────────────────────────────────────────────────
WARMUP_RUNS = 2
RECORDED_TRIALS = 3

# ── Quality Eval Tasks ─────────────────────────────────────────────────────
EVAL_TASKS = "arc_easy,arc_challenge,hellaswag,winogrande,piqa,mmlu,gsm8k"

# ── CSV Schema ─────────────────────────────────────────────────────────────
CSV_COLUMNS = [
    "comparison", "model_name", "param_count", "quant_method", "runner",
    "prompt_tokens", "gen_tokens", "threads", "trial",
    "prefill_toks", "decode_toks", "peak_ram_mb", "idle_ram_mb", "wall_time_s",
]

SPEED_CSV = os.path.join(RESULTS_DIR, "speed_benchmarks.csv")
QUALITY_CSV = os.path.join(RESULTS_DIR, "quality_benchmarks.csv")
