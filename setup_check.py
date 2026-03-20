#!/usr/bin/env python3
"""Pre-flight check: verify all dependencies and infrastructure are ready."""

import shutil
import subprocess
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from configs.benchmark_config import BITNET_REPO, BITNET_MODELS, OLLAMA_MODELS

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"

def check(label, ok, fix=""):
    status = f"{GREEN}✓{RESET}" if ok else f"{RED}✗{RESET}"
    print(f"  {status} {label}")
    if not ok and fix:
        print(f"      Fix: {fix}")
    return ok

def main():
    all_ok = True
    print("\n=== Python Dependencies ===")
    for pkg in ["pandas", "matplotlib", "seaborn", "psutil", "requests", "lm_eval", "transformers", "torch", "accelerate"]:
        try:
            __import__(pkg)
            check(pkg, True)
        except ImportError:
            all_ok = check(pkg, False, f"pip install {pkg}") and all_ok

    print("\n=== Ollama ===")
    ollama_path = shutil.which("ollama")
    all_ok = check("ollama installed", ollama_path is not None, "Install from https://ollama.ai") and all_ok

    if ollama_path:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        installed = result.stdout if result.returncode == 0 else ""
        for model_name in OLLAMA_MODELS:
            found = model_name.split(":")[0] in installed
            all_ok = check(f"model: {model_name}", found, f"ollama pull {model_name}") and all_ok

    print("\n=== BitNet ===")
    bitnet_exists = os.path.isdir(BITNET_REPO)
    check("BitNet repo cloned", bitnet_exists,
          f"git clone --recursive https://github.com/microsoft/BitNet.git {BITNET_REPO}")

    if bitnet_exists:
        for name, info in BITNET_MODELS.items():
            model_exists = os.path.exists(info["path"])
            check(f"model: {name}", model_exists,
                  f"Download and run setup_env.py — see execution_plan.md Phase 1a")

    print("\n=== Results Directory ===")
    results_dir = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    check("results/ exists", os.path.isdir(results_dir))

    print()
    if all_ok:
        print(f"{GREEN}All checks passed! Ready to run benchmarks.{RESET}")
    else:
        print(f"{YELLOW}Some checks failed — see fixes above.{RESET}")
    print()

if __name__ == "__main__":
    main()
