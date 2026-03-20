# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Benchmarking project comparing native 1.58-bit (ternary) LLMs against post-training quantized models for local inference. Primary metrics are RAM usage and tokens per second; secondary metric is model quality via lm-evaluation-harness.

Three comparison scenarios:
1. **BitNet-b1.58-2B-4T** (native ternary) vs conventional small models (Qwen2.5 1.5B, Llama 3.2 3B, Gemma 2 2B) at various GGUF quantization levels
2. **Native ternary** (BitNet 2.4B) vs **post-training ternary** (Llama3-8B-1.58) — same bit-width, same runtime, isolates training approach
3. (Optional) Falcon3 1.58-bit scaling curve across 1B/3B/7B/10B

## Architecture

- **BitNet models** run via `bitnet.cpp` using `e2e_benchmark.py` for speed and `run_inference.py` for generation
- **Baseline models** run via Ollama (wraps llama.cpp), queried through REST API at `http://localhost:11434/api/generate`
- **Quality evals** use EleutherAI's `lm-evaluation-harness` (`lm_eval`). BitNet quality is measured via BF16 master weights through HuggingFace transformers (not bitnet.cpp)
- **Analysis** in Jupyter notebooks with pandas, matplotlib, seaborn

## Environment Setup

```bash
# BitNet side
git clone --recursive https://github.com/microsoft/BitNet.git
conda create -n bitnet-cpp python=3.9
conda activate bitnet-cpp
pip install -r requirements.txt

# Ollama side — install from ollama.ai, then:
ollama pull qwen2.5:1.5b
ollama pull llama3.2:3b
ollama pull gemma2:2b

# Python dependencies
pip install jupyter pandas matplotlib seaborn psutil requests
pip install lm-eval transformers torch accelerate
```

## Key Commands

```bash
# BitNet inference
python run_inference.py -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf -p "prompt" -n 32

# BitNet benchmarking
python e2e_benchmark.py -m MODEL -p {prompt_tokens} -n {gen_tokens} -t {threads}

# Ollama quality eval
lm_eval --model local-completions \
  --model_args model=qwen2.5:1.5b,base_url=http://localhost:11434/v1 \
  --tasks arc_easy,arc_challenge,hellaswag,winogrande,piqa,mmlu \
  --batch_size auto

# Verify Ollama quant level
ollama show MODEL --modelfile
```

## Data Format

Results stored in CSV with columns: `comparison`, `model_name`, `param_count`, `quant_method`, `runner`, `prompt_tokens`, `gen_tokens`, `threads`, `trial`, `prefill_toks`, `decode_toks`, `peak_ram_mb`, `idle_ram_mb`, `wall_time_s`.

Speed benchmarks use 5 prompt configurations (short-short through long-long) with 2 warm-up runs and 3+ recorded trials per configuration. Thread counts tested: 1, 2, 4, and max physical cores.

## Important Notes

- BitNet speed benchmarks must use bitnet.cpp with ternary-packed weights; BF16 weights are only for quality eval
- Ollama timing comes from response JSON fields: `prompt_eval_duration`, `eval_duration`, `prompt_eval_count`, `eval_count`
- RAM measured as peak RSS via `psutil`
- Different runners (bitnet.cpp vs llama.cpp/Ollama) mean speed differences partly reflect runtime optimization, not just model properties
