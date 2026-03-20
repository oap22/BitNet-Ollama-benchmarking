# Execution Plan: BitNet vs Ollama Benchmarking

## Phase 1: Environment Setup

### 1a. BitNet Infrastructure
```bash
git clone --recursive https://github.com/microsoft/BitNet.git
cd BitNet
conda create -n bitnet-cpp python=3.9
conda activate bitnet-cpp
pip install -r requirements.txt

# Download and prepare flagship model
huggingface-cli download microsoft/BitNet-b1.58-2B-4T-gguf --local-dir models/BitNet-b1.58-2B-4T
python setup_env.py -md models/BitNet-b1.58-2B-4T -q i2_s

# Download post-training ternary model (Comparison 2)
# Llama3-8B-1.58-100B-tokens — download and prepare with i2_s quantization

# Verify
python run_inference.py -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf -p "Hello" -n 32
```

### 1b. Ollama Infrastructure
```bash
# Install Ollama from https://ollama.ai, then pull models:
ollama pull qwen2.5:1.5b
ollama pull llama3.2:3b
ollama pull gemma2:2b

# Verify quant levels match expectations
ollama show qwen2.5:1.5b --modelfile
ollama show llama3.2:3b --modelfile
ollama show gemma2:2b --modelfile

# For specific quant variants (Q8_0, Q2_K), check available tags
# at ollama.com/library or create custom Modelfiles pointing to GGUF files
```

**Important:** Default Ollama pulls may not correspond to exact quant levels (e.g., Q4_K_M vs Q4_0). Verify and document what you actually get.

### 1c. Python Dependencies
```bash
pip install jupyter pandas matplotlib seaborn psutil requests
pip install lm-eval transformers torch accelerate
```

---

## Phase 2: Speed Benchmarks (Jupyter Notebook)

### Models to Benchmark

| Model | Params | Quant | Runner |
|-------|--------|-------|--------|
| BitNet-b1.58-2B-4T | 2.4B | Native 1.58-bit (i2_s) | bitnet.cpp |
| Qwen2.5 1.5B Instruct | 1.5B | GGUF Q4_K_M | Ollama |
| Qwen2.5 1.5B Instruct | 1.5B | GGUF Q8_0 | Ollama |
| Llama 3.2 3B Instruct | 3.2B | GGUF Q4_K_M | Ollama |
| Llama 3.2 3B Instruct | 3.2B | GGUF Q2_K | Ollama |
| Gemma 2 2B Instruct | 2.6B | GGUF Q4_K_M | Ollama |
| Llama3-8B-1.58 | 8.0B | Post-training 1.58-bit (i2_s) | bitnet.cpp |

### Prompt Configurations

| Config | Prompt Tokens | Gen Tokens | Purpose |
|--------|--------------|------------|---------|
| short-short | 64 | 64 | Baseline latency |
| short-long | 64 | 256 | Generation-heavy |
| medium-medium | 256 | 128 | Balanced |
| long-short | 512 | 64 | Prefill-heavy |
| long-long | 512 | 256 | Stress test |

### Protocol
- **Thread counts:** 1, 2, 4, max physical cores
- **Per config/thread combo:** 2 warm-up runs (discarded), then 3+ recorded trials
- **Metrics collected per trial:** prefill tok/s, decode tok/s, peak RAM (MB), idle RAM (MB), wall time (s)

### Data Collection

**BitNet models** — spawn `e2e_benchmark.py` via subprocess, parse stdout:
```bash
python e2e_benchmark.py -m MODEL -p {prompt_tokens} -n {gen_tokens} -t {threads}
```

**Ollama models** — POST to REST API, parse response JSON:
```python
resp = requests.post("http://localhost:11434/api/generate", json={
    "model": model_name,
    "prompt": prompt_text,
    "options": {"num_predict": gen_tokens}
})
# Extract: prompt_eval_duration, eval_duration, prompt_eval_count, eval_count
# prefill_toks = prompt_eval_count * 1e9 / prompt_eval_duration
# decode_toks  = eval_count * 1e9 / eval_duration
```

**RAM** — measure via `psutil` (peak RSS of the inference process)

### Output
All rows appended to `results/speed_benchmarks.csv` with the unified schema:
```
comparison, model_name, param_count, quant_method, runner, prompt_tokens, gen_tokens,
threads, trial, prefill_toks, decode_toks, peak_ram_mb, idle_ram_mb, wall_time_s
```

---

## Phase 3: Quality Evals

### Benchmark Suite

| Benchmark | Task | Metric |
|-----------|------|--------|
| ARC-Easy | arc_easy | acc_norm |
| ARC-Challenge | arc_challenge | acc_norm |
| HellaSwag | hellaswag | acc_norm |
| WinoGrande | winogrande | acc |
| PIQA | piqa | acc_norm |
| MMLU | mmlu | acc |
| GSM8K | gsm8k | acc (strict-match) |

### Ollama Quality Evals
```bash
# Run for each Ollama model variant
lm_eval --model local-completions \
  --model_args model=qwen2.5:1.5b,base_url=http://localhost:11434/v1 \
  --tasks arc_easy,arc_challenge,hellaswag,winogrande,piqa,mmlu,gsm8k \
  --batch_size auto
```

### BitNet Quality Evals
```bash
# Use BF16 master weights through HuggingFace (NOT bitnet.cpp)
# Quality is mathematically identical; lm-eval doesn't support bitnet.cpp natively
lm_eval --model hf \
  --model_args pretrained=microsoft/bitnet-b1.58-2B-4T-bf16,dtype=bfloat16 \
  --tasks arc_easy,arc_challenge,hellaswag,winogrande,piqa,mmlu,gsm8k \
  --batch_size auto
```

### Output
Store quality results in `results/quality_benchmarks.csv` or alongside speed data.

---

## Phase 4: Analysis (Jupyter Notebook)

### Speed Visualizations
1. **Bar charts** — decode tok/s per model at each prompt config
2. **Line plots** — tok/s vs thread count for each model
3. **Scatter plot** — tok/s vs RAM (upper-left = best; fast + small)
4. **Heatmap** — prefill/decode tok/s across prompt configs x thread counts

### Quality Visualizations
1. **Grouped bar charts** — benchmark scores per model (one cluster per benchmark)
2. **Comparison table** — all scores side-by-side + published BitNet tech report numbers
3. **Sanity check** — flag divergence from published results

### Combined "Bang for Buck" Summary
- **X-axis:** Peak RAM (MB)
- **Y-axis:** Average benchmark score (mean across lm-eval tasks)
- **Point size:** Decode tok/s
- Directly answers: "Which model gives best quality at smallest memory footprint while being fast?"

---

## Key Caveats

1. **Different runners** — bitnet.cpp vs Ollama (llama.cpp) means speed differences partly reflect runtime optimization, not just model properties
2. **Training data uncontrolled** — different models trained on different data; cannot fully isolate architecture effects
3. **Single hardware config** — results are platform-specific (ARM vs x86 matters significantly for BitNet)
4. **Verify Ollama quant levels** — `ollama show MODEL --modelfile` to confirm actual quantization
5. **All models must be instruction-tuned** — BitNet-b1.58-2B-4T uses SFT+DPO; baselines need instruction-tuned variants too
6. **lm-eval limitations** — measures multiple-choice/exact-match accuracy, not open-ended generation quality
