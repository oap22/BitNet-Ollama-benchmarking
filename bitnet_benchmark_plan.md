# BitNet Benchmarking Plan

## Objective

Evaluate whether native 1.58-bit (ternary) LLMs offer a better quality/efficiency tradeoff than conventional post-training quantized models for local inference. Metrics: **RAM usage** and **tokens per second (tok/s)**. Secondary: qualitative output quality.

---

## Metrics

### Primary: RAM

- Measure peak RSS (Resident Set Size) of the inference process during generation
- Use `psutil` in Python or `/proc/{pid}/status` VmRSS on Linux
- Record at idle (model loaded, no inference) and at peak (during generation)

### Primary: Tokens Per Second

- Measure **prompt processing** (prefill) and **token generation** (decode) separately
- BitNet: `e2e_benchmark.py` reports both; parse its stdout
- Ollama: use the `/api/generate` endpoint which returns timing metadata in the response JSON (`prompt_eval_duration`, `eval_duration`, `prompt_eval_count`, `eval_count`)
- For each configuration, run **3+ trials** and report mean ± std dev
- Always do **2 warm-up runs** before recording (first run includes model loading overhead)

### Secondary: Model Quality (lm-evaluation-harness)

- Use EleutherAI's `lm-evaluation-harness` — the same framework used in the BitNet tech report
- Results are directly comparable to published numbers
- Run entirely locally, no API costs
- Target benchmarks (matching BitNet tech report):
  - **ARC-Easy / ARC-Challenge** — science question answering
  - **HellaSwag** — commonsense reasoning
  - **WinoGrande** — coreference resolution
  - **PIQA** — physical intuition
  - **MMLU** — broad knowledge across 57 subjects
  - **HumanEval** — code generation (optional, requires execution sandbox)
  - **GSM8K** — grade school math reasoning
- For each model, record accuracy (or normalized accuracy where applicable)

---

## Comparisons

### Comparison 1: Flagship Ternary vs Best Conventional Small Models

**Question:** Does the best natively trained ternary model compete with the best conventional small models at various quantization levels?

| Side | Model | Params | Precision/Quant | Runner |
|------|-------|--------|-----------------|--------|
| BitNet | BitNet-b1.58-2B-4T | 2.4B | Native 1.58-bit (i2_s) | bitnet.cpp |
| Baseline | Qwen2.5 1.5B Instruct | 1.5B | GGUF Q4_K_M | Ollama |
| Baseline | Qwen2.5 1.5B Instruct | 1.5B | GGUF Q8_0 | Ollama |
| Baseline | Llama 3.2 3B Instruct | 3.2B | GGUF Q4_K_M | Ollama |
| Baseline | Llama 3.2 3B Instruct | 3.2B | GGUF Q2_K | Ollama |
| Baseline | Gemma 2 2B Instruct | 2.6B | GGUF Q4_K_M | Ollama |

**Why these models:** The BitNet tech report (arxiv 2504.12285) already compared against Qwen2.5, LLaMA 3.2, and Gemma. We replicate and extend by adding GGUF quant variants and measuring on our own hardware. Parameter counts bracket the 2.4B BitNet model. All are instruction-tuned.

**Key confounds to note:** Different training data, different training methods (LLaMA 3.2 1B uses pruning+distillation, Gemma uses distillation, BitNet is trained from scratch). These cannot be controlled — acknowledge in results.

### Comparison 2: Native Ternary vs Post-Training Ternary

**Question:** Is it better to train a small model natively at 1.58 bits, or to take a larger full-precision model and crush it down to 1.58 bits?

| Side | Model | Params | Precision/Quant | Runner |
|------|-------|--------|-----------------|--------|
| Native | BitNet-b1.58-2B-4T | 2.4B | Native 1.58-bit (i2_s) | bitnet.cpp |
| Post-train | Llama3-8B-1.58-100B-tokens | 8.0B | Post-training 1.58-bit (i2_s) | bitnet.cpp |

**Why this comparison:** Same inference framework (bitnet.cpp), same kernel optimizations, same bit-width. The only variable is *how the model became ternary*. The 8B model has 3.3x more parameters but was not trained for ternary — it was quantized after the fact. This isolates the native-vs-post-training question.

**Note:** Llama3-8B-1.58 was only trained on 100B tokens at 1.58-bit (the name says so). The original Llama 3 8B was trained on far more. So this is really testing "big model, aggressive post-training quantization" vs "small model, native quantization with massive token budget."

### Comparison 3 (Optional): Falcon3 1.58-bit Scaling Curve

**Question:** How does post-training ternary quantization scale across model sizes?

| Model | Params | Runner |
|-------|--------|--------|
| Falcon3-1B-Instruct-1.58bit | 1B | bitnet.cpp |
| Falcon3-3B-Instruct-1.58bit | 3B | bitnet.cpp |
| Falcon3-7B-Instruct-1.58bit | 7B | bitnet.cpp |
| Falcon3-10B-Instruct-1.58bit | 10B | bitnet.cpp |

**Purpose:** Plot tok/s and RAM against parameter count for a single model family all at the same bit-width. Gives you a scaling curve. Compare each point against BitNet-b1.58-2B-4T to see where the crossover happens (if it does) between "more parameters, post-training quantized" and "fewer parameters, natively trained."

---

## Prompt Matrix

### Speed Benchmarks (tok/s and RAM)

Use synthetic prompts of controlled length. Vary prompt length and generation length independently.

| Config Name | Prompt Tokens (-p) | Generation Tokens (-n) | Purpose |
|-------------|--------------------|-----------------------|---------|
| short-short | 64 | 64 | Baseline latency |
| short-long | 64 | 256 | Generation-heavy workload |
| medium-medium | 256 | 128 | Balanced workload |
| long-short | 512 | 64 | Prefill-heavy workload |
| long-long | 512 | 256 | Stress test |

For BitNet: use `e2e_benchmark.py -m MODEL -p {prompt_tokens} -n {gen_tokens} -t {threads}`

For Ollama: send requests to `http://localhost:11434/api/generate` with a prompt padded to the target token count. Parse timing from the response JSON.

**Thread count:** Test at 1, 2, 4, and max physical cores. Report all, but highlight the "best" thread count for each model.

### Quality Benchmarks (lm-evaluation-harness)

Run the same benchmark suite that the BitNet tech report used. This gives you directly comparable numbers.

**Benchmarks to run:**

| Benchmark | Task Name in Harness | Metric | What It Tests |
|-----------|---------------------|--------|---------------|
| ARC-Easy | `arc_easy` | acc_norm | Science QA (easy) |
| ARC-Challenge | `arc_challenge` | acc_norm | Science QA (hard) |
| HellaSwag | `hellaswag` | acc_norm | Commonsense reasoning |
| WinoGrande | `winogrande` | acc | Coreference resolution |
| PIQA | `piqa` | acc_norm | Physical intuition |
| MMLU | `mmlu` | acc | Broad knowledge (57 subjects) |
| GSM8K | `gsm8k` | acc (strict-match) | Grade school math |

**For Ollama models:**
```bash
# lm-evaluation-harness supports local OpenAI-compatible APIs
# Ollama exposes one at http://localhost:11434/v1
lm_eval --model local-completions \
  --model_args model=qwen2.5:1.5b,base_url=http://localhost:11434/v1 \
  --tasks arc_easy,arc_challenge,hellaswag,winogrande,piqa,mmlu \
  --batch_size auto
```

**For BitNet models:**
- lm-evaluation-harness does not natively support bitnet.cpp
- Two options:
  1. Use `run_inference_server.py` from the BitNet repo (if it exposes an OpenAI-compatible endpoint) and point lm_eval at it
  2. Use the BF16 master weights (`microsoft/bitnet-b1.58-2B-4T-bf16`) through HuggingFace transformers for eval only — quality should be identical since the weights are mathematically equivalent, just unpacked. **This is the recommended approach for quality eval** since speed isn't what you're measuring here
- Verify: compare a few outputs from bitnet.cpp and the BF16 weights to confirm they match

**Important:** The BF16 weights route measures model quality only, NOT inference speed. Speed benchmarks must still use bitnet.cpp with the ternary-packed weights.

---

## Infrastructure Setup

### BitNet Side

```bash
# Clone and build
git clone --recursive https://github.com/microsoft/BitNet.git
cd BitNet
conda create -n bitnet-cpp python=3.9
conda activate bitnet-cpp
pip install -r requirements.txt

# Download and prepare the flagship model
huggingface-cli download microsoft/BitNet-b1.58-2B-4T-gguf --local-dir models/BitNet-b1.58-2B-4T
python setup_env.py -md models/BitNet-b1.58-2B-4T -q i2_s

# Download post-training ternary model for Comparison 2
# (check setup_env.py --hf-repo options or download manually)

# Verify it runs
python run_inference.py -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf -p "Hello" -n 32
```

### Ollama Side

```bash
# Install Ollama (https://ollama.ai)
# Pull models at specific quant levels
ollama pull qwen2.5:1.5b          # default quant
ollama pull llama3.2:3b
ollama pull gemma2:2b

# For specific quant levels, use Modelfiles or pull from 
# specific tags — check ollama.com/library for available tags

# Verify
ollama run qwen2.5:1.5b "Hello"
```

### Jupyter Environment

```bash
pip install jupyter pandas matplotlib seaborn psutil requests
```

Access BitNet via subprocess calls. Access Ollama via its REST API (`requests` library) or the `ollama` Python package.

### lm-evaluation-harness

```bash
pip install lm-eval

# For running BitNet quality evals via BF16 weights:
pip install transformers torch accelerate

# Verify installation
lm_eval --tasks list
```

---

## Data Collection Format

Store all results in a single CSV/DataFrame with these columns:

| Column | Description |
|--------|-------------|
| comparison | Which comparison (1, 2, or 3) |
| model_name | Human-readable model name |
| param_count | Parameter count in billions |
| quant_method | "native_ternary", "post_training_ternary", "gguf_q4km", "gguf_q8", "gguf_q2k" |
| runner | "bitnet_cpp" or "ollama" |
| prompt_tokens | Number of prompt tokens |
| gen_tokens | Number of generated tokens |
| threads | Thread count used |
| trial | Trial number (1, 2, 3...) |
| prefill_toks | Prompt processing speed (tok/s) |
| decode_toks | Token generation speed (tok/s) |
| peak_ram_mb | Peak RAM usage in MB |
| idle_ram_mb | RAM at idle (model loaded) |
| wall_time_s | Total wall clock time |

---

## Analysis Plan

### Speed Analysis
- Bar charts: decode tok/s per model at each prompt config
- Line plots: tok/s vs thread count for each model
- Scatter plot: tok/s vs RAM — this is the "bang for the buck" chart. Models in the upper-left (fast + small) win.

### Quality Analysis
- Grouped bar chart: benchmark scores per model (one cluster per benchmark, one bar per model)
- Table: all benchmark scores side-by-side, including published BitNet tech report numbers for comparison
- Highlight where your measured numbers diverge from published results (sanity check)

### The "Bang for Buck" Summary
- Combined scatter plot: x-axis = peak RAM (MB), y-axis = average benchmark score (mean across all lm-eval tasks), point size = decode tok/s
- This single chart answers "which model gives me the best quality at the smallest memory footprint while also being fast?"

---

## Known Limitations to Acknowledge

1. **Training data is not controlled.** Different models trained on different data. Cannot isolate architecture effects.
2. **Different runners.** BitNet uses bitnet.cpp, baselines use Ollama (which wraps llama.cpp). Kernel implementations differ. Speed differences may partly reflect runtime optimization quality, not just model properties.
3. **Single hardware config.** Results are specific to whatever machine you run on. ARM vs x86 will show very different relative performance for BitNet.
4. **Benchmark limitations.** lm-evaluation-harness measures multiple-choice and exact-match accuracy. It doesn't capture open-ended generation quality, conversational ability, or instruction-following nuance. Good benchmark scores don't guarantee a good chat experience.
5. **Ollama quant levels.** Default Ollama pulls may not always correspond to exact quant levels. Verify actual quantization via `ollama show MODEL --modelfile`.
6. **BitNet-b1.58-2B-4T is instruction-tuned (SFT + DPO).** Ensure baseline models are also instruction-tuned variants for fair quality comparison.
