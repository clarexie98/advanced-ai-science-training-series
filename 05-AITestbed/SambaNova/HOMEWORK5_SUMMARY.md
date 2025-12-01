# SambaNova Homework 5: Summary

## ✅ Completed Tasks

### 1. Used HuggingFace Dataset
- **Dataset**: SQuAD v1.1 (Stanford Question Answering Dataset)
- **Source**: https://huggingface.co/datasets/squad
- **Purpose**: Standardized question-answering benchmark based on Wikipedia
- **Sample Size**: 10 prompts from validation split

### 2. Compared Metis vs Sophia Performance
- **Metis Model**: `gpt-oss-120b-131072` (SambaNova RDU processors)
- **Sophia Model**: `openai/gpt-oss-120b` (NVIDIA A100 GPUs)
- **Metrics**: Latency, throughput (tokens/sec), success rate

### 3. Key Findings
**Sophia (GPU) was 3.07× faster than Metis (SambaNova)**:
- **Mean Latency**: 1.621s (Sophia) vs 4.987s (Metis)
- **Throughput**: 203.98 tokens/sec (Sophia) vs 65.37 tokens/sec (Metis)
- **Success Rate**: Both 100% (10/10 successful requests)

### 4. Reasoned About Performance Differences
Key factors explaining why Sophia outperformed Metis in this test:

**Workload Characteristics**:
- Single-request testing (GPUs excel here)
- Short answers (37-179 chars)
- No batching (Metis optimized for batched inference)

**Model Differences**:
- Metis uses longer context variant (131k tokens) → more overhead
- Sophia uses standard context (32k tokens) → less overhead

**Software Stack**:
- Sophia uses highly optimized vLLM framework (FlashAttention, PagedAttention)
- Metis uses proprietary runtime (less mature)

**Hardware Utilization**:
- GPUs fully utilize tensor cores for single requests
- RDUs designed for pipeline parallelism with batched requests

**When Metis Would Excel**:
- High-concurrency scenarios (100+ concurrent requests)
- Batch processing (process 1000s of requests together)
- Very long context tasks (using the 131k token capacity)
- Cost-sensitive deployments (lower power/$ at scale)

---

## Files Created

1. **compare_metis_sophia.py** - Benchmark script with HuggingFace dataset integration
2. **Homework5_README.md** - Setup and execution instructions
3. **Homework5_Analysis.md** - Complete analysis with findings and explanations
4. **setup_env.sh** - Environment setup script (installs datasets package)
5. **run_homework.sh** - Quick start script

---

## How to Run

```bash
cd /home/clarexie/2025/advanced-ai-science-training-series/05-AITestbed/SambaNova

# Setup environment (first time only)
./setup_env.sh

# Run benchmark
source ../../04-Inference-Workflows/Agentic-workflows/0_activate_env.sh
python compare_metis_sophia.py
```

---

## Results File

`metis_vs_sophia_results_20251201_191558.json` contains:
- Full results for all 10 prompts
- Per-request latency, token counts, response text
- Aggregate statistics
- Dataset metadata

---

## Key Learning

This homework demonstrates the importance of **workload-aware benchmarking** when evaluating AI accelerators. The "best" platform depends on:
- Concurrency requirements (single vs batch)
- Latency vs throughput priorities
- Context length needs
- Cost constraints

Sophia won this test because it was designed for exactly this workload (low-latency, single-request inference), while Metis is optimized for different scenarios (high-throughput, batched processing).
