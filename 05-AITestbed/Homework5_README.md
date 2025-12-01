# Homework 5: 

## Task 1: Cerebras - Llama-7B Batch Size Comparison

Run the Llama-7B example for different batch sizes and compare the performance.

**Issue:** Unable to complete due to access user node

I cannot access the Cerebras user nodes (cer-usn-01 or cer-usn-02) from the Cerebras login node. I get "Permission denied (hostbased)" errors when trying to SSH to the user nodes. I attempted to debug and run the homework on the login node, but that didn't work since the CS-3 system needs to be accessed from the user nodes.

I contacted Paige Kinsley and Murali Emani about this issue. Unfortunately, as of the homework deadline (I requested an extention till Dec 5), this access issue remains unresolved.

---

## Task 2: SambaNova - GPT-OSS Performance on Metis vs Sophia



For the dataset, I used **HuggingFace SQuAD v1.1** (Stanford Question Answering Dataset), which contains question-answer pairs based on Wikipedia articles. I tested 10 prompts from the validation split, all related to Super Bowl 50. The prompts followed the format: "Context: [passage] Question: [question] Answer:" to test extractive question answering.

## Models Used
- **Metis**: `gpt-oss-120b-131072` 
- **Sophia**: `openai/gpt-oss-120b` 

### Performance Comparison

| Metric | Metis (SambaNova) | Sophia (GPU) | Ratio |
|--------|-------------------|--------------|-------|
| **Mean Latency** | 4.987s | 1.621s | **3.07× faster (Sophia)** |
| **Median Latency** | 5.044s | 1.761s | 2.86× faster |
| **Min Latency** | 4.465s | 0.597s | 7.48× faster |
| **Max Latency** | 5.535s | 3.133s | 1.77× faster |
| **Throughput** | 65.37 tokens/sec | 203.98 tokens/sec | **3.12× higher (Sophia)** |
| **Avg Response Length** | 326.0 tokens | 330.7 tokens | Similar |
| **Success Rate** | 10/10 (100%) | 10/10 (100%) | Equal |

Sophia (GPU) significantly outperformed Metis, delivering responses 3× faster with 3× higher throughput. Both platforms achieved 100% success rate with similar response quality.

### Analysis

The results were initially surprising since SambaNova markets their RDUs as inference-optimized hardware, yet Sophia outperformed Metis by 3×. After digging into the results, I realized the performance gap comes down to workload mismatch. My benchmark tested single requests with short answers (37-179 characters), which is exactly what GPUs are optimized for - low-latency individual requests using tensor cores. SambaNova's dataflow architecture, on the other hand, is designed for high-throughput batch processing where multiple requests flow through the pipeline simultaneously. I essentially tested Sophia's strength while bypassing Metis's advantages.

The model variants also played a role. Metis uses the 131k token context window variant while Sophia uses the standard 32k version. That longer context support adds computational overhead through larger position embeddings and attention matrices, even for my short prompts. The software stack matters too - Sophia runs vLLM with years of GPU-specific optimizations like FlashAttention-2 and PagedAttention, while SambaNova's runtime is less mature. Plus there's the latency variance: Metis showed more consistent performance (std dev 0.311s) compared to Sophia (0.839s), which hints that RDUs might be better for production systems needing predictable SLAs.

Where would Metis actually excel? Probably in high-concurrency scenarios with 100+ simultaneous requests, batch processing workloads, or tasks using that full 131k context window. The dataflow architecture should handle parallel streams more efficiently than GPUs at scale, and RDUs are more power-efficient for sustained high-volume inference. My single-request test is similar to the Homework 3 producer-consumer results - DragonHPC only showed its advantages at large data sizes. You need to test the right workload to see specialized hardware benefits.


### Thoughts

The benchmark shows that hardware performance depends heavily on the workload. Sophia won this test because I tested exactly what GPUs are optimized for: low-latency single requests. A fair comparison would need to test batching, concurrency, and long-context scenarios where SambaNova's architecture should shine. This is similar to how the DragonHPC + DDict approach in Homework 3 only showed its advantages at large data sizes - you need to test the right workload to see the benefits of specialized hardware.

This result was initially surprising since SambaNova markets their RDUs as inference-optimized hardware. After analyzing the results, here are the key factors:

### 1. Workload Characteristics
- I tested single requests (not batched)
- Short answers (37-179 characters)
- GPUs excel at low-latency, single-request inference
- SambaNova's dataflow architecture is designed for high-throughput batching, which I didn't test

### 2. Model Variant Differences
- Metis uses the 131k token context window variant
- Sophia uses the standard 32k context window
- Longer context support adds computational overhead (larger position embeddings, attention matrices) even for short prompts like mine

### 3. Software Stack Maturity
- Sophia uses vLLM, a highly optimized GPU inference framework with FlashAttention-2 and PagedAttention
- vLLM has years of GPU-specific optimizations
- SambaNova's runtime is less mature and publicly documented

### 4. Hardware Utilization
- Single requests fully utilize GPU tensor cores for matrix operations
- RDU's dataflow architecture is designed to excel when processing many requests simultaneously (pipeline parallelism)
- My test didn't leverage Metis's strengths

## When Would Metis Excel?

Based on the architecture differences, Metis would likely outperform Sophia in these scenarios:
- **High-concurrency workloads** (100+ concurrent requests) - Dataflow architecture handles parallel streams efficiently
- **Batch processing** (processing 1000s of requests together) - RDU optimized for throughput over latency
- **Very long context tasks** (using the 131k token capacity) - Metis built to handle extended contexts
- **Cost-sensitive deployments** (lower power consumption per token at scale) - RDU more power-efficient than GPUs
- **Deterministic latency** (production systems needing consistent SLAs) - Less variance than GPU (std dev: 0.311s vs 0.839s)

My single-request benchmark essentially tested Sophia's strengths while bypassing Metis's advantages. A more comprehensive benchmark would test batching, concurrency, and long-context scenarios.



## Files

- `compare_metis_sophia.py` - Benchmark script that loads SQuAD and tests both platforms
- `setup_env.sh` - Installs the HuggingFace datasets package
- `metis_vs_sophia_results_*.json` - Raw results with per-request latency and token counts

## How to Run

```bash
cd /home/clarexie/2025/advanced-ai-science-training-series/05-AITestbed/SambaNova

# First time setup (installs datasets package)
./setup_env.sh

# Run benchmark
source ../../04-Inference-Workflows/Agentic-workflows/0_activate_env.sh
python compare_metis_sophia.py
```

The script will:
1. Download SQuAD v1.1 from HuggingFace
2. Test 10 prompts on Metis (takes ~50 seconds)
3. Wait 5 seconds
4. Test the same 10 prompts on Sophia (takes ~16 seconds)
5. Print comparison report
6. Save detailed results to JSON file


