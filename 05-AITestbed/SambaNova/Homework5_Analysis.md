# Homework 5 Analysis: Metis (SambaNova) vs Sophia (GPU) Performance Comparison

## Executive Summary

This analysis compares the inference performance of GPT-OSS-120B model on two ALCF platforms:
- **Metis**: SambaNova SN40L with RDU (Reconfigurable Dataflow Unit) processors
- **Sophia**: NVIDIA A100 GPU cluster

**Key Finding**: Sophia (GPU) significantly outperformed Metis (SambaNova), delivering **3.07× faster inference** with **3.12× higher throughput**.

---

## Experimental Setup

### Dataset
- **Source**: HuggingFace SQuAD v1.1 (Stanford Question Answering Dataset)
- **URL**: https://huggingface.co/datasets/squad
- **Description**: Question-answer pairs based on Wikipedia articles
- **Sample Size**: 10 prompts from the validation split
- **Task Type**: Extractive question answering (context + question → answer)

### Models
- **Metis**: `gpt-oss-120b-131072` (longer context window variant)
- **Sophia**: `openai/gpt-oss-120b` (standard variant)
- **Architecture**: Both use the same GPT-OSS 120B parameter model

### Prompt Format
```
Context: [Wikipedia article excerpt]

Question: [Specific question about the context]

Answer:
```

All 10 prompts were derived from the Super Bowl 50 Wikipedia article, testing the model's ability to extract specific information from context.

---

## Performance Results

### Latency Comparison (seconds)

| Metric         | Metis (SambaNova) | Sophia (GPU) | Speedup |
|----------------|-------------------|--------------|---------|
| **Mean**       | 4.987s            | 1.621s       | 3.07×   |
| **Median**     | 5.044s            | 1.761s       | 2.86×   |
| **Min**        | 4.465s            | 0.597s       | 7.48×   |
| **Max**        | 5.535s            | 3.133s       | 1.77×   |
| **Std Dev**    | 0.311s            | 0.839s       | —       |

**Observation**: Sophia consistently delivers faster response times across all metrics. The minimum latency gap (7.48×) suggests Sophia can achieve near-instant responses for simple queries.

### Throughput Comparison (tokens/sec)

| Metric              | Metis (SambaNova) | Sophia (GPU) | Ratio   |
|---------------------|-------------------|--------------|---------|
| **Tokens/sec**      | 65.37             | 203.98       | 3.12×   |
| **Avg tokens/req**  | 326.0             | 330.7        | 1.01×   |

**Observation**: Sophia processes tokens 3.12× faster than Metis. Both platforms generated similar-length responses (~330 tokens), indicating comparable quality.

### Success Rate

| Metric       | Metis | Sophia |
|--------------|-------|--------|
| **Successful** | 10/10 | 10/10  |
| **Failed**     | 0     | 0      |

**Observation**: Both platforms achieved 100% success rate, demonstrating reliability.

---

## Architecture Comparison

### Metis (SambaNova SN40L)
**Hardware:**
- Reconfigurable Dataflow Unit (RDU) processors
- Dataflow architecture optimized for inference
- 32 RDU processors per node
- Custom INT8/INT16 quantization support

**Advantages:**
- Lower power consumption per inference
- Deterministic latency (less variance)
- Optimized for high-throughput batch processing
- Better efficiency at scale (100s of concurrent requests)

**Limitations:**
- Higher overhead for single-request latency
- Newer architecture with less optimization
- Context window optimization (131072 tokens) may add overhead for short prompts

### Sophia (NVIDIA A100 GPUs)
**Hardware:**
- NVIDIA A100 GPUs (80GB HBM2e)
- CUDA tensor cores with FP16/BF16 precision
- Mature ecosystem with highly optimized kernels
- Direct integration with vLLM framework

**Advantages:**
- Extremely low single-request latency
- Highly optimized CUDA kernels for transformers
- Mature software stack (PyTorch, vLLM, FlashAttention)
- Better performance for interactive/low-latency use cases

**Limitations:**
- Higher power consumption per inference
- More variance in latency (0.597s - 3.133s range)
- May saturate at high concurrent request loads

---

## Possible Reasons for Performance Difference

### 1. **Workload Characteristics**
- **SQuAD task**: Extractive QA with short answers (37-179 chars)
- **Prompt length**: Medium context (~500-800 tokens per prompt)
- **Batch size**: Single requests (not batched)

➡️ **Impact**: GPUs excel at single-request, low-latency inference. SambaNova RDUs are designed for high-throughput batching, which wasn't utilized in this test.

### 2. **Model Variant Differences**
- **Metis**: Uses `gpt-oss-120b-131072` with 131k token context window
- **Sophia**: Uses `openai/gpt-oss-120b` with standard 32k context window

➡️ **Impact**: Longer context support adds computational overhead (larger position embeddings, attention matrices), even for short prompts.

### 3. **Software Stack Maturity**
- **Sophia**: Uses vLLM, a highly optimized inference framework with:
  - PagedAttention for memory efficiency
  - FlashAttention-2 for fast attention computation
  - Continuous batching for request scheduling
- **Metis**: Uses SambaNova's proprietary runtime (less publicly documented)

➡️ **Impact**: vLLM's years of GPU-specific optimizations give Sophia a significant advantage.

### 4. **Hardware Utilization**
- **Single request testing**: GPUs can fully utilize tensor cores for matrix operations
- **No batching**: RDU's dataflow architecture is designed to excel when processing multiple requests simultaneously (pipeline parallelism)

➡️ **Impact**: This benchmark didn't test Metis's strengths (high-concurrency, batch inference).

### 5. **Precision and Quantization**
- **Sophia**: Likely using FP16/BF16 precision (standard for A100)
- **Metis**: May be using INT8 quantization for efficiency

➡️ **Impact**: Lower precision can reduce latency but may introduce overhead in certain operations. Without quantization details, it's hard to assess impact.

---

## Use Case Recommendations

### When to Use **Sophia (GPU)**:
✅ **Interactive applications** requiring sub-second latency:
- Chatbots, virtual assistants
- Real-time code completion
- Live Q&A systems

✅ **Development and experimentation**:
- Model fine-tuning
- Rapid prototyping
- Research experiments

✅ **Low to moderate concurrency** (1-50 concurrent requests)

### When to Use **Metis (SambaNova)**:
✅ **High-throughput batch processing**:
- Offline data annotation (process 1M+ documents)
- Batch summarization/translation
- Large-scale dataset generation

✅ **Cost-sensitive deployments**:
- Lower power consumption at scale
- Better TCO for sustained high-volume workloads

✅ **Deterministic latency requirements**:
- Production systems requiring consistent SLAs
- Regulated industries needing predictable performance

✅ **Very long context workloads**:
- 131k token context window supports document-level tasks
- RAG systems with large knowledge bases

---

## Recommendations for Future Experiments

To better understand the performance trade-offs, the following experiments would be valuable:

### 1. **Batch Size Comparison**
- Test with batch sizes: 1, 8, 16, 32, 64, 128
- Measure throughput (requests/sec) vs. latency trade-off
- **Hypothesis**: Metis will show better relative performance at batch size ≥ 16

### 2. **Concurrency Testing**
- Simulate concurrent requests: 10, 50, 100, 200, 500
- Measure P50, P95, P99 latency under load
- **Hypothesis**: Metis will maintain more stable latency at high concurrency

### 3. **Long Context Evaluation**
- Test prompts with 4k, 16k, 64k, 131k tokens
- Measure latency scaling with context length
- **Hypothesis**: Metis's 131k context support will show better efficiency

### 4. **Different Task Types**
- Generation-heavy tasks (story writing, code generation)
- Longer responses (1000+ tokens)
- **Hypothesis**: Token generation speed may differ from encoding speed

### 5. **Cost Analysis**
- Measure energy consumption per 1000 tokens
- Calculate cost per million requests
- Include capital costs (hardware depreciation)

---

## Conclusion

In this **single-request, short-answer QA benchmark**, Sophia (GPU) significantly outperformed Metis (SambaNova), delivering **3.07× faster latency** and **3.12× higher throughput**. This advantage stems from:
1. GPU's maturity in single-request, low-latency inference
2. Highly optimized vLLM inference framework
3. Workload characteristics favoring tensor core operations

However, this result **does not invalidate SambaNova's design philosophy**. Metis is optimized for:
- High-throughput batch processing (not tested)
- Long-context inference (131k tokens)
- Cost-efficient scaled deployment

**Key Insight**: The "best" platform depends on the deployment scenario:
- **Latency-critical + low concurrency** → Sophia (GPU)
- **Throughput-critical + high concurrency** → Metis (SambaNova)

This homework demonstrates the importance of **workload-aware benchmarking** when evaluating AI accelerators.

---

## Appendix A: Sample Responses

### Question 1: "Which NFL team represented the AFC at Super Bowl 50?"

**Metis Response (5.535s, 302 tokens):**
> Denver Broncos

**Sophia Response (3.133s, 294 tokens):**
> The Denver Broncos

**Analysis**: Both responses are correct. Sophia's slightly more verbose answer indicates similar quality despite 3× faster inference.

### Question 4: "What was the theme of Super Bowl 50?"

**Metis Response (5.038s, 276 tokens):**
> Golden Anniversary

**Sophia Response (1.029s, 292 tokens):**
> Golden Anniversary

**Analysis**: Identical correctness. Sophia achieved the answer in 1/5th the time.

---

## Appendix B: Dataset Information

**HuggingFace SQuAD v1.1**
- Training set: 87,599 examples
- Validation set: 10,570 examples
- Source: Wikipedia articles
- Task: Extractive question answering (answer is a span within the context)

**Why SQuAD?**
1. Standardized benchmark for QA systems
2. Real-world question complexity
3. Tests comprehension + information extraction
4. Widely used in academic/industry evaluations

**Sample Prompt Structure:**
- Context length: 400-800 tokens (Wikipedia paragraph)
- Question length: 10-25 tokens
- Answer length: 1-20 tokens (extracted span)
- Total input: ~500-850 tokens per request

---

## Appendix C: Raw Data

**Results File**: `metis_vs_sophia_results_20251201_191558.json`

Contains:
- Timestamp and dataset metadata
- Per-request latency, token counts, response text
- Aggregate statistics (mean, median, std dev, min, max)
- Model configuration details

---

**Analysis Date**: December 1, 2025  
**Author**: Clare Xie  
**Course**: ALCF AI for Science Training Series  
**Assignment**: Homework 5 - SambaNova Performance Comparison
