# SambaNova Homework: Metis vs Sophia Performance Comparison

## Overview

This homework compares the inference performance of the GPT-OSS-120B model on two different platforms:
- **Metis**: SambaNova SN40L (specialized RDU processors optimized for inference)
- **Sophia**: ALCF GPU cluster (general-purpose A100 GPUs)

**Dataset**: HuggingFace SQuAD (Stanford Question Answering Dataset)
- URL: https://huggingface.co/datasets/squad
- Description: Question-answer pairs based on Wikipedia articles
- Purpose: Tests the model's ability to extract information from context

## Files

- `compare_metis_sophia.py` - Main benchmark script
- `Homework5_Analysis.md` - Analysis and findings (to be created after running)

## Setup Instructions

### 1. Navigate to the SambaNova directory

```bash
cd /home/clarexie/2025/advanced-ai-science-training-series/05-AITestbed/SambaNova
```

### 2. Ensure authentication is set up

```bash
cd ../
python inference_auth_token.py authenticate
```

If you've already authenticated (which you have from Homework 4), you're good to go!

### 3. Verify token is valid

```bash
python inference_auth_token.py get_time_until_token_expiration --units hours
```

## Running the Benchmark

### Default Test (10 samples from SQuAD)

```bash
cd SambaNova
source ../../04-Inference-Workflows/Agentic-workflows/0_activate_env.sh
python compare_metis_sophia.py
```

This runs 10 SQuAD samples on both Metis and Sophia (~3-5 minutes).

### Custom Sample Size

Edit `compare_metis_sophia.py` line 354:
```python
# Change num_samples parameter:
dataset_samples = load_huggingface_dataset(num_samples=10)  # Change 10 to desired number
```

Then run:
```bash
python compare_metis_sophia.py
```

## What the Script Does

1. **Loads HuggingFace Dataset**: 
   - Automatically downloads SQuAD v1.1 from HuggingFace
   - Uses 10 question-answer samples (configurable)
   - Each sample includes context passage and a question
   - Format: "Context: [passage]\n\nQuestion: [question]\n\nAnswer:"

2. **Benchmarks Metis (SambaNova)**:
   - Model: `gpt-oss-120b-131072` (longer context window)
   - Endpoint: https://inference-api.alcf.anl.gov/resource_server/metis/api/v1
   - Measures latency (time to first token + generation)
   - Collects token usage statistics
   - Records response quality

3. **Benchmarks Sophia (GPU Cluster)**:
   - Model: `openai/gpt-oss-120b`
   - Endpoint: https://data-portal-dev.cels.anl.gov/resource_server/sophia/vllm/v1
   - Repeats the same prompts for fair comparison
   - Collects same metrics

4. **Calculates Statistics**:
   - Mean, min, max, median latency
   - Throughput (tokens/second)
   - Success rates
   - Speedup factor

5. **Generates Report**:
   - Console output with formatted comparison table
   - JSON file with detailed results
   - Timestamp for tracking

## Expected Output

You'll see output like this:

```
======================================================================
Benchmarking METIS
======================================================================

[1/5] Testing prompt: Explain the concept of quantum entanglement...
  Run 1: 2.345s, 487 tokens, Response length: 1853 chars

[2/5] Testing prompt: What is the difference between supervised...
  Run 1: 1.987s, 423 tokens, Response length: 1645 chars

...

======================================================================
PERFORMANCE COMPARISON REPORT
======================================================================

📊 LATENCY COMPARISON (seconds)
Metric               Metis (SambaNova)         Sophia (GPU)         
----------------------------------------------------------------------
Mean Latency                          2.156                    3.421
Min Latency                           1.987                    2.876
Max Latency                           2.543                    4.123
Median Latency                        2.234                    3.387

🚀 THROUGHPUT COMPARISON
Metric               Metis (SambaNova)         Sophia (GPU)         
----------------------------------------------------------------------
Tokens/sec                           212.34                   142.67
Avg tokens/req                        458.2                    487.3

✅ SUCCESS RATE
Metric               Metis (SambaNova)         Sophia (GPU)         
----------------------------------------------------------------------
Successful                                 5                        5
Failed                                     0                        0

⚡ SPEEDUP: Metis is 1.59x faster than Sophia

💾 Detailed results saved to: metis_vs_sophia_results_20251201_143022.json
```

## Analyzing Results

After running, you should observe:

### Expected Performance Patterns

**Metis (SambaNova) Advantages:**
- **Lower latency** - RDU processors optimized for inference
- **Higher throughput** - Purpose-built dataflow architecture
- **Consistent performance** - Less variance in response times
- **Lower power consumption** - Specialized silicon vs GPUs

**Sophia (GPU) Characteristics:**
- **General-purpose** - A100s designed for training AND inference
- **Flexible** - Can run diverse workloads
- **Potentially higher latency** - Not inference-optimized
- **Good for small batches** - But Metis excels at single-request throughput

### Key Metrics to Compare

1. **Mean Latency**: Average time per request
2. **Throughput**: Tokens generated per second
3. **Consistency**: Standard deviation of latencies
4. **Response Quality**: Compare actual responses (manual review)

## Using the Web UI for Qualitative Testing

While the script measures quantitative performance, you can also test qualitatively:

### 1. Access the Web UI

Go to: https://inference.alcf.anl.gov/

### 2. Test Metis Model

- Select `gpt-oss-120b-131072` (Metis)
- Ask one of the scientific questions
- Note the response time and quality

### 3. Test Sophia Model

- Select `openai/gpt-oss-120b` (Sophia)  
- Ask the SAME question
- Compare response time and quality

### 4. Document Observations

Note any differences in:
- Response latency (perceived speed)
- Answer quality/accuracy
- Answer length/detail
- Ability to handle complex questions

## Understanding the Differences

### Hardware Architecture

**SambaNova RDU (Reconfigurable Dataflow Unit)**:
- Dataflow architecture (vs control flow in GPUs)
- Optimized for inference workloads
- Lower precision (INT8/FP16) for speed
- Dedicated memory hierarchy

**NVIDIA A100 GPU**:
- CUDA-based parallel processing
- Designed for training (higher precision)
- General-purpose tensor cores
- Shared memory across workloads

### Why Metis Should Be Faster

1. **Dataflow Optimization**: RDUs execute operations as data arrives, reducing latency
2. **Lower Precision**: Inference-optimized with INT8 quantization
3. **No Context Switching**: Dedicated inference workload
4. **Optimized Memory**: Faster data movement for transformer models

### Possible Scenarios Where Sophia Might Compete

1. **Batch Processing**: GPUs excel at large batch sizes
2. **Complex Reasoning**: Higher precision might help edge cases
3. **Multi-task**: GPU handles diverse workloads simultaneously
4. **Memory-bound Tasks**: A100 has 80GB HBM2e

## Homework Deliverables

After running the benchmark, create:

1. **`Homework5_Analysis.md`** with:
   - Summary of findings
   - Performance comparison table
   - Analysis of why differences occur
   - Hardware architecture comparison
   - Use cases for each platform

2. **JSON results file** (auto-generated by script)

3. **Optional**: Screenshots from Web UI testing

## Troubleshooting

### Authentication Issues

```bash
# Re-authenticate if token expired
cd /home/clarexie/2025/advanced-ai-science-training-series/05-AITestbed
python inference_auth_token.py authenticate --force
```

### Model Not Found

If you get model errors, check available models:

```bash
access_token=$(python ../inference_auth_token.py get_access_token)
curl -X GET "https://inference-api.alcf.anl.gov/resource_server/list-endpoints" \
     -H "Authorization: Bearer ${access_token}" | jq -C '.'
```

### Connection Timeouts

If requests timeout:
- Reduce number of prompts in script
- Add longer delays between requests
- Check network connection

### Import Errors

Make sure you're in the correct directory:
```bash
cd /home/clarexie/2025/advanced-ai-science-training-series/05-AITestbed/SambaNova
```

The script looks for `inference_auth_token.py` in the parent directory.

## Advanced Options

### Use a Different HuggingFace Dataset

To use an actual HuggingFace dataset, modify the script:

```python
from datasets import load_dataset

# Load dataset
dataset = load_dataset("squad", split="validation[:100]")

# Extract prompts
prompts = [f"Answer this question: {item['question']}" 
           for item in dataset]
```

### Adjust Parameters

In the script, you can modify:
- `temperature`: Control randomness (0.0-1.0)
- `max_tokens`: Limit response length
- `num_runs`: Run multiple times per prompt for statistics

### Test Different Models

Try comparing:
- `gpt-oss-120b-131072` (Metis version with longer context)
- `Llama-4-Maverick-17B-128E-Instruct` (smaller model)

## Next Steps

1. Run the quick test (5 prompts)
2. Review the console output
3. Examine the JSON results file
4. Run the full test (15 prompts) if time permits
5. Write your analysis in `Homework5_Analysis.md`
6. Commit and push to GitHub

Good luck! 🚀
