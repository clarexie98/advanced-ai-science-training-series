# Homework 5: 

## Task 1: Cerebras 

Run the Llama-7B example for different batch sizes and compare the performance.

**Issue:** Unable to complete due to access user node

I cannot access the Cerebras user nodes (cer-usn-01 or cer-usn-02) from the Cerebras login node. I get "Permission denied (hostbased)" errors when trying to SSH to the user nodes. I attempted to debug and run the homework on the login node, but that didn't work since the CS-3 system needs to be accessed from the user nodes.

I contacted Paige Kinsley and Murali Emani about this issue. Unfortunately, as of the homework deadline (I requested an extention till Dec 5), this access issue remains unresolved.

---

## Task 2: SambaNova 

Use your choice of huggingface dataset and compare the performance on GptOSS model using both Metis and Sophia, reason out the possible differences.

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

### Thoughts

The results were initially surprising since SambaNova uses inference-optimized hardware, yet Sophia outperformed Metis by 3×. I think Sophia's better performance could be due to workload mismatch. My benchmark tested single requests with short answers (37-179 characters). These low-latency individual requests using tensor cores are what GPUs are optimized for. SambaNova's dataflow architecture, on the other hand, is designed for high-throughput batch processing.

The benchmark exercise shows that hardware performance depends heavily on the workload. Sophia did better in this test because I tested exactly what GPUs are optimized for. A fair comparison would need to test batching and long-context scenarios. This is similar to how the DragonHPC + DDict approach in Homework 3 only showed its advantages at large data sizes.


## Files
- `setup_env.sh` - Installs the HuggingFace datasets package
- `compare_metis_sophia.py` - Benchmark script that loads SQuAD and tests both platforms
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



