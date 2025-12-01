# Homework 5 Analysis: Metis vs Sophia Performance Comparison

**Student**: Clare Xie  
**Date**: December 1, 2025  
**Course**: AI Science Training Series  

---

## Executive Summary

This analysis compares the inference performance of the GPT-OSS-120B model running on two distinct hardware platforms at ALCF:
- **Metis**: SambaNova SN40L with RDU (Reconfigurable Dataflow Unit) processors
- **Sophia**: GPU cluster with NVIDIA A100 accelerators

[Fill in after running experiments: Brief 2-3 sentence summary of key findings]

---

## Experimental Setup

### Dataset
- **Source**: Scientific Q&A prompts covering physics, biology, chemistry, and machine learning
- **Size**: [5 or 15] prompts tested
- **Domains**: Quantum mechanics, photosynthesis, machine learning, genetics, thermodynamics

### Model Configuration
- **Model**: `openai/gpt-oss-120b` (120 billion parameters)
- **Temperature**: 0.7
- **Max Tokens**: 512
- **Runs per Prompt**: 1

### Testing Methodology
1. Sequential testing to avoid load interference
2. Same prompts used for both platforms
3. Identical model parameters
4. Automated latency measurement
5. Token usage tracking

---

## Results

### Performance Metrics

| Metric | Metis (SambaNova) | Sophia (GPU) | Difference |
|--------|-------------------|--------------|------------|
| **Mean Latency (s)** | [FILL IN] | [FILL IN] | [CALCULATE %] |
| **Min Latency (s)** | [FILL IN] | [FILL IN] | [CALCULATE %] |
| **Max Latency (s)** | [FILL IN] | [FILL IN] | [CALCULATE %] |
| **Median Latency (s)** | [FILL IN] | [FILL IN] | [CALCULATE %] |
| **Throughput (tokens/s)** | [FILL IN] | [FILL IN] | [CALCULATE %] |
| **Avg Tokens per Request** | [FILL IN] | [FILL IN] | [CALCULATE %] |
| **Success Rate** | [FILL IN] | [FILL IN] | - |
| **Speedup Factor** | - | - | [X]x |

### Detailed Observations

#### Latency Distribution
[After running experiments, describe the distribution]
- Consistency of response times
- Presence of outliers
- Variance between requests

#### Throughput Analysis
[Analyze tokens per second]
- How does it vary across prompts?
- Impact of prompt complexity
- Relationship to response length

#### Response Quality
[Manual assessment of responses]
- Accuracy of scientific information
- Completeness of answers
- Coherence and clarity
- Any differences in response style

---

## Analysis of Performance Differences

### Hardware Architecture Comparison

#### SambaNova RDU (Metis)
**Architecture**:
- Reconfigurable Dataflow Unit processors
- Dataflow execution model (operations execute as data arrives)
- 16 RDU processors per SambaRack (32 total)
- Purpose-built for inference workloads

**Advantages**:
1. **Dataflow Optimization**: Eliminates control flow overhead
2. **Lower Latency**: Optimized for single-request throughput
3. **Efficient Memory Hierarchy**: Designed for transformer models
4. **Lower Precision Inference**: INT8/FP16 quantization for speed
5. **Deterministic Performance**: Dedicated inference resources

**Trade-offs**:
- Specialized for inference (not training)
- Fixed model formats
- Less flexible for arbitrary workloads

#### NVIDIA A100 GPU (Sophia)
**Architecture**:
- CUDA-based parallel processing
- 80GB HBM2e memory
- 3rd generation Tensor Cores
- General-purpose architecture

**Advantages**:
1. **Versatility**: Handles training and inference
2. **High Precision**: FP64/FP32 available for accuracy
3. **Large Memory**: 80GB for massive models/batches
4. **Mature Ecosystem**: Extensive CUDA libraries
5. **Batch Efficiency**: Excellent for large batch sizes

**Trade-offs**:
- Higher latency for single requests
- Power consumption higher
- Shared resources across workloads

### Why the Performance Difference?

#### Expected: Metis Faster Than Sophia

**Reasons**:
1. **Inference Optimization**: RDUs designed specifically for model serving
2. **Dataflow vs Control Flow**: Eliminates instruction fetch/decode overhead
3. **Quantization**: Lower precision (INT8) runs faster without quality loss
4. **Memory Bandwidth**: Optimized data movement for transformers
5. **No Training Overhead**: Dedicated inference path

#### Potential Scenarios Where Sophia Competes

1. **Large Batch Processing**: A100s excel with batch size > 32
2. **Memory-Bound Models**: 80GB HBM2e vs RDU memory constraints
3. **High Precision Needs**: Scientific computing requiring FP64
4. **Dynamic Workloads**: Mixed training/inference/preprocessing

### Quantitative Analysis

[After running experiments, fill in]

**Speedup Calculation**:
```
Speedup = Sophia_Latency / Metis_Latency
        = [FILL IN] / [FILL IN]
        = [X.XX]x
```

**Throughput Improvement**:
```
Throughput_Improvement = (Metis_Throughput - Sophia_Throughput) / Sophia_Throughput × 100%
                       = ([FILL IN] - [FILL IN]) / [FILL IN] × 100%
                       = [X.X]%
```

**Efficiency Metrics**:
- **Latency Reduction**: [X]% faster on Metis
- **Throughput Gain**: [X]% more tokens/sec on Metis
- **Consistency**: [Compare standard deviations]

---

## Use Case Recommendations

### When to Use Metis (SambaNova)

**Ideal Scenarios**:
1. ✅ **Real-time Inference**: Chatbots, interactive applications
2. ✅ **Single-Request Throughput**: User-facing services
3. ✅ **Latency-Sensitive**: Applications where milliseconds matter
4. ✅ **Production Serving**: Stable, predictable inference workloads
5. ✅ **Energy Efficiency**: Lower power consumption important

**Example Applications**:
- Scientific Q&A systems
- Code completion services
- Medical diagnosis assistants
- Real-time translation
- Interactive AI tutors

### When to Use Sophia (GPU Cluster)

**Ideal Scenarios**:
1. ✅ **Large Batch Processing**: Offline processing of datasets
2. ✅ **Model Fine-tuning**: Need training capabilities
3. ✅ **Research/Experimentation**: Flexibility for diverse tasks
4. ✅ **High Precision Needed**: FP32/FP64 for scientific accuracy
5. ✅ **Memory-Intensive Models**: Models > 100B parameters

**Example Applications**:
- Dataset annotation at scale
- Model training and fine-tuning
- Scientific simulations
- Batch data processing
- Multi-task inference pipelines

---

## Conclusions

### Key Findings

1. [Summarize main performance difference]
2. [Hardware architecture impact]
3. [Use case implications]

### Lessons Learned

1. **Hardware Specialization Matters**: Purpose-built inference accelerators provide measurable advantages
2. **Trade-offs Exist**: Flexibility vs optimization
3. **Workload Matching**: Choose hardware based on specific needs

### Future Work

1. Test with different batch sizes (1, 8, 32, 64)
2. Compare power consumption and efficiency
3. Evaluate on different model architectures (CNN, ViT, etc.)
4. Assess cost-per-inference metrics
5. Test with longer context lengths

---

## Appendix

### A. Sample Responses

[Include 2-3 example responses from each platform for qualitative comparison]

**Example 1: Quantum Entanglement**

*Metis Response*:
```
[Paste response]
```

*Sophia Response*:
```
[Paste response]
```

*Comparison*: [Brief notes on differences]

---

### B. Raw Performance Data

[Link to JSON results file]

File: `metis_vs_sophia_results_YYYYMMDD_HHMMSS.json`

---

### C. References

1. SambaNova SN40L Documentation: https://docs.sambanova.ai/
2. NVIDIA A100 Architecture: https://www.nvidia.com/en-us/data-center/a100/
3. ALCF AI Testbed: https://www.alcf.anl.gov/alcf-ai-testbed
4. Transformer Inference Optimization: [Relevant papers]

---

## Acknowledgments

This work was conducted as part of the ALCF AI Science Training Series. Thanks to the ALCF team for providing access to both Metis and Sophia platforms.
