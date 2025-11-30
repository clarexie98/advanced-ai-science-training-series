# Homework 2: Tensor Parallelism Analysis

For this assignment, I explored how tensor parallelism affects training performance by running three experiments with an 8-layer transformer model on Polaris. I started with a baseline configuration using TP=1. Then I increased the tensor parallelism to TP=2, splitting the model across pairs of GPUs, and finally TP=4, where all four GPUs collaborated on the same model split into four pieces. Each run used the ezpz framework's FSDP with tensor parallelism support, and I collected timing and performance metrics to compare how the different configurations performed.

## Results

I found TP=4 performed the best, completing training in 64.79 seconds compared to 76.74 seconds for the baseline TP=1 configuration. That's about an 18% speedup, which I wasn't expecting since tensor parallelism adds communication between GPUs. TP=2 fell in between at 75.37 seconds, showing only a marginal 2% improvement. Looking at the per-iteration timings, TP=4 averaged 0.119 seconds per iteration compared to 0.121 seconds for TP=1. The forward pass times were essentially the same across all three configurations (around 0.047-0.048 seconds), but the backward pass showed the most variation, with TP=2 being fastest at 0.071 seconds and TP=1 slowest at 0.073 seconds. Memory usage stayed pretty consistent around 2.1-2.2GB per GPU across all runs, which makes sense since the model is small enough to fit comfortably on each A100.

The device mesh configurations were interesting to observe. With TP=1, I had four independent data parallel replicas `[[0], [1], [2], [3]]`, where each GPU maintained the full model and processed different batches. TP=2 created two tensor parallel groups `[[0, 1], [2, 3]]`, so GPUs 0 and 1 shared one model instance while GPUs 2 and 3 shared another. With TP=4, all GPUs collaborated on a single model instance `[[0, 1, 2, 3]]`, with no data parallelism at all.

## Thoughts

I noticed that the speedup isn't linear—we're getting 18% faster with 4 GPUs doing tensor parallelism instead of data parallelism, but it's not 4x faster or anything close. This tells me we're starting to hit the point where communication overhead matters. For this relatively small 22M parameter model, the compute-to-communication ratio isn't huge, so we're not seeing massive gains. I'd expect tensor parallelism to do better with much larger models where the computation per layer is substantial enough that the communication becomes negligible in comparison.

I think for relatively small models like this one, tensor parallelism is more about performance optimization than memory necessity. All three configurations used roughly the same amount of memory per GPU because the model easily fits on each GPU.

---

**Experiment outputs:**
- TP=1: `outputs/ezpz-fsdp-tp/2025-11-30-075954/`
- TP=2: `outputs/ezpz-fsdp-tp/2025-11-30-080129/`
- TP=4: `outputs/ezpz-fsdp-tp/2025-11-30-080310/`
