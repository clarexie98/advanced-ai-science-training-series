# Homework 1:

## 1. Single GPU 

I implemented `pytorch_single_gpu.py` which runs successfully on Polaris, completing 10 epochs of training in 16.38s on synthetic transformer data.

## 2. Rank Counting Methods

I created `pytorch_ddp_mpi_only.py` using `MPI.COMM_WORLD.Get_rank()` with `RANK % torch.cuda.device_count()` for local rank. Tested with 4 ranks (0-3 with local ranks 0-3), completed training in 4.50s. The MPI approach is portable while PALS is simpler but system-specific.

## 3. Tensor Dimensions

I implemented `pytorch_ddp_large_tensors.py` testing baseline (2048,1,512) which completed in 2.37s with 0.86 GB allocated memory. Attempted larger feature dimension (2048,1,1024) but failed with "feature number must equal d_model" error because the default PyTorch Transformer model requires feature dimensions to match its d_model parameter (512 by default).

## 4. Cross-Node Communication Costs

I attempted to test cross-node communication by creating `submit_2nodes_prof.pbs` and `run_ddp_prof_2nodes.sh`, but `qsub` was rejected by the queue. Single-rank profiling completed in 20.68s, generating traces in `./traces/pytorch_2p8/cuda_pt_2p8_1nodes_E10_R1_2025-11-30-073354`.

## 5. I/O Format Comparison

I created `pytorch_ddp_io_comparison.py` testing 7 formats. Results: in-memory (3.08s), HDF5 compressed (59.90s - extremely slow), HDF5 uncompressed (2.89s), PyTorch .pt (2.84s), NumPy .npz compressed (2.83s), NumPy .npz uncompressed (2.91s), NumPy .npy files (2.83s). Compression saves ~10% storage (75MB vs 84MB) but HDF5 compressed is 20x slower due to decompression overhead.

## 6. Large Tensors and Data Types

I implemented `pytorch_ddp_large_dtype_comparison.py` testing sequence length variations. Results: Baseline FP32 (3.58s, 1.02GB memory), Large_Seq_50 FP32 (3.36s, 1.19GB), FP16 (4.07s, 1.19GB), BF16 (3.90s, 1.07GB), Large_Seq_100 FP32 (3.44s, 1.32GB), FP16 (3.97s, 1.21GB), BF16 (3.91s, 1.05GB). BF16 consistently used less memory than FP32. Feature dimension tests (2048, 4096) all failed with d_model errors.

