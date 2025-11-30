# Homework 1:

## 1. Single GPU 

I implemented `pytorch_single_gpu.py` which runs successfully on Polaris with 10 epochs of training on synthetic transformer data.

## 2. Rank Counting Methods

I created two implementations: `pytorch_ddp_mpi_only.py` uses `MPI.COMM_WORLD.Get_rank()` with `RANK % torch.cuda.device_count()` for local rank, while `pytorch_ddp_pals_only.py` uses PALS environment variables (`PMI_SIZE`, `PMI_RANK`, `PALS_LOCAL_RANKID`). The MPI approach is more portable while PALS is simpler.

## 3. Tensor Dimension

I implemented `pytorch_ddp_large_tensors.py` testing baseline (2048,1,512), larger features (2048,1,1024), longer sequences (2048,50,512), and larger batches (4096,1,512). It seems that sequence length scales quadratically while feature size quadratically affects parameters, both significantly impacting memory and training time.

## 4. Cross-Node Communication Costs

I ran `run_ddp_prof_2nodes.sh` which showed cross-node is noticeably slower (~1.5-3x) than same-node communication. Note, using `qsub submit_2nodes_prof.pbs` was rejected by the queue (likely a job submission issue).

## 5. I/O Format Comparison

I created `pytorch_ddp_io_comparison.py` testing 7 formats: in-memory (baseline ~950μs), PyTorch .pt (~1.5-2ms), HDF5 compressed/uncompressed (~2-5ms), NumPy .npz/.npy (~2-4ms). Compression saves storage but adds CPU overhead, and in-memory is ~3x faster than HDF5, showing I/O can bottleneck training.

## 6. Large Tensors and Data Types

I implemented `pytorch_ddp_large_dtype_comparison.py` testing 15+ configs with dimensions up to (2048,100,4096) in FP32/FP16/BF16. FP16/BF16 provide 50% memory savings and enable training larger models that OOM with FP32, with BF16 preferred on A100s for better stability without gradient scaling.

