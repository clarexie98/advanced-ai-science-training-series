from mpi4py import MPI
import os, socket, time
import argparse
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import profile, ProfilerActivity
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Large Tensor Data Type Comparison")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--trace-dir", type=str, default="./traces/large_dtype/")
    args = parser.parse_args()
    return args

args = parse_args()

# DDP Setup
SIZE = MPI.COMM_WORLD.Get_size()
RANK = MPI.COMM_WORLD.Get_rank()
LOCAL_RANK = os.environ.get('PALS_LOCAL_RANKID')
os.environ['RANK'] = str(RANK)
os.environ['WORLD_SIZE'] = str(SIZE)
MASTER_ADDR = socket.gethostname() if RANK == 0 else None
MASTER_ADDR = MPI.COMM_WORLD.bcast(MASTER_ADDR, root=0)
os.environ['MASTER_ADDR'] = f"{MASTER_ADDR}.hsn.cm.polaris.alcf.anl.gov"
os.environ['MASTER_PORT'] = str(2345)

torch.distributed.init_process_group(backend='nccl', init_method='env://', 
                                     rank=int(RANK), world_size=int(SIZE))
torch.cuda.set_device(int(LOCAL_RANK))
device = torch.device('cuda')

# ============================================================================
# Test Configurations: Large 2nd and 3rd dimensions
# ============================================================================

configs = [
    # Format: (name, src_shape, tgt_shape, dtype)
    ("Baseline_FP32", (2048, 1, 512), (2048, 20, 512), torch.float32),
    
    # Large 3rd dimension (feature dimension)
    ("Large_Feature_2048_FP32", (2048, 1, 2048), (2048, 20, 2048), torch.float32),
    ("Large_Feature_2048_FP16", (2048, 1, 2048), (2048, 20, 2048), torch.float16),
    ("Large_Feature_2048_BF16", (2048, 1, 2048), (2048, 20, 2048), torch.bfloat16),
    
    ("Large_Feature_4096_FP32", (2048, 1, 4096), (2048, 20, 4096), torch.float32),
    ("Large_Feature_4096_FP16", (2048, 1, 4096), (2048, 20, 4096), torch.float16),
    ("Large_Feature_4096_BF16", (2048, 1, 4096), (2048, 20, 4096), torch.bfloat16),
    
    # Large 2nd dimension (sequence length)
    ("Large_Seq_50_FP32", (2048, 1, 512), (2048, 50, 512), torch.float32),
    ("Large_Seq_50_FP16", (2048, 1, 512), (2048, 50, 512), torch.float16),
    ("Large_Seq_50_BF16", (2048, 1, 512), (2048, 50, 512), torch.bfloat16),
    
    ("Large_Seq_100_FP32", (2048, 1, 512), (2048, 100, 512), torch.float32),
    ("Large_Seq_100_FP16", (2048, 1, 512), (2048, 100, 512), torch.float16),
    ("Large_Seq_100_BF16", (2048, 1, 512), (2048, 100, 512), torch.bfloat16),
    
    # Both large
    ("Large_Both_2048_FP32", (2048, 1, 2048), (2048, 50, 2048), torch.float32),
    ("Large_Both_2048_FP16", (2048, 1, 2048), (2048, 50, 2048), torch.float16),
    ("Large_Both_2048_BF16", (2048, 1, 2048), (2048, 50, 2048), torch.bfloat16),
]

results = []

for config_name, src_shape, tgt_shape, dtype in configs:
    if RANK == 0:
        print("\n" + "=" * 80)
        print(f"Configuration: {config_name}")
        print(f"  src shape: {src_shape}, tgt shape: {tgt_shape}")
        print(f"  dtype: {dtype}")
        print("=" * 80)
    
    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats(device)
    
    # Create tensors
    src = torch.rand(src_shape, dtype=dtype)
    tgt = torch.rand(tgt_shape, dtype=dtype)
    
    # Calculate memory footprint
    src_size_mb = src.element_size() * src.nelement() / (1024**2)
    tgt_size_mb = tgt.element_size() * tgt.nelement() / (1024**2)
    total_data_mb = src_size_mb + tgt_size_mb
    
    if RANK == 0:
        print(f"  Data size: src={src_size_mb:.2f} MB, tgt={tgt_size_mb:.2f} MB, total={total_data_mb:.2f} MB")
        print(f"  Element size: {src.element_size()} bytes")
    
    # Create dataset
    dataset = torch.utils.data.TensorDataset(src, tgt)
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, shuffle=True, num_replicas=SIZE, rank=RANK, seed=0)
    loader = torch.utils.data.DataLoader(dataset, sampler=sampler, 
                                         batch_size=args.batch_size)
    
    # Create model with matching dtype
    try:
        model = torch.nn.Transformer(batch_first=True)
        
        # For FP16/BF16, we need mixed precision training
        if dtype in [torch.float16, torch.bfloat16]:
            scaler = torch.cuda.amp.GradScaler(enabled=(dtype == torch.float16))
        
        optimizer = torch.optim.Adam(model.parameters(), lr=(0.001*SIZE))
        criterion = torch.nn.CrossEntropyLoss()
        model.train()
        model = model.to(device)
        
        # Keep criterion in FP32 for numerical stability
        criterion = criterion.to(device)
        model = DDP(model)
        
        # Profile
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        schedule = torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=1)
        prof = profile(activities=activities, record_shapes=True, 
                       schedule=schedule, profile_memory=True)
        prof.start()
        
        # Training
        start_t = time.time()
        successful_batches = 0
        
        for epoch in range(args.epochs):
            sampler.set_epoch(epoch)
            
            for source, targets in loader:
                source = source.to(device)
                targets = targets.to(device)
                
                optimizer.zero_grad()
                
                # Mixed precision training for FP16/BF16
                if dtype in [torch.float16, torch.bfloat16]:
                    with torch.cuda.amp.autocast(dtype=dtype):
                        output = model(source, targets)
                        # Convert to FP32 for loss calculation
                        loss = criterion(output.float(), targets.float())
                    
                    if dtype == torch.float16:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()
                else:
                    output = model(source, targets)
                    loss = criterion(output, targets)
                    loss.backward()
                    optimizer.step()
                
                successful_batches += 1
                prof.step()
        
        elapsed = time.time() - start_t
        
        # Memory stats
        memory_allocated = torch.cuda.max_memory_allocated(device) / (1024**3)
        memory_reserved = torch.cuda.max_memory_reserved(device) / (1024**3)
        
        if RANK == 0:
            throughput = successful_batches / elapsed
            print(f"  Training time: {elapsed:.2f}s")
            print(f"  Throughput: {throughput:.2f} batches/sec")
            print(f"  Peak GPU Memory - Allocated: {memory_allocated:.2f} GB, Reserved: {memory_reserved:.2f} GB")
        
        # Save profile
        os.makedirs(args.trace_dir, exist_ok=True)
        prof.export_chrome_trace(f"{args.trace_dir}/{config_name}-{RANK}-of-{SIZE}.json")
        
        results.append({
            'name': config_name,
            'dtype': str(dtype).split('.')[-1],
            'src_shape': src_shape,
            'tgt_shape': tgt_shape,
            'data_size_mb': total_data_mb,
            'time': elapsed,
            'throughput': successful_batches / elapsed,
            'memory_gb': memory_allocated,
            'batches': successful_batches
        })
        
    except RuntimeError as e:
        if RANK == 0:
            print(f"  ERROR: {str(e)}")
            print(f"  Configuration too large for available GPU memory")
        results.append({
            'name': config_name,
            'dtype': str(dtype).split('.')[-1],
            'src_shape': src_shape,
            'tgt_shape': tgt_shape,
            'data_size_mb': total_data_mb,
            'time': -1,
            'throughput': -1,
            'memory_gb': -1,
            'batches': 0,
            'error': 'OOM'
        })
    
    # Cleanup
    torch.cuda.empty_cache()
    del model, optimizer, criterion, dataset, loader, src, tgt
    torch.distributed.barrier(device_ids=[torch.cuda.current_device()])

# ============================================================================
# Summary
# ============================================================================

if RANK == 0:
    print("\n" + "=" * 100)
    print("SUMMARY: Large Tensor Data Type Comparison")
    print("=" * 100)
    print(f"{'Configuration':<30} {'DType':<8} {'Data(MB)':<12} {'Time(s)':<10} {'Throughput':<12} {'Memory(GB)':<12}")
    print("-" * 100)
    
    for r in results:
        if 'error' in r:
            print(f"{r['name']:<30} {r['dtype']:<8} {r['data_size_mb']:<12.1f} {'OOM':<10} {'-':<12} {'-':<12}")
        else:
            print(f"{r['name']:<30} {r['dtype']:<8} {r['data_size_mb']:<12.1f} "
                  f"{r['time']:<10.2f} {r['throughput']:<12.2f} {r['memory_gb']:<12.2f}")
    
    print("=" * 100)
    
    # Insights
    print("\nKey Insights:")
    print("1. Memory savings with FP16/BF16 vs FP32")
    print("2. Impact of sequence length (2nd dimension) on memory and speed")
    print("3. Impact of feature dimension (3rd dimension) on memory and speed")
    print("4. Trade-offs between precision and performance")

torch.distributed.destroy_process_group()