from mpi4py import MPI
import os, socket, time
import argparse
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import h5py
import numpy as np
from torch.profiler import profile, ProfilerActivity
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="I/O Format Comparison")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--trace-dir", type=str, default="./traces/io_comparison/")
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
torch.manual_seed(0)

# Generate data once
src = torch.rand((2048, 1, 512))
tgt = torch.rand((2048, 20, 512))

# ============================================================================
# Dataset Classes for Different Formats
# ============================================================================

class MemoryDataset(torch.utils.data.Dataset):
    """Baseline: In-memory tensors"""
    def __init__(self, src, tgt):
        self.src = src
        self.tgt = tgt
    
    def __len__(self):
        return len(self.src)
    
    def __getitem__(self, idx):
        return self.src[idx], self.tgt[idx]

class HDF5Dataset(torch.utils.data.Dataset):
    """HDF5 file format"""
    def __init__(self, filepath):
        self.filepath = filepath
        self.hdf5_file = h5py.File(self.filepath, "r")
        self.src = self.hdf5_file["src"]
        self.tgt = self.hdf5_file["tgt"]
    
    def __len__(self):
        return len(self.src)
    
    def __getitem__(self, idx):
        return torch.tensor(self.src[idx], dtype=torch.float32), \
               torch.tensor(self.tgt[idx], dtype=torch.float32)

class PTDataset(torch.utils.data.Dataset):
    """PyTorch .pt file format"""
    def __init__(self, filepath):
        self.data = torch.load(filepath)
        self.src = self.data['src']
        self.tgt = self.data['tgt']
    
    def __len__(self):
        return len(self.src)
    
    def __getitem__(self, idx):
        return self.src[idx], self.tgt[idx]

class NPZDataset(torch.utils.data.Dataset):
    """NumPy .npz file format"""
    def __init__(self, filepath):
        data = np.load(filepath)
        self.src = torch.from_numpy(data['src']).float()
        self.tgt = torch.from_numpy(data['tgt']).float()
    
    def __len__(self):
        return len(self.src)
    
    def __getitem__(self, idx):
        return self.src[idx], self.tgt[idx]

class NumpyDataset(torch.utils.data.Dataset):
    """Individual .npy files"""
    def __init__(self, src_file, tgt_file):
        self.src = torch.from_numpy(np.load(src_file)).float()
        self.tgt = torch.from_numpy(np.load(tgt_file)).float()
    
    def __len__(self):
        return len(self.src)
    
    def __getitem__(self, idx):
        return self.src[idx], self.tgt[idx]

# ============================================================================
# Create Files (Rank 0 only)
# ============================================================================

if RANK == 0:
    print("=" * 80)
    print("Creating dataset files in different formats...")
    print("=" * 80)
    
    # 1. HDF5
    start = time.time()
    with h5py.File("tensor_dataset.h5", "w") as f:
        f.create_dataset("src", data=src.numpy(), compression="gzip")
        f.create_dataset("tgt", data=tgt.numpy(), compression="gzip")
    print(f"HDF5 (compressed): {time.time() - start:.3f}s")
    
    # 2. HDF5 uncompressed
    start = time.time()
    with h5py.File("tensor_dataset_raw.h5", "w") as f:
        f.create_dataset("src", data=src.numpy())
        f.create_dataset("tgt", data=tgt.numpy())
    print(f"HDF5 (uncompressed): {time.time() - start:.3f}s")
    
    # 3. PyTorch .pt
    start = time.time()
    torch.save({'src': src, 'tgt': tgt}, 'tensor_dataset.pt')
    print(f"PyTorch .pt: {time.time() - start:.3f}s")
    
    # 4. NumPy .npz (compressed)
    start = time.time()
    np.savez_compressed('tensor_dataset.npz', src=src.numpy(), tgt=tgt.numpy())
    print(f"NumPy .npz (compressed): {time.time() - start:.3f}s")
    
    # 5. NumPy .npz (uncompressed)
    start = time.time()
    np.savez('tensor_dataset_raw.npz', src=src.numpy(), tgt=tgt.numpy())
    print(f"NumPy .npz (uncompressed): {time.time() - start:.3f}s")
    
    # 6. Individual .npy files
    start = time.time()
    np.save('tensor_src.npy', src.numpy())
    np.save('tensor_tgt.npy', tgt.numpy())
    print(f"Individual .npy files: {time.time() - start:.3f}s")
    
    # File sizes
    print("\nFile sizes:")
    for fname in ['tensor_dataset.h5', 'tensor_dataset_raw.h5', 'tensor_dataset.pt',
                  'tensor_dataset.npz', 'tensor_dataset_raw.npz']:
        size_mb = os.path.getsize(fname) / (1024**2)
        print(f"  {fname}: {size_mb:.2f} MB")
    src_size = os.path.getsize('tensor_src.npy') / (1024**2)
    tgt_size = os.path.getsize('tensor_tgt.npy') / (1024**2)
    print(f"  tensor_src.npy + tensor_tgt.npy: {src_size + tgt_size:.2f} MB")
    print()

torch.distributed.barrier(device_ids=[torch.cuda.current_device()])

# ============================================================================
# Test Each Format
# ============================================================================

formats = [
    ("In-Memory", lambda: MemoryDataset(src, tgt)),
    ("HDF5 (compressed)", lambda: HDF5Dataset("tensor_dataset.h5")),
    ("HDF5 (uncompressed)", lambda: HDF5Dataset("tensor_dataset_raw.h5")),
    ("PyTorch .pt", lambda: PTDataset("tensor_dataset.pt")),
    ("NumPy .npz (compressed)", lambda: NPZDataset("tensor_dataset.npz")),
    ("NumPy .npz (uncompressed)", lambda: NPZDataset("tensor_dataset_raw.npz")),
    ("NumPy .npy files", lambda: NumpyDataset("tensor_src.npy", "tensor_tgt.npy")),
]

results = []

for format_name, dataset_fn in formats:
    if RANK == 0:
        print("=" * 80)
        print(f"Testing: {format_name}")
        print("=" * 80)
    
    # Create dataset
    dataset = dataset_fn()
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, shuffle=True, num_replicas=SIZE, rank=RANK, seed=0)
    loader = torch.utils.data.DataLoader(dataset, sampler=sampler, 
                                         batch_size=args.batch_size)
    
    # Create fresh model for each test
    model = torch.nn.Transformer(batch_first=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=(0.001*SIZE))
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    model = model.to(device)
    criterion = criterion.to(device)
    model = DDP(model)
    
    # Profile
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    schedule = torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=1)
    prof = profile(activities=activities, record_shapes=True, 
                   schedule=schedule, profile_memory=True)
    prof.start()
    
    # Training
    data_load_times = []
    start_t = time.time()
    
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        epoch_data_time = 0
        
        for source, targets in loader:
            data_start = time.time()
            source = source.to(device)
            targets = targets.to(device)
            epoch_data_time += time.time() - data_start
            
            optimizer.zero_grad()
            output = model(source, targets)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            prof.step()
        
        data_load_times.append(epoch_data_time)
    
    total_time = time.time() - start_t
    avg_data_time = np.mean(data_load_times)
    
    if RANK == 0:
        print(f"Total training time: {total_time:.2f}s")
        print(f"Average data loading time per epoch: {avg_data_time:.3f}s")
        print(f"Compute time (estimated): {total_time - sum(data_load_times):.2f}s")
    
    # Save profiling
    os.makedirs(args.trace_dir, exist_ok=True)
    safe_name = format_name.replace(" ", "_").replace("(", "").replace(")", "").replace(".", "")
    prof.export_chrome_trace(f"{args.trace_dir}/{safe_name}-{RANK}-of-{SIZE}.json")
    
    results.append({
        'format': format_name,
        'total_time': total_time,
        'data_time': sum(data_load_times),
        'avg_data_time': avg_data_time
    })
    
    # Cleanup
    del model, optimizer, criterion, dataset, loader
    torch.cuda.empty_cache()
    torch.distributed.barrier(device_ids=[torch.cuda.current_device()])

# ============================================================================
# Summary
# ============================================================================

if RANK == 0:
    print("\n" + "=" * 80)
    print("SUMMARY: I/O Format Comparison")
    print("=" * 80)
    print(f"{'Format':<30} {'Total Time':<15} {'Data Load Time':<20} {'Avg/Epoch':<15}")
    print("-" * 80)
    for r in results:
        print(f"{r['format']:<30} {r['total_time']:<15.2f} {r['data_time']:<20.2f} {r['avg_data_time']:<15.3f}")
    print("=" * 80)

torch.distributed.destroy_process_group()