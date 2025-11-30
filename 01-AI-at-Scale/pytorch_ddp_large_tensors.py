from mpi4py import MPI
import os, socket, time
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

# Setup DDP
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

# Experiment with different dimensions
configs = [
    {"name": "baseline", "src": (2048, 1, 512), "tgt": (2048, 20, 512)},
    {"name": "larger_seq", "src": (2048, 1, 1024), "tgt": (2048, 20, 1024)},
    {"name": "longer_tgt", "src": (2048, 1, 512), "tgt": (2048, 50, 512)},
    {"name": "larger_batch", "src": (4096, 1, 512), "tgt": (4096, 20, 512)},
]

for config in configs:
    if RANK == 0:
        print(f"\n{'='*60}")
        print(f"Testing configuration: {config['name']}")
        print(f"src shape: {config['src']}, tgt shape: {config['tgt']}")
        print(f"{'='*60}\n")
    
    src = torch.rand(config['src'])
    tgt = torch.rand(config['tgt'])
    dataset = torch.utils.data.TensorDataset(src, tgt)
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, shuffle=True, num_replicas=SIZE, rank=RANK, seed=0)
    loader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=32)
    
    model = torch.nn.Transformer(batch_first=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=(0.001*SIZE))
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    model = model.to(device)
    criterion = criterion.to(device)
    model = DDP(model)
    
    start_t = time.time()
    for epoch in range(5):  # Reduced epochs for testing
        sampler.set_epoch(epoch)
        for source, targets in loader:
            source = source.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            output = model(source, targets)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
    
    if RANK == 0:
        print(f'{config["name"]} - Training time: {time.time() - start_t:.2f}s')
        # Calculate memory usage
        memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
        memory_reserved = torch.cuda.memory_reserved(device) / 1024**3
        print(f'GPU Memory - Allocated: {memory_allocated:.2f} GB, Reserved: {memory_reserved:.2f} GB\n')
    
    torch.cuda.empty_cache()

torch.distributed.destroy_process_group()