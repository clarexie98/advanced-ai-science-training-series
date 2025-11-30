from mpi4py import MPI
import os
import socket
import time
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

# DDP: Rank counting using mpi4py only
SIZE = MPI.COMM_WORLD.Get_size()
RANK = MPI.COMM_WORLD.Get_rank()

# Calculate local rank using device_count
local_size = torch.cuda.device_count()
LOCAL_RANK = RANK % local_size

os.environ['RANK'] = str(RANK)
os.environ['WORLD_SIZE'] = str(SIZE)
os.environ['LOCAL_RANK'] = str(LOCAL_RANK)

# Get master address via MPI
MASTER_ADDR = socket.gethostname() if RANK == 0 else None
MASTER_ADDR = MPI.COMM_WORLD.bcast(MASTER_ADDR, root=0)
os.environ['MASTER_ADDR'] = f"{MASTER_ADDR}.hsn.cm.polaris.alcf.anl.gov"
os.environ['MASTER_PORT'] = str(2345)

print(f"MPI: Hi from rank {RANK} of {SIZE} with local rank {LOCAL_RANK}")

# Initialize process group
torch.distributed.init_process_group(backend='nccl', init_method='env://', 
                                     rank=RANK, world_size=SIZE)

# Pin GPU to local rank
torch.cuda.set_device(LOCAL_RANK)
device = torch.device('cuda')
torch.manual_seed(0)

# Create dataset
src = torch.rand((2048, 1, 512))
tgt = torch.rand((2048, 20, 512))
dataset = torch.utils.data.TensorDataset(src, tgt)

sampler = torch.utils.data.distributed.DistributedSampler(
    dataset, shuffle=True, num_replicas=SIZE, rank=RANK, seed=0)
loader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=32)

# Create and wrap model
model = torch.nn.Transformer(batch_first=True)
optimizer = torch.optim.Adam(model.parameters(), lr=(0.001*SIZE))
criterion = torch.nn.CrossEntropyLoss()
model.train()
model = model.to(device)
criterion = criterion.to(device)
model = DDP(model)

start_t = time.time()
for epoch in range(10):
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
    print(f'Total training time: {time.time() - start_t:.2f}s')

torch.distributed.destroy_process_group()