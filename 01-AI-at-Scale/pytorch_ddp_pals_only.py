import os
import socket
import time
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

# DDP: Rank counting using PALS only
SIZE = int(os.environ.get('PMI_SIZE', 1))  # Total number of ranks
RANK = int(os.environ.get('PMI_RANK', 0))  # Global rank
LOCAL_RANK = int(os.environ.get('PALS_LOCAL_RANKID', 0))  # Local rank on node

os.environ['RANK'] = str(RANK)
os.environ['WORLD_SIZE'] = str(SIZE)

# Get master address
if RANK == 0:
    MASTER_ADDR = socket.gethostname()
    # Broadcast via environment (in real scenario, use file or other mechanism)
    os.environ['MASTER_ADDR'] = f"{MASTER_ADDR}.hsn.cm.polaris.alcf.anl.gov"
else:
    # Worker ranks need to know master address - this is simplified
    # In production, use a coordination mechanism
    pass

os.environ['MASTER_PORT'] = str(2345)
print(f"PALS: Hi from rank {RANK} of {SIZE} with local rank {LOCAL_RANK}")

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