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

# Create and save data in .pt format
src = torch.rand((2048, 1, 512))
tgt = torch.rand((2048, 20, 512))

if RANK == 0:
    torch.save({'src': src, 'tgt': tgt}, 'tensor_dataset.pt')
    print(".pt dataset created successfully!")

torch.distributed.barrier(device_ids=[torch.cuda.current_device()])

# Custom Dataset for .pt files
class PTTensorDataset(torch.utils.data.Dataset):
    def __init__(self, pt_file_path):
        self.data = torch.load(pt_file_path)
        self.src = self.data['src']
        self.tgt = self.data['tgt']
    
    def __len__(self):
        return len(self.src)
    
    def __getitem__(self, idx):
        return self.src[idx], self.tgt[idx]

dataset = PTTensorDataset('tensor_dataset.pt')
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
    print(f'Total training time (.pt format): {time.time() - start_t:.2f}s')

torch.distributed.destroy_process_group()