import torch
import time

# Set device to CUDA
device = torch.device('cuda')
torch.manual_seed(0)

# Create synthetic data
src = torch.rand((2048, 1, 512))
tgt = torch.rand((2048, 20, 512))
dataset = torch.utils.data.TensorDataset(src, tgt)
loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# Create model
model = torch.nn.Transformer(batch_first=True)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()
model.train()
model = model.to(device)
criterion = criterion.to(device)

print("Starting single GPU training...")
start_t = time.time()

for epoch in range(10):
    for source, targets in loader:
        source = source.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()

        output = model(source, targets)
        loss = criterion(output, targets)

        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/10 completed")

print(f'Total training time: {time.time() - start_t:.2f}s')