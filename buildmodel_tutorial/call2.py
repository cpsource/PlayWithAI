import torch
import torch.utils.data as data

# Create a dataset
dataset = data.Dataset(...)

# Create a data loader
train_loader = data.DataLoader(dataset, batch_size=128, shuffle=True)

# Iterate over the data
for i, (data, target) in enumerate(train_loader):
    x = data.to(device)
    y = target.to(device)

    # Forward pass
    logits = model(x)

    # Compute loss
    loss = criterion(logits, y)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
