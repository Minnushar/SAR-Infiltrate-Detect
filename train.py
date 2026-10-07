import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import SARPatchDataset
from src.autoencoder import SARAutoencoder


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)


# Load dataset
dataset = SARPatchDataset()

# DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True
)


# Create model
model = SARAutoencoder().to(device)

# Loss function
criterion = nn.MSELoss()

# Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# Training
epochs = 20

print("\nStarting training...\n")

for epoch in range(epochs):

    model.train()

    total_loss = 0

    for batch in dataloader:

        batch = batch.to(device)

        # Forward pass
        output = model(batch)

        # Reconstruction loss
        loss = criterion(output, batch)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(dataloader)

    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Loss: {average_loss:.6f}"
    )


# Save trained model
torch.save(
    model.state_dict(),
    "models/sar_autoencoder.pth"
)

print("\nTraining complete!")
print("Model saved to: models/sar_autoencoder.pth")