# === PyTorch Implementation ===

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR

# Reproducibility
SEED = 1
rng = np.random.default_rng(SEED)

# Number of observations
n = 1000  

# Ground truth Cobb–Douglas parameter (capital share)
alpha = 0.7  

# Source of randomness
# Productivity factor A is drawn from a Beta(8, 2) distribution.
# Its mean is E[A] = a / (a + b) = 0.8.
a, b = 8.0, 2.0

# Inputs: capital and labor
# Capital (K) and labor (L) endowments are sampled uniformly on [0, 1].
K = rng.uniform(low=0, high=1.0, size=n)
L = rng.uniform(low=0, high=1.0, size=n)

# Stack regressors into a single matrix X = [K, L].
X = np.column_stack([K, L]) 

# Productivity shocks
# One independent draw A_i ~ Beta(8, 2) per observation.
A = rng.beta(a, b, size=n)

# Output of the representative firm
# Cobb–Douglas production function:
# Y_i = A_i * K_i^alpha * L_i^(1 - alpha)
Y = (A * (K**alpha) * (L**(1.0 - alpha)))[:, None]

# Define input size, hidden size, output size
n_input, n_hidden, n_output = X.shape[1], 10, Y.shape[1]

# Create model with two hidden layers
class NeuralNetwork(nn.Module):
    def __init__(self):
        # Define the network architecture
        super(NeuralNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden), 
            nn.ReLU(),
            nn.Linear(n_hidden, n_output),
            nn.Sigmoid()
        )
        # Apply He initialization
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
    # Define forward pass
    def forward(self, x):
        return self.net(x)

# Instantiate the model
model = NeuralNetwork()

# Loss function: MSE
loss_fn = nn.MSELoss()
# Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# Learning rate scheduler (halve LR every 10 epochs)
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

# Create DataLoader for batching
dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(Y))
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training loop
epochs = 50
model.train()
for epoch in range(epochs):
    # Initialize running loss
    running_loss = 0.0
    # Iterate over batches
    for x_batch, y_batch in dataloader:
        # Forward pass
        y_pred = model(x_batch)
        # Compute loss
        loss = loss_fn(y_pred, y_batch)
        # Zero gradients
        optimizer.zero_grad()
        # Backpropagation
        loss.backward()
        # Optimization step
        optimizer.step()
        # Accumulate loss
        running_loss += loss.item() * x_batch.size(0)
    # Step the scheduler
    scheduler.step()
    # Compute average loss for the epoch
    epoch_loss = running_loss / len(dataloader.dataset)
    # Print epoch loss
    print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")

## Plot data generating process (for mean A) and model predictions in 1x2 plot using heatmaps
A_mean = a / (a + b)
K_grid = np.linspace(0, 1, 1000)
L_grid = np.linspace(0, 1, 1000)
K_mesh, L_mesh = np.meshgrid(K_grid, L_grid)
Y_dgp = A_mean * (K_mesh**alpha) * (L_mesh**(1.0 - alpha))
X_mesh = np.column_stack([K_mesh.ravel(), L_mesh.ravel()])

model.eval()
with torch.inference_mode():
    Y_pred_mesh = model(torch.FloatTensor(X_mesh)).numpy().reshape(K_mesh.shape)
Y_error_mesh = np.abs(Y_pred_mesh - Y_dgp)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Use a shared color scale for DGP and prediction
vmin = min(Y_dgp.min(), Y_pred_mesh.min())
vmax = max(Y_dgp.max(), Y_pred_mesh.max())

# DGP heatmap
im0 = axes[0].imshow(Y_dgp, extent=(0, 1, 0, 1), origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
axes[0].set_title('Data Generating Process')

# Model prediction heatmap
im1 = axes[1].imshow(Y_pred_mesh, extent=(0, 1, 0, 1), origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
axes[1].set_title('Neural Network Predictions')

# Shared colorbar for the first two heatmaps
fig.colorbar(im0, ax=axes[:2], fraction=0.046, pad=0.04)

# Error heatmap
im2 = axes[2].imshow(Y_error_mesh, extent=(0, 1, 0, 1), origin='lower', cmap='inferno')
axes[2].set_title('Absolute Error')
fig.colorbar(im2, ax=axes[2])

plt.show()