import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# Define the VAE model
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            GCNConv(input_dim, hidden_dim),
            nn.ReLU(),
            GCNConv(hidden_dim, latent_dim * 2)
        )
        self.decoder = nn.Sequential(
            GCNConv(latent_dim, hidden_dim),
            nn.ReLU(),
            GCNConv(hidden_dim, input_dim)
        )
        self.latent_dim = latent_dim

    def encode(self, x, edge_index):
        z = self.encoder(x, edge_index).view(-1, 2, self.latent_dim)
        mu, logvar = z[:, 0, :], z[:, 1, :]
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, edge_index):
        return self.decoder(z, edge_index)

    def forward(self, x, edge_index):
        mu, logvar = self.encode(x, edge_index)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, edge_index), mu, logvar

# Create a simple linear graph
x = torch.tensor([[0], [1], [2], [3]], dtype=torch.float)
edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long)

# Create a VAE instance
input_dim = 1
hidden_dim = 16
latent_dim = 8
vae = VAE(input_dim, hidden_dim, latent_dim)

# Define the loss function
def loss_function(recon_x, x, mu, logvar):
    recon_loss = nn.MSELoss()(recon_x, x)
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_divergence

# Define the optimizer
optimizer = optim.Adam(vae.parameters(), lr=0.001)

# Training pipeline
def train():
    vae.train()
    optimizer.zero_grad()
    recon_x, mu, logvar = vae(x, edge_index)
    loss = loss_function(recon_x, x, mu, logvar)
    loss.backward()
    optimizer.step()

# Train the VAE
num_epochs = 100
for epoch in range(num_epochs):
    train()

# Reconstruct the input graph
vae.eval()
with torch.no_grad():
    reconstructed_x, _, _ = vae(x, edge_index)

print("Original graph:")
print(x)
print("Reconstructed graph:")
print(reconstructed_x)

