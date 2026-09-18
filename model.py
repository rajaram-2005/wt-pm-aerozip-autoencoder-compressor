import torch
import torch.nn as nn

class AeroZipCompressor(nn.Module):
    def __init__(self, input_dim=64, latent_dim=8):
        super(AeroZipCompressor, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return latent, reconstructed

def compression_ratio(input_dim=64, latent_dim=8):
    return input_dim / latent_dim

# Training:
# model = AeroZipCompressor()
# loss = nn.MSELoss()(reconstructed, x)
