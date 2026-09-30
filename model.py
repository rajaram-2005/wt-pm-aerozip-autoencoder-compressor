"""MLP autoencoder for compressing high-dimensional turbine telemetry frames.

This module defines the architecture and nominal compression-ratio helper only:
it ships no data loader, packet serializer, training loop or trained weights.
Run ``python model.py`` to build the reference configuration and print its
summary. The wt-pm platform imports this file as-is and calls
``AeroZipCompressor`` and ``compression_ratio``, so keep those signatures
stable.
"""
import torch
import torch.nn as nn


class AeroZipCompressor(nn.Module):
    """Symmetrical bottleneck autoencoder (input_dim -> 32 -> latent_dim -> 32 -> input_dim).

    input_dim: number of telemetry features per frame (default 64).
    latent_dim: width of the compressed latent representation (default 8).
    """

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
        """Encode ``x`` into the bottleneck and reconstruct it.

        Returns ``(latent, reconstructed)`` with shapes ``(..., latent_dim)``
        and ``(..., input_dim)``.
        """
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return latent, reconstructed


def compression_ratio(input_dim=64, latent_dim=8):
    """Return the nominal feature-dimension ratio ``input_dim / latent_dim``."""
    return input_dim / latent_dim


if __name__ == "__main__":
    # Smoke test: build the reference configuration (64-feature frame -> 8-D
    # latent bottleneck) and print the architecture. Nothing is loaded or
    # trained here.
    model = AeroZipCompressor(input_dim=64, latent_dim=8)
    n_params = sum(p.numel() for p in model.parameters())
    ratio = compression_ratio(64, 8)
    print(model)
    print(f"Parameters: {n_params:,} | Compression ratio: {ratio:.1f}:1 (64 -> 8)")
