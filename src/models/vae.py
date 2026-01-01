"""
Basic VAE (Variational Autoencoder) for Music Clustering
Implementation for Easy Task
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class BasicVAE(nn.Module):
    """
    Basic fully-connected VAE for music feature extraction
    Suitable for flattened audio features (MFCC, etc.)
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 128,
        hidden_dims: list = [512, 256],
        dropout: float = 0.2
    ):
        super(BasicVAE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        
        # Build encoder
        encoder_layers = []
        in_dim = input_dim
        
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = h_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Build decoder
        decoder_layers = []
        hidden_dims_reversed = list(reversed(hidden_dims))
        
        in_dim = latent_dim
        for h_dim in hidden_dims_reversed:
            decoder_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = h_dim
        
        decoder_layers.append(nn.Linear(hidden_dims_reversed[-1], input_dim))
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent space parameters"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation"""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass"""
        # Store original shape
        original_shape = x.shape
        
        # Flatten input if needed
        if x.dim() > 2:
            batch_size = x.size(0)
            x = x.view(batch_size, -1)
        
        # Encode
        mu, logvar = self.encode(x)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Decode
        recon = self.decode(z)
        
        # Reshape to match original input
        if len(original_shape) > 2:
            recon = recon.view(original_shape)
        
        return recon, mu, logvar
    
    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Get latent representation (for clustering)"""
        # Flatten input if needed
        if x.dim() > 2:
            batch_size = x.size(0)
            x = x.view(batch_size, -1)
        
        mu, _ = self.encode(x)
        return mu


def vae_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0
) -> Dict[str, torch.Tensor]:
    """
    VAE loss function
    Args:
        recon_x: reconstructed input
        x: original input
        mu: latent mean
        logvar: latent log variance
        beta: weight for KL divergence (beta=1 for standard VAE)
    """
    # Flatten tensors
    if recon_x.dim() > 2:
        batch_size = recon_x.size(0)
        recon_x = recon_x.reshape(batch_size, -1)
        x = x.reshape(batch_size, -1)
    
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    
    # KL divergence
    # KL(N(mu, sigma) || N(0, 1))
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total loss
    total_loss = recon_loss + beta * kl_loss
    
    return {
        'loss': total_loss,
        'recon_loss': recon_loss,
        'kl_loss': kl_loss
    }


class Autoencoder(nn.Module):
    """
    Standard Autoencoder (for baseline comparison)
    No variational component
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 128,
        hidden_dims: list = [512, 256],
        dropout: float = 0.2
    ):
        super(Autoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Build encoder
        encoder_layers = []
        in_dim = input_dim
        
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = h_dim
        
        encoder_layers.append(nn.Linear(hidden_dims[-1], latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Build decoder
        decoder_layers = []
        hidden_dims_reversed = list(reversed(hidden_dims))
        
        in_dim = latent_dim
        for h_dim in hidden_dims_reversed:
            decoder_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = h_dim
        
        decoder_layers.append(nn.Linear(hidden_dims_reversed[-1], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode to latent space"""
        if x.dim() > 2:
            batch_size = x.size(0)
            x = x.view(batch_size, -1)
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space"""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass"""
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z
    
    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Get latent representation"""
        return self.encode(x)


def autoencoder_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor
) -> torch.Tensor:
    """Autoencoder reconstruction loss"""
    if recon_x.dim() > 2:
        batch_size = recon_x.size(0)
        recon_x = recon_x.view(batch_size, -1)
        x = x.view(batch_size, -1)
    
    return F.mse_loss(recon_x, x, reduction='mean')


if __name__ == "__main__":
    # Test BasicVAE
    print("Testing BasicVAE:")
    
    # Create model
    input_dim = 128 * 1293  # Example: mel-spectrogram (128 mels, 1293 time frames)
    model = BasicVAE(
        input_dim=input_dim,
        latent_dim=128,
        hidden_dims=[512, 256]
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    batch_size = 8
    x = torch.randn(batch_size, input_dim)
    
    recon, mu, logvar = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Reconstruction shape: {recon.shape}")
    print(f"Mu shape: {mu.shape}")
    print(f"Logvar shape: {logvar.shape}")
    
    # Test loss
    loss_dict = vae_loss(recon, x, mu, logvar)
    print(f"\nLoss: {loss_dict['loss'].item():.4f}")
    print(f"Recon loss: {loss_dict['recon_loss'].item():.4f}")
    print(f"KL loss: {loss_dict['kl_loss'].item():.4f}")
    
    # Test latent extraction
    z = model.get_latent(x)
    print(f"\nLatent shape: {z.shape}")
    
    print("\n" + "="*60)
    print("Testing Autoencoder:")
    
    ae_model = Autoencoder(
        input_dim=input_dim,
        latent_dim=128,
        hidden_dims=[512, 256]
    )
    
    print(f"Model parameters: {sum(p.numel() for p in ae_model.parameters()):,}")
    
    recon, z = ae_model(x)
    print(f"Reconstruction shape: {recon.shape}")
    print(f"Latent shape: {z.shape}")
    
    loss = autoencoder_loss(recon, x)
    print(f"Loss: {loss.item():.4f}")
