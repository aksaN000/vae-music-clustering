"""
Beta-VAE and Conditional VAE (CVAE) for Disentangled Representations
Implementation for Hard Task
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List, Optional
from .conv_vae import ConvVAE


class BetaVAE(ConvVAE):
    """
    Beta-VAE for learning disentangled representations
    Uses higher beta value to increase KL weight
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        input_height: int = 128,
        input_width: int = 1292,  # Fixed: actual data dimension
        latent_dim: int = 128,
        hidden_channels: List[int] = [32, 64, 128, 256],
        beta: float = 4.0,
        dropout: float = 0.2
    ):
        super(BetaVAE, self).__init__(
            input_channels=input_channels,
            input_height=input_height,
            input_width=input_width,
            latent_dim=latent_dim,
            hidden_channels=hidden_channels,
            dropout=dropout
        )
        
        self.beta = beta
    
    def loss_function(
        self,
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Beta-VAE loss with higher KL weight"""
        # Flatten both tensors to avoid shape mismatches
        # Use reshape instead of view to handle non-contiguous tensors
        recon_x = recon_x.reshape(recon_x.size(0), -1)
        x = x.reshape(x.size(0), -1)
        
        # Reconstruction loss
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        
        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss with beta weighting
        total_loss = recon_loss + self.beta * kl_loss
        
        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'weighted_kl': self.beta * kl_loss
        }


class ConditionalVAE(nn.Module):
    """
    Conditional VAE (CVAE) for controlled generation
    Conditions on language, genre, or other labels
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        input_height: int = 128,
        input_width: int = 1293,
        latent_dim: int = 128,
        num_classes: int = 5,  # Number of languages or genres
        hidden_channels: List[int] = [32, 64, 128, 256],
        dropout: float = 0.2,
        condition_embedding_dim: int = 32
    ):
        super(ConditionalVAE, self).__init__()
        
        self.input_channels = input_channels
        self.input_height = input_height
        self.input_width = input_width
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.condition_embedding_dim = condition_embedding_dim
        self.hidden_channels = hidden_channels
        
        # Condition embedding
        self.condition_embedding = nn.Embedding(num_classes, condition_embedding_dim)
        
        # Encoder (takes input + condition)
        encoder_layers = []
        in_channels = input_channels + 1  # +1 for condition channel
        
        for h_channels in hidden_channels:
            encoder_layers.extend([
                nn.Conv2d(in_channels, h_channels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(h_channels),
                nn.LeakyReLU(0.2),
                nn.Dropout2d(dropout)
            ])
            in_channels = h_channels
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Calculate flattened size
        self.flat_size = self._get_flat_size()
        
        # Latent space (conditioned)
        self.fc_mu = nn.Linear(self.flat_size + condition_embedding_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_size + condition_embedding_dim, latent_dim)
        
        # Decoder input (latent + condition)
        self.decoder_input = nn.Linear(latent_dim + condition_embedding_dim, self.flat_size)
        
        # Decoder
        decoder_layers = []
        hidden_channels_reversed = list(reversed(hidden_channels))
        
        for i in range(len(hidden_channels_reversed) - 1):
            decoder_layers.extend([
                nn.ConvTranspose2d(
                    hidden_channels_reversed[i],
                    hidden_channels_reversed[i + 1],
                    kernel_size=4,
                    stride=2,
                    padding=1
                ),
                nn.BatchNorm2d(hidden_channels_reversed[i + 1]),
                nn.LeakyReLU(0.2),
                nn.Dropout2d(dropout)
            ])
        
        # Final layer
        decoder_layers.append(
            nn.ConvTranspose2d(
                hidden_channels_reversed[-1],
                input_channels,
                kernel_size=4,
                stride=2,
                padding=1
            )
        )
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def _get_flat_size(self) -> int:
        """Calculate flattened size"""
        with torch.no_grad():
            dummy_input = torch.zeros(
                1, self.input_channels + 1, self.input_height, self.input_width
            )
            dummy_output = self.encoder(dummy_input)
            return dummy_output.view(1, -1).size(1)
    
    def _broadcast_condition(
        self,
        condition: torch.Tensor,
        height: int,
        width: int
    ) -> torch.Tensor:
        """Broadcast condition to spatial dimensions"""
        # condition: (batch_size,)
        batch_size = condition.size(0)
        
        # Create spatial condition map
        # First convert to float tensor filled with condition values
        condition_float = condition.float().view(batch_size, 1, 1, 1)  # (B, 1, 1, 1)
        condition_map = condition_float.expand(batch_size, 1, height, width)
        
        return condition_map
    
    def encode(
        self,
        x: torch.Tensor,
        condition: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode with condition"""
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        batch_size = x.size(0)
        
        # Broadcast condition to spatial dimensions
        condition_map = self._broadcast_condition(
            condition,
            self.input_height,
            self.input_width
        )
        
        # Concatenate input with condition
        x_cond = torch.cat([x, condition_map], dim=1)
        
        # Encode
        h = self.encoder(x_cond)
        h = h.view(batch_size, -1)
        
        # Get condition embedding
        cond_emb = self.condition_embedding(condition)
        
        # Concatenate with condition
        h_cond = torch.cat([h, cond_emb], dim=1)
        
        # Latent parameters
        mu = self.fc_mu(h_cond)
        logvar = self.fc_logvar(h_cond)
        
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Decode with condition"""
        # Get condition embedding
        cond_emb = self.condition_embedding(condition)
        
        # Concatenate latent with condition
        z_cond = torch.cat([z, cond_emb], dim=1)
        
        # Decode
        h = self.decoder_input(z_cond)
        
        batch_size = h.size(0)
        h_channels = self.hidden_channels[-1]
        h_height = self.input_height // (2 ** len(self.hidden_channels))
        h_width = self.input_width // (2 ** len(self.hidden_channels))
        
        h = h.view(batch_size, h_channels, h_height, h_width)
        
        recon = self.decoder(h)
        
        # Crop to original size if needed
        if recon.size(2) != self.input_height or recon.size(3) != self.input_width:
            recon = F.interpolate(
                recon,
                size=(self.input_height, self.input_width),
                mode='bilinear',
                align_corners=False
            )
        
        return recon
    
    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass"""
        mu, logvar = self.encode(x, condition)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, condition)
        
        return recon, mu, logvar
    
    def get_latent(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Get latent representation"""
        mu, _ = self.encode(x, condition)
        return mu
    
    def loss_function(
        self,
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """CVAE loss"""
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        total_loss = recon_loss + kl_loss
        
        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss
        }


class FactorVAE(ConvVAE):
    """
    Factor-VAE for disentanglement
    Uses Total Correlation penalty
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        input_height: int = 128,
        input_width: int = 1293,
        latent_dim: int = 128,
        hidden_channels: List[int] = [32, 64, 128, 256],
        gamma: float = 6.0,
        dropout: float = 0.2
    ):
        super(FactorVAE, self).__init__(
            input_channels=input_channels,
            input_height=input_height,
            input_width=input_width,
            latent_dim=latent_dim,
            hidden_channels=hidden_channels,
            dropout=dropout
        )
        
        self.gamma = gamma
        
        # Discriminator for Total Correlation estimation
        self.discriminator = nn.Sequential(
            nn.Linear(latent_dim, 1000),
            nn.LeakyReLU(0.2),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2),
            nn.Linear(1000, 2)
        )
    
    def permute_dims(self, z: torch.Tensor) -> torch.Tensor:
        """Permute latent dimensions independently"""
        batch_size, latent_dim = z.size()
        
        z_perm = z.clone()
        for i in range(latent_dim):
            perm_idx = torch.randperm(batch_size)
            z_perm[:, i] = z_perm[perm_idx, i]
        
        return z_perm


if __name__ == "__main__":
    print("Testing Beta-VAE:")
    
    model = BetaVAE(
        input_channels=1,
        input_height=128,
        input_width=1293,
        latent_dim=128,
        beta=4.0
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    batch_size = 4
    x = torch.randn(batch_size, 1, 128, 1293)
    
    recon, mu, logvar = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Reconstruction shape: {recon.shape}")
    
    loss_dict = model.loss_function(recon, x, mu, logvar)
    print(f"Total loss: {loss_dict['loss'].item():.4f}")
    print(f"KL loss (weighted): {loss_dict['weighted_kl'].item():.4f}")
    
    print("\n" + "="*60)
    print("Testing Conditional VAE:")
    
    cvae = ConditionalVAE(
        input_channels=1,
        input_height=128,
        input_width=1293,
        latent_dim=128,
        num_classes=5  # 5 languages
    )
    
    print(f"Model parameters: {sum(p.numel() for p in cvae.parameters()):,}")
    
    # Test with conditions (language labels)
    conditions = torch.randint(0, 5, (batch_size,))
    
    recon, mu, logvar = cvae(x, conditions)
    print(f"Reconstruction shape: {recon.shape}")
    print(f"Latent shape: {mu.shape}")
    
    # Test conditional generation
    z = torch.randn(batch_size, 128)
    new_conditions = torch.tensor([0, 1, 2, 3, 4])[:batch_size]
    generated = cvae.decode(z, new_conditions)
    print(f"Generated shape: {generated.shape}")
