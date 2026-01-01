"""
Convolutional VAE for Music Spectrograms
Implementation for Medium Task
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List


class ConvVAE(nn.Module):
    """
    Convolutional VAE for 2D spectrograms (mel-spectrograms)
    More suitable for spatial patterns in audio
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        input_height: int = 128,  # n_mels
        input_width: int = 1292,  # Actual data: 165376 = 128×1292
        latent_dim: int = 128,
        hidden_channels: List[int] = [32, 64, 128, 256],
        dropout: float = 0.2
    ):
        super(ConvVAE, self).__init__()
        
        self.input_channels = input_channels
        self.input_height = input_height
        self.input_width = input_width
        self.latent_dim = latent_dim
        self.hidden_channels = hidden_channels
        
        # Encoder
        encoder_layers = []
        in_channels = input_channels
        
        for h_channels in hidden_channels:
            encoder_layers.extend([
                nn.Conv2d(in_channels, h_channels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(h_channels),
                nn.LeakyReLU(0.2),
                nn.Dropout2d(dropout)
            ])
            in_channels = h_channels
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Calculate flattened size after convolutions
        self.flat_size = self._get_flat_size()
        
        # Latent space
        self.fc_mu = nn.Linear(self.flat_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_size, latent_dim)
        
        # Decoder input
        self.decoder_input = nn.Linear(latent_dim, self.flat_size)
        
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
        """Calculate flattened size after encoder"""
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.input_channels, self.input_height, self.input_width)
            dummy_output = self.encoder(dummy_input)
            return dummy_output.view(1, -1).size(1)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent parameters"""
        # Reshape flattened input to 2D if needed
        if x.dim() == 2:  # [batch, flat_features]
            x = x.view(x.size(0), self.input_height, self.input_width)
        
        # Add channel dimension if needed
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        
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
        h = self.decoder_input(z)
        
        # Reshape to feature map
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
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass"""
        # Store original shape for reshaping output
        original_shape = x.shape
        
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        
        # Crop spatial dimensions if needed (before flattening)
        if recon.dim() == 4:  # [batch, channels, height, width]
            if len(original_shape) >= 2:
                # Calculate target dimensions
                if len(original_shape) == 2:  # Flattened input: [batch, height*width]
                    target_h = self.input_height
                    target_w = self.input_width
                elif len(original_shape) == 3:  # 2D input: [batch, height, width]
                    target_h = original_shape[1]
                    target_w = original_shape[2]
                else:  # 4D input: [batch, channels, height, width]
                    target_h = original_shape[2]
                    target_w = original_shape[3]
                
                # Crop if dimensions don't match
                if recon.size(2) != target_h or recon.size(3) != target_w:
                    recon = recon[:, :, :target_h, :target_w]
        
        # Flatten reconstruction to match input shape if input was flattened
        if len(original_shape) == 2:  # Input was [batch, flat_features]
            recon = recon.view(recon.size(0), -1)
        elif len(original_shape) == 3:  # Input was [batch, height, width]
            recon = recon.squeeze(1)  # Remove channel dimension
        
        return recon, mu, logvar
    
    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Get latent representation"""
        mu, _ = self.encode(x)
        return mu


class ResidualBlock(nn.Module):
    """Residual block for deeper architectures"""
    
    def __init__(self, channels: int, dropout: float = 0.2):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout2d(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        
        out = out + residual
        out = F.relu(out)
        
        return out


class DeepConvVAE(nn.Module):
    """
    Deeper Convolutional VAE with residual connections
    Better for complex audio patterns
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        input_height: int = 128,
        input_width: int = 1292,
        latent_dim: int = 128,
        hidden_channels: List[int] = [32, 64, 128, 256],
        num_residual_blocks: int = 2,
        dropout: float = 0.2
    ):
        super(DeepConvVAE, self).__init__()
        
        self.input_channels = input_channels
        self.input_height = input_height
        self.input_width = input_width
        self.latent_dim = latent_dim
        self.hidden_channels = hidden_channels
        
        # Encoder with residual blocks
        encoder_layers = []
        in_channels = input_channels
        
        for h_channels in hidden_channels:
            # Downsampling
            encoder_layers.extend([
                nn.Conv2d(in_channels, h_channels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(h_channels),
                nn.LeakyReLU(0.2),
                nn.Dropout2d(dropout)
            ])
            
            # Residual blocks
            for _ in range(num_residual_blocks):
                encoder_layers.append(ResidualBlock(h_channels, dropout))
            
            in_channels = h_channels
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Calculate flattened size
        self.flat_size = self._get_flat_size()
        
        # Latent space
        self.fc_mu = nn.Linear(self.flat_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_size, latent_dim)
        
        # Decoder input
        self.decoder_input = nn.Linear(latent_dim, self.flat_size)
        
        # Decoder with residual blocks
        decoder_layers = []
        hidden_channels_reversed = list(reversed(hidden_channels))
        
        for i in range(len(hidden_channels_reversed) - 1):
            # Residual blocks
            for _ in range(num_residual_blocks):
                decoder_layers.append(ResidualBlock(hidden_channels_reversed[i], dropout))
            
            # Upsampling
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
        
        # Final residual blocks and output
        for _ in range(num_residual_blocks):
            decoder_layers.append(ResidualBlock(hidden_channels_reversed[-1], dropout))
        
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
        """Calculate flattened size after encoder"""
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.input_channels, self.input_height, self.input_width)
            dummy_output = self.encoder(dummy_input)
            return dummy_output.view(1, -1).size(1)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input"""
        # Reshape flattened input to 2D if needed
        if x.dim() == 2:  # [batch, flat_features]
            x = x.view(x.size(0), self.input_height, self.input_width)
        
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent"""
        h = self.decoder_input(z)
        
        batch_size = h.size(0)
        h_channels = self.hidden_channels[-1]
        h_height = self.input_height // (2 ** len(self.hidden_channels))
        h_width = self.input_width // (2 ** len(self.hidden_channels))
        
        h = h.view(batch_size, h_channels, h_height, h_width)
        recon = self.decoder(h)
        
        if recon.size(2) != self.input_height or recon.size(3) != self.input_width:
            recon = F.interpolate(
                recon,
                size=(self.input_height, self.input_width),
                mode='bilinear',
                align_corners=False
            )
        
        return recon
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass"""
        # Store original shape for reshaping output
        original_shape = x.shape
        
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        
        # Crop/adjust dimensions BEFORE flattening
        if recon.dim() == 4:  # [batch, channels, height, width]
            # Crop spatial dimensions to match input
            if len(original_shape) >= 2:
                target_h = original_shape[-2] if len(original_shape) >= 3 else self.input_height
                target_w = original_shape[-1] if len(original_shape) >= 2 else self.input_width
                if recon.size(2) != target_h or recon.size(3) != target_w:
                    recon = recon[:, :, :target_h, :target_w]
        
        # Now flatten reconstruction to match input shape
        if len(original_shape) == 2:  # Input was [batch, flat_features]
            recon = recon.view(recon.size(0), -1)
        elif len(original_shape) == 3:  # Input was [batch, height, width]
            recon = recon.squeeze(1)  # Remove channel dimension
        
        return recon, mu, logvar
    
    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Get latent representation"""
        mu, _ = self.encode(x)
        return mu


if __name__ == "__main__":
    print("Testing ConvVAE:")
    
    # Create model
    model = ConvVAE(
        input_channels=1,
        input_height=128,
        input_width=1293,
        latent_dim=128,
        hidden_channels=[32, 64, 128, 256]
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 1, 128, 1293)
    
    recon, mu, logvar = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Reconstruction shape: {recon.shape}")
    print(f"Latent shape: {mu.shape}")
    
    # Test latent extraction
    z = model.get_latent(x)
    print(f"Extracted latent shape: {z.shape}")
    
    print("\n" + "="*60)
    print("Testing DeepConvVAE:")
    
    deep_model = DeepConvVAE(
        input_channels=1,
        input_height=128,
        input_width=1293,
        latent_dim=128,
        hidden_channels=[32, 64, 128, 256],
        num_residual_blocks=2
    )
    
    print(f"Model parameters: {sum(p.numel() for p in deep_model.parameters()):,}")
    
    recon, mu, logvar = deep_model(x)
    print(f"Reconstruction shape: {recon.shape}")
    print(f"Latent shape: {mu.shape}")
