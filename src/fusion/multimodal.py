"""
Multi-modal Fusion Strategies
Combines audio and lyrics representations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class EarlyFusion(nn.Module):
    """
    Early Fusion: Concatenate features before encoding
    """
    
    def __init__(
        self,
        audio_dim: int,
        lyrics_dim: int,
        output_dim: int,
        hidden_dims: list = [512, 256]
    ):
        super(EarlyFusion, self).__init__()
        
        # Concatenate and project
        layers = []
        in_dim = audio_dim + lyrics_dim
        
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            in_dim = h_dim
        
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        self.fusion = nn.Sequential(*layers)
    
    def forward(
        self,
        audio_features: torch.Tensor,
        lyrics_features: torch.Tensor
    ) -> torch.Tensor:
        """Fuse audio and lyrics features"""
        # Flatten if needed
        if audio_features.dim() > 2:
            audio_features = audio_features.view(audio_features.size(0), -1)
        if lyrics_features.dim() > 2:
            lyrics_features = lyrics_features.view(lyrics_features.size(0), -1)
        
        # Concatenate
        combined = torch.cat([audio_features, lyrics_features], dim=1)
        
        # Fuse
        fused = self.fusion(combined)
        
        return fused


class LateFusion(nn.Module):
    """
    Late Fusion: Process modalities separately, then combine
    """
    
    def __init__(
        self,
        audio_dim: int,
        lyrics_dim: int,
        latent_dim: int,
        hidden_dims: list = [512, 256]
    ):
        super(LateFusion, self).__init__()
        
        # Audio encoder
        audio_layers = []
        in_dim = audio_dim
        for h_dim in hidden_dims:
            audio_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            in_dim = h_dim
        audio_layers.append(nn.Linear(hidden_dims[-1], latent_dim))
        self.audio_encoder = nn.Sequential(*audio_layers)
        
        # Lyrics encoder
        lyrics_layers = []
        in_dim = lyrics_dim
        for h_dim in hidden_dims:
            lyrics_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            in_dim = h_dim
        lyrics_layers.append(nn.Linear(hidden_dims[-1], latent_dim))
        self.lyrics_encoder = nn.Sequential(*lyrics_layers)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.ReLU()
        )
    
    def forward(
        self,
        audio_features: torch.Tensor,
        lyrics_features: torch.Tensor
    ) -> torch.Tensor:
        """Fuse audio and lyrics features"""
        # Flatten if needed
        if audio_features.dim() > 2:
            audio_features = audio_features.view(audio_features.size(0), -1)
        if lyrics_features.dim() > 2:
            lyrics_features = lyrics_features.view(lyrics_features.size(0), -1)
        
        # Encode separately
        audio_latent = self.audio_encoder(audio_features)
        lyrics_latent = self.lyrics_encoder(lyrics_features)
        
        # Concatenate and fuse
        combined = torch.cat([audio_latent, lyrics_latent], dim=1)
        fused = self.fusion(combined)
        
        return fused


class AttentionFusion(nn.Module):
    """
    Attention-based Fusion: Use cross-modal attention
    """
    
    def __init__(
        self,
        audio_dim: int,
        lyrics_dim: int,
        latent_dim: int,
        num_heads: int = 8,
        hidden_dims: list = [512, 256]
    ):
        super(AttentionFusion, self).__init__()
        
        self.latent_dim = latent_dim
        
        # Project to same dimension
        self.audio_projection = nn.Linear(audio_dim, latent_dim)
        self.lyrics_projection = nn.Linear(lyrics_dim, latent_dim)
        
        # Multi-head cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Fusion MLP
        fusion_layers = []
        in_dim = latent_dim * 2
        for h_dim in hidden_dims:
            fusion_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            in_dim = h_dim
        fusion_layers.append(nn.Linear(hidden_dims[-1], latent_dim))
        
        self.fusion_mlp = nn.Sequential(*fusion_layers)
    
    def forward(
        self,
        audio_features: torch.Tensor,
        lyrics_features: torch.Tensor
    ) -> torch.Tensor:
        """Fuse with cross-attention"""
        # Flatten if needed
        if audio_features.dim() > 2:
            audio_features = audio_features.view(audio_features.size(0), -1)
        if lyrics_features.dim() > 2:
            lyrics_features = lyrics_features.view(lyrics_features.size(0), -1)
        
        # Project to same dimension
        audio_proj = self.audio_projection(audio_features).unsqueeze(1)  # (B, 1, D)
        lyrics_proj = self.lyrics_projection(lyrics_features).unsqueeze(1)  # (B, 1, D)
        
        # Cross-attention: audio attends to lyrics
        audio_attended, _ = self.cross_attention(
            query=audio_proj,
            key=lyrics_proj,
            value=lyrics_proj
        )
        
        # Cross-attention: lyrics attends to audio
        lyrics_attended, _ = self.cross_attention(
            query=lyrics_proj,
            key=audio_proj,
            value=audio_proj
        )
        
        # Combine attended features
        audio_attended = audio_attended.squeeze(1)
        lyrics_attended = lyrics_attended.squeeze(1)
        
        combined = torch.cat([audio_attended, lyrics_attended], dim=1)
        
        # Final fusion
        fused = self.fusion_mlp(combined)
        
        return fused


class WeightedFusion(nn.Module):
    """
    Weighted Fusion: Learn weights for each modality
    """
    
    def __init__(
        self,
        audio_dim: int,
        lyrics_dim: int,
        latent_dim: int
    ):
        super(WeightedFusion, self).__init__()
        
        # Project to same dimension
        self.audio_projection = nn.Linear(audio_dim, latent_dim)
        self.lyrics_projection = nn.Linear(lyrics_dim, latent_dim)
        
        # Learnable weights
        self.audio_weight = nn.Parameter(torch.tensor(0.5))
        self.lyrics_weight = nn.Parameter(torch.tensor(0.5))
    
    def forward(
        self,
        audio_features: torch.Tensor,
        lyrics_features: torch.Tensor
    ) -> torch.Tensor:
        """Weighted combination"""
        # Flatten if needed
        if audio_features.dim() > 2:
            audio_features = audio_features.view(audio_features.size(0), -1)
        if lyrics_features.dim() > 2:
            lyrics_features = lyrics_features.view(lyrics_features.size(0), -1)
        
        # Project
        audio_proj = self.audio_projection(audio_features)
        lyrics_proj = self.lyrics_projection(lyrics_features)
        
        # Normalize weights
        weights = F.softmax(torch.stack([self.audio_weight, self.lyrics_weight]), dim=0)
        
        # Weighted sum
        fused = weights[0] * audio_proj + weights[1] * lyrics_proj
        
        return fused


class MultimodalVAE(nn.Module):
    """
    Multi-modal VAE that incorporates fusion
    """
    
    def __init__(
        self,
        audio_vae,
        lyrics_dim: int,
        latent_dim: int,
        fusion_type: str = 'attention'
    ):
        super(MultimodalVAE, self).__init__()
        
        self.audio_vae = audio_vae
        self.latent_dim = latent_dim
        self.fusion_type = fusion_type
        
        # Get audio latent dimension
        audio_latent_dim = audio_vae.latent_dim
        
        # Fusion module
        if fusion_type == 'early':
            self.fusion = EarlyFusion(
                audio_dim=audio_latent_dim,
                lyrics_dim=lyrics_dim,
                output_dim=latent_dim
            )
        elif fusion_type == 'late':
            self.fusion = LateFusion(
                audio_dim=audio_latent_dim,
                lyrics_dim=lyrics_dim,
                latent_dim=latent_dim
            )
        elif fusion_type == 'attention':
            self.fusion = AttentionFusion(
                audio_dim=audio_latent_dim,
                lyrics_dim=lyrics_dim,
                latent_dim=latent_dim
            )
        elif fusion_type == 'weighted':
            self.fusion = WeightedFusion(
                audio_dim=audio_latent_dim,
                lyrics_dim=lyrics_dim,
                latent_dim=latent_dim
            )
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
    
    def forward(
        self,
        audio_features: torch.Tensor,
        lyrics_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass"""
        # Get audio VAE latent
        audio_recon, audio_mu, audio_logvar = self.audio_vae(audio_features)
        
        # Fuse with lyrics
        fused_latent = self.fusion(audio_mu, lyrics_features)
        
        return audio_recon, audio_mu, audio_logvar, fused_latent
    
    def get_latent(
        self,
        audio_features: torch.Tensor,
        lyrics_features: torch.Tensor
    ) -> torch.Tensor:
        """Get fused latent representation"""
        audio_mu, _ = self.audio_vae.encode(audio_features)
        fused_latent = self.fusion(audio_mu, lyrics_features)
        return fused_latent


if __name__ == "__main__":
    print("Testing Fusion Modules:")
    
    batch_size = 8
    audio_dim = 128 * 1293
    lyrics_dim = 768  # XLM-RoBERTa embedding dimension
    latent_dim = 128
    
    audio_features = torch.randn(batch_size, audio_dim)
    lyrics_features = torch.randn(batch_size, lyrics_dim)
    
    # Test Early Fusion
    print("\n1. Early Fusion:")
    early_fusion = EarlyFusion(audio_dim, lyrics_dim, latent_dim)
    fused = early_fusion(audio_features, lyrics_features)
    print(f"   Output shape: {fused.shape}")
    
    # Test Late Fusion
    print("\n2. Late Fusion:")
    late_fusion = LateFusion(audio_dim, lyrics_dim, latent_dim)
    fused = late_fusion(audio_features, lyrics_features)
    print(f"   Output shape: {fused.shape}")
    
    # Test Attention Fusion
    print("\n3. Attention Fusion:")
    attention_fusion = AttentionFusion(audio_dim, lyrics_dim, latent_dim)
    fused = attention_fusion(audio_features, lyrics_features)
    print(f"   Output shape: {fused.shape}")
    
    # Test Weighted Fusion
    print("\n4. Weighted Fusion:")
    weighted_fusion = WeightedFusion(audio_dim, lyrics_dim, latent_dim)
    fused = weighted_fusion(audio_features, lyrics_features)
    print(f"   Output shape: {fused.shape}")
    print(f"   Audio weight: {weighted_fusion.audio_weight.item():.4f}")
    print(f"   Lyrics weight: {weighted_fusion.lyrics_weight.item():.4f}")
