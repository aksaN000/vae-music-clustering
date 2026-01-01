"""
Inspect and load trained VAE models from checkpoints
"""

import torch
import numpy as np
from pathlib import Path
import sys

def load_checkpoint(checkpoint_path):
    """Load and inspect a checkpoint file"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print("="*70)
    print(f"Checkpoint: {Path(checkpoint_path).name}")
    print("="*70)
    
    # Show basic info
    if 'epoch' in checkpoint:
        print(f"Epoch: {checkpoint['epoch']}")
    
    if 'best_val_loss' in checkpoint:
        print(f"Best Val Loss: {checkpoint['best_val_loss']:.4f}")
    
    # Show training history
    if 'history' in checkpoint:
        history = checkpoint['history']
        print(f"\nTraining History (last 5 epochs):")
        print("-" * 70)
        
        if 'train_loss' in history and len(history['train_loss']) > 0:
            print(f"Train Loss: {history['train_loss'][-5:]}")
        if 'train_recon_loss' in history and len(history['train_recon_loss']) > 0:
            print(f"Train Recon: {history['train_recon_loss'][-5:]}")
        if 'train_kl_loss' in history and len(history['train_kl_loss']) > 0:
            print(f"Train KL: {history['train_kl_loss'][-5:]}")
        
        if 'val_loss' in history and len(history['val_loss']) > 0:
            print(f"\nVal Loss: {history['val_loss'][-5:]}")
        if 'val_recon_loss' in history and len(history['val_recon_loss']) > 0:
            print(f"Val Recon: {history['val_recon_loss'][-5:]}")
        if 'val_kl_loss' in history and len(history['val_kl_loss']) > 0:
            print(f"Val KL: {history['val_kl_loss'][-5:]}")
    
    # Show model architecture info
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        total_params = sum(p.numel() for p in state_dict.values() if isinstance(p, torch.Tensor))
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Number of layers: {len(state_dict)}")
    
    print("="*70)
    print()
    
    return checkpoint


def load_and_use_model(checkpoint_path, model_class):
    """Load model for inference"""
    from src.models.vae import BasicVAE
    from src.models.conv_vae import ConvVAE
    from src.models.beta_vae import BetaVAE
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Get model config from checkpoint or use defaults
    if 'model_config' in checkpoint:
        config = checkpoint['model_config']
        model = model_class(**config)
    else:
        # Try to infer from state dict
        print("Warning: No model config found, using default parameters")
        model = model_class(input_dim=165376, latent_dim=128)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Model loaded successfully from {Path(checkpoint_path).name}")
    return model


def extract_latent_representations(model, dataloader, device='cuda'):
    """Extract latent representations from dataset"""
    model = model.to(device)
    model.eval()
    
    latent_vectors = []
    labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, dict):
                x = batch['features'].to(device)
                if 'language' in batch:
                    labels.extend(batch['language'].cpu().numpy())
            else:
                x = batch.to(device)
            
            # Get latent representation
            mu, _ = model.encode(x)
            latent_vectors.append(mu.cpu().numpy())
    
    latent_vectors = np.concatenate(latent_vectors, axis=0)
    
    print(f"Extracted {latent_vectors.shape[0]} latent vectors")
    print(f"Latent dimension: {latent_vectors.shape[1]}")
    
    return latent_vectors, labels


if __name__ == "__main__":
    # Example usage
    checkpoint_dir = Path("results/checkpoints/basic")
    
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
    else:
        # Load best model by default
        checkpoint_path = checkpoint_dir / "best_model.pt"
    
    if not Path(checkpoint_path).exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print(f"\nAvailable checkpoints in {checkpoint_dir}:")
        for ckpt in sorted(checkpoint_dir.glob("*.pt")):
            print(f"  - {ckpt.name}")
        sys.exit(1)
    
    # Inspect checkpoint
    checkpoint = load_checkpoint(checkpoint_path)
    
    # Show how to load the model
    print("\nTo use this model for inference:")
    print("="*70)
    print("""
from src.models.vae import BasicVAE
import torch

# Load the model
checkpoint = torch.load('results/checkpoints/basic/best_model.pt')
model = BasicVAE(input_dim=165376, latent_dim=128)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Extract latent representations
with torch.no_grad():
    mu, logvar = model.encode(your_data)
    latent_vectors = mu.cpu().numpy()

# Use for clustering
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=10)
clusters = kmeans.fit_predict(latent_vectors)
""")
    print("="*70)
