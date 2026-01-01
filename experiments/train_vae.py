"""
Main Training Script for VAE Models
Supports all VAE variants: Basic, Conv, Beta, CVAE, VaDE
"""

import sys
sys.path.append('.')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import argparse
import yaml
from pathlib import Path

from src.data.dataset import AudioOnlyDataset, MultimodalDataset, get_dataloader
from src.models.vae import BasicVAE
from src.models.conv_vae import ConvVAE, DeepConvVAE
from src.models.beta_vae import BetaVAE, ConditionalVAE
from src.models.vade import VaDE
from src.training.trainer import VAETrainer


def get_model(model_type: str, config: dict, input_dim: int):
    """Create model based on type"""
    
    latent_dim = config['latent_dim']
    hidden_dims = config['hidden_dims']
    
    if model_type == 'basic':
        model = BasicVAE(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            dropout=config.get('dropout', 0.2)
        )
    
    elif model_type == 'conv':
        model = ConvVAE(
            input_channels=1,
            input_height=128,  # n_mels
            input_width=1293,  # 30s at 22050Hz
            latent_dim=latent_dim,
            hidden_channels=hidden_dims,
            dropout=config.get('dropout', 0.2)
        )
    
    elif model_type == 'deep_conv':
        model = DeepConvVAE(
            input_channels=1,
            input_height=128,
            input_width=1292,
            latent_dim=latent_dim,
            hidden_channels=hidden_dims,
            num_residual_blocks=2,
            dropout=config.get('dropout', 0.2)
        )
    
    elif model_type == 'beta':
        model = BetaVAE(
            input_channels=1,
            input_height=128,
            input_width=1293,
            latent_dim=latent_dim,
            hidden_channels=hidden_dims,
            beta=config.get('beta', 4.0),
            dropout=config.get('dropout', 0.2)
        )
    
    elif model_type == 'cvae':
        # Get number of classes from config (can condition on language or genre)
        num_classes = config.get('num_languages', 5)  # Default to 5 languages
        if args.condition == 'genre':
            num_classes = config.get('num_genres', 45)  # Use 45 genres if specified
        
        model = ConditionalVAE(
            input_channels=1,
            input_height=128,
            input_width=1292,
            latent_dim=latent_dim,
            num_classes=num_classes,
            hidden_channels=hidden_dims,
            dropout=config.get('dropout', 0.2)
        )
    
    elif model_type == 'vade':
        model = VaDE(
            input_dim=input_dim,
            latent_dim=latent_dim,
            n_clusters=config.get('n_clusters', 15),
            hidden_dims=hidden_dims,
            dropout=config.get('dropout', 0.2)
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def main(args):
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() and config['optimization']['use_gpu'] else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    # Load dataset
    print("\nLoading dataset...")
    # Determine data path
    if args.data_path:
        data_path = args.data_path
    elif args.modality == 'audio':
        data_path = "data/features/audio_only_dataset.pkl"
    else:
        data_path = "data/features/multimodal_dataset.pkl"
    
    if args.modality == 'audio':
        dataset = AudioOnlyDataset(
            data_path=data_path,
            feature_type=config['multimodal']['audio_feature']
        )
    elif args.modality == 'multimodal':
        dataset = MultimodalDataset(
            data_path=data_path,
            feature_type=config['multimodal']['audio_feature']
        )
    else:
        raise ValueError(f"Unknown modality: {args.modality}")
    
    print(f"Total samples: {len(dataset)}")
    
    # Split into train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = get_dataloader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    val_loader = get_dataloader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    # Calculate input dimension
    sample = dataset[0]
    if 'features' in sample:
        sample_features = sample['features']
    elif 'audio_features' in sample:
        sample_features = sample['audio_features']
    
    if sample_features.dim() == 3:
        input_dim = sample_features.size(0) * sample_features.size(1) * sample_features.size(2)
    elif sample_features.dim() == 2:
        input_dim = sample_features.size(0) * sample_features.size(1)
    else:
        input_dim = sample_features.size(0)
    
    print(f"\nInput dimension: {input_dim}")
    
    # Create model
    print(f"\nCreating {args.model} model...")
    model = get_model(args.model, config['model'], input_dim)
    
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {num_params:,}")
    print(f"Trainable parameters: {num_trainable:,}")
    
    # VaDE pretraining
    if args.model == 'vade':
        print("\nPre-training VaDE...")
        model.pretrain(train_loader, device, epochs=10)
    
    # Determine model name for saving (include modality if multimodal, condition for cvae)
    model_name = args.model
    if args.modality == 'multimodal':
        model_name = f"{args.model}_multimodal"
    elif args.model == 'cvae' and args.condition:
        model_name = f"cvae_{args.condition}"
    
    # Create trainer
    trainer = VAETrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        device=device,
        mixed_precision=config['training']['mixed_precision'],
        gradient_clip=config['training']['gradient_clip'],
        log_interval=config['logging']['log_interval'],
        save_dir=f"{config['logging']['output_dir']}/checkpoints/{model_name}",
        condition_type=args.condition if args.model == 'cvae' else None
    )
    
    # Train
    print("\nStarting training...")
    history = trainer.train(
        num_epochs=config['training']['epochs'],
        early_stopping_patience=config['training']['early_stopping_patience']
    )
    
    # Save final model
    final_path = Path(config['logging']['output_dir']) / 'checkpoints' / model_name / 'final_model.pt'
    torch.save(model.state_dict(), final_path)
    print(f"\nSaved final model to: {final_path}")
    
    # Plot training curves
    from src.visualization.plots import ClusterVisualizer
    viz = ClusterVisualizer(output_dir=f"{config['logging']['output_dir']}/visualizations/{model_name}")
    viz.plot_training_curves(history, save_name='training_curves.png')
    
    print("\nTraining complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VAE models for music clustering")
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--model', type=str, default='conv',
                       choices=['basic', 'conv', 'deep_conv', 'beta', 'cvae', 'vade'],
                       help='Model type')
    parser.add_argument('--modality', type=str, default='audio',
                       choices=['audio', 'multimodal'],
                       help='Data modality')
    parser.add_argument('--condition', type=str, default='language',
                       choices=['language', 'genre'],
                       help='Conditioning variable for CVAE (language or genre)')
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to dataset pickle file (default: data/features/audio_only_dataset.pkl or multimodal_dataset.pkl)')
    
    args = parser.parse_args()
    main(args)
