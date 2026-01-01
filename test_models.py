"""
Model Testing & Comparison Script
Demonstrates and proves unique features of each VAE model
"""

import sys
sys.path.append('.')

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
from scipy.stats import entropy

from src.data.dataset import AudioOnlyDataset
from src.models.vae import BasicVAE
from src.models.conv_vae import ConvVAE
from src.models.beta_vae import BetaVAE, ConditionalVAE
from src.models.vade import VaDE


class ModelTester:
    """Test and compare VAE models"""
    
    def __init__(self, checkpoint_dir='checkpoints', output_dir='results/model_tests'):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_model(self, model_name, checkpoint_path):
        """Load trained model"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if model_name == 'basic_vae':
            model = BasicVAE(input_dim=165504, latent_dim=128)
        elif model_name == 'conv_vae':
            model = ConvVAE(latent_dim=128)
        elif model_name == 'beta_vae':
            model = BetaVAE(latent_dim=128, beta=4.0)
        elif model_name == 'cvae':
            model = ConditionalVAE(latent_dim=128, num_classes=5)
        elif model_name == 'vade':
            model = VaDE(input_dim=165504, latent_dim=128, n_clusters=50)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval()
        
        return model
    
    def test_smoothness(self, model_name='basic_vae'):
        """Test 1: Latent space smoothness (Basic VAE)"""
        print("\n" + "="*80)
        print("TEST 1: LATENT SPACE SMOOTHNESS (Basic VAE Feature)")
        print("="*80)
        
        checkpoint = self.checkpoint_dir / model_name / 'best_model.pt'
        if not checkpoint.exists():
            print(f"❌ Model not found: {checkpoint}")
            print("   Train the model first: python experiments/train_vae.py --model basic_vae")
            return
        
        model = self.load_model(model_name, checkpoint)
        
        # Get two random samples
        dataset = AudioOnlyDataset()
        idx1, idx2 = np.random.choice(len(dataset), 2, replace=False)
        
        with torch.no_grad():
            x1 = dataset[idx1]['features'].unsqueeze(0).to(self.device)
            x2 = dataset[idx2]['features'].unsqueeze(0).to(self.device)
            
            # Get latent codes
            if hasattr(model, 'encoder'):
                mu1, _ = model.encode(x1.view(1, -1))
                mu2, _ = model.encode(x2.view(1, -1))
            else:
                mu1, _ = model.encoder(x1)
                mu2, _ = model.encoder(x2)
            
            # Interpolate in latent space
            alphas = np.linspace(0, 1, 10)
            interpolations = []
            
            for alpha in alphas:
                z_interp = (1 - alpha) * mu1 + alpha * mu2
                
                if hasattr(model, 'decoder'):
                    recon = model.decode(z_interp)
                else:
                    recon = model.decoder(z_interp)
                
                interpolations.append(recon.cpu().numpy())
        
        # Measure smoothness (L2 distance between consecutive interpolations)
        distances = []
        for i in range(len(interpolations) - 1):
            dist = np.linalg.norm(interpolations[i+1] - interpolations[i])
            distances.append(dist)
        
        smoothness = np.std(distances)  # Lower std = smoother
        
        print(f"\n✓ Interpolation smoothness: {smoothness:.6f}")
        print(f"  → Lower value = smoother latent space")
        print(f"  → Basic VAE should have smooth transitions")
        print(f"\n📊 Step-by-step distances:")
        for i, d in enumerate(distances):
            print(f"     Step {i+1}: {d:.6f}")
        
        # Visualize
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        for i, (ax, interp) in enumerate(zip(axes.flat, interpolations)):
            spec = interp[0].reshape(128, -1)
            ax.imshow(spec, aspect='auto', origin='lower', cmap='viridis')
            ax.set_title(f'α={alphas[i]:.1f}')
            ax.axis('off')
        
        plt.suptitle(f'{model_name.upper()}: Latent Space Interpolation (Smoothness Test)', fontsize=14)
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{model_name}_smoothness.png', dpi=150)
        print(f"\n✓ Visualization saved: {self.output_dir / f'{model_name}_smoothness.png'}")
        
        return smoothness
    
    def test_filters(self, model_name='conv_vae'):
        """Test 2: Learned convolutional filters (Conv VAE)"""
        print("\n" + "="*80)
        print("TEST 2: LEARNED CONVOLUTIONAL FILTERS (Conv VAE Feature)")
        print("="*80)
        
        checkpoint = self.checkpoint_dir / model_name / 'best_model.pt'
        if not checkpoint.exists():
            print(f"❌ Model not found: {checkpoint}")
            return
        
        model = self.load_model(model_name, checkpoint)
        
        # Extract first conv layer filters
        first_conv = None
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                first_conv = module
                break
        
        if first_conv is None:
            print("❌ No convolutional layers found")
            return
        
        filters = first_conv.weight.data.cpu().numpy()
        print(f"\n✓ First conv layer: {filters.shape}")
        print(f"  → {filters.shape[0]} filters of size {filters.shape[2]}×{filters.shape[3]}")
        
        # Visualize filters
        n_filters = min(32, filters.shape[0])
        fig, axes = plt.subplots(4, 8, figsize=(16, 8))
        
        for i, ax in enumerate(axes.flat):
            if i < n_filters:
                filt = filters[i, 0]  # First input channel
                ax.imshow(filt, cmap='seismic', vmin=-np.abs(filt).max(), vmax=np.abs(filt).max())
                ax.set_title(f'Filter {i+1}', fontsize=8)
            ax.axis('off')
        
        plt.suptitle(f'{model_name.upper()}: Learned Convolutional Filters', fontsize=14)
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{model_name}_filters.png', dpi=150)
        print(f"\n✓ Visualization saved: {self.output_dir / f'{model_name}_filters.png'}")
        print("\n📊 Filter statistics:")
        print(f"  → Mean absolute value: {np.abs(filters).mean():.6f}")
        print(f"  → Std: {filters.std():.6f}")
        print(f"  → Filters detect: frequency bands, rhythmic patterns, spectral shapes")
        
        return filters
    
    def test_disentanglement(self, model_name='beta_vae'):
        """Test 3: Disentanglement (Beta-VAE)"""
        print("\n" + "="*80)
        print("TEST 3: DISENTANGLEMENT (Beta-VAE Feature)")
        print("="*80)
        
        checkpoint = self.checkpoint_dir / model_name / 'best_model.pt'
        if not checkpoint.exists():
            print(f"❌ Model not found: {checkpoint}")
            return
        
        model = self.load_model(model_name, checkpoint)
        dataset = AudioOnlyDataset()
        
        # Get a sample
        sample_idx = np.random.randint(len(dataset))
        x = dataset[sample_idx]['features'].unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Get latent code
            if hasattr(model, 'encoder'):
                mu, _ = model.encode(x.view(1, -1))
            else:
                mu, _ = model.encoder(x)
            
            # Traverse individual dimensions
            n_dims = 8  # Test first 8 dimensions
            traversal_range = np.linspace(-3, 3, 7)
            
            fig, axes = plt.subplots(n_dims, len(traversal_range), figsize=(14, 16))
            
            for dim_idx in range(n_dims):
                for i, val in enumerate(traversal_range):
                    z = mu.clone()
                    z[0, dim_idx] = val  # Change only this dimension
                    
                    if hasattr(model, 'decoder'):
                        recon = model.decode(z)
                    else:
                        recon = model.decoder(z)
                    
                    spec = recon[0].cpu().numpy().reshape(128, -1)
                    axes[dim_idx, i].imshow(spec, aspect='auto', origin='lower', cmap='viridis')
                    
                    if i == 0:
                        axes[dim_idx, i].set_ylabel(f'Dim {dim_idx}', fontsize=10)
                    if dim_idx == 0:
                        axes[dim_idx, i].set_title(f'z={val:.1f}', fontsize=8)
                    axes[dim_idx, i].axis('off')
        
        plt.suptitle(f'{model_name.upper()}: Latent Dimension Traversal (Disentanglement)', fontsize=14)
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{model_name}_disentanglement.png', dpi=150)
        print(f"\n✓ Visualization saved: {self.output_dir / f'{model_name}_disentanglement.png'}")
        print("\n📊 Disentanglement test:")
        print("  → Each row shows variation of ONE latent dimension")
        print("  → Good disentanglement = each dimension controls ONE factor")
        print("  → Beta-VAE should show independent changes per dimension")
        
    def test_separation(self, model_name='cvae', condition='language'):
        """Test 4: Class separation (Conditional VAE)"""
        print("\n" + "="*80)
        print(f"TEST 4: CLASS SEPARATION (Conditional VAE - {condition})")
        print("="*80)
        
        checkpoint = self.checkpoint_dir / model_name / 'best_model.pt'
        if not checkpoint.exists():
            print(f"❌ Model not found: {checkpoint}")
            return
        
        model = self.load_model(model_name, checkpoint)
        dataset = AudioOnlyDataset()
        
        # Extract latent codes and labels
        latents = []
        labels = []
        
        with torch.no_grad():
            for i in range(min(500, len(dataset))):
                sample = dataset[i]
                x = sample['features'].unsqueeze(0).to(self.device)
                
                if hasattr(model, 'encoder'):
                    mu, _ = model.encode(x.view(1, -1))
                else:
                    mu, _ = model.encoder(x)
                
                latents.append(mu.cpu().numpy())
                labels.append(sample[condition])
        
        latents = np.vstack(latents)
        labels = np.array(labels)
        
        # t-SNE visualization
        print("\n📊 Computing t-SNE...")
        tsne = TSNE(n_components=2, random_state=42)
        latents_2d = tsne.fit_transform(latents)
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 10))
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            mask = labels == label
            ax.scatter(latents_2d[mask, 0], latents_2d[mask, 1],
                      c=[color], label=f'{label}', alpha=0.6, s=20)
        
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_title(f'{model_name.upper()}: Class Separation ({condition})', fontsize=14)
        ax.set_xlabel('t-SNE Dimension 1')
        ax.set_ylabel('t-SNE Dimension 2')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{model_name}_separation_{condition}.png', dpi=150, bbox_inches='tight')
        print(f"\n✓ Visualization saved: {self.output_dir / f'{model_name}_separation_{condition}.png'}")
        print(f"\n✓ Found {len(unique_labels)} classes")
        print("  → Conditional VAE should show distinct, well-separated clusters")
        
    def test_soft_clustering(self, model_name='vade'):
        """Test 5: Soft clustering (VaDE)"""
        print("\n" + "="*80)
        print("TEST 5: SOFT CLUSTERING (VaDE Feature)")
        print("="*80)
        
        checkpoint = self.checkpoint_dir / model_name / 'best_model.pt'
        if not checkpoint.exists():
            print(f"❌ Model not found: {checkpoint}")
            return
        
        model = self.load_model(model_name, checkpoint)
        dataset = AudioOnlyDataset()
        
        # Get soft cluster assignments for samples
        assignments = []
        confidences = []
        
        with torch.no_grad():
            for i in range(min(100, len(dataset))):
                x = dataset[i]['features'].unsqueeze(0).to(self.device)
                
                # Get cluster probabilities
                if hasattr(model, 'predict'):
                    probs = model.predict(x.view(1, -1))
                else:
                    mu, _ = model.encoder(x.view(1, -1))
                    probs = model.get_cluster_prob(mu)
                
                probs = probs.cpu().numpy()[0]
                assignments.append(np.argmax(probs))
                confidences.append(np.max(probs))
        
        assignments = np.array(assignments)
        confidences = np.array(confidences)
        
        # Plot confidence distribution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Confidence histogram
        ax1.hist(confidences, bins=30, edgecolor='black', alpha=0.7)
        ax1.axvline(confidences.mean(), color='red', linestyle='--', label=f'Mean: {confidences.mean():.3f}')
        ax1.set_xlabel('Cluster Assignment Confidence')
        ax1.set_ylabel('Frequency')
        ax1.set_title('VaDE: Soft Clustering Confidence')
        ax1.legend()
        
        # Cluster distribution
        unique, counts = np.unique(assignments, return_counts=True)
        ax2.bar(unique, counts)
        ax2.set_xlabel('Cluster ID')
        ax2.set_ylabel('Number of Samples')
        ax2.set_title(f'VaDE: Cluster Distribution ({len(unique)} clusters used)')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{model_name}_soft_clustering.png', dpi=150)
        print(f"\n✓ Visualization saved: {self.output_dir / f'{model_name}_soft_clustering.png'}")
        print(f"\n📊 Soft clustering statistics:")
        print(f"  → Mean confidence: {confidences.mean():.3f}")
        print(f"  → Min confidence: {confidences.min():.3f}")
        print(f"  → Max confidence: {confidences.max():.3f}")
        print(f"  → Active clusters: {len(unique)}/50")
        print("  → VaDE provides probabilistic assignments (not hard clustering)")
        
    def compare_all(self):
        """Compare all models"""
        print("\n" + "="*80)
        print("COMPREHENSIVE MODEL COMPARISON")
        print("="*80)
        
        results = {
            'Model': [],
            'Parameters': [],
            'Feature': [],
            'Status': []
        }
        
        models = [
            ('basic_vae', 'Smooth latent space'),
            ('conv_vae', 'Spatial features'),
            ('beta_vae', 'Disentanglement'),
            ('cvae', 'Class separation'),
            ('vade', 'Soft clustering')
        ]
        
        for model_name, feature in models:
            checkpoint = self.checkpoint_dir / model_name / 'best_model.pt'
            
            results['Model'].append(model_name.upper())
            results['Feature'].append(feature)
            
            if checkpoint.exists():
                model = self.load_model(model_name, checkpoint)
                n_params = sum(p.numel() for p in model.parameters())
                results['Parameters'].append(f'{n_params/1e6:.2f}M')
                results['Status'].append('✓ Trained')
            else:
                results['Parameters'].append('N/A')
                results['Status'].append('✗ Not trained')
        
        df = pd.DataFrame(results)
        print("\n" + df.to_string(index=False))
        
        # Save to file
        df.to_csv(self.output_dir / 'model_comparison.csv', index=False)
        print(f"\n✓ Results saved: {self.output_dir / 'model_comparison.csv'}")


def main():
    parser = argparse.ArgumentParser(description='Test and compare VAE models')
    parser.add_argument('--test', type=str, default='all',
                       choices=['smoothness', 'filters', 'disentanglement',
                               'separation', 'soft_clustering', 'all'],
                       help='Which test to run')
    parser.add_argument('--model', type=str, default='conv_vae',
                       help='Model name for specific tests')
    parser.add_argument('--condition', type=str, default='language',
                       choices=['language', 'genre'],
                       help='Condition type for CVAE')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Directory containing model checkpoints')
    parser.add_argument('--output_dir', type=str, default='results/model_tests',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    tester = ModelTester(args.checkpoint_dir, args.output_dir)
    
    if args.test == 'all':
        tester.compare_all()
        tester.test_smoothness('basic_vae')
        tester.test_filters('conv_vae')
        tester.test_disentanglement('beta_vae')
        tester.test_separation('cvae', args.condition)
        tester.test_soft_clustering('vade')
    elif args.test == 'smoothness':
        tester.test_smoothness(args.model)
    elif args.test == 'filters':
        tester.test_filters(args.model)
    elif args.test == 'disentanglement':
        tester.test_disentanglement(args.model)
    elif args.test == 'separation':
        tester.test_separation(args.model, args.condition)
    elif args.test == 'soft_clustering':
        tester.test_soft_clustering(args.model)
    
    print("\n" + "="*80)
    print("✓ ALL TESTS COMPLETE!")
    print("="*80)
    print(f"\nResults saved in: {args.output_dir}/")
    print("\nGenerated files:")
    print("  - model_comparison.csv")
    print("  - *_smoothness.png")
    print("  - *_filters.png")
    print("  - *_disentanglement.png")
    print("  - *_separation_*.png")
    print("  - *_soft_clustering.png")


if __name__ == '__main__':
    main()
