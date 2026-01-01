"""
Universal Evaluation Script for All VAE Models
Handles: basic, conv, conv_multimodal, beta, cvae, vade
"""

import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    normalized_mutual_info_score, adjusted_rand_score
)
from pathlib import Path
import pickle
import json
from datetime import datetime
import sys
import argparse

from src.models.vae import BasicVAE
from src.models.conv_vae import ConvVAE
from src.models.beta_vae import BetaVAE, ConditionalVAE
from src.models.vade import VaDE
from src.data.dataset import AudioOnlyDataset, MultimodalDataset


def load_model_and_data(model_name='basic', checkpoint_name='best_model.pt', data_path=None):
    """Universal model and data loader - reads checkpoint to get actual config"""
    
    # Determine data path
    if data_path is None:
        data_path = "data/features/audio_only_dataset.pkl"
    
    checkpoint_dir = Path(f'results/checkpoints/{model_name}')
    checkpoint_path = checkpoint_dir / checkpoint_name
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Get config from checkpoint or use defaults
    model_config = checkpoint.get('config', {})
    hidden_dims = model_config.get('hidden_dims', [16, 32, 64])
    
    print(f"Model config: hidden_dims={hidden_dims}")
    
    # Initialize model based on type
    if model_name == 'basic':
        model = BasicVAE(
            input_dim=165376,
            latent_dim=128,
            hidden_dims=hidden_dims,
            dropout=0.2
        )
        dataset = AudioOnlyDataset(
            data_path=data_path,
            feature_type='melspec'
        )
        
    elif model_name == 'conv':
        model = ConvVAE(
            input_channels=1,
            input_height=128,
            input_width=1292,
            latent_dim=128,
            hidden_channels=hidden_dims,
            dropout=0.2
        )
        dataset = AudioOnlyDataset(
            data_path=data_path,
            feature_type='melspec'
        )
        
    elif model_name == 'beta':
        model = BetaVAE(
            input_channels=1,
            input_height=128,
            input_width=1292,
            latent_dim=128,
            hidden_channels=hidden_dims,
            beta=4.0,
            dropout=0.2
        )
        dataset = AudioOnlyDataset(
            data_path=data_path,
            feature_type='melspec'
        )
        
    elif model_name in ['cvae', 'cvae_genre', 'cvae_language']:
        # Determine number of classes based on condition type
        if 'genre' in model_name:
            num_classes = 45  # 45 genres
        else:
            num_classes = 5   # 5 languages (default)
        
        model = ConditionalVAE(
            input_channels=1,
            input_height=128,
            input_width=1292,
            latent_dim=128,
            num_classes=num_classes,
            hidden_channels=hidden_dims,
            dropout=0.2
        )
        dataset = AudioOnlyDataset(
            data_path=data_path,
            feature_type='melspec'
        )
        
    elif model_name == 'vade':
        # VaDE uses fully connected, get actual dimensions from checkpoint
        state_dict = checkpoint['model_state_dict']
        
        # Infer hidden dims from encoder layers
        encoder_dims = []
        i = 0
        while f'encoder.{i}.weight' in state_dict:
            encoder_dims.append(state_dict[f'encoder.{i}.weight'].shape[0])
            i += 4  # Skip batchnorm, relu, dropout
        
        # Get n_clusters from GMM parameters
        n_clusters = state_dict['pi'].shape[0] if 'pi' in state_dict else 10
        
        print(f"VaDE config from checkpoint: hidden_dims={encoder_dims}, n_clusters={n_clusters}")
        
        model = VaDE(
            input_dim=165376,
            latent_dim=128,
            n_clusters=n_clusters,
            hidden_dims=encoder_dims if encoder_dims else [16, 32, 64],
            dropout=0.2
        )
        dataset = AudioOnlyDataset(
            data_path=data_path,
            feature_type='melspec'
        )
        
    elif model_name == 'conv_multimodal':
        model = ConvVAE(
            input_channels=1,
            input_height=128,
            input_width=1292,
            latent_dim=128,
            hidden_channels=hidden_dims,
            dropout=0.2
        )
        dataset = MultimodalDataset(
            data_path="data/features/multimodal_dataset.pkl",
            feature_type='melspec'
        )
        
    else:
        raise ValueError(f"Unknown model type: {model_name}")
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    
    print(f"✓ Model loaded: {checkpoint.get('epoch', 'N/A')} epochs")
    print(f"✓ Best val loss: {checkpoint.get('best_val_loss', 'N/A'):.4f}")
    print(f"✓ Dataset: {len(dataset)} samples")
    
    return model, dataset, checkpoint


def extract_latent_vectors(model, dataset, model_name, device='cuda'):
    """Extract latent representations - handles all model types"""
    model = model.to(device)
    model.eval()
    
    latent_vectors = []
    languages = []
    genres = []
    
    print("\n🔄 Extracting latent representations...")
    
    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            
            # Get features based on dataset type
            if 'audio_features' in sample:
                # Multimodal dataset
                x = sample['audio_features'].unsqueeze(0).to(device)
            else:
                # Audio-only dataset
                x = sample['features'].unsqueeze(0).to(device)
            
            # Flatten for BasicVAE and VaDE
            if model_name in ['basic', 'vade']:
                if x.dim() > 2:
                    x = x.view(x.size(0), -1)
            
            # Extract latent (handle conditional models)
            if model_name in ['cvae', 'cvae_genre', 'cvae_language']:
                # ConditionalVAE needs condition parameter
                if 'genre' in model_name:
                    condition = sample['genre'] if 'genre' in sample else torch.tensor([0])
                else:
                    condition = sample['language'] if 'language' in sample else torch.tensor([0])
                
                if not torch.is_tensor(condition):
                    condition = torch.tensor([condition])
                condition = condition.to(device)
                mu, _ = model.encode(x, condition)
            else:
                mu, _ = model.encode(x)
            
            latent_vectors.append(mu.cpu().numpy())
            
            # Get labels
            if 'language' in sample:
                lang = sample['language']
                languages.append(lang.item() if torch.is_tensor(lang) else lang)
            if 'genre' in sample:
                genre = sample['genre']
                genres.append(genre.item() if torch.is_tensor(genre) else genre)
    
    latent_vectors = np.concatenate(latent_vectors, axis=0)
    print(f"✓ Extracted {latent_vectors.shape[0]} vectors × {latent_vectors.shape[1]} dims")
    
    return latent_vectors, np.array(languages), np.array(genres)


def compute_cluster_purity(clusters, true_labels):
    """Compute cluster purity"""
    if len(true_labels) == 0:
        return 0.0
    
    total_correct = 0
    for cluster_id in np.unique(clusters):
        cluster_mask = clusters == cluster_id
        cluster_labels = true_labels[cluster_mask]
        if len(cluster_labels) > 0:
            most_common = np.bincount(cluster_labels).argmax()
            total_correct += np.sum(cluster_labels == most_common)
    
    return total_correct / len(true_labels)


def evaluate_clustering(latent_vectors, languages, genres, n_clusters_range=[5, 10, 15]):
    """Comprehensive clustering evaluation"""
    
    results = {
        'kmeans': {},
        'hierarchical': {},
        'gmm': {},
        'dbscan': {},
        'pca_baseline': {}
    }
    
    print("\n📊 Running clustering algorithms...")
    
    # PCA Baseline
    print("\n  → PCA + K-Means baseline...")
    pca = PCA(n_components=50)
    pca_features = pca.fit_transform(latent_vectors)
    
    for n_clusters in n_clusters_range:
        kmeans_pca = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        pca_clusters = kmeans_pca.fit_predict(pca_features)
        
        results['pca_baseline'][n_clusters] = {
            'silhouette': silhouette_score(pca_features, pca_clusters),
            'calinski_harabasz': calinski_harabasz_score(pca_features, pca_clusters),
            'davies_bouldin': davies_bouldin_score(pca_features, pca_clusters),
            'language_purity': compute_cluster_purity(pca_clusters, languages) if len(languages) > 0 else 0,
            'genre_purity': compute_cluster_purity(pca_clusters, genres) if len(genres) > 0 else 0,
            'language_nmi': normalized_mutual_info_score(languages, pca_clusters) if len(languages) > 0 else 0,
            'language_ari': adjusted_rand_score(languages, pca_clusters) if len(languages) > 0 else 0
        }
    
    # K-Means on VAE latent space
    print("  → K-Means...")
    for n_clusters in n_clusters_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(latent_vectors)
        
        results['kmeans'][n_clusters] = {
            'silhouette': silhouette_score(latent_vectors, clusters),
            'calinski_harabasz': calinski_harabasz_score(latent_vectors, clusters),
            'davies_bouldin': davies_bouldin_score(latent_vectors, clusters),
            'language_purity': compute_cluster_purity(clusters, languages) if len(languages) > 0 else 0,
            'genre_purity': compute_cluster_purity(clusters, genres) if len(genres) > 0 else 0,
            'language_nmi': normalized_mutual_info_score(languages, clusters) if len(languages) > 0 else 0,
            'language_ari': adjusted_rand_score(languages, clusters) if len(languages) > 0 else 0
        }
    
    # Hierarchical
    print("  → Hierarchical...")
    for n_clusters in n_clusters_range:
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
        clusters = hierarchical.fit_predict(latent_vectors)
        
        results['hierarchical'][n_clusters] = {
            'silhouette': silhouette_score(latent_vectors, clusters),
            'calinski_harabasz': calinski_harabasz_score(latent_vectors, clusters),
            'davies_bouldin': davies_bouldin_score(latent_vectors, clusters),
            'language_purity': compute_cluster_purity(clusters, languages) if len(languages) > 0 else 0,
            'genre_purity': compute_cluster_purity(clusters, genres) if len(genres) > 0 else 0,
            'language_nmi': normalized_mutual_info_score(languages, clusters) if len(languages) > 0 else 0,
            'language_ari': adjusted_rand_score(languages, clusters) if len(languages) > 0 else 0
        }
    
    # Gaussian Mixture
    print("  → Gaussian Mixture Models...")
    for n_clusters in n_clusters_range:
        gmm = GaussianMixture(n_components=n_clusters, random_state=42)
        clusters = gmm.fit_predict(latent_vectors)
        
        results['gmm'][n_clusters] = {
            'silhouette': silhouette_score(latent_vectors, clusters),
            'calinski_harabasz': calinski_harabasz_score(latent_vectors, clusters),
            'davies_bouldin': davies_bouldin_score(latent_vectors, clusters),
            'language_purity': compute_cluster_purity(clusters, languages) if len(languages) > 0 else 0,
            'genre_purity': compute_cluster_purity(clusters, genres) if len(genres) > 0 else 0,
            'language_nmi': normalized_mutual_info_score(languages, clusters) if len(languages) > 0 else 0,
            'language_ari': adjusted_rand_score(languages, clusters) if len(languages) > 0 else 0
        }
    
    # DBSCAN
    print("  → DBSCAN...")
    for eps in [0.5, 1.0, 2.0]:
        dbscan = DBSCAN(eps=eps, min_samples=5)
        clusters = dbscan.fit_predict(latent_vectors)
        
        n_clusters_found = len(set(clusters)) - (1 if -1 in clusters else 0)
        
        if n_clusters_found > 1:
            mask = clusters != -1
            results['dbscan'][f'eps_{eps}'] = {
                'n_clusters': n_clusters_found,
                'n_noise': np.sum(clusters == -1),
                'silhouette': silhouette_score(latent_vectors[mask], clusters[mask]) if np.sum(mask) > 1 else 0,
                'language_purity': compute_cluster_purity(clusters[mask], languages[mask]) if len(languages) > 0 and np.sum(mask) > 0 else 0,
                'genre_purity': compute_cluster_purity(clusters[mask], genres[mask]) if len(genres) > 0 and np.sum(mask) > 0 else 0
            }
        else:
            results['dbscan'][f'eps_{eps}'] = {'n_clusters': n_clusters_found, 'note': 'Too few clusters'}
    
    return results


def visualize_latent_space(latent_vectors, languages, genres, model_name, output_dir):
    """Create visualizations"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n🎨 Creating visualizations...")
    
    # t-SNE visualization
    print("  → t-SNE projection...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(latent_vectors) - 1))
    tsne_coords = tsne.fit_transform(latent_vectors)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # By language
    if len(languages) > 0:
        scatter = axes[0].scatter(tsne_coords[:, 0], tsne_coords[:, 1], 
                                 c=languages, cmap='tab10', alpha=0.6, s=20)
        axes[0].set_title(f'{model_name.upper()} - Latent Space by Language', fontsize=14)
        axes[0].set_xlabel('t-SNE 1')
        axes[0].set_ylabel('t-SNE 2')
        plt.colorbar(scatter, ax=axes[0], label='Language')
    
    # By genre
    if len(genres) > 0:
        scatter = axes[1].scatter(tsne_coords[:, 0], tsne_coords[:, 1], 
                                 c=genres, cmap='tab20', alpha=0.6, s=20)
        axes[1].set_title(f'{model_name.upper()} - Latent Space by Genre', fontsize=14)
        axes[1].set_xlabel('t-SNE 1')
        axes[1].set_ylabel('t-SNE 2')
        plt.colorbar(scatter, ax=axes[1], label='Genre')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'latent_space_tsne.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {output_dir / 'latent_space_tsne.png'}")


def save_results(results, model_name, checkpoint_info, output_dir):
    """Save evaluation results"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to Python native types for JSON serialization
    def convert_to_native(obj):
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_native(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    # Save metrics JSON
    output_file = output_dir / 'metrics.json'
    
    full_results = {
        'model': model_name,
        'timestamp': datetime.now().isoformat(),
        'checkpoint': {
            'epoch': checkpoint_info.get('epoch', 'N/A'),
            'val_loss': float(checkpoint_info.get('best_val_loss', 0)),
        },
        'clustering_metrics': convert_to_native(results)
    }
    
    with open(output_file, 'w') as f:
        json.dump(full_results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # Print summary
    print("\n" + "="*70)
    print(f"📈 EVALUATION SUMMARY - {model_name.upper()}")
    print("="*70)
    
    for algo in ['kmeans', 'hierarchical', 'gmm']:
        if algo in results and results[algo]:
            print(f"\n{algo.upper()}:")
            for k, metrics in results[algo].items():
                print(f"  n_clusters={k}:")
                print(f"    Silhouette: {metrics['silhouette']:.4f}")
                print(f"    Davies-Bouldin: {metrics['davies_bouldin']:.4f}")
                print(f"    Language Purity: {metrics['language_purity']:.4f}")
                print(f"    Language NMI: {metrics['language_nmi']:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate VAE models')
    parser.add_argument('--model', type=str, required=True,
                       choices=['basic', 'conv', 'conv_multimodal', 'beta', 'cvae', 
                               'cvae_genre', 'cvae_language', 'vade'],
                       help='Model type to evaluate')
    parser.add_argument('--checkpoint', type=str, default='best_model.pt',
                       help='Checkpoint filename')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to dataset pickle file (default: data/features/audio_only_dataset.pkl or multimodal_dataset.pkl)')
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"🔬 EVALUATING {args.model.upper()} MODEL")
    print(f"{'='*70}\n")
    
    # Load model and data
    model, dataset, checkpoint = load_model_and_data(args.model, args.checkpoint, args.data_path)
    
    # Extract latent representations
    latent_vectors, languages, genres = extract_latent_vectors(
        model, dataset, args.model, device=args.device
    )
    
    # Run clustering evaluation
    results = evaluate_clustering(latent_vectors, languages, genres)
    
    # Create visualizations
    output_dir = f'results/evaluations/{args.model}'
    visualize_latent_space(latent_vectors, languages, genres, args.model, output_dir)
    
    # Save results
    save_results(results, args.model, checkpoint, output_dir)
    
    print(f"\n✅ Evaluation complete for {args.model}!")
    print(f"📁 Results saved to: {output_dir}\n")


if __name__ == '__main__':
    main()
