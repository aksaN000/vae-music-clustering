"""
Run Clustering Experiments on Trained VAE Models
"""

import sys
sys.path.append('.')

import torch
import numpy as np
import argparse
import yaml
import pandas as pd
from pathlib import Path

from src.data.dataset import AudioOnlyDataset, MultimodalDataset, get_dataloader
from src.models.vae import BasicVAE
from src.models.conv_vae import ConvVAE
from src.models.beta_vae import BetaVAE, ConditionalVAE
from src.models.vade import VaDE
from src.clustering.cluster import ClusteringAlgorithms, extract_latent_features
from src.clustering.evaluation import ClusteringMetrics
from src.visualization.plots import ClusterVisualizer


def load_model(model_type: str, checkpoint_path: str, config: dict, device):
    """Load trained model"""
    
    # Determine input dimension (placeholder, will be overwritten)
    input_dim = 128 * 1293
    
    # Create model
    latent_dim = config['latent_dim']
    hidden_dims = config['hidden_dims']
    
    if model_type == 'basic':
        model = BasicVAE(input_dim=input_dim, latent_dim=latent_dim, hidden_dims=hidden_dims)
    elif model_type == 'conv':
        model = ConvVAE(latent_dim=latent_dim, hidden_channels=hidden_dims)
    elif model_type == 'beta':
        model = BetaVAE(latent_dim=latent_dim, hidden_channels=hidden_dims, beta=config.get('beta', 4.0))
    elif model_type == 'cvae':
        model = ConditionalVAE(latent_dim=latent_dim, num_classes=5, hidden_channels=hidden_dims)
    elif model_type == 'vade':
        model = VaDE(input_dim=input_dim, latent_dim=latent_dim, n_clusters=config.get('n_clusters', 15))
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model


def main(args):
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Load dataset
    print("Loading dataset...")
    if args.modality == 'audio':
        dataset = AudioOnlyDataset(data_path="data/features/audio_only_dataset.pkl")
    else:
        dataset = MultimodalDataset(data_path="data/features/multimodal_dataset.pkl")
    
    dataloader = get_dataloader(dataset, batch_size=32, shuffle=False, num_workers=8)
    print(f"Dataset size: {len(dataset)}\n")
    
    # Load model
    print(f"Loading {args.model} model from {args.checkpoint}...")
    model = load_model(args.model, args.checkpoint, config['model'], device)
    print(f"Model loaded successfully\n")
    
    # Extract latent features
    print("Extracting latent features...")
    features, labels_dict = extract_latent_features(model, dataloader, device)
    print(f"Features shape: {features.shape}")
    print(f"Labels: {list(labels_dict.keys())}\n")
    
    # Initialize clustering and evaluation
    clustering_alg = ClusteringAlgorithms()
    evaluator = ClusteringMetrics()
    visualizer = ClusterVisualizer(
        output_dir=f"results/visualizations/{args.model}/clustering"
    )
    
    # Results storage
    all_results = []
    
    # Get ground truth labels for evaluation
    if 'language' in labels_dict and len(labels_dict['language']) > 0:
        true_labels = labels_dict['language']
        has_labels = True
        print("Using language labels for supervised evaluation\n")
    else:
        true_labels = None
        has_labels = False
        print("No labels available, using only unsupervised metrics\n")
    
    # Run clustering with different methods and n_clusters
    for method in config['clustering']['methods']:
        print(f"\n{'='*60}")
        print(f"Method: {method.upper()}")
        print('='*60)
        
        if method == 'dbscan':
            # DBSCAN doesn't need n_clusters
            print(f"  Running DBSCAN with eps={config['clustering']['dbscan_eps']}...")
            
            pred_labels, info = clustering_alg.dbscan(
                features,
                eps=config['clustering']['dbscan_eps'],
                min_samples=config['clustering']['dbscan_min_samples']
            )
            
            print(f"  Found {info['n_clusters']} clusters")
            print(f"  Noise points: {info['n_noise']}")
            
            # Evaluate
            if info['n_clusters'] > 1:
                metrics = evaluator.evaluate_all(features, pred_labels, true_labels if has_labels else None)
                metrics['method'] = f"{method}"
                metrics['n_clusters'] = info['n_clusters']
                all_results.append(metrics)
                
                evaluator.print_metrics(metrics, title=f"{method.upper()} Results")
            
        else:
            # Methods that need n_clusters
            for n_clusters in config['clustering']['n_clusters']:
                print(f"\n  n_clusters = {n_clusters}")
                
                try:
                    pred_labels, info = clustering_alg.cluster(
                        features,
                        method=method,
                        n_clusters=n_clusters
                    )
                    
                    # Evaluate
                    metrics = evaluator.evaluate_all(features, pred_labels, true_labels if has_labels else None)
                    metrics['method'] = f"{method}_k{n_clusters}"
                    metrics['n_clusters'] = n_clusters
                    all_results.append(metrics)
                    
                    evaluator.print_metrics(metrics, title=f"{method.upper()} (k={n_clusters}) Results")
                    
                    # Visualize best performing
                    if n_clusters == 5:  # Number of languages
                        # UMAP visualization
                        if has_labels:
                            lang_names = {0: 'Arabic', 1: 'Bangla', 2: 'English', 3: 'Hindi', 4: 'Spanish'}
                            visualizer.plot_latent_space_2d(
                                features,
                                true_labels,
                                method='umap',
                                title=f'{method.upper()} - Ground Truth (Languages)',
                                label_names=lang_names,
                                save_name=f'{method}_k{n_clusters}_ground_truth_umap.png'
                            )
                        
                        visualizer.plot_latent_space_2d(
                            features,
                            pred_labels,
                            method='umap',
                            title=f'{method.upper()} Clustering (k={n_clusters})',
                            save_name=f'{method}_k{n_clusters}_predicted_umap.png'
                        )
                        
                        # Cluster distribution
                        if has_labels:
                            visualizer.plot_cluster_distribution(
                                pred_labels,
                                true_labels,
                                gt_names=['Arabic', 'Bangla', 'English', 'Hindi', 'Spanish'],
                                save_name=f'{method}_k{n_clusters}_distribution.png'
                            )
                
                except Exception as e:
                    print(f"  Error: {e}")
                    continue
    
    # Save results
    results_df = pd.DataFrame(all_results)
    results_path = Path(f"results/metrics/{args.model}_clustering_results.csv")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_path, index=False)
    print(f"\n\nResults saved to: {results_path}")
    
    # Plot comparison
    if len(results_df) > 0:
        visualizer.plot_metrics_comparison(
            results_df,
            save_name='metrics_comparison.png'
        )
    
    # Print best results
    print("\n" + "="*60)
    print("BEST RESULTS")
    print("="*60)
    
    if 'silhouette_score' in results_df.columns:
        best_silhouette = results_df.loc[results_df['silhouette_score'].idxmax()]
        print(f"\nBest Silhouette Score: {best_silhouette['silhouette_score']:.4f}")
        print(f"  Method: {best_silhouette['method']}")
    
    if 'adjusted_rand_index' in results_df.columns:
        best_ari = results_df.loc[results_df['adjusted_rand_index'].idxmax()]
        print(f"\nBest ARI: {best_ari['adjusted_rand_index']:.4f}")
        print(f"  Method: {best_ari['method']}")
    
    print("\nClustering complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run clustering on VAE latent space")
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--model', type=str, required=True,
                       choices=['basic', 'conv', 'beta', 'cvae', 'vade'])
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--modality', type=str, default='audio',
                       choices=['audio', 'multimodal'])
    
    args = parser.parse_args()
    main(args)
