"""
Baseline Comparison: PCA + K-Means, Autoencoder + K-Means
"""

import sys
sys.path.append('.')

import torch
import numpy as np
import argparse
import yaml
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA

from src.data.dataset import AudioOnlyDataset, get_dataloader
from src.models.vae import Autoencoder
from src.clustering.cluster import ClusteringAlgorithms
from src.clustering.evaluation import ClusteringMetrics
from src.visualization.plots import ClusterVisualizer


def train_autoencoder(train_loader, input_dim, latent_dim, device, epochs=50):
    """Train standard autoencoder"""
    model = Autoencoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=[512, 256]
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("Training Autoencoder...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            x = batch['features'].to(device)
            
            # Flatten
            if x.dim() > 2:
                x = x.view(x.size(0), -1)
            
            recon, z = model(x)
            loss = torch.nn.functional.mse_loss(recon, x)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")
    
    return model


def extract_ae_features(model, dataloader, device):
    """Extract features from autoencoder"""
    model.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            x = batch['features'].to(device)
            
            if x.dim() > 2:
                x = x.view(x.size(0), -1)
            
            z = model.encode(x)
            all_features.append(z.cpu().numpy())
            
            if 'language' in batch:
                all_labels.extend(batch['language'].cpu().numpy())
    
    features = np.concatenate(all_features, axis=0)
    labels = np.array(all_labels) if all_labels else None
    
    return features, labels


def main(args):
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Load dataset
    print("Loading dataset...")
    dataset = AudioOnlyDataset(data_path="data/features/audio_only_dataset.pkl")
    dataloader = get_dataloader(dataset, batch_size=32, shuffle=False, num_workers=8)
    
    # Get a sample to determine input dimension
    sample = dataset[0]['features']
    if sample.dim() == 3:
        input_dim = sample.size(0) * sample.size(1) * sample.size(2)
    else:
        input_dim = sample.numel()
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Input dimension: {input_dim}\n")
    
    # Extract all features for raw clustering and PCA
    print("Extracting raw features...")
    all_features = []
    all_labels = []
    
    for batch in dataloader:
        features = batch['features']
        if features.dim() > 2:
            features = features.view(features.size(0), -1)
        all_features.append(features.numpy())
        
        if 'language' in batch:
            all_labels.extend(batch['language'].numpy())
    
    raw_features = np.concatenate(all_features, axis=0)
    true_labels = np.array(all_labels) if all_labels else None
    has_labels = true_labels is not None
    
    print(f"Raw features shape: {raw_features.shape}")
    
    # Initialize
    clustering_alg = ClusteringAlgorithms()
    evaluator = ClusteringMetrics()
    visualizer = ClusterVisualizer(output_dir="results/visualizations/baseline")
    
    all_results = []
    
    # === 1. Raw Features + K-Means ===
    print("\n" + "="*60)
    print("BASELINE 1: Raw Features + K-Means")
    print("="*60)
    
    for n_clusters in [5, 10, 15]:
        print(f"\n  k = {n_clusters}")
        pred_labels, _ = clustering_alg.kmeans(raw_features, n_clusters, n_init=10)
        
        metrics = evaluator.evaluate_all(raw_features, pred_labels, true_labels if has_labels else None)
        metrics['method'] = f'raw_kmeans_k{n_clusters}'
        metrics['n_clusters'] = n_clusters
        all_results.append(metrics)
        
        evaluator.print_metrics(metrics, f"Raw K-Means (k={n_clusters})")
    
    # === 2. PCA + K-Means ===
    print("\n" + "="*60)
    print("BASELINE 2: PCA + K-Means")
    print("="*60)
    
    latent_dim = config['baseline']['pca_components']
    print(f"\nApplying PCA (n_components={latent_dim})...")
    
    pca = PCA(n_components=latent_dim, random_state=42)
    pca_features = pca.fit_transform(raw_features)
    
    print(f"Explained variance: {pca.explained_variance_ratio_.sum():.4f}")
    
    for n_clusters in [5, 10, 15]:
        print(f"\n  k = {n_clusters}")
        pred_labels, _ = clustering_alg.kmeans(pca_features, n_clusters, n_init=10)
        
        metrics = evaluator.evaluate_all(pca_features, pred_labels, true_labels if has_labels else None)
        metrics['method'] = f'pca_kmeans_k{n_clusters}'
        metrics['n_clusters'] = n_clusters
        all_results.append(metrics)
        
        evaluator.print_metrics(metrics, f"PCA + K-Means (k={n_clusters})")
        
        if n_clusters == 5:
            visualizer.plot_latent_space_2d(
                pca_features,
                pred_labels,
                method='umap',
                title=f'PCA + K-Means (k={n_clusters})',
                save_name=f'pca_kmeans_k{n_clusters}_umap.png'
            )
    
    # === 3. Autoencoder + K-Means ===
    print("\n" + "="*60)
    print("BASELINE 3: Autoencoder + K-Means")
    print("="*60)
    
    # Train autoencoder
    train_loader = get_dataloader(dataset, batch_size=32, shuffle=True, num_workers=8)
    ae_model = train_autoencoder(train_loader, input_dim, latent_dim, device, epochs=50)
    
    # Extract features
    print("\nExtracting autoencoder features...")
    ae_features, _ = extract_ae_features(ae_model, dataloader, device)
    print(f"AE features shape: {ae_features.shape}")
    
    for n_clusters in [5, 10, 15]:
        print(f"\n  k = {n_clusters}")
        pred_labels, _ = clustering_alg.kmeans(ae_features, n_clusters, n_init=10)
        
        metrics = evaluator.evaluate_all(ae_features, pred_labels, true_labels if has_labels else None)
        metrics['method'] = f'ae_kmeans_k{n_clusters}'
        metrics['n_clusters'] = n_clusters
        all_results.append(metrics)
        
        evaluator.print_metrics(metrics, f"AE + K-Means (k={n_clusters})")
        
        if n_clusters == 5:
            visualizer.plot_latent_space_2d(
                ae_features,
                pred_labels,
                method='umap',
                title=f'Autoencoder + K-Means (k={n_clusters})',
                save_name=f'ae_kmeans_k{n_clusters}_umap.png'
            )
    
    # Save results
    results_df = pd.DataFrame(all_results)
    results_path = Path("results/metrics/baseline_results.csv")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_path, index=False)
    print(f"\n\nResults saved to: {results_path}")
    
    # Plot comparison
    visualizer.plot_metrics_comparison(
        results_df,
        save_name='baseline_comparison.png'
    )
    
    # Print summary
    print("\n" + "="*60)
    print("BASELINE SUMMARY")
    print("="*60)
    
    for method_prefix in ['raw_kmeans', 'pca_kmeans', 'ae_kmeans']:
        method_results = results_df[results_df['method'].str.startswith(method_prefix)]
        if len(method_results) > 0 and 'silhouette_score' in method_results.columns:
            best = method_results.loc[method_results['silhouette_score'].idxmax()]
            print(f"\n{method_prefix.upper()}:")
            print(f"  Best Silhouette: {best['silhouette_score']:.4f} (k={int(best['n_clusters'])})")
            if 'adjusted_rand_index' in best:
                print(f"  ARI: {best['adjusted_rand_index']:.4f}")
    
    print("\nBaseline comparison complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run baseline clustering methods")
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    
    args = parser.parse_args()
    main(args)
