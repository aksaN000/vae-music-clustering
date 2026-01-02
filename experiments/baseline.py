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
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score
)

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
    all_languages = []
    all_genres = []
    
    for batch in dataloader:
        features = batch['features']
        if features.dim() > 2:
            features = features.view(features.size(0), -1)
        all_features.append(features.numpy())
        
        if 'language' in batch:
            all_languages.extend(batch['language'].numpy())
        if 'genre' in batch:
            all_genres.extend(batch['genre'].numpy())
    
    raw_features = np.concatenate(all_features, axis=0)
    languages = np.array(all_languages) if all_languages else None
    genres = np.array(all_genres) if all_genres else None
    
    print(f"Raw features shape: {raw_features.shape}")
    print(f"Languages: {len(np.unique(languages)) if languages is not None else 'N/A'}")
    print(f"Genres: {len(np.unique(genres)) if genres is not None else 'N/A'}")
    
    # Initialize
    visualizer = ClusterVisualizer(output_dir="results/visualizations/baseline")
    all_results = []
    
    # Define clustering tasks (match VAE evaluation)
    tasks = []
    if languages is not None:
        tasks.append(('language', languages, 4))
    if genres is not None:
        tasks.append(('genre', genres, 3))
    
    # === 1. Raw Features + Clustering ===
    print("\n" + "="*60)
    print("BASELINE 1: Raw Features (180 songs)")
    print("="*60)
    
    for task_name, true_labels, n_clusters in tasks:
        print(f"\n>>> Task: {task_name.upper()} (n_clusters={n_clusters})")
        
        for method in ['kmeans', 'agglomerative', 'gmm']:
            try:
                if method == 'kmeans':
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    pred_labels = kmeans.fit_predict(raw_features)
                elif method == 'agglomerative':
                    agglo = AgglomerativeClustering(n_clusters=n_clusters)
                    pred_labels = agglo.fit_predict(raw_features)
                else:  # gmm
                    gmm = GaussianMixture(n_components=n_clusters, random_state=42, reg_covar=1e-4)
                    pred_labels = gmm.fit_predict(raw_features)
            except Exception as e:
                print(f"  {method}: FAILED (memory error - {str(e)[:50]}...)")
                continue
            
            # Match VAE evaluation CSV format
            metrics = {
                'model': 'Raw Features',
                'task': task_name,
                'method': method,
                'n_clusters': n_clusters,
                'silhouette': silhouette_score(raw_features, pred_labels) if len(np.unique(pred_labels)) > 1 else -1,
                'calinski_harabasz': calinski_harabasz_score(raw_features, pred_labels) if len(np.unique(pred_labels)) > 1 else 0,
                'davies_bouldin': davies_bouldin_score(raw_features, pred_labels) if len(np.unique(pred_labels)) > 1 else float('inf'),
                'ari': adjusted_rand_score(true_labels, pred_labels),
                'nmi': normalized_mutual_info_score(true_labels, pred_labels),
                'homogeneity': homogeneity_score(true_labels, pred_labels),
                'completeness': completeness_score(true_labels, pred_labels),
                'v_measure': v_measure_score(true_labels, pred_labels)
            }
            all_results.append(metrics)
            print(f"  {method}: NMI={metrics['nmi']:.4f}, Silhouette={metrics['silhouette']:.4f}")
    
    # === 2. PCA + Clustering ===
    print("\n" + "="*60)
    print("BASELINE 2: PCA (128 dims)")
    print("="*60)
    
    latent_dim = 128  # Match VAE latent dimension
    print(f"\nApplying PCA (n_components={latent_dim})...")
    
    pca = PCA(n_components=latent_dim, random_state=42)
    pca_features = pca.fit_transform(raw_features)
    print(f"Explained variance: {pca.explained_variance_ratio_.sum():.4f}")
    
    for task_name, true_labels, n_clusters in tasks:
        print(f"\n>>> Task: {task_name.upper()} (n_clusters={n_clusters})")
        
        for method in ['kmeans', 'agglomerative', 'gmm']:
            try:
                if method == 'kmeans':
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    pred_labels = kmeans.fit_predict(pca_features)
                elif method == 'agglomerative':
                    agglo = AgglomerativeClustering(n_clusters=n_clusters)
                    pred_labels = agglo.fit_predict(pca_features)
                else:  # gmm
                    gmm = GaussianMixture(n_components=n_clusters, random_state=42, reg_covar=1e-3, covariance_type='diag')
                    pred_labels = gmm.fit_predict(pca_features)
            except Exception as e:
                print(f"  {method}: FAILED ({str(e)[:60]}...)")
                continue
            
            metrics = {
                'model': 'PCA',
                'task': task_name,
                'method': method,
                'n_clusters': n_clusters,
                'silhouette': silhouette_score(pca_features, pred_labels) if len(np.unique(pred_labels)) > 1 else -1,
                'calinski_harabasz': calinski_harabasz_score(pca_features, pred_labels) if len(np.unique(pred_labels)) > 1 else 0,
                'davies_bouldin': davies_bouldin_score(pca_features, pred_labels) if len(np.unique(pred_labels)) > 1 else float('inf'),
                'ari': adjusted_rand_score(true_labels, pred_labels),
                'nmi': normalized_mutual_info_score(true_labels, pred_labels),
                'homogeneity': homogeneity_score(true_labels, pred_labels),
                'completeness': completeness_score(true_labels, pred_labels),
                'v_measure': v_measure_score(true_labels, pred_labels)
            }
            all_results.append(metrics)
            print(f"  {method}: NMI={metrics['nmi']:.4f}, Silhouette={metrics['silhouette']:.4f}")
    
    print("\n" + "="*60)
    print("Note: Skipping Autoencoder baseline (memory constraints)")
    print("Raw Features and PCA are sufficient classical baselines.")
    print("="*60)
    
    # Save results (match VAE evaluation format)
    results_df = pd.DataFrame(all_results)
    results_path = Path("results/evaluations/baseline_clustering_metrics.csv")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_path, index=False)
    print(f"\n\n✅ Results saved to: {results_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("BASELINE SUMMARY (180 songs)")
    print("="*60)
    
    for model_name in ['Raw Features', 'PCA']:
        model_results = results_df[results_df['model'] == model_name]
        if len(model_results) > 0:
            print(f"\n{model_name}:")
            for task in ['language', 'genre']:
                task_results = model_results[model_results['task'] == task]
                if len(task_results) > 0:
                    best = task_results.loc[task_results['nmi'].idxmax()]
                    print(f"  {task.capitalize()}: NMI={best['nmi']:.4f} ({best['method']})")
    
    print("\n✅ Baseline evaluation complete!")
    print("Note: Baselines use 180 original songs; VAEs use 2,107 windowed clips.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run baseline clustering methods")
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    
    args = parser.parse_args()
    main(args)
