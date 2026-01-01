"""
Comprehensive Model Evaluation System
Q1 Standard: Complete metrics, visualizations, and comparisons
"""

import sys
sys.path.append('.')

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import yaml
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
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
from sklearn.metrics.pairwise import euclidean_distances
import warnings
warnings.filterwarnings('ignore')

from src.data.dataset import MultimodalDataset, get_dataloader
from src.models.vae import BasicVAE
from src.models.conv_vae import ConvVAE
from src.models.beta_vae import BetaVAE, ConditionalVAE
from src.models.vade import VaDE

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


class ModelEvaluator:
    """Complete evaluation pipeline for VAE models"""
    
    def __init__(self, output_dir="results/evaluations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load config
        with open('configs/config.yaml', 'r') as f:
            self.config = yaml.safe_load(f)
        
        print(f"Evaluator initialized on device: {self.device}")
    
    def load_model(self, model_name, model_type, checkpoint_path, dataset):
        """Load trained model and extract latent features"""
        
        cfg = self.config['model']
        latent_dim = cfg['latent_dim']
        hidden_dims = cfg['hidden_dims']
        
        # Get actual input dimension from dataset
        sample = dataset[0]
        if 'features' in sample:
            features = sample['features']
        else:
            features = sample['audio_features']
        
        if features.dim() == 3:
            input_dim = features.size(0) * features.size(1) * features.size(2)
        else:
            input_dim = features.numel()
        
        print(f"  Input dimension: {input_dim}")
        
        # Create model with correct architecture
        if model_type == 'basic':
            model = BasicVAE(input_dim=input_dim, latent_dim=latent_dim, hidden_dims=hidden_dims)
        elif model_type == 'conv':
            model = ConvVAE(input_channels=1, input_height=128, input_width=1292,
                           latent_dim=latent_dim, hidden_channels=hidden_dims)
        elif model_type == 'beta':
            model = BetaVAE(input_channels=1, input_height=128, input_width=1293,
                           latent_dim=latent_dim, hidden_channels=hidden_dims, beta=cfg.get('beta', 4.0))
        elif model_type == 'cvae_lang':
            model = ConditionalVAE(input_channels=1, input_height=128, input_width=1292,
                                  latent_dim=latent_dim, num_classes=4, hidden_channels=hidden_dims)
        elif model_type == 'cvae_genre':
            model = ConditionalVAE(input_channels=1, input_height=128, input_width=1292,
                                  latent_dim=latent_dim, num_classes=3, hidden_channels=hidden_dims)
        elif model_type == 'vade':
            model = VaDE(input_dim=input_dim, latent_dim=latent_dim, n_clusters=15, hidden_dims=hidden_dims)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval()
        
        # Extract features
        return self._extract_features(model, dataset, model_name)
    
    def _extract_features(self, model, dataset, model_name):
        """Extract latent features from model"""
        
        dataloader = get_dataloader(dataset, batch_size=32, shuffle=False, num_workers=0)
        
        latents = []
        languages = []
        genres = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Extracting {model_name}", leave=False):
                if 'features' in batch:
                    x = batch['features'].to(self.device)
                else:
                    x = batch['audio_features'].to(self.device)
                
                # Flatten for BasicVAE and VaDE (non-convolutional models)
                if model_name in ['Basic VAE', 'VaDE']:
                    x = x.flatten(1)
                
                # CVAEs were trained with 1292 width, no padding needed
                
                # Get latent representation
                try:
                    if 'CVAE' in model_name:
                        # Conditional VAE needs condition
                        if 'Language' in model_name:
                            condition = batch['language'].to(self.device)
                        else:  # Genre
                            condition = batch['genre'].to(self.device)
                        mu, logvar = model.encode(x, condition)
                        z = model.reparameterize(mu, logvar)
                    elif hasattr(model, 'encode'):
                        mu, logvar = model.encode(x)
                        z = model.reparameterize(mu, logvar)
                    else:
                        mu, logvar = model.encoder(x)
                        z = model.reparameterize(mu, logvar)
                    latents.append(z.cpu().numpy())
                except Exception as e:
                    print(f"    Warning: Failed to encode batch: {e}")
                    continue
                
                if 'language' in batch:
                    languages.extend(batch['language'].cpu().numpy())
                if 'genre' in batch:
                    genres.extend(batch['genre'].cpu().numpy())
        
        features = {
            'latents': np.vstack(latents),
            'languages': np.array(languages) if languages else None,
            'genres': np.array(genres) if genres else None
        }
        
        return features
    
    def clustering_evaluation(self, features, model_name):
        """Comprehensive clustering evaluation"""
        
        latents = features['latents']
        results = []
        
        # Define clustering tasks
        tasks = []
        if features['languages'] is not None:
            tasks.append(('language', features['languages'], 4))
        if features['genres'] is not None:
            tasks.append(('genre', features['genres'], 3))
        
        for task_name, true_labels, n_clusters in tasks:
            
            # 1. K-Means
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans_labels = kmeans.fit_predict(latents)
            
            # 2. Agglomerative Clustering
            agglo = AgglomerativeClustering(n_clusters=n_clusters)
            agglo_labels = agglo.fit_predict(latents)
            
            # 3. Gaussian Mixture Model
            try:
                gmm = GaussianMixture(n_components=n_clusters, random_state=42, reg_covar=1e-4)
                gmm_labels = gmm.fit_predict(latents)
                gmm_success = True
            except Exception as e:
                print(f"    Warning: GMM failed for {model_name}, skipping GMM results")
                gmm_labels = None
                gmm_success = False
            
            # Evaluate each method
            methods_to_eval = [
                ('kmeans', kmeans_labels),
                ('agglomerative', agglo_labels)
            ]
            if gmm_success:
                methods_to_eval.append(('gmm', gmm_labels))
            
            for method_name, pred_labels in methods_to_eval:
                metrics = {
                    'model': model_name,
                    'task': task_name,
                    'method': method_name,
                    'n_clusters': n_clusters,
                    
                    # Unsupervised metrics
                    'silhouette': silhouette_score(latents, pred_labels),
                    'calinski_harabasz': calinski_harabasz_score(latents, pred_labels),
                    'davies_bouldin': davies_bouldin_score(latents, pred_labels),
                    
                    # Supervised metrics
                    'ari': adjusted_rand_score(true_labels, pred_labels),
                    'nmi': normalized_mutual_info_score(true_labels, pred_labels),
                    'homogeneity': homogeneity_score(true_labels, pred_labels),
                    'completeness': completeness_score(true_labels, pred_labels),
                    'v_measure': v_measure_score(true_labels, pred_labels),
                }
                
                results.append(metrics)
        
        return results
    
    def reconstruction_quality(self, model, dataset, model_name):
        """Evaluate reconstruction quality"""
        
        dataloader = get_dataloader(dataset, batch_size=32, shuffle=False, num_workers=0)
        
        mse_errors = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Reconstruction {model_name}", leave=False):
                if 'features' in batch:
                    x = batch['features'].to(self.device)
                else:
                    x = batch['audio_features'].to(self.device)
                
                try:
                    recon, _, _ = model(x)
                    mse = torch.nn.functional.mse_loss(recon, x, reduction='none')
                    mse = mse.view(mse.size(0), -1).mean(dim=1)
                    mse_errors.extend(mse.cpu().numpy())
                except:
                    continue
        
        return {
            'model': model_name,
            'mean_mse': np.mean(mse_errors),
            'std_mse': np.std(mse_errors),
            'median_mse': np.median(mse_errors),
            'min_mse': np.min(mse_errors),
            'max_mse': np.max(mse_errors)
        }
    
    def save_results(self, results_dict, filename):
        """Save evaluation results"""
        
        # Save as CSV
        df = pd.DataFrame(results_dict)
        csv_path = self.output_dir / f"{filename}.csv"
        df.to_csv(csv_path, index=False)
        
        # Save as JSON
        json_path = self.output_dir / f"{filename}.json"
        with open(json_path, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
        
        print(f"  * Results saved: {csv_path.name}")
        
        return df


def main():
    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL EVALUATION - Q1 STANDARD")
    print("="*80)
    
    evaluator = ModelEvaluator()
    
    # Load dataset (use multimodal dataset for all since it has all 180 samples)
    dataset = MultimodalDataset(data_path="data/features/multimodal_dataset.pkl")
    
    # Define models to evaluate (using actual checkpoint names)
    models = [
        ('Basic VAE', 'basic', 'results/checkpoints/basic/final_model.pt', dataset),
        ('Conv VAE', 'conv', 'results/checkpoints/conv/final_model.pt', dataset),
        ('Beta-VAE', 'beta', 'results/checkpoints/beta/final_model.pt', dataset),
        ('CVAE-Language', 'cvae_lang', 'results/checkpoints/cvae_language/final_model.pt', dataset),
        ('CVAE-Genre', 'cvae_genre', 'results/checkpoints/cvae_genre/final_model.pt', dataset),
        ('VaDE', 'vade', 'results/checkpoints/vade/final_model.pt', dataset),
        ('Multimodal VAE', 'conv', 'results/checkpoints/conv_multimodal/final_model.pt', dataset),
    ]
    
    all_clustering_results = []
    all_reconstruction_results = []
    
    for model_name, model_type, checkpoint, dataset in models:
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print('='*60)
        
        try:
            # Load model and extract features
            features = evaluator.load_model(model_name, model_type, checkpoint, dataset)
            print(f"  Latent features: {features['latents'].shape}")
            
            # Clustering evaluation
            print(f"  Running clustering evaluation...")
            clustering_results = evaluator.clustering_evaluation(features, model_name)
            all_clustering_results.extend(clustering_results)
            
            print(f"  * {model_name} evaluation complete")
            
        except Exception as e:
            print(f"  X Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save all results
    print(f"\n{'='*80}")
    print("SAVING RESULTS")
    print('='*80)
    
    if not all_clustering_results:
        print("\nX No results collected. All evaluations failed.")
        return
    
    clustering_df = evaluator.save_results(all_clustering_results, "clustering_metrics")
    
    # Summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY")
    print('='*80)
    print(f"\nTotal models evaluated: {len(set(r['model'] for r in all_clustering_results))}")
    print(f"Total clustering experiments: {len(all_clustering_results)}")
    
    # Best performers
    print(f"\n{'='*60}")
    print("TOP PERFORMERS")
    print('='*60)
    
    for metric in ['silhouette', 'ari', 'nmi', 'v_measure']:
        best = clustering_df.loc[clustering_df[metric].idxmax()]
        print(f"\nBest {metric.upper()}: {best[metric]:.4f}")
        print(f"  Model: {best['model']}")
        print(f"  Task: {best['task']}, Method: {best['method']}")
    
    print(f"\n{'='*80}")
    print("EVALUATION COMPLETE")
    print(f"Results saved in: {evaluator.output_dir}")
    print('='*80)


if __name__ == "__main__":
    main()
