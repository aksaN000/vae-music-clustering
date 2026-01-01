"""
Individual Model Evaluation Script
Generates detailed evaluation reports and visualizations for each trained model
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    adjusted_rand_score, normalized_mutual_info_score, v_measure_score,
    confusion_matrix, classification_report
)
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import models
from src.models.vae import BasicVAE
from src.models.conv_vae import ConvVAE
from src.models.beta_vae import BetaVAE, ConditionalVAE
from src.models.vade import VaDE
from src.data.dataset import MultimodalDataset


class IndividualModelEvaluator:
    """Evaluate individual models and generate detailed reports"""
    
    def __init__(self, checkpoint_dir, results_dir, device='cuda'):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.results_dir = Path(results_dir)
        self.device = device
        
        # Model configurations
        self.models_config = {
            'basic': {'class': BasicVAE, 'name': 'Basic VAE', 'input_dim': 165376, 'latent_dim': 128, 'hidden_dims': [16, 32, 64]},
            'conv': {'class': ConvVAE, 'name': 'Conv VAE', 'input_channels': 1, 'input_height': 128, 'input_width': 1292, 'latent_dim': 128, 'hidden_channels': [16, 32, 64]},
            'beta': {'class': BetaVAE, 'name': 'Beta-VAE', 'input_channels': 1, 'input_height': 128, 'input_width': 1293, 'latent_dim': 128, 'hidden_channels': [16, 32, 64]},
            'cvae_language': {'class': ConditionalVAE, 'name': 'CVAE-Language', 'input_channels': 1, 'input_height': 128, 'input_width': 1292, 'latent_dim': 128, 'num_classes': 4, 'hidden_channels': [16, 32, 64]},
            'cvae_genre': {'class': ConditionalVAE, 'name': 'CVAE-Genre', 'input_channels': 1, 'input_height': 128, 'input_width': 1292, 'latent_dim': 128, 'num_classes': 3, 'hidden_channels': [16, 32, 64]},
            'vade': {'class': VaDE, 'name': 'VaDE', 'input_dim': 165376, 'latent_dim': 128, 'n_clusters': 15, 'hidden_dims': [16, 32, 64]},
            'conv_multimodal': {'class': ConvVAE, 'name': 'Multimodal VAE', 'input_channels': 1, 'input_height': 128, 'input_width': 1292, 'latent_dim': 128, 'hidden_channels': [16, 32, 64]}
        }
        
        # Clustering methods
        self.clustering_methods = ['kmeans', 'agglomerative', 'gmm']
        
    def load_model(self, model_key):
        """Load a trained model"""
        config = self.models_config[model_key]
        checkpoint_path = self.checkpoint_dir / model_key / 'final_model.pt'
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Create model
        if config['class'] == BasicVAE:
            model = config['class'](config['input_dim'], config['latent_dim'], 
                                   hidden_dims=config.get('hidden_dims', [512, 256]))
        elif config['class'] == ConvVAE:
            model = config['class'](input_channels=config['input_channels'],
                                   input_height=config.get('input_height', 128),
                                   input_width=config.get('input_width', 1292),
                                   latent_dim=config['latent_dim'],
                                   hidden_channels=config.get('hidden_channels', [16, 32, 64]))
        elif config['class'] == BetaVAE:
            model = config['class'](input_channels=config['input_channels'],
                                   input_height=config.get('input_height', 128),
                                   input_width=config.get('input_width', 1293),
                                   latent_dim=config['latent_dim'],
                                   hidden_channels=config.get('hidden_channels', [16, 32, 64]))
        elif config['class'] == ConditionalVAE:
            model = config['class'](input_channels=config['input_channels'],
                                   input_height=config.get('input_height', 128),
                                   input_width=config.get('input_width', 1292),
                                   latent_dim=config['latent_dim'],
                                   num_classes=config['num_classes'],
                                   hidden_channels=config.get('hidden_channels', [16, 32, 64]))
        elif config['class'] == VaDE:
            model = config['class'](config['input_dim'], config['latent_dim'], 
                                   config['n_clusters'],
                                   hidden_dims=config.get('hidden_dims', [16, 32, 64]))
        
        # Load weights
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval()
        
        return model, config
    
    def extract_embeddings(self, model, dataloader, model_key):
        """Extract latent embeddings from model"""
        embeddings = []
        language_labels = []
        genre_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Extracting {model_key} embeddings"):
                # Handle different dataset formats
                if 'audio' in batch:
                    x = batch['audio'].to(self.device)
                elif 'features' in batch:
                    x = batch['features'].to(self.device)
                elif 'audio_features' in batch:
                    x = batch['audio_features'].to(self.device)
                else:
                    raise KeyError(f"No audio data found in batch. Keys: {batch.keys()}")
                
                # Handle different model types
                if model_key == 'basic':
                    x = x.flatten(1)
                    mu, logvar = model.encode(x)
                    z = model.reparameterize(mu, logvar)
                elif model_key == 'vade':
                    x = x.flatten(1)
                    mu, logvar = model.encode(x)
                    z = model.reparameterize(mu, logvar)
                elif model_key in ['cvae_language', 'cvae_genre']:
                    # Use condition for CVAE
                    condition = batch['language'] if model_key == 'cvae_language' else batch['genre']
                    condition = condition.to(self.device)
                    
                    # ConditionalVAE expects input_width=1292, let's NOT pad
                    # The model was trained with 1292 width
                    mu, logvar = model.encode(x, condition)
                    z = model.reparameterize(mu, logvar)
                else:
                    # Conv models
                    mu, logvar = model.encode(x)
                    z = model.reparameterize(mu, logvar)
                
                embeddings.append(z.cpu().numpy())
                language_labels.append(batch['language'].numpy())
                genre_labels.append(batch['genre'].numpy())
        
        embeddings = np.concatenate(embeddings, axis=0)
        language_labels = np.concatenate(language_labels, axis=0)
        genre_labels = np.concatenate(genre_labels, axis=0)
        
        return embeddings, language_labels, genre_labels
    
    def perform_clustering(self, embeddings, method, n_clusters):
        """Perform clustering"""
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        elif method == 'agglomerative':
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        elif method == 'gmm':
            clusterer = GaussianMixture(n_components=n_clusters, random_state=42, 
                                       reg_covar=1e-4, max_iter=200)
        
        try:
            if method == 'gmm':
                predictions = clusterer.fit_predict(embeddings)
            else:
                predictions = clusterer.fit_predict(embeddings)
            return predictions
        except Exception as e:
            print(f"Clustering failed: {e}")
            return None
    
    def calculate_metrics(self, embeddings, predictions, true_labels):
        """Calculate clustering metrics"""
        metrics = {}
        
        # Internal metrics (no labels needed)
        try:
            metrics['silhouette'] = float(silhouette_score(embeddings, predictions))
        except:
            metrics['silhouette'] = 0.0
        
        try:
            metrics['davies_bouldin'] = float(davies_bouldin_score(embeddings, predictions))
        except:
            metrics['davies_bouldin'] = 0.0
        
        try:
            metrics['calinski_harabasz'] = float(calinski_harabasz_score(embeddings, predictions))
        except:
            metrics['calinski_harabasz'] = 0.0
        
        # External metrics (with labels)
        try:
            metrics['ari'] = float(adjusted_rand_score(true_labels, predictions))
        except:
            metrics['ari'] = 0.0
        
        try:
            metrics['nmi'] = float(normalized_mutual_info_score(true_labels, predictions))
        except:
            metrics['nmi'] = 0.0
        
        try:
            metrics['v_measure'] = float(v_measure_score(true_labels, predictions))
        except:
            metrics['v_measure'] = 0.0
        
        return metrics
    
    def visualize_confusion_matrix(self, true_labels, predictions, label_names, save_path):
        """Create confusion matrix visualization"""
        cm = confusion_matrix(true_labels, predictions)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=range(len(np.unique(predictions))),
                   yticklabels=label_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Cluster')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def visualize_cluster_distribution(self, predictions, true_labels, label_names, save_path):
        """Visualize cluster distribution"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Predicted clusters
        unique, counts = np.unique(predictions, return_counts=True)
        axes[0].bar(unique, counts, color='steelblue', alpha=0.7)
        axes[0].set_xlabel('Cluster ID')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Predicted Cluster Distribution')
        axes[0].grid(axis='y', alpha=0.3)
        
        # True labels
        unique_true, counts_true = np.unique(true_labels, return_counts=True)
        axes[1].bar(range(len(counts_true)), counts_true, 
                   tick_label=label_names, color='coral', alpha=0.7)
        axes[1].set_xlabel('True Label')
        axes[1].set_ylabel('Count')
        axes[1].set_title('True Label Distribution')
        axes[1].grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def visualize_latent_space(self, embeddings, labels, label_names, save_path, method='tsne'):
        """Visualize latent space using t-SNE or PCA"""
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
            title = 't-SNE Latent Space'
        else:
            reducer = PCA(n_components=2, random_state=42)
            title = 'PCA Latent Space'
        
        embeddings_2d = reducer.fit_transform(embeddings)
        
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                            c=labels, cmap='tab10', alpha=0.6, s=50)
        
        # Create legend
        handles, _ = scatter.legend_elements()
        plt.legend(handles, label_names, title="Labels", loc='best')
        
        plt.title(title)
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_metrics_table(self, results, save_path):
        """Create metrics comparison table"""
        rows = []
        for method, metrics in results.items():
            row = {
                'Method': method.upper(),
                'Silhouette': f"{metrics['silhouette']:.4f}",
                'Davies-Bouldin': f"{metrics['davies_bouldin']:.4f}",
                'Calinski-Harabasz': f"{metrics['calinski_harabasz']:.2f}",
                'ARI': f"{metrics['ari']:.4f}",
                'NMI': f"{metrics['nmi']:.4f}",
                'V-Measure': f"{metrics['v_measure']:.4f}"
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        fig, ax = plt.subplots(figsize=(14, 3))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=df.values, colLabels=df.columns,
                        cellLoc='center', loc='center',
                        colWidths=[0.15, 0.12, 0.15, 0.15, 0.12, 0.12, 0.12])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header
        for i in range(len(df.columns)):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(df) + 1):
            for j in range(len(df.columns)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#E7E6E6')
        
        plt.title('Clustering Metrics Summary', fontsize=14, weight='bold', pad=20)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self, model_key, results, config):
        """Generate text report"""
        report = []
        report.append("=" * 80)
        report.append(f"EVALUATION REPORT: {config['name']}")
        report.append("=" * 80)
        report.append("")
        report.append(f"Model Type: {config['name']}")
        report.append(f"Latent Dimension: {config['latent_dim']}")
        report.append(f"Checkpoint: {model_key}/final_model.pt")
        report.append("")
        report.append("-" * 80)
        report.append("CLUSTERING RESULTS")
        report.append("-" * 80)
        report.append("")
        
        for task in ['language', 'genre']:
            report.append(f"\n{'='*40}")
            report.append(f"Task: {task.upper()} Classification")
            report.append(f"{'='*40}\n")
            
            for method in self.clustering_methods:
                metrics = results[task][method]
                report.append(f"\n{method.upper()} Clustering:")
                report.append(f"  Silhouette Score:      {metrics['silhouette']:.4f}")
                report.append(f"  Davies-Bouldin Index:  {metrics['davies_bouldin']:.4f}")
                report.append(f"  Calinski-Harabasz:     {metrics['calinski_harabasz']:.2f}")
                report.append(f"  Adjusted Rand Index:   {metrics['ari']:.4f}")
                report.append(f"  Normalized Mutual Info: {metrics['nmi']:.4f}")
                report.append(f"  V-Measure Score:       {metrics['v_measure']:.4f}")
        
        report.append("\n" + "=" * 80)
        report.append("END OF REPORT")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def evaluate_model(self, model_key):
        """Evaluate a single model"""
        print(f"\n{'='*80}")
        print(f"Evaluating: {self.models_config[model_key]['name']}")
        print(f"{'='*80}\n")
        
        # Create output directory
        output_dir = self.results_dir / 'evaluations' / model_key
        viz_dir = self.results_dir / 'visualizations' / model_key
        output_dir.mkdir(parents=True, exist_ok=True)
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
        model, config = self.load_model(model_key)
        
        # Load dataset
        dataset = MultimodalDataset(
            data_path='data/features/multimodal_dataset.pkl',
            feature_type='melspec'
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
        
        # Extract embeddings
        embeddings, language_labels, genre_labels = self.extract_embeddings(
            model, dataloader, model_key
        )
        
        print(f"Embeddings shape: {embeddings.shape}")
        print(f"Language labels: {len(np.unique(language_labels))} classes")
        print(f"Genre labels: {len(np.unique(genre_labels))} classes\n")
        
        # Evaluation results
        results = {'language': {}, 'genre': {}}
        
        # Language-based clustering
        print("Language-based clustering:")
        language_names = ['Arabic', 'English', 'Hindi', 'Spanish']
        for method in self.clustering_methods:
            print(f"  - {method.upper()}...", end=' ')
            predictions = self.perform_clustering(embeddings, method, n_clusters=4)
            if predictions is not None:
                metrics = self.calculate_metrics(embeddings, predictions, language_labels)
                results['language'][method] = metrics
                print(f"ARI: {metrics['ari']:.4f}, NMI: {metrics['nmi']:.4f}")
                
                # Save confusion matrix
                self.visualize_confusion_matrix(
                    language_labels, predictions, language_names,
                    viz_dir / f'confusion_matrix_language_{method}.png'
                )
            else:
                print("FAILED")
        
        # Genre-based clustering
        print("\nGenre-based clustering:")
        genre_names = ['Hip-Hop', 'Pop', 'Rock']
        for method in self.clustering_methods:
            print(f"  - {method.upper()}...", end=' ')
            predictions = self.perform_clustering(embeddings, method, n_clusters=3)
            if predictions is not None:
                metrics = self.calculate_metrics(embeddings, predictions, genre_labels)
                results['genre'][method] = metrics
                print(f"ARI: {metrics['ari']:.4f}, NMI: {metrics['nmi']:.4f}")
                
                # Save confusion matrix
                self.visualize_confusion_matrix(
                    genre_labels, predictions, genre_names,
                    viz_dir / f'confusion_matrix_genre_{method}.png'
                )
            else:
                print("FAILED")
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        
        # Latent space (language)
        self.visualize_latent_space(
            embeddings, language_labels, language_names,
            viz_dir / 'latent_space_language_tsne.png', method='tsne'
        )
        self.visualize_latent_space(
            embeddings, language_labels, language_names,
            viz_dir / 'latent_space_language_pca.png', method='pca'
        )
        
        # Latent space (genre)
        self.visualize_latent_space(
            embeddings, genre_labels, genre_names,
            viz_dir / 'latent_space_genre_tsne.png', method='tsne'
        )
        self.visualize_latent_space(
            embeddings, genre_labels, genre_names,
            viz_dir / 'latent_space_genre_pca.png', method='pca'
        )
        
        # Cluster distributions
        # Use best clustering method (kmeans for simplicity)
        lang_pred = self.perform_clustering(embeddings, 'kmeans', n_clusters=4)
        self.visualize_cluster_distribution(
            lang_pred, language_labels, language_names,
            viz_dir / 'cluster_distribution_language.png'
        )
        
        genre_pred = self.perform_clustering(embeddings, 'kmeans', n_clusters=3)
        self.visualize_cluster_distribution(
            genre_pred, genre_labels, genre_names,
            viz_dir / 'cluster_distribution_genre.png'
        )
        
        # Metrics tables
        self.create_metrics_table(
            results['language'],
            viz_dir / 'metrics_table_language.png'
        )
        self.create_metrics_table(
            results['genre'],
            viz_dir / 'metrics_table_genre.png'
        )
        
        # Save results
        with open(output_dir / 'evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Generate text report
        report = self.generate_report(model_key, results, config)
        with open(output_dir / 'evaluation_report.txt', 'w') as f:
            f.write(report)
        
        print(f"\n* Report saved: {output_dir / 'evaluation_report.txt'}")
        print(f"* Results saved: {output_dir / 'evaluation_results.json'}")
        print(f"* Visualizations saved: {viz_dir}/ (12 files)")
        
        return results


def main():
    # Setup
    checkpoint_dir = Path('results/checkpoints')
    results_dir = Path('results')
    
    evaluator = IndividualModelEvaluator(checkpoint_dir, results_dir)
    
    # Evaluate all models
    all_results = {}
    model_keys = ['basic', 'conv', 'beta', 'cvae_language', 'cvae_genre', 'vade', 'conv_multimodal']
    
    print("\n" + "="*80)
    print("INDIVIDUAL MODEL EVALUATION")
    print("="*80)
    print(f"\nEvaluating {len(model_keys)} models...")
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Results directory: {results_dir}")
    
    for i, model_key in enumerate(model_keys, 1):
        print(f"\n[{i}/{len(model_keys)}] Processing {model_key}...")
        try:
            results = evaluator.evaluate_model(model_key)
            all_results[model_key] = results
        except Exception as e:
            print(f"ERROR: Failed to evaluate {model_key}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"\nEvaluated {len(all_results)}/{len(model_keys)} models successfully")
    print(f"\nResults saved to:")
    print(f"  - results/evaluations/<model>/")
    print(f"  - results/visualizations/<model>/")


if __name__ == '__main__':
    main()
