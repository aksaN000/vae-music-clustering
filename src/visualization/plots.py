"""
Visualization Utilities
t-SNE, UMAP, cluster plots, reconstruction examples
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from umap import UMAP
from pathlib import Path
import torch
from typing import Optional, Dict, List
import pandas as pd


class ClusterVisualizer:
    """Visualization utilities for clustering results"""
    
    def __init__(self, output_dir: str = 'results/visualizations'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        sns.set_style('whitegrid')
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12
    
    def plot_latent_space_2d(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        method: str = 'umap',
        title: str = 'Latent Space Visualization',
        label_names: Optional[Dict] = None,
        save_name: str = 'latent_space.png'
    ):
        """
        Plot 2D projection of latent space
        Args:
            features: Latent features (n_samples, latent_dim)
            labels: Labels for coloring
            method: 'tsne' or 'umap'
            title: Plot title
            label_names: Dictionary mapping label indices to names
            save_name: Filename to save
        """
        print(f"Computing {method.upper()} projection...")
        
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        elif method == 'umap':
            reducer = UMAP(n_components=2, random_state=42, n_neighbors=15)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        reduced = reducer.fit_transform(features)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 10))
        
        unique_labels = np.unique(labels)
        colors = plt.cm.get_cmap('tab10', len(unique_labels))
        
        for idx, label in enumerate(unique_labels):
            mask = labels == label
            label_name = label_names.get(label, str(label)) if label_names else str(label)
            
            ax.scatter(
                reduced[mask, 0],
                reduced[mask, 1],
                c=[colors(idx)],
                label=label_name,
                alpha=0.6,
                s=50,
                edgecolors='black',
                linewidth=0.5
            )
        
        ax.set_xlabel(f'{method.upper()} Dimension 1', fontsize=14)
        ax.set_ylabel(f'{method.upper()} Dimension 2', fontsize=14)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved to: {save_path}")
    
    def plot_cluster_distribution(
        self,
        labels: np.ndarray,
        ground_truth: np.ndarray,
        cluster_names: Optional[List[str]] = None,
        gt_names: Optional[List[str]] = None,
        save_name: str = 'cluster_distribution.png'
    ):
        """
        Plot cluster distribution against ground truth
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Cluster distribution
        unique_clusters, cluster_counts = np.unique(labels, return_counts=True)
        
        if cluster_names:
            cluster_labels = [cluster_names[c] if c < len(cluster_names) else f"Cluster {c}" 
                            for c in unique_clusters]
        else:
            cluster_labels = [f"Cluster {c}" for c in unique_clusters]
        
        ax1.bar(range(len(unique_clusters)), cluster_counts, color=sns.color_palette('Set2'))
        ax1.set_xticks(range(len(unique_clusters)))
        ax1.set_xticklabels(cluster_labels, rotation=45, ha='right')
        ax1.set_xlabel('Clusters', fontsize=12)
        ax1.set_ylabel('Count', fontsize=12)
        ax1.set_title('Predicted Cluster Distribution', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # Ground truth distribution
        unique_gt, gt_counts = np.unique(ground_truth, return_counts=True)
        
        if gt_names:
            gt_labels = [gt_names[g] if g < len(gt_names) else f"Class {g}" 
                        for g in unique_gt]
        else:
            gt_labels = [f"Class {g}" for g in unique_gt]
        
        ax2.bar(range(len(unique_gt)), gt_counts, color=sns.color_palette('Set1'))
        ax2.set_xticks(range(len(unique_gt)))
        ax2.set_xticklabels(gt_labels, rotation=45, ha='right')
        ax2.set_xlabel('Ground Truth Classes', fontsize=12)
        ax2.set_ylabel('Count', fontsize=12)
        ax2.set_title('Ground Truth Distribution', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved to: {save_path}")
    
    def plot_confusion_matrix(
        self,
        true_labels: np.ndarray,
        pred_labels: np.ndarray,
        class_names: Optional[List[str]] = None,
        save_name: str = 'confusion_matrix.png'
    ):
        """Plot confusion matrix"""
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(true_labels, pred_labels)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names if class_names else 'auto',
            yticklabels=class_names if class_names else 'auto',
            cbar_kws={'label': 'Count'}
        )
        
        plt.xlabel('Predicted Cluster', fontsize=12)
        plt.ylabel('True Class', fontsize=12)
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved to: {save_path}")
    
    def plot_reconstruction_comparison(
        self,
        model,
        dataloader,
        device,
        n_examples: int = 5,
        save_name: str = 'reconstructions.png'
    ):
        """Plot original vs reconstructed spectrograms"""
        model.eval()
        
        with torch.no_grad():
            batch = next(iter(dataloader))
            
            if 'features' in batch:
                x = batch['features'][:n_examples].to(device)
            elif 'audio_features' in batch:
                x = batch['audio_features'][:n_examples].to(device)
            else:
                print("No suitable features in batch")
                return
            
            # Get reconstruction
            if hasattr(model, 'forward'):
                output = model(x)
                if isinstance(output, tuple):
                    recon = output[0]
                else:
                    recon = output
            else:
                print("Model has no forward method")
                return
            
            x = x.cpu().numpy()
            recon = recon.cpu().numpy()
        
        # Plot
        fig, axes = plt.subplots(n_examples, 2, figsize=(12, 3*n_examples))
        
        for i in range(n_examples):
            # Original
            if x[i].ndim == 3:
                img_orig = x[i][0]  # Take first channel
            else:
                img_orig = x[i]
            
            axes[i, 0].imshow(img_orig, aspect='auto', cmap='viridis', origin='lower')
            axes[i, 0].set_title(f'Original {i+1}')
            axes[i, 0].set_ylabel('Frequency')
            
            # Reconstruction
            if recon[i].ndim == 3:
                img_recon = recon[i][0]
            else:
                img_recon = recon[i]
            
            axes[i, 1].imshow(img_recon, aspect='auto', cmap='viridis', origin='lower')
            axes[i, 1].set_title(f'Reconstructed {i+1}')
        
        axes[-1, 0].set_xlabel('Time')
        axes[-1, 1].set_xlabel('Time')
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved to: {save_path}")
    
    def plot_training_curves(
        self,
        history: Dict[str, List[float]],
        save_name: str = 'training_curves.png'
    ):
        """Plot training loss curves"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Total loss
        axes[0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
        if 'val_loss' in history and history['val_loss']:
            axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Total Loss')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Reconstruction loss
        axes[1].plot(epochs, history['train_recon_loss'], 'b-', label='Train', linewidth=2)
        if 'val_recon_loss' in history and history['val_recon_loss']:
            axes[1].plot(epochs, history['val_recon_loss'], 'r-', label='Validation', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Reconstruction Loss')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        # KL loss
        axes[2].plot(epochs, history['train_kl_loss'], 'b-', label='Train', linewidth=2)
        if 'val_kl_loss' in history and history['val_kl_loss']:
            axes[2].plot(epochs, history['val_kl_loss'], 'r-', label='Validation', linewidth=2)
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Loss')
        axes[2].set_title('KL Divergence Loss')
        axes[2].legend()
        axes[2].grid(alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved to: {save_path}")
    
    def plot_metrics_comparison(
        self,
        results_df: pd.DataFrame,
        save_name: str = 'metrics_comparison.png'
    ):
        """Plot comparison of different methods"""
        metrics = ['silhouette_score', 'calinski_harabasz_index', 
                  'adjusted_rand_index', 'normalized_mutual_info', 'cluster_purity']
        
        available_metrics = [m for m in metrics if m in results_df.columns]
        
        n_metrics = len(available_metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 5))
        
        if n_metrics == 1:
            axes = [axes]
        
        for ax, metric in zip(axes, available_metrics):
            results_df.plot(x='method', y=metric, kind='bar', ax=ax, color='skyblue')
            ax.set_title(metric.replace('_', ' ').title())
            ax.set_xlabel('')
            ax.set_ylabel('Score')
            ax.grid(axis='y', alpha=0.3)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved to: {save_path}")


if __name__ == "__main__":
    print("Visualization utilities - import and use in analysis scripts")
