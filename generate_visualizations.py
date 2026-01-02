"""
Comprehensive Visualization System
Q1 Standard: Publication-ready plots and figures
"""

import sys
sys.path.append('.')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']


class Visualizer:
    """Publication-quality visualization system"""
    
    def __init__(self, results_dir="results/evaluations", output_dir="results/visualizations"):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Color palette
        self.colors = sns.color_palette("husl", 8)
        
        print(f"Visualizer initialized. Output: {self.output_dir}")
    
    def load_results(self):
        """Load evaluation results including baselines"""
        
        # Load VAE results
        csv_path = self.results_dir / "clustering_metrics.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"VAE results not found: {csv_path}")
        
        df_vae = pd.read_csv(csv_path)
        print(f"Loaded {len(df_vae)} VAE results")
        
        # Load baseline results
        baseline_path = self.results_dir / "baseline_clustering_metrics.csv"
        if baseline_path.exists():
            df_baseline = pd.read_csv(baseline_path)
            print(f"Loaded {len(df_baseline)} baseline results")
            
            # Merge
            df = pd.concat([df_vae, df_baseline], ignore_index=True)
            print(f"Total: {len(df)} results (VAE + Baselines)")
        else:
            print("Warning: Baseline results not found, continuing with VAEs only")
            df = df_vae
        
        return df
    
    def plot_metrics_comparison(self, df):
        """Compare key metrics across models"""
        
        metrics = ['silhouette', 'ari', 'nmi', 'v_measure', 'calinski_harabasz', 'davies_bouldin']
        
        # Aggregate by model
        model_perf = df.groupby('model')[metrics].mean()
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            data = model_perf[metric].sort_values(ascending=False)
            
            ax.barh(range(len(data)), data.values, color=self.colors[:len(data)])
            ax.set_yticks(range(len(data)))
            ax.set_yticklabels(data.index)
            ax.set_xlabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()} Score')
            ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / "metrics_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  * Saved: {output_path.name}")
    
    def plot_model_ranking(self, df):
        """Overall model ranking"""
        
        # Calculate overall score
        metrics = ['silhouette', 'ari', 'nmi', 'v_measure']
        model_scores = df.groupby('model')[metrics].mean()
        model_scores['overall'] = model_scores.mean(axis=1)
        model_scores = model_scores.sort_values('overall', ascending=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.barh(range(len(model_scores)), model_scores['overall'], 
                      color=self.colors[:len(model_scores)])
        
        ax.set_yticks(range(len(model_scores)))
        ax.set_yticklabels(model_scores.index)
        ax.set_xlabel('Overall Score (Average of Key Metrics)')
        ax.set_title('Overall Model Performance Ranking', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (idx, row) in enumerate(model_scores.iterrows()):
            ax.text(row['overall'] + 0.01, i, f"{row['overall']:.4f}", 
                   va='center', fontsize=9)
        
        plt.tight_layout()
        output_path = self.output_dir / "model_ranking.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  * Saved: {output_path.name}")
    
    def plot_task_comparison(self, df):
        """Compare performance by task (language vs genre)"""
        
        metrics = ['silhouette', 'ari', 'nmi', 'v_measure']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            # Group by model and task
            pivot = df.pivot_table(values=metric, index='model', columns='task', aggfunc='mean')
            
            pivot.plot(kind='bar', ax=ax, color=[self.colors[0], self.colors[1]])
            ax.set_title(f'{metric.replace("_", " ").title()} by Task', fontweight='bold')
            ax.set_xlabel('Model')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.legend(title='Task', loc='best')
            ax.grid(axis='y', alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        output_path = self.output_dir / "task_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  * Saved: {output_path.name}")
    
    def plot_method_comparison(self, df):
        """Compare clustering methods"""
        
        metrics = ['silhouette', 'ari', 'nmi']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            # Group by method
            method_perf = df.groupby('method')[metric].mean().sort_values(ascending=False)
            
            ax.bar(range(len(method_perf)), method_perf.values, 
                  color=self.colors[:len(method_perf)])
            ax.set_xticks(range(len(method_perf)))
            ax.set_xticklabels(method_perf.index, rotation=0)
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()} by Method', fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / "method_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  * Saved: {output_path.name}")
    
    def plot_heatmap(self, df):
        """Heatmap of normalized performance"""
        
        metrics = ['silhouette', 'ari', 'nmi', 'v_measure', 'calinski_harabasz']
        
        # Aggregate by model
        model_perf = df.groupby('model')[metrics].mean()
        
        # Normalize each metric to [0, 1]
        normalized = (model_perf - model_perf.min()) / (model_perf.max() - model_perf.min())
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.heatmap(normalized, annot=True, fmt='.3f', cmap='YlOrRd', 
                   cbar_kws={'label': 'Normalized Score'}, ax=ax)
        
        ax.set_title('Normalized Performance Heatmap', fontsize=14, fontweight='bold')
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Models')
        
        plt.tight_layout()
        output_path = self.output_dir / "performance_heatmap.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  * Saved: {output_path.name}")
    
    def create_summary_report(self, df):
        """Create text summary report"""
        
        output_path = self.output_dir / "summary_report.txt"
        
        with open(output_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("VISUALIZATION SUMMARY REPORT\n")
            f.write("="*80 + "\n\n")
            
            # Overall statistics
            f.write("OVERALL STATISTICS\n")
            f.write("-"*80 + "\n")
            f.write(f"Total Models: {df['model'].nunique()}\n")
            f.write(f"Total Experiments: {len(df)}\n")
            f.write(f"Tasks: {', '.join(df['task'].unique())}\n")
            f.write(f"Methods: {', '.join(df['method'].unique())}\n\n")
            
            # Best performers
            f.write("BEST PERFORMERS\n")
            f.write("-"*80 + "\n")
            
            metrics = ['silhouette', 'ari', 'nmi', 'v_measure']
            for metric in metrics:
                best_idx = df[metric].idxmax()
                best_row = df.loc[best_idx]
                f.write(f"\nBest {metric.upper()}: {best_row[metric]:.4f}\n")
                f.write(f"  Model: {best_row['model']}\n")
                f.write(f"  Task: {best_row['task']}\n")
                f.write(f"  Method: {best_row['method']}\n")
            
            f.write("\n" + "="*80 + "\n")
        
        print(f"  * Saved: {output_path.name}")


def main():
    print("\n" + "="*80)
    print("VISUALIZATION GENERATION")
    print("="*80)
    
    visualizer = Visualizer()
    
    # Load results
    print(f"\n{'='*60}")
    print("Loading Results")
    print('='*60)
    df = visualizer.load_results()
    
    # Generate plots
    print(f"\n{'='*60}")
    print("Generating Visualizations")
    print('='*60)
    
    visualizer.plot_metrics_comparison(df)
    visualizer.plot_model_ranking(df)
    visualizer.plot_task_comparison(df)
    visualizer.plot_method_comparison(df)
    visualizer.plot_heatmap(df)
    visualizer.create_summary_report(df)
    
    print(f"\n{'='*80}")
    print("VISUALIZATION COMPLETE")
    print(f"All plots saved in: {visualizer.output_dir}")
    print('='*80)


if __name__ == "__main__":
    main()
