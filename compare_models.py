"""
Compare multiple trained VAE models
Run this after training and evaluating all models
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd


def load_all_evaluations():
    """Load evaluation results from all models"""
    results_dir = Path('results/evaluations')
    all_results = {}
    
    for model_dir in results_dir.glob('*'):
        if model_dir.is_dir():
            json_file = model_dir / 'evaluation_results.json'
            if json_file.exists():
                with open(json_file, 'r') as f:
                    all_results[model_dir.name] = json.load(f)
    
    return all_results


def create_comparison_table(all_results):
    """Create comprehensive comparison table"""
    
    print("\n" + "="*100)
    print("MODEL COMPARISON - CLUSTERING PERFORMANCE")
    print("="*100)
    
    # Prepare data for table
    models = list(all_results.keys())
    algorithms = ['kmeans', 'hierarchical', 'gmm']
    
    for algo in algorithms:
        print(f"\n{algo.upper()} CLUSTERING:")
        print("-"*100)
        
        # Header
        header = f"{'Model':<15}"
        header += f"{'Silhouette':<15}{'Davies-Bouldin':<18}{'Calinski-Har.':<15}"
        header += f"{'NMI':<10}{'ARI':<10}{'Purity':<10}"
        print(header)
        print("-"*100)
        
        for model in models:
            if algo in all_results[model]['clustering_metrics']:
                metrics = all_results[model]['clustering_metrics'][algo]
                row = f"{model:<15}"
                row += f"{metrics.get('silhouette_score', 0):<15.4f}"
                row += f"{metrics.get('davies_bouldin_index', 0):<18.4f}"
                row += f"{metrics.get('calinski_harabasz_index', 0):<15.2f}"
                row += f"{metrics.get('normalized_mutual_info_score', 0):<10.4f}"
                row += f"{metrics.get('adjusted_rand_score', 0):<10.4f}"
                row += f"{metrics.get('cluster_purity', 0):<10.4f}"
                print(row)
    
    # PCA Baseline comparison
    print(f"\n\nPCA BASELINE COMPARISON:")
    print("-"*100)
    header = f"{'Model':<15}{'Sil. Improve %':<18}{'DB Improve %':<18}{'CH Improve %':<18}"
    print(header)
    print("-"*100)
    
    for model in models:
        if 'improvement_over_baseline' in all_results[model]:
            imp = all_results[model]['improvement_over_baseline']
            row = f"{model:<15}"
            row += f"{imp.get('silhouette_improvement_pct', 0):<18.2f}"
            row += f"{imp.get('davies_bouldin_improvement_pct', 0):<18.2f}"
            row += f"{imp.get('calinski_harabasz_improvement_pct', 0):<18.2f}"
            print(row)
    
    print("="*100)


def create_comparison_plots(all_results):
    """Create visual comparisons"""
    save_path = Path('results/visualizations/comparisons')
    save_path.mkdir(parents=True, exist_ok=True)
    
    models = list(all_results.keys())
    n_models = len(models)
    
    # Extract metrics for plotting
    metrics_data = {
        'silhouette': [],
        'davies_bouldin': [],
        'calinski_harabasz': [],
        'nmi': [],
        'ari': [],
        'purity': []
    }
    
    for model in models:
        kmeans_metrics = all_results[model]['clustering_metrics'].get('kmeans', {})
        metrics_data['silhouette'].append(kmeans_metrics.get('silhouette_score', 0))
        metrics_data['davies_bouldin'].append(kmeans_metrics.get('davies_bouldin_index', 0))
        metrics_data['calinski_harabasz'].append(kmeans_metrics.get('calinski_harabasz_index', 0))
        metrics_data['nmi'].append(kmeans_metrics.get('normalized_mutual_info_score', 0))
        metrics_data['ari'].append(kmeans_metrics.get('adjusted_rand_score', 0))
        metrics_data['purity'].append(kmeans_metrics.get('cluster_purity', 0))
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Model Comparison - K-Means Clustering Metrics', fontsize=16, fontweight='bold')
    
    # Silhouette Score
    axes[0, 0].bar(models, metrics_data['silhouette'], color='steelblue')
    axes[0, 0].set_title('Silhouette Score (Higher is Better)')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Davies-Bouldin Index
    axes[0, 1].bar(models, metrics_data['davies_bouldin'], color='coral')
    axes[0, 1].set_title('Davies-Bouldin Index (Lower is Better)')
    axes[0, 1].set_ylabel('Index')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Calinski-Harabasz Index
    axes[0, 2].bar(models, metrics_data['calinski_harabasz'], color='seagreen')
    axes[0, 2].set_title('Calinski-Harabasz Index (Higher is Better)')
    axes[0, 2].set_ylabel('Index')
    axes[0, 2].tick_params(axis='x', rotation=45)
    axes[0, 2].grid(axis='y', alpha=0.3)
    
    # NMI
    if any(metrics_data['nmi']):
        axes[1, 0].bar(models, metrics_data['nmi'], color='mediumpurple')
        axes[1, 0].set_title('Normalized Mutual Information (Higher is Better)')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(axis='y', alpha=0.3)
    
    # ARI
    if any(metrics_data['ari']):
        axes[1, 1].bar(models, metrics_data['ari'], color='gold')
        axes[1, 1].set_title('Adjusted Rand Index (Higher is Better)')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(axis='y', alpha=0.3)
    
    # Purity
    if any(metrics_data['purity']):
        axes[1, 2].bar(models, metrics_data['purity'], color='lightcoral')
        axes[1, 2].set_title('Cluster Purity (Higher is Better)')
        axes[1, 2].set_ylabel('Score')
        axes[1, 2].tick_params(axis='x', rotation=45)
        axes[1, 2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path / 'model_comparison_metrics.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved comparison plot to {save_path}/model_comparison_metrics.png")
    plt.close()
    
    # Clustering algorithm comparison within each model
    for model in models:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'{model.upper()} - Clustering Algorithm Comparison', fontsize=14, fontweight='bold')
        
        algorithms = ['kmeans', 'hierarchical', 'gmm']
        algo_metrics = all_results[model]['clustering_metrics']
        
        sil_scores = [algo_metrics[a]['silhouette_score'] for a in algorithms if a in algo_metrics]
        db_scores = [algo_metrics[a]['davies_bouldin_index'] for a in algorithms if a in algo_metrics]
        ch_scores = [algo_metrics[a]['calinski_harabasz_index'] for a in algorithms if a in algo_metrics]
        
        algo_names = [a for a in algorithms if a in algo_metrics]
        
        axes[0].bar(algo_names, sil_scores, color=['steelblue', 'coral', 'seagreen'])
        axes[0].set_title('Silhouette Score')
        axes[0].set_ylabel('Score')
        axes[0].grid(axis='y', alpha=0.3)
        
        axes[1].bar(algo_names, db_scores, color=['steelblue', 'coral', 'seagreen'])
        axes[1].set_title('Davies-Bouldin Index')
        axes[1].set_ylabel('Index')
        axes[1].grid(axis='y', alpha=0.3)
        
        axes[2].bar(algo_names, ch_scores, color=['steelblue', 'coral', 'seagreen'])
        axes[2].set_title('Calinski-Harabasz Index')
        axes[2].set_ylabel('Index')
        axes[2].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        model_save_path = Path(f'results/visualizations/{model}')
        model_save_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(model_save_path / 'clustering_algorithm_comparison.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved algorithm comparison for {model}")
        plt.close()


def save_comparison_summary():
    """Save comparison summary to JSON"""
    all_results = load_all_evaluations()
    
    summary = {
        "comparison_date": pd.Timestamp.now().isoformat(),
        "models_compared": list(all_results.keys()),
        "best_performers": {},
        "all_models": {}
    }
    
    # Find best performer for each metric
    for model_name, results in all_results.items():
        kmeans = results['clustering_metrics'].get('kmeans', {})
        summary["all_models"][model_name] = {
            "silhouette": kmeans.get('silhouette_score', 0),
            "davies_bouldin": kmeans.get('davies_bouldin_index', 999),
            "calinski_harabasz": kmeans.get('calinski_harabasz_index', 0),
            "nmi": kmeans.get('normalized_mutual_info_score', 0),
            "ari": kmeans.get('adjusted_rand_score', 0),
            "purity": kmeans.get('cluster_purity', 0)
        }
    
    # Determine winners
    if summary["all_models"]:
        summary["best_performers"]["silhouette"] = max(summary["all_models"].items(), 
                                                        key=lambda x: x[1]['silhouette'])[0]
        summary["best_performers"]["davies_bouldin"] = min(summary["all_models"].items(),
                                                           key=lambda x: x[1]['davies_bouldin'])[0]
        summary["best_performers"]["calinski_harabasz"] = max(summary["all_models"].items(),
                                                              key=lambda x: x[1]['calinski_harabasz'])[0]
    
    # Save
    save_path = Path('results/evaluations/model_comparison.json')
    with open(save_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Saved comparison summary to {save_path}")
    
    return summary


if __name__ == "__main__":
    print("Loading all model evaluations...")
    all_results = load_all_evaluations()
    
    if not all_results:
        print("❌ No evaluation results found!")
        print("Please run evaluate_results.py for each model first:")
        print("  python evaluate_results.py --model basic")
        print("  python evaluate_results.py --model conv")
        print("  python evaluate_results.py --model beta")
        exit(1)
    
    print(f"Found {len(all_results)} model(s): {', '.join(all_results.keys())}")
    
    # Create comparison table
    create_comparison_table(all_results)
    
    # Create comparison plots
    create_comparison_plots(all_results)
    
    # Save comparison summary
    summary = save_comparison_summary()
    
    print("\n" + "="*100)
    print("MODEL COMPARISON COMPLETE!")
    print("="*100)
    print(f"\nResults saved to:")
    print(f"  - results/evaluations/model_comparison.json")
    print(f"  - results/visualizations/comparisons/model_comparison_metrics.png")
    print(f"  - results/visualizations/<model>/clustering_algorithm_comparison.png")
