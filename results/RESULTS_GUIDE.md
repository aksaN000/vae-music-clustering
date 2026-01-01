# Results Organization Guide

Complete guide to training, evaluating, and comparing VAE models.

## 📁 Directory Structure

```
results/
├── checkpoints/          # Trained model weights
│   ├── basic/
│   ├── conv/
│   ├── beta/
│   ├── cvae/
│   └── vade/
├── evaluations/          # JSON metrics and statistics
│   ├── basic/
│   │   └── evaluation_results.json
│   ├── conv/
│   ├── beta/
│   └── model_comparison.json
└── visualizations/       # Plots and visualizations
    ├── basic/
    │   ├── latent_space_visualization.png
    │   ├── training_history.png
    │   ├── comparison_pca_vs_vae.png
    │   └── clustering_algorithm_comparison.png
    ├── conv/
    ├── beta/
    └── comparisons/
        └── model_comparison_metrics.png
```

## 🚀 Workflow

### Step 1: Train Models

```bash
# Basic VAE
python experiments/train_vae.py --model basic --modality audio

# Convolutional VAE
python experiments/train_vae.py --model conv --modality audio

# Beta-VAE
python experiments/train_vae.py --model beta --modality audio

# Conditional VAE
python experiments/train_vae.py --model cvae --condition language

# VaDE
python experiments/train_vae.py --model vade --n_clusters 15
```

### Step 2: Evaluate Individual Models

```bash
# Evaluate basic VAE
python evaluate_results.py --model basic

# Evaluate conv VAE
python evaluate_results.py --model conv

# Evaluate beta VAE
python evaluate_results.py --model beta
```

**This generates for each model:**
- ✅ All clustering metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz, NMI, ARI, Purity)
- ✅ Multiple clustering algorithms (K-Means, Hierarchical, GMM)
- ✅ PCA baseline comparison
- ✅ Visualizations (t-SNE, PCA, training curves)
- ✅ Comprehensive JSON report

### Step 3: Compare All Models

```bash
# After evaluating all models, compare them
python compare_models.py
```

**This generates:**
- ✅ Side-by-side comparison table
- ✅ Comparison visualizations
- ✅ Best performer identification
- ✅ model_comparison.json summary

## 📊 Evaluation Metrics Included

### Unsupervised Metrics (No labels needed)
- **Silhouette Score** (0 to 1, higher is better) - Measures cluster separation
- **Davies-Bouldin Index** (0 to ∞, lower is better) - Measures cluster compactness
- **Calinski-Harabasz Index** (0 to ∞, higher is better) - Ratio of between-cluster to within-cluster variance

### Supervised Metrics (Requires true labels)
- **Normalized Mutual Information (NMI)** (0 to 1, higher is better) - Measures information shared between clusters and labels
- **Adjusted Rand Index (ARI)** (-1 to 1, higher is better) - Measures similarity between clusterings
- **Cluster Purity** (0 to 1, higher is better) - Fraction of correctly clustered samples

## 📈 Clustering Algorithms Tested

Each model is evaluated with:
1. **K-Means** - Centroid-based clustering
2. **Hierarchical Agglomerative** - Bottom-up clustering
3. **Gaussian Mixture Model (GMM)** - Probabilistic clustering

## 📄 JSON Structure

```json
{
  "metadata": { ... },
  "training": {
    "epochs_trained": 81,
    "best_val_loss": 0.5195,
    "final_train_recon": 0.5209,
    "final_train_kl": 0.0285
  },
  "clustering_metrics": {
    "kmeans": {
      "silhouette_score": 0.4205,
      "davies_bouldin_index": 0.7277,
      "calinski_harabasz_index": 1707.42,
      "normalized_mutual_info_score": 0.0198,
      "adjusted_rand_score": 0.0067,
      "cluster_purity": 0.4799
    },
    "hierarchical": { ... },
    "gmm": { ... },
    "pca_baseline": { ... }
  },
  "improvement_over_baseline": {
    "silhouette_improvement_pct": 1024.03,
    "davies_bouldin_improvement_pct": 76.19,
    "calinski_harabasz_improvement_pct": 4191.05
  },
  "cluster_statistics": { ... },
  "latent_space_statistics": { ... },
  "performance_summary": {
    "best_clustering_algorithm": "kmeans",
    "overall_conclusion": "VAE significantly outperforms PCA baseline"
  }
}
```

## 🎯 Quick Reference

### Check specific checkpoint
```bash
python inspect_checkpoint.py results/checkpoints/basic/best_model.pt
```

### Evaluate specific checkpoint
```bash
python evaluate_results.py --model basic --checkpoint checkpoint_epoch_50.pt
```

### View JSON results
```bash
# Windows PowerShell
Get-Content results/evaluations/basic/evaluation_results.json | ConvertFrom-Json

# Linux/Mac
cat results/evaluations/basic/evaluation_results.json | jq
```

## 📝 Notes

- All metrics are computed automatically
- NMI, ARI, and Purity require language/genre labels from dataset
- Visualizations use t-SNE for 2D projection (can be slow for large datasets)
- Comparison script works with any subset of trained models
- All results are timestamped for tracking

## 🏆 Expected Results

**Basic VAE:**
- Silhouette: ~0.42 (Good separation)
- Davies-Bouldin: ~0.73 (Compact clusters)
- Calinski-Harabasz: ~1700 (Well-defined)
- 1000%+ improvement over PCA baseline

**Conv VAE:**
- Expected to perform better on spectrogram structure
- Should show higher clustering metrics

**Beta-VAE:**
- Best disentanglement
- May have lower reconstruction but better clustering
