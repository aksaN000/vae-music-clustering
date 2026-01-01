# Evaluation & Analysis System

## Overview
Q1-standard evaluation, visualization, and comparison framework for VAE models.

## Structure

```
results/
├── evaluations/       # Clustering metrics (CSV, JSON)
├── visualizations/    # Publication-quality plots (300 DPI PNG)
└── comparisons/       # Statistical analysis & reports
```

## Scripts

### 1. `evaluate_all_models.py`
**Comprehensive model evaluation**

- **Loads all 7 models**: Basic VAE, Conv VAE, Beta-VAE, CVAE (Language), CVAE (Genre), VaDE, Multimodal VAE
- **Extracts latent representations** from audio/multimodal features
- **Clustering evaluation** with 3 methods: K-Means, Agglomerative, GMM
- **Metrics** (8 comprehensive measures):
  - Silhouette Score (cluster quality)
  - Calinski-Harabasz Index (cluster separation)
  - Davies-Bouldin Index (cluster compactness)
  - Adjusted Rand Index (ground truth agreement)
  - Normalized Mutual Information (information-theoretic)
  - Homogeneity (cluster purity)
  - Completeness (class coverage)
  - V-Measure (harmonic mean of homogeneity & completeness)

**Outputs:**
- `results/evaluations/clustering_metrics.csv` - All metrics in tabular format
- `results/evaluations/detailed_results.json` - Full results with metadata

**Usage:**
```bash
python evaluate_all_models.py
```

### 2. `generate_visualizations.py`
**Publication-quality visualizations**

- **6 plot types:**
  1. Metrics Comparison (6 key metrics across models)
  2. Model Ranking (overall performance)
  3. Task Comparison (language vs genre clustering)
  4. Method Comparison (K-Means vs Agglomerative vs GMM)
  5. Performance Heatmap (normalized scores matrix)
  6. Summary Report (text-based findings)

- **Quality settings:**
  - 300 DPI (publication-ready)
  - Arial font
  - Consistent color palette
  - Professional styling

**Outputs:**
- `results/visualizations/*.png` - All plots
- `results/visualizations/summary_report.txt` - Text summary

**Usage:**
```bash
python generate_visualizations.py
```

### 3. `compare_all_models.py`
**Statistical analysis & comparisons**

- **Pairwise comparisons**: T-tests between all model pairs
- **Effect sizes**: Cohen's d for practical significance
- **Task analysis**: Best models for each task
- **Method analysis**: Clustering method performance
- **LaTeX tables**: Publication-ready tables
- **Markdown report**: Comprehensive findings

**Outputs:**
- `results/comparisons/summary_table.csv` - Aggregated statistics
- `results/comparisons/pairwise_comparisons.csv` - Statistical tests
- `results/comparisons/task_analysis.csv` - Task-specific results
- `results/comparisons/method_analysis.csv` - Clustering method comparison
- `results/comparisons/table_latex.tex` - LaTeX table
- `results/comparisons/comparison_report.md` - Full report

**Usage:**
```bash
python compare_all_models.py
```

### 4. `run_evaluation_pipeline.py`
**Master pipeline script**

Runs all three steps in sequence:
1. Evaluate → 2. Visualize → 3. Compare

**Usage:**
```bash
python run_evaluation_pipeline.py
```

## Quick Start

```bash
# Run complete pipeline
python run_evaluation_pipeline.py

# Or run individually
python evaluate_all_models.py
python generate_visualizations.py
python compare_all_models.py
```

## Requirements

All models must be trained first:
- `results/checkpoints/basic_vae/best_model.pth`
- `results/checkpoints/conv_vae/best_model.pth`
- `results/checkpoints/beta_vae/best_model.pth`
- `results/checkpoints/conditional_vae_language/best_model.pth`
- `results/checkpoints/conditional_vae_genre/best_model.pth`
- `results/checkpoints/vade/best_model.pth`
- `results/checkpoints/multimodal_vae/best_model.pth`

## Metrics Guide

### Clustering Quality (Internal)
- **Silhouette Score** [-1, 1]: Higher is better. Measures cluster cohesion and separation.
- **Calinski-Harabasz Index** [0, ∞): Higher is better. Ratio of between-cluster to within-cluster variance.
- **Davies-Bouldin Index** [0, ∞): Lower is better. Average similarity between clusters.

### Ground Truth Agreement (External)
- **Adjusted Rand Index** [-1, 1]: Higher is better. Adjusted for chance agreement.
- **Normalized Mutual Information** [0, 1]: Higher is better. Information shared between clusterings.
- **Homogeneity** [0, 1]: Higher is better. Each cluster contains only one class.
- **Completeness** [0, 1]: Higher is better. All class members in same cluster.
- **V-Measure** [0, 1]: Higher is better. Harmonic mean of homogeneity and completeness.

## Interpretation

### Best Overall Model
Model with highest average across all metrics.

### Task-Specific Champions
- **Language Clustering**: Model best at separating languages
- **Genre Clustering**: Model best at separating genres

### Clustering Method Preference
K-Means vs Agglomerative vs GMM - which works best for latent space.

## Output Examples

### Evaluation CSV
```
model,task,method,n_samples,silhouette,ari,nmi,...
basic_vae,language,kmeans,180,0.3245,0.5678,...
conv_vae,language,kmeans,180,0.4521,0.6234,...
...
```

### Comparison Report
Markdown document with:
- Executive summary
- Performance rankings
- Task-specific analysis
- Statistical significance
- Key findings

### Visualizations
- Bar charts for metric comparison
- Heatmaps for normalized performance
- Task-specific breakdowns
- Method comparisons

## Notes

- All scripts use config.yaml for model specifications
- Automatically creates output directories
- Progress bars for long operations
- Error handling with informative messages
- Publication-ready outputs (300 DPI, proper formatting)
