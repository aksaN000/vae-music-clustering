# Model Comparison Report

## Executive Summary

- **Total Models**: 9
- **Total Experiments**: 52
- **Tasks**: language, genre
- **Clustering Methods**: kmeans, agglomerative, gmm

## Overall Performance Rankings

| Rank | Model | Overall Score | Silhouette | ARI | NMI | V-Measure |
|------|-------|---------------|------------|-----|-----|------------|
| 1 | CVAE-Language | 0.0286 | 0.0097 | 0.0199 | 0.0425 | 0.0425 |
| 2 | Basic VAE | 0.0279 | 0.0356 | 0.0067 | 0.0348 | 0.0348 |
| 3 | Raw Features | 0.0225 | 0.0365 | 0.0082 | 0.0226 | 0.0226 |
| 4 | Conv VAE | 0.0200 | 0.0098 | 0.0099 | 0.0301 | 0.0301 |
| 5 | PCA | 0.0198 | 0.0237 | 0.0091 | 0.0232 | 0.0232 |
| 6 | Multimodal VAE | 0.0148 | 0.0100 | 0.0043 | 0.0225 | 0.0225 |
| 7 | CVAE-Genre | 0.0110 | 0.0172 | -0.0024 | 0.0147 | 0.0147 |
| 8 | Beta-VAE | 0.0089 | 0.0034 | -0.0002 | 0.0163 | 0.0163 |
| 9 | VaDE | 0.0047 | -0.0033 | -0.0016 | 0.0119 | 0.0119 |

## Task-Specific Performance

### Language Clustering

- **Best ARI**: 0.0208 (Conv VAE)
- **Best NMI**: 0.0402 (CVAE-Language)
- **Best Silhouette**: 0.0420 (Raw Features)

### Genre Clustering

- **Best ARI**: 0.0595 (CVAE-Language)
- **Best NMI**: 0.0739 (CVAE-Language)
- **Best Silhouette**: 0.0776 (Basic VAE)

## Clustering Method Comparison

| Method | Avg Silhouette | Avg ARI | Avg NMI |
|--------|----------------|---------|----------|
| agglomerative | 0.0192 | 0.0081 | 0.0288 |
| gmm | 0.0058 | 0.0042 | 0.0225 |
| kmeans | 0.0191 | 0.0052 | 0.0216 |

## Key Findings

1. **Best Overall Model**: CVAE-Language (Overall Score: 0.0286)
2. **Best for Language**: Conv VAE (ARI: 0.0114)
2. **Best for Genre**: CVAE-Language (ARI: 0.0373)
3. **Best Clustering Method**: agglomerative
