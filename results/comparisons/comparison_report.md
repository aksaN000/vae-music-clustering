# Model Comparison Report

## Executive Summary

- **Total Models**: 7
- **Total Experiments**: 42
- **Tasks**: language, genre
- **Clustering Methods**: kmeans, agglomerative, gmm

## Overall Performance Rankings

| Rank | Model | Overall Score | Silhouette | ARI | NMI | V-Measure |
|------|-------|---------------|------------|-----|-----|------------|
| 1 | Conv VAE | 0.0229 | 0.0385 | 0.0016 | 0.0257 | 0.0257 |
| 2 | CVAE-Genre | 0.0163 | 0.0101 | 0.0084 | 0.0233 | 0.0233 |
| 3 | Multimodal VAE | 0.0132 | 0.0092 | 0.0020 | 0.0209 | 0.0209 |
| 4 | VaDE | 0.0123 | 0.0083 | 0.0021 | 0.0194 | 0.0194 |
| 5 | Basic VAE | 0.0112 | 0.0111 | 0.0016 | 0.0160 | 0.0160 |
| 6 | CVAE-Language | 0.0065 | 0.0085 | -0.0036 | 0.0105 | 0.0105 |
| 7 | Beta-VAE | 0.0063 | 0.0038 | -0.0021 | 0.0118 | 0.0118 |

## Task-Specific Performance

### Language Clustering

- **Best ARI**: 0.0212 (CVAE-Genre)
- **Best NMI**: 0.0539 (Multimodal VAE)
- **Best Silhouette**: 0.0226 (Conv VAE)

### Genre Clustering

- **Best ARI**: 0.0112 (VaDE)
- **Best NMI**: 0.0214 (Multimodal VAE)
- **Best Silhouette**: 0.1647 (Conv VAE)

## Clustering Method Comparison

| Method | Avg Silhouette | Avg ARI | Avg NMI |
|--------|----------------|---------|----------|
| agglomerative | 0.0194 | -0.0019 | 0.0141 |
| gmm | 0.0086 | 0.0064 | 0.0246 |
| kmeans | 0.0103 | -0.0002 | 0.0159 |

## Key Findings

1. **Best Overall Model**: Conv VAE (Overall Score: 0.0229)
2. **Best for Language**: CVAE-Genre (ARI: 0.0122)
2. **Best for Genre**: CVAE-Genre (ARI: 0.0046)
3. **Best Clustering Method**: gmm
