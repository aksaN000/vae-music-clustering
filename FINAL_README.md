# Multi-Lingual Music Clustering with VAE
### Neural Networks Course Project - January 10, 2026

## ✓ Project Status: READY TO TRAIN

**Dataset Verified:**
- ✓ 5 Languages: Arabic, Bangla, English, Hindi, Spanish
- ✓ 45 Genres across all languages
- ✓ 1,119 audio files
- ✓ 677 audio-lyrics matched pairs (60.5%)
- ✓ Audio-lyrics ID matching validated

**Hardware Optimizations:**
- ✓ 15-core CPU parallel processing
- ✓ GPU with mixed precision (FP16)
- ✓ DataLoader with persistent workers
- ✓ Automatic gradient scaling

---

## Quick Start (< 5 minutes)

### 1. Verify Dataset
```bash
python verify_dataset.py
```
This confirms all files are properly organized and matched.

### 2. Test Training (Single Model)
```bash
python quick_start.py
```
Trains Conv VAE on audio data (50 epochs) as a quick test.

### 3. Full Training Pipeline
```bash
python train_all.py
```
Trains all models systematically:
- **Phase 1**: Basic audio models (Basic VAE, Conv VAE, Beta-VAE)
- **Phase 2**: Conditional models (CVAE for language & genre)
- **Phase 3**: Joint clustering (VaDE)
- **Phase 4**: Multimodal (audio + lyrics)

---

## Manual Training Examples

### Audio-Only Models
```bash
# Basic VAE
python experiments/train_vae.py --model basic_vae --modality audio --epochs 100

# Convolutional VAE
python experiments/train_vae.py --model conv_vae --modality audio --epochs 100

# Beta-VAE (disentangled representations)
python experiments/train_vae.py --model beta_vae --modality audio --epochs 100
```

### Conditional VAE (Language or Genre)
```bash
# Condition on language (5 classes)
python experiments/train_vae.py --model cvae --modality audio --condition language --epochs 100

# Condition on genre (45 classes)
python experiments/train_vae.py --model cvae --modality audio --condition genre --epochs 100
```

### Variational Deep Embedding (Joint Clustering)
```bash
# VaDE with 50 clusters (5 languages + 45 genres)
python experiments/train_vae.py --model vade --modality audio --epochs 100
```

### Multi-Modal (Audio + Lyrics)
```bash
# Uses 677 matched audio-lyrics pairs
python experiments/train_vae.py --model conv_vae --modality multimodal --epochs 100
```

---

## Clustering & Evaluation

### Run Clustering on Trained Models
```bash
# K-Means, Agglomerative, DBSCAN, GMM
python experiments/run_clustering.py --checkpoint checkpoints/conv_vae/best_model.pt --model conv

# Test multiple cluster counts (5, 10, 15, 20, 30, 45, 50)
python experiments/run_clustering.py --checkpoint checkpoints/conv_vae/best_model.pt --model conv --n_clusters 45
```

### Baseline Comparisons
```bash
# PCA + K-Means, Autoencoder + K-Means
python experiments/baseline.py
```

---

## Project Structure

```
newestNN/
├── data/
│   ├── audio/                    # 1,119 audio files
│   │   ├── arabic/ (11 genres)   # 96 files
│   │   ├── bangla/ (5 genres)    # 189 files
│   │   ├── english/ (7 genres)   # 537 files
│   │   ├── hindi/ (5 genres)     # 100 files
│   │   └── spanish/ (17 genres)  # 197 files
│   ├── processed_lyrics/         # 677 lyrics files
│   └── features/                 # Cached features (auto-generated)
│
├── src/
│   ├── data/                     # Data processing
│   │   ├── audio_processor.py    # Mel-spec, MFCC extraction (15-core parallel)
│   │   ├── lyrics_processor.py   # XLM-RoBERTa embeddings (multilingual)
│   │   ├── data_matcher.py       # Audio-lyrics pairing with ID validation
│   │   └── dataset.py            # PyTorch datasets (5 lang + 45 genre labels)
│   │
│   ├── models/                   # VAE architectures
│   │   ├── vae.py                # Basic VAE & Autoencoder
│   │   ├── conv_vae.py           # ConvVAE & DeepConvVAE
│   │   ├── beta_vae.py           # Beta-VAE, ConditionalVAE (5 or 45 classes)
│   │   └── vade.py               # VaDE (VAE + GMM, 50 clusters)
│   │
│   ├── fusion/                   # Multi-modal strategies
│   │   └── multimodal.py         # Early/Late/Attention/Weighted fusion
│   │
│   ├── clustering/               # Clustering & evaluation
│   │   ├── cluster.py            # K-Means, Agglomerative, DBSCAN, GMM
│   │   └── evaluation.py         # 6 metrics (Silhouette, CH, DB, ARI, NMI, Purity)
│   │
│   ├── training/                 # Training infrastructure
│   │   └── trainer.py            # GPU-optimized (FP16, gradient clipping)
│   │
│   └── visualization/            # Result visualization
│       └── plots.py              # t-SNE, UMAP, cluster distributions
│
├── experiments/                  # Experiment runners
│   ├── train_vae.py              # Main training script
│   ├── run_clustering.py         # Clustering experiments
│   └── baseline.py               # PCA + AE baselines
│
├── configs/
│   └── config.yaml               # Hyperparameters (5 lang, 45 genre, 50 clusters)
│
├── verify_dataset.py             # Dataset verification tool
├── quick_start.py                # Quick test training
└── train_all.py                  # Full pipeline launcher
```

---

## Configuration (config.yaml)

### Key Parameters
- **Languages**: 5 (Arabic, Bangla, English, Hindi, Spanish)
- **Genres**: 45 unique genres
- **Clusters**: 50 (5 languages + 45 genres)
- **Latent Dim**: 128
- **Batch Size**: 32
- **Workers**: 15 (CPU parallelization)
- **Mixed Precision**: FP16 (GPU memory efficiency)

### Clustering Test Counts
```yaml
clustering:
  n_clusters: [5, 10, 15, 20, 30, 45, 50]
```
- **5**: Language clustering
- **45**: Genre clustering
- **50**: Combined language + genre

---

## Evaluation Metrics

### Clustering Quality (Unsupervised)
1. **Silhouette Score**: Cohesion vs. separation (-1 to 1, higher better)
2. **Calinski-Harabasz Index**: Variance ratio (higher better)
3. **Davies-Bouldin Index**: Cluster similarity (lower better)

### Clustering Accuracy (Supervised)
4. **Adjusted Rand Index (ARI)**: Agreement with true labels (0 to 1)
5. **Normalized Mutual Information (NMI)**: Shared information (0 to 1)
6. **Cluster Purity**: Dominant class per cluster (0 to 1)

---

## Hardware Requirements

### Minimum
- CPU: 4+ cores
- RAM: 16 GB
- GPU: 4 GB VRAM (CUDA-compatible)
- Storage: 10 GB

### Recommended (Your Setup)
- CPU: 15 cores ✓
- RAM: 32+ GB
- GPU: 8+ GB VRAM ✓
- Storage: 20+ GB

### Training Time Estimates
- **Audio processing**: ~15 minutes (one-time, cached)
- **Lyrics processing**: ~5 minutes (one-time, cached)
- **Single VAE model**: 2-4 hours (100 epochs)
- **Full pipeline**: ~20 hours (all models)

---

## Task Coverage

### Easy Tasks ✓
- [x] Data preprocessing (audio + lyrics, 15-core parallel)
- [x] Basic VAE implementation
- [x] K-Means clustering
- [x] Basic evaluation metrics

### Medium Tasks ✓
- [x] Convolutional VAE (spectrograms)
- [x] Beta-VAE (disentangled)
- [x] Multiple clustering algorithms (4 types)
- [x] Comprehensive evaluation (6 metrics)
- [x] GPU optimization (FP16, DataLoader)

### Hard Tasks ✓
- [x] Multi-modal fusion (4 strategies)
- [x] Conditional VAE (language + genre)
- [x] VaDE (VAE + GMM joint clustering)
- [x] Cross-lingual evaluation (5 languages)
- [x] Genre classification (45 classes)

---

## Output Files

### Checkpoints
```
checkpoints/
├── basic_vae/
│   └── best_model.pt
├── conv_vae/
│   └── best_model.pt
├── beta_vae/
│   └── best_model.pt
├── cvae_language/
│   └── best_model.pt
├── cvae_genre/
│   └── best_model.pt
└── vade/
    └── best_model.pt
```

### Results
```
results/
├── clustering_results.csv        # Metrics for all experiments
├── latent_representations/       # Extracted embeddings
├── visualizations/               # t-SNE, UMAP plots
│   ├── tsne_latent_space.png
│   ├── umap_latent_space.png
│   ├── cluster_distribution.png
│   └── language_genre_breakdown.png
└── training_logs/                # Loss curves, metrics
```

---

## Troubleshooting

### Out of Memory (GPU)
```bash
# Reduce batch size in config.yaml
batch_size: 16  # instead of 32
```

### Slow Data Loading
```bash
# Reduce workers if CPU-bound
num_workers: 8  # instead of 15
```

### Audio Files Not Found
```bash
# Verify paths in data/audio/{language}/{genre}/
python verify_dataset.py
```

### Lyrics Matching Issues
```bash
# Check ID extraction in data_matcher.py
# Files should match pattern: {lang}_{id}_Title.mp3 and {lang}_{id}.txt
```

---

## Citation

```bibtex
@project{multilingual_music_vae_2026,
  title={Multi-Lingual Music Clustering with Variational Autoencoders},
  author={Neural Networks Course Project},
  year={2026},
  languages={Arabic, Bangla, English, Hindi, Spanish},
  genres={45 unique genres},
  models={BasicVAE, ConvVAE, BetaVAE, ConditionalVAE, VaDE}
}
```

---

## Support

**Dataset Issues**: Run `python verify_dataset.py`  
**Training Issues**: Check GPU availability with `nvidia-smi`  
**Code Issues**: Review experiment logs in `results/training_logs/`

**Project Deadline**: January 10, 2026  
**Status**: ✓ Ready for training and evaluation
