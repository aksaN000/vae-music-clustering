# Quick Start Guide

## Complete Project Implementation

This project implements comprehensive VAE-based music clustering with multi-modal fusion.

## Installation

```powershell
# Create virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA support (for GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Running the Complete Pipeline

### Option 1: Automated Pipeline (Recommended)

Run everything with one command:

```powershell
python run_all.py
```

This will:
1. Extract audio features from all 1120 files (15 cores parallel processing)
2. Process lyrics with multilingual transformers (677 files)
3. Match audio-lyrics pairs
4. Train all 5 VAE models (Basic, Conv, Beta, CVAE, VaDE)
5. Run clustering experiments
6. Generate all visualizations and metrics
7. Create baseline comparisons

### Option 2: Step-by-Step Execution

#### Step 1: Data Processing

```powershell
# Extract audio features (uses all 15 CPU cores)
python src/data/audio_processor.py --audio_dir data/audio --n_workers 15

# Process lyrics with multilingual transformers
python src/data/lyrics_processor.py --lyrics_dir data/processed_lyrics

# Match audio-lyrics pairs
python src/data/data_matcher.py
```

#### Step 2: Train Models

```powershell
# Easy Task: Basic VAE
python experiments/train_vae.py --model basic --modality audio

# Medium Task: Convolutional VAE
python experiments/train_vae.py --model conv --modality audio

# Hard Task: Beta-VAE
python experiments/train_vae.py --model beta --modality audio

# Hard Task: Conditional VAE
python experiments/train_vae.py --model cvae --modality audio

# VaDE with GMM priors
python experiments/train_vae.py --model vade --modality audio
```

#### Step 3: Run Clustering

```powershell
# Run clustering on trained model
python experiments/run_clustering.py --model conv --checkpoint results/checkpoints/conv/best_model.pt

# Try other models
python experiments/run_clustering.py --model beta --checkpoint results/checkpoints/beta/best_model.pt
python experiments/run_clustering.py --model vade --checkpoint results/checkpoints/vade/best_model.pt
```

#### Step 4: Baseline Comparisons

```powershell
python experiments/baseline.py
```

## Project Structure Summary

```
project/
├── data/
│   ├── audio/              # 1,120 audio files (5 languages)
│   ├── processed_lyrics/   # 677 lyrics files
│   └── features/           # Cached features (auto-generated)
│
├── src/
│   ├── data/               # Data processing pipeline
│   ├── models/             # All 5 VAE architectures
│   ├── clustering/         # Clustering + evaluation
│   ├── fusion/             # Multi-modal fusion
│   ├── training/           # Training with GPU optimization
│   └── visualization/      # Plotting utilities
│
├── experiments/
│   ├── train_vae.py        # Training script
│   ├── run_clustering.py   # Clustering experiments
│   └── baseline.py         # Baseline comparisons
│
├── results/
│   ├── checkpoints/        # Trained models
│   ├── visualizations/     # All plots
│   └── metrics/            # CSV results
│
├── configs/
│   └── config.yaml         # Experiment configuration
│
├── requirements.txt
├── README.md
└── run_all.py              # Automated pipeline
```

## Key Features

### Data Processing
- **Audio**: Mel-spectrograms (128 bands) + MFCC (40 coefficients)
- **Lyrics**: XLM-RoBERTa embeddings (original, romanized, translated)
- **Parallel Processing**: Uses all 15 CPU cores
- **Caching**: Features cached for fast re-runs

### Models Implemented
1. **Basic VAE**: Fully-connected encoder-decoder (Easy Task)
2. **Convolutional VAE**: 2D CNN for spectrograms (Medium Task)
3. **Beta-VAE**: Disentangled representations (Hard Task)
4. **Conditional VAE**: Language/genre conditioning (Hard Task)
5. **VaDE**: VAE + GMM for joint clustering (Hard Task)

### Multi-Modal Fusion
- Early Fusion: Concatenate features
- Late Fusion: Process separately, then combine
- Attention Fusion: Cross-modal attention
- Weighted Fusion: Learnable weights

### Clustering Methods
- K-Means (multiple initializations)
- Agglomerative Hierarchical
- DBSCAN (density-based)
- Gaussian Mixture Models

### Evaluation Metrics (All 6)
1. Silhouette Score
2. Calinski-Harabasz Index
3. Davies-Bouldin Index
4. Adjusted Rand Index (ARI)
5. Normalized Mutual Information (NMI)
6. Cluster Purity

### Visualizations
- t-SNE/UMAP latent space projections
- Cluster distribution plots
- Confusion matrices
- Training curves
- Reconstruction examples
- Metrics comparison charts

## Performance Optimizations

### GPU Acceleration
- Mixed precision training (FP16)
- Automatic GPU detection
- Memory-efficient batch processing

### CPU Parallelization
- 15 workers for data loading
- Parallel feature extraction
- Efficient caching

## Configuration

Edit `configs/config.yaml` to change:
- Batch size, learning rate, epochs
- Model architecture (latent dim, hidden dims)
- Clustering parameters (n_clusters, methods)
- Data processing settings (sample rate, n_mels, n_mfcc)

## Troubleshooting

### Out of Memory
- Reduce `batch_size` in config.yaml
- Reduce `num_workers` if RAM limited
- Use smaller `hidden_dims`

### CUDA Errors
- Check CUDA version: `python -c "import torch; print(torch.version.cuda)"`
- Reinstall PyTorch with correct CUDA version
- Set `use_gpu: false` in config to use CPU only

### Slow Training
- Enable mixed precision: `mixed_precision: true`
- Increase `num_workers` for data loading
- Use smaller dataset for testing

## Expected Runtime

On a system with 15 cores + 1 GPU:
- Audio processing: ~30-60 minutes (1120 files)
- Lyrics processing: ~20-40 minutes (677 files)
- Training per model: ~1-3 hours (depends on dataset size)
- Clustering: ~5-10 minutes per model
- Complete pipeline: ~8-12 hours

## Results

After completion, check:
- `results/visualizations/`: All plots organized by model
- `results/metrics/`: CSV files with all metrics
- `results/checkpoints/`: Trained model weights

Best models saved as `best_model.pt` in each checkpoint directory.

## Next Steps

1. **Analyze Results**: Compare metrics across models
2. **Write Paper**: Use visualizations and metrics for NeurIPS-style report
3. **Fine-tune**: Adjust hyperparameters in config.yaml
4. **Experiment**: Try multi-modal fusion with different strategies

## Citation

If you use this code:

```bibtex
@article{mostakim2026vae,
  title={Variational Autoencoders for Multi-Modal Hybrid-Language Music Clustering},
  author={Mostakim, Moin},
  year={2026}
}
```

## Contact

Course: Neural Networks  
Due: January 10th, 2026
