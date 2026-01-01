# Reproducibility Guide - 7 VAE Models

## Complete Training Commands (No Overlap)

Each command creates a **separate, unique directory** for checkpoints, evaluations, and visualizations.

### 1. Basic VAE (Audio-only)
```bash
python experiments/train_vae.py --model basic --modality audio
```
**Saves to**: `results/checkpoints/basic/`

### 2. Convolutional VAE (Audio-only)
```bash
python experiments/train_vae.py --model conv --modality audio
```
**Saves to**: `results/checkpoints/conv/`

### 3. Convolutional VAE (Multimodal: Audio + Lyrics)
```bash
python experiments/train_vae.py --model conv --modality multimodal
```
**Saves to**: `results/checkpoints/conv_multimodal/`

### 4. Beta-VAE (Audio-only)
```bash
python experiments/train_vae.py --model beta --modality audio --beta 4.0
```
**Saves to**: `results/checkpoints/beta/`

### 5. Conditional VAE - Genre (45 genres)
```bash
python experiments/train_vae.py --model cvae --modality audio --condition genre
```
**Saves to**: `results/checkpoints/cvae_genre/`

### 6. Conditional VAE - Language (5 languages)
```bash
python experiments/train_vae.py --model cvae --modality audio --condition language
```
**Saves to**: `results/checkpoints/cvae_language/`

### 7. VaDE (Clustering with GMM)
```bash
python experiments/train_vae.py --model vade --modality audio --n_clusters 50
```
**Saves to**: `results/checkpoints/vade/`

---

## Evaluation Commands

After training, evaluate each model:

```bash
# 1. Basic VAE
python evaluate_results.py --model basic

# 2. Conv VAE (Audio)
python evaluate_results.py --model conv

# 3. Conv VAE (Multimodal)
python evaluate_results.py --model conv_multimodal

# 4. Beta-VAE
python evaluate_results.py --model beta

# 5. CVAE - Genre
python evaluate_results.py --model cvae_genre

# 6. CVAE - Language
python evaluate_results.py --model cvae_language

# 7. VaDE
python evaluate_results.py --model vade
```

---

## Directory Structure

```
results/
├── checkpoints/
│   ├── basic/              # Model 1
│   ├── conv/               # Model 2
│   ├── conv_multimodal/    # Model 3
│   ├── beta/               # Model 4
│   ├── cvae_genre/         # Model 5 (NEW)
│   ├── cvae_language/      # Model 6
│   └── vade/               # Model 7
├── evaluations/
│   ├── basic/
│   ├── conv/
│   ├── conv_multimodal/
│   ├── beta/
│   ├── cvae_genre/
│   ├── cvae_language/
│   └── vade/
└── visualizations/
    ├── basic/
    ├── conv/
    ├── conv_multimodal/
    ├── beta/
    ├── cvae_genre/
    ├── cvae_language/
    └── vade/
```

---

## Current Status

**Completed Models** (✅ Trained & Evaluated):
1. ✅ BasicVAE (81 epochs, val_loss=0.5195)
2. ⏳ ConvVAE audio (partially trained, 53 epochs)
3. ✅ ConvVAE multimodal (96 epochs, val_loss=0.5845)
4. ✅ BetaVAE (91 epochs, val_loss=0.5467)
5. ⏹️ CVAE Genre (not yet trained)
6. ✅ CVAE Language (97 epochs, val_loss=0.5050)
7. ✅ VaDE (100 epochs, val_loss=18.2317)

**Models Remaining to Train**:
- [ ] ConvVAE audio (resume or restart)
- [ ] CVAE Genre conditioning

---

## Training Order (Recommended)

For fastest completion, train missing models in parallel terminals:

**Terminal 1:**
```bash
python experiments/train_vae.py --model conv --modality audio
```

**Terminal 2:**
```bash
python experiments/train_vae.py --model cvae --modality audio --condition genre
```

Estimated time: ~2-3 hours per model (100 epochs on RTX 3060)

---

## Verification Commands

Check if all 7 models exist:
```bash
# List checkpoints
ls results/checkpoints/

# Expected output:
# basic, beta, conv, conv_multimodal, cvae_genre, cvae_language, vade
```

Check evaluation status:
```bash
# List evaluations
ls results/evaluations/

# Each model should have: metrics.json, latent_space_tsne.png
```

---

## Key Implementation Details

1. **No Overlaps**: Each model has unique save directory
2. **Naming Logic**:
   - Multimodal Conv: Appends `_multimodal` suffix
   - CVAE variants: Appends `_genre` or `_language` suffix
   - Others: Use base model name
3. **Universal Evaluation**: `evaluate_results.py` handles all 7 model types
4. **Configuration**:
   - Hidden dims: `[16, 32, 64]` (reduced for memory efficiency)
   - Latent dim: 128
   - KL annealing: 0 → 1.0 over 20 epochs
   - Mixed precision: Enabled (FP16)

---

## Troubleshooting

**Issue**: Model directory already exists
**Solution**: Either delete old directory or use existing checkpoint

**Issue**: CUDA out of memory
**Solution**: Reduce batch size in `configs/config.yaml` (default: 32)

**Issue**: Evaluation fails for CVAE
**Solution**: Ensure model name matches exactly (`cvae_genre` or `cvae_language`)

---

## Complete Reproduction Steps

1. **Setup environment**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare data** (if not already done):
   ```bash
   python src/data/audio_processor.py
   python src/data/lyrics_processor.py
   python src/data/data_matcher.py
   ```

3. **Train all 7 models** (use commands from top of this file)

4. **Evaluate all 7 models** (use evaluation commands above)

5. **Compare results**:
   ```bash
   python compare_models.py  # If comparison script exists
   ```

---

**Last Updated**: January 1, 2026
**System**: Python 3.12.6, PyTorch 2.6.0+cu124, RTX 3060
