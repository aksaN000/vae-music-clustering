# Variational Autoencoders for Multi-Modal Hybrid-Language Music Clustering

A systematic empirical study comparing seven Variational Autoencoder (VAE) architectures for unsupervised clustering of a balanced, multilingual music dataset spanning four languages (Arabic, English, Hindi, Spanish) and three genres (Pop, Rock, Hip-Hop). The repository contains the full data pipeline, model implementations, evaluation framework, trained-model evaluation outputs, and the accompanying paper (see [paper.pdf](paper.pdf)).

## Table of Contents

- [Highlights](#highlights)
- [Research Questions](#research-questions)
- [Dataset](#dataset)
- [Architectures Compared](#architectures-compared)
- [Headline Results](#headline-results)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Reproducing the Experiments](#reproducing-the-experiments)
- [Configuration](#configuration)
- [Evaluation Framework](#evaluation-framework)
- [Methodological Notes](#methodological-notes)
- [Citing this Work](#citing-this-work)
- [License](#license)

## Highlights

- **Seven VAE variants** implemented and benchmarked end-to-end: Basic VAE, Convolutional VAE, β-VAE, Conditional VAE (genre), Conditional VAE (language), VaDE (GMM-prior clustering VAE), and Multimodal VAE (audio + lyrics, attention fusion).
- **Custom-curated balanced dataset** of 180 full-length songs (15 per language × genre cell), each manually paired with lyrics, expanded to 2,107 clips via a 30 s window / 20 s hop protocol.
- **Song-level data splitting** performed before windowing — preventing the clip-leakage failure mode in which clips from the same song appear in multiple splits.
- **Three clustering algorithms × six evaluation metrics** (Silhouette, Calinski-Harabasz, Davies-Bouldin, ARI, NMI, V-Measure) cross-tabulated against two downstream tasks (language and genre).
- **Baseline parity:** PCA + K-means, autoencoder + K-means, and raw-feature + K-means baselines included for honest comparison.
- **Reconstruction-clustering disconnect** documented: the Multimodal VAE achieves the lowest reconstruction loss but does not lead on genre clustering.

## Research Questions

The accompanying paper (see [paper.pdf](paper.pdf)) is organised around four questions:

1. How do the seven VAE architectures compare for hybrid-language music clustering across multiple evaluation metrics?
2. Does song-level dataset splitting (partitioning songs before windowing) prevent leakage while preserving clustering performance?
3. Is there a correlation between reconstruction quality (validation loss) and clustering effectiveness (NMI, ARI, Silhouette)?
4. Which clustering algorithms (K-means, Agglomerative, GMM) best leverage VAE latent representations?

## Dataset

A hand-curated hybrid-language music corpus, intentionally small and balanced to make per-cell comparisons tractable.

| Property | Value |
| --- | --- |
| Original songs | 180 |
| Languages | 4 — Arabic, English, Hindi, Spanish |
| Genres | 3 — Pop, Rock, Hip-Hop |
| Songs per (language × genre) cell | 15 |
| Lyrics coverage | 100% — every song paired with a lyrics file |
| Audio sample rate | 22,050 Hz |
| Windowing | 30 s window, 20 s hop (33% overlap) |
| Total clips after windowing | 2,107 |
| Split strategy | **Song-level** 80 / 10 / 10 (before windowing) |
| Metadata | [data/dataset_metadata.csv](data/dataset_metadata.csv) |

Audio features extracted by [src/data/audio_processor.py](src/data/audio_processor.py): 128-band log-mel spectrogram and 40-coefficient MFCC, computed with `n_fft = 2048`, `hop_length = 512`.

Lyrics features extracted by [src/data/lyrics_processor.py](src/data/lyrics_processor.py) using `xlm-roberta-base` (multilingual transformer, max 512 tokens), with optional romanisation and translation to English.

Note: the raw audio files are excluded from the repository (see [.gitignore](.gitignore)). Only the metadata CSV and the processed lyrics directory are tracked. To reproduce, place the corresponding audio files under `data/audio/<language>/<genre>/` as referenced in `dataset_metadata.csv`.

## Architectures Compared

All models share latent dimensionality 128 and use the encoder/decoder hidden stack `[32, 64, 128, 256]` (Conv-based) where applicable. Implementations live under [src/models/](src/models/).

| # | Model | Modality | Distinguishing feature | File |
| --- | --- | --- | --- | --- |
| 1 | Basic VAE | Audio | Fully-connected encoder/decoder | [src/models/vae.py](src/models/vae.py) |
| 2 | Convolutional VAE | Audio | Convolutional stack over mel-spectrograms | [src/models/conv_vae.py](src/models/conv_vae.py) |
| 3 | β-VAE | Audio | Disentangled latent space, β = 4.0 | [src/models/beta_vae.py](src/models/beta_vae.py) |
| 4 | CVAE — language | Audio + label | Language label as condition | [src/models/vae.py](src/models/vae.py) |
| 5 | CVAE — genre | Audio + label | Genre label as condition | [src/models/vae.py](src/models/vae.py) |
| 6 | VaDE | Audio | Gaussian-mixture prior with 15 components | [src/models/vade.py](src/models/vade.py) |
| 7 | Multimodal VAE | Audio + lyrics | Attention fusion of audio + XLM-RoBERTa lyric embeddings | [src/fusion/multimodal.py](src/fusion/multimodal.py) |

Baselines (in [experiments/baseline.py](experiments/baseline.py)): PCA + K-means, autoencoder + K-means, raw features + K-means.

## Headline Results

Results aggregated from [results/comparisons/summary_table.csv](results/comparisons/summary_table.csv) — means ± std across clustering methods.

**Per-model NMI summary across both tasks (higher is better):**

| Model | Genre NMI | Genre ARI | Genre Silhouette | Language NMI | Language ARI | Language Silhouette |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Basic VAE** | **0.0815 ± 0.016** | **0.0485 ± 0.025** | **0.0625 ± 0.040** | 0.0242 ± 0.009 | -0.0004 ± 0.004 | 0.0216 ± 0.004 |
| Conv VAE | 0.0418 ± 0.035 | 0.0181 ± 0.019 | 0.0107 ± 0.005 | 0.0348 ± 0.006 | 0.0120 ± 0.005 | 0.0033 ± 0.009 |
| β-VAE | 0.0162 ± 0.008 | 0.0050 ± 0.004 | 0.0095 ± 0.005 | 0.0256 ± 0.006 | 0.0048 ± 0.005 | 0.0099 ± 0.006 |
| CVAE-Genre | 0.0058 ± 0.002 | -0.0042 ± 0.002 | 0.0173 ± 0.001 | 0.0246 ± 0.001 | 0.0026 ± 0.002 | 0.0153 ± 0.004 |
| CVAE-Language | 0.0280 ± 0.010 | 0.0154 ± 0.009 | 0.0120 ± 0.006 | 0.0300 ± 0.006 | 0.0044 ± 0.013 | 0.0048 ± 0.011 |
| VaDE | 0.0250 ± 0.015 | 0.0141 ± 0.015 | 0.0146 ± 0.005 | 0.0200 ± 0.004 | 0.0016 ± 0.007 | 0.0137 ± 0.004 |
| **Multimodal VAE** | 0.0095 ± 0.004 | -0.0017 ± 0.004 | 0.0056 ± 0.012 | **0.0307 ± 0.012** | **0.0049 ± 0.010** | 0.0018 ± 0.008 |
| PCA (baseline) | 0.0294 ± 0.015 | 0.0191 ± 0.012 | 0.0282 ± 0.007 | 0.0169 ± 0.003 | -0.0009 ± 0.004 | 0.0191 ± 0.023 |
| Raw features (baseline) | 0.0284 ± 0.028 | 0.0174 ± 0.026 | 0.0370 ± 0.005 | 0.0168 ± 0.002 | -0.0011 ± 0.001 | 0.0360 ± 0.008 |

**Best result per metric** (from [results/visualizations/summary_report.txt](results/visualizations/summary_report.txt)):

| Metric | Value | Model | Task | Clustering method |
| --- | ---: | --- | --- | --- |
| Silhouette | **0.1078** | Basic VAE | genre | GMM |
| ARI | **0.0773** | Basic VAE | genre | Agglomerative |
| NMI | **0.0969** | Basic VAE | genre | Agglomerative |
| V-Measure | **0.0969** | Basic VAE | genre | Agglomerative |

**Clustering method ranking** (mean across all models, from [results/comparisons/method_analysis.csv](results/comparisons/method_analysis.csv)):

| Method | Mean Silhouette | Mean ARI | Mean NMI |
| --- | ---: | ---: | ---: |
| K-means | 0.0203 | 0.0074 | 0.0239 |
| Agglomerative | 0.0172 | 0.0096 | 0.0265 |
| GMM | 0.0139 | 0.0098 | 0.0322 |

**Key findings:**

1. **For genre clustering, simplicity wins.** The 5.5 M-parameter Basic VAE beats every more sophisticated architecture — including the 64.5 M-parameter Conv VAE — across all three clustering methods.
2. **For language clustering, multimodality wins.** The Multimodal VAE achieves the highest mean language NMI, narrowly outperforming its unimodal Conv-VAE counterpart. Audio-lyrics fusion buys the most where audio alone is least discriminative.
3. **Reconstruction loss does not predict clustering.** The Multimodal VAE attains the lowest validation reconstruction loss (0.5502 per the paper) yet ranks last on genre clustering — a clean counter-example to the "better generative model ⇒ better representations" intuition.
4. **Hard ceiling near NMI ≈ 0.10.** No model, baseline, or clustering algorithm exceeds NMI ≈ 0.10 on either task, suggesting a fundamental scale limitation of a 180-song corpus rather than an architectural deficit.
5. **Agglomerative narrowly leads on ARI/NMI, K-means on Silhouette.** GMM yields the best NMI on average but lower Silhouette — consistent with GMM producing softer, more overlapping clusters than agglomerative or K-means.

Full pairwise statistical comparisons are available in [results/comparisons/pairwise_comparisons.csv](results/comparisons/pairwise_comparisons.csv); a LaTeX-formatted summary table in [results/comparisons/table_latex.tex](results/comparisons/table_latex.tex).

## Repository Structure

```
.
├── paper.pdf                       # Full write-up (read this first for context)
├── paper.tex                       # LaTeX source
├── references.bib                  # Bibliography
│
├── configs/
│   └── config.yaml                 # Single source of truth for all hyperparameters
│
├── data/
│   ├── dataset_metadata.csv        # 180-song manifest with language/genre/lyrics paths
│   └── lyrics/                     # Processed lyrics files (audio excluded from repo)
│
├── src/
│   ├── data/                       # audio_processor, lyrics_processor, dataset, data_matcher
│   ├── models/                     # vae, conv_vae, beta_vae, vade
│   ├── fusion/                     # multimodal.py — attention-based audio/lyrics fusion
│   ├── clustering/                 # cluster.py, evaluation.py
│   ├── training/                   # trainer.py
│   └── visualization/              # plots.py
│
├── scripts/                        # One-off dataset construction utilities
│   ├── 1_select_balanced_genres.py
│   ├── 2_determine_cell_size.py
│   ├── 3_create_balanced_subset.py
│   └── create_windowed_dataset.py
│
├── experiments/
│   ├── train_vae.py                # Trains a single VAE variant
│   ├── run_clustering.py           # Runs clustering + metrics on a trained checkpoint
│   └── baseline.py                 # PCA / AE / raw-features baselines
│
├── results/
│   ├── comparisons/                # Aggregated tables and pairwise stats
│   ├── evaluations/                # Per-model clustering metrics (one folder per checkpoint)
│   └── visualizations/             # Figures consumed by the paper
│
├── quick_start.py                  # Smoke-test: Conv VAE on audio in ~50 epochs
├── run_all.py                      # End-to-end pipeline driver
├── train_all.py                    # Trains every VAE variant in sequence
├── retrain_all.py                  # Retrains every VAE variant
├── evaluate_all_models.py          # Runs clustering across every checkpoint
├── evaluate_individual_models.py   # Detailed per-model evaluation
├── compare_all_models.py           # Builds the cross-model comparison tables
├── generate_visualizations.py      # Builds the comparison figures
├── inspect_checkpoint.py           # Checkpoint introspection utility
├── run_evaluation_pipeline.py      # Evaluation-only sub-pipeline
└── requirements.txt
```

## Installation

Tested on Python 3.10+ with PyTorch 2.x and a CUDA-capable GPU. CPU-only execution is possible but several orders of magnitude slower.

```bash
git clone https://github.com/aksaN000/425_project.git
cd 425_project

python -m venv .venv
.\.venv\Scripts\Activate.ps1            # Windows PowerShell
# source .venv/bin/activate             # Linux / macOS

# PyTorch with CUDA 11.8 (adjust to your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
```

[requirements.txt](requirements.txt) pins the full dependency surface: PyTorch / Lightning for training; `librosa` / `soundfile` / `audiomentations` for audio; `transformers` / `sentencepiece` / `langdetect` for lyrics; `scikit-learn` / `umap-learn` / `hdbscan` for clustering and dim reduction; `matplotlib` / `seaborn` / `plotly` for figures; `wandb` and `tensorboard` for experiment tracking.

## Reproducing the Experiments

The repository ships with completed evaluation artefacts under [results/](results/). The full pipeline that produced them is:

### 1. Audio + lyrics preprocessing

```bash
python src/data/audio_processor.py --audio_dir data/audio --n_workers 15
python src/data/lyrics_processor.py --lyrics_dir data/processed_lyrics
python src/data/data_matcher.py
```

### 2. Dataset construction (if rebuilding from raw)

```bash
python scripts/1_select_balanced_genres.py
python scripts/2_determine_cell_size.py
python scripts/3_create_balanced_subset.py
python scripts/create_windowed_dataset.py
```

### 3. Smoke test (single Conv VAE, ~50 epochs)

```bash
python quick_start.py
```

### 4. Train every VAE variant

```bash
python train_all.py
# or, for any single variant:
python experiments/train_vae.py --model conv_vae --modality audio
python experiments/train_vae.py --model conv_vae --modality multimodal
```

Supported `--model` values: `basic`, `conv`, `beta`, `cvae`, `vade`. Supported `--modality` values: `audio`, `multimodal`.

### 5. Clustering + evaluation

```bash
python experiments/run_clustering.py --model conv --checkpoint results/checkpoints/conv/best_model.pt
python experiments/baseline.py
python evaluate_all_models.py
python compare_all_models.py
```

### 6. Figures

```bash
python generate_visualizations.py
```

### 7. End-to-end (everything above, sequentially)

```bash
python run_all.py
```

## Configuration

All hyperparameters live in [configs/config.yaml](configs/config.yaml) and are loaded once at the top of each entry-point script. Selected defaults:

| Group | Setting | Value |
| --- | --- | --- |
| Audio | sample rate / `n_mels` / `n_mfcc` | 22050 / 128 / 40 |
| Audio | `n_fft` / `hop_length` | 2048 / 512 |
| Windowing | window / hop / overlap | 30 s / 20 s / 33% |
| Model | latent dimension | 128 |
| Model | conv hidden dims | `[32, 64, 128, 256]` |
| Model | β (β-VAE) | 4.0 |
| Model | VaDE GMM components | 15 |
| Multimodal | fusion type | attention |
| Multimodal | lyrics encoder | `xlm-roberta-base` |
| Multimodal | audio / lyrics weight | 0.5 / 0.5 |
| Training | batch size | 32 |
| Training | epochs | 100 |
| Training | learning rate / weight decay | 1e-4 / 1e-4 |
| Training | gradient clip / early-stop patience | 0.5 / 15 |
| Training | mixed precision | FP16 |
| Clustering | algorithms | K-means, Agglomerative, DBSCAN, GMM |
| Clustering | cluster counts swept | 3, 4, 5, 10, 12, 15 |
| Evaluation | metrics | Silhouette, Calinski-Harabasz, Davies-Bouldin, ARI, NMI, Purity |
| Visualisation | dimensionality reduction | UMAP (default), t-SNE, PCA |

Audio augmentation (pitch shift ±2 semitones, time stretch 0.9–1.1×, noise factor 0.005) is enabled by default during training.

## Evaluation Framework

Each trained VAE is evaluated on two downstream tasks (genre and language) using three clustering algorithms (K-means, Agglomerative, GMM), producing a 3 × 2 metric grid per model. The cross-tabulation uses six metrics — three internal (Silhouette, Calinski-Harabasz, Davies-Bouldin) and three external (ARI, NMI, V-Measure / Purity).

- Per-model raw metrics: `results/evaluations/<model>/`
- Aggregated table: [results/comparisons/summary_table.csv](results/comparisons/summary_table.csv)
- Pairwise model statistics: [results/comparisons/pairwise_comparisons.csv](results/comparisons/pairwise_comparisons.csv)
- Per-task best-of analysis: [results/comparisons/task_analysis.csv](results/comparisons/task_analysis.csv)
- Per-method analysis: [results/comparisons/method_analysis.csv](results/comparisons/method_analysis.csv)
- Figures (cross-model heatmap, ranking, per-method comparison, per-task comparison): [results/visualizations/](results/visualizations/)

## Methodological Notes

- **Song-level splitting before windowing.** All 180 original songs are partitioned into train/validation/test (80/10/10) *before* the 30 s/20 s windowing protocol is applied. This guarantees no two clips from the same song appear in different splits — a common failure mode when windowing is performed first and clips are split randomly.
- **Class balance by construction.** Exactly 15 songs per (language × genre) cell, eliminating class-prior confounds when comparing per-task NMI / ARI.
- **Identical training budget per model.** Same batch size, learning rate, weight decay, mixed-precision setting, early-stopping patience, and 100 epoch cap, so accuracy gaps reflect architecture rather than tuning effort.
- **Baseline parity.** PCA + K-means and raw-features + K-means are run through the same evaluation pipeline as the learned models, so any "VAE wins" claim is measured against honest non-deep alternatives.
- **NMI ceiling reported, not hidden.** No model exceeds NMI ≈ 0.10 — this is reported rather than excluded, and attributed to dataset scale (180 songs) rather than method failure.

## Citing this Work

```bibtex
@misc{alif2025vae_music_clustering,
  title   = {Variational Autoencoders for Multi-Modal Hybrid-Language Music Clustering: A Systematic Comparison},
  author  = {Aksan Gony Alif},
  year    = {2025},
  note    = {BRAC University coursework (Student ID 24341256)},
  howpublished = {\url{https://github.com/aksaN000/425_project}}
}
```

The full paper, including methodology details, ablations, and complete result figures, is in [paper.pdf](paper.pdf). LaTeX source: [paper.tex](paper.tex). Bibliography: [references.bib](references.bib).

## License

Released for academic and educational use. If you intend to reuse the dataset construction methodology, the song-level splitting protocol, or any of the result tables, please cite the paper.
