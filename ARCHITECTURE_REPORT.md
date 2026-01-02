# Architecture Configuration Report - All Models

**Generated:** January 2, 2026  
**Dataset:** 2107 windowed samples (30s clips from 180 full songs)  
**Input:** Mel spectrograms 128×1292 (165,376 features)

---

## 📊 Model Overview Summary

| Model | Type | Parameters | Latent Dim | Compression | Key Feature |
|-------|------|------------|------------|-------------|-------------|
| **Basic VAE** | FC | 5.5M | 128 | 1292:1 | Baseline fully-connected |
| **Conv VAE** | CNN | 64.5M | 128 | 1292:1 | Spatial patterns in spectrograms |
| **Beta-VAE** | CNN | 64.5M | 128 | 1292:1 | Disentangled representations (β=4.0) |
| **CVAE-Language** | CNN | 69.7M | 128 | 1292:1 | Conditioned on 4 languages |
| **CVAE-Genre** | CNN | 69.7M | 128 | 1292:1 | Conditioned on 3 genres |
| **VaDE** | FC | 169.9M | 128 | 1292:1 | VAE + GMM for clustering (15 clusters) |
| **Multimodal** | Hybrid | TBD | 128 | Variable | Audio + Lyrics fusion |

---

## 1️⃣ Basic VAE (Baseline)

**Architecture Type:** Fully-connected (FC)  
**Status:** ✅ Trained (2107 samples, 79 epochs)  
**Parameters:** 5,488,032 (~5.5M)

### Configuration
```yaml
Input: 165,376 (128×1292 flattened)
Latent Dim: 128
Hidden Dims: [16, 32, 64]
Dropout: 0.2
Batch Norm: Yes
Activation: ReLU
```

### Architecture Flow
```
Input (165,376)
    ↓
Encoder: FC(165376→16) → BN → ReLU → Dropout(0.2)
         FC(16→32) → BN → ReLU → Dropout(0.2)
         FC(32→64) → BN → ReLU → Dropout(0.2)
    ↓
Latent Space: FC(64→128) [μ, logσ²]
    ↓ Reparameterization
    ↓
Decoder: FC(128→64) → BN → ReLU → Dropout(0.2)
         FC(64→32) → BN → ReLU → Dropout(0.2)
         FC(32→16) → BN → ReLU → Dropout(0.2)
         FC(16→165376)
    ↓
Output (165,376)
```

### Current Performance
- **Genre Clustering:** NMI=0.0651, ARI=0.0258, Silhouette=0.0793
- **Language Clustering:** NMI=0.0314, ARI=0.0051, Silhouette=0.0256
- **Training:** 1.5 min (79 epochs), val_loss=0.6213
- **Compression Ratio:** 1292:1 (165,376→128)

### Issues
❌ **VERY SHALLOW** - Only 3 layers, cannot capture complex patterns  
❌ **Extreme compression** - 1292:1 ratio loses too much information  
❌ **Poor clustering** - NMI < 0.1 indicates weak separation  
⚠️ **Attempting deeper architecture (latent=288, hidden=[80,160,320]) made results WORSE**

---

## 2️⃣ Conv VAE (Convolutional)

**Architecture Type:** 2D Convolutional Neural Network  
**Status:** ⏳ Not yet trained  
**Parameters:** 64,458,049 (~64.5M) - **12x more than Basic VAE**

### Configuration
```yaml
Input Channels: 1 (grayscale spectrogram)
Input Height: 128 (n_mels)
Input Width: 1292 (time frames)
Latent Dim: 128
Hidden Channels: [32, 64, 128, 256]
Dropout: 0.2 (Dropout2d)
Kernel Size: 4×4
Stride: 2
Padding: 1
```

### Architecture Flow
```
Input (1×128×1292)
    ↓
Encoder: Conv2d(1→32, k=4, s=2, p=1) → BN2d → LeakyReLU(0.2) → Dropout2d
         Conv2d(32→64, k=4, s=2, p=1) → BN2d → LeakyReLU(0.2) → Dropout2d
         Conv2d(64→128, k=4, s=2, p=1) → BN2d → LeakyReLU(0.2) → Dropout2d
         Conv2d(128→256, k=4, s=2, p=1) → BN2d → LeakyReLU(0.2) → Dropout2d
    ↓ Flatten
Latent Space: FC(flat_size→128) [μ, logσ²]
    ↓ Reparameterization
    ↓
Decoder: FC(128→flat_size)
    ↓ Reshape (256×H×W)
    ↓
Decoder: ConvT2d(256→128, k=4, s=2, p=1) → BN2d → LeakyReLU → Dropout2d
         ConvT2d(128→64, k=4, s=2, p=1) → BN2d → LeakyReLU → Dropout2d
         ConvT2d(64→32, k=4, s=2, p=1) → BN2d → LeakyReLU → Dropout2d
         ConvT2d(32→1, k=4, s=2, p=1)
    ↓
Output (1×128×1292)
```

### Advantages
✅ **Spatial awareness** - Captures 2D patterns in spectrograms (frequency×time)  
✅ **Local features** - Convolutional kernels detect local acoustic patterns  
✅ **Translation invariance** - Same pattern recognized anywhere in spectrogram  
✅ **12x more parameters** - Much higher capacity than Basic VAE  
✅ **Recommended by diagnostics** - "Try Conv VAE instead of Basic VAE"

### Why Conv VAE?
- Spectrograms are 2D images (frequency×time)
- Convolutional layers preserve spatial structure
- Better suited for audio than fully-connected layers
- Standard architecture for spectrogram processing

---

## 3️⃣ Beta-VAE (Disentangled)

**Architecture Type:** Convolutional (same as Conv VAE)  
**Status:** ⏳ Not yet trained  
**Parameters:** 64,458,049 (~64.5M) - **Same as Conv VAE**

### Configuration
```yaml
Base: ConvVAE (inherits all Conv VAE layers)
Latent Dim: 128
Hidden Channels: [32, 64, 128, 256]
Beta (β): 4.0  ← KEY DIFFERENCE
Dropout: 0.2
```

### Architecture Flow
**IDENTICAL to Conv VAE** except for loss function:

```python
# Beta-VAE Loss
total_loss = recon_loss + β × kl_loss

Where:
  recon_loss = MSE(reconstruction, original)
  kl_loss = KL(q(z|x) || p(z))
  β = 4.0 (higher weight on KL divergence)
```

### Key Differences from Conv VAE
- **Higher β value (4.0)** - Increases KL divergence weight
- **Disentanglement** - Encourages independent latent dimensions
- **Trade-off** - Sacrifices reconstruction quality for better separation

### Purpose
✅ **Disentangled representations** - Each latent dimension captures independent factor  
✅ **Better clustering** - Separated features easier to cluster  
⚠️ **Reconstruction quality** - May be worse than standard VAE due to high β

### Beta Value Impact
- **β = 1.0** → Standard VAE (balanced)
- **β = 4.0** → Strong disentanglement (current)
- **β > 4.0** → Extreme disentanglement (may degrade reconstruction)

---

## 4️⃣ CVAE-Language (Conditional VAE)

**Architecture Type:** Conditional Convolutional  
**Status:** ⏳ Not yet trained  
**Parameters:** 69,709,761 (~69.7M)

### Configuration
```yaml
Base: Similar to Conv VAE with conditioning
Latent Dim: 128
Hidden Channels: [32, 64, 128, 256]
Num Classes: 4 (Arabic, English, Hindi, Spanish)
Condition Embedding: 32 dimensions
Dropout: 0.2
```

### Architecture Flow
```
Input (1×128×1292) + Language Label
    ↓
Condition Embedding: Embedding(4→32)
    ↓
Encoder: Conv2d layers (same as Conv VAE)
         + Condition embedding concatenated
    ↓
Latent Space: FC(flat_size+32→128) [μ, logσ²]
    ↓ Reparameterization + Condition
    ↓
Decoder: FC(128+32→flat_size)
         ConvT2d layers (same as Conv VAE)
    ↓
Output (1×128×1292)
```

### Key Features
✅ **Controlled generation** - Can generate samples for specific language  
✅ **Language-aware** - Explicitly models language variations  
✅ **Label conditioning** - Uses language labels during training  
✅ **32D embeddings** - Learnable language representations

### Purpose
- Learn language-specific patterns
- Better separation of language clusters
- Enable controlled generation (e.g., "generate Arabic hiphop")

---

## 5️⃣ CVAE-Genre (Conditional VAE)

**Architecture Type:** Conditional Convolutional  
**Status:** ⏳ Not yet trained  
**Parameters:** 69,709,729 (~69.7M)

### Configuration
```yaml
Base: Same as CVAE-Language
Latent Dim: 128
Hidden Channels: [32, 64, 128, 256]
Num Classes: 3 (hiphop, pop, rock)
Condition Embedding: 32 dimensions
Dropout: 0.2
```

### Architecture Flow
**IDENTICAL to CVAE-Language** except conditioning on genre (3 classes) instead of language (4 classes)

### Key Features
✅ **Genre-aware** - Explicitly models genre variations  
✅ **Controlled generation** - Can generate samples for specific genre  
✅ **Label conditioning** - Uses genre labels during training  
✅ **Current best task** - Genre clustering (NMI=0.0651) already 2x better than language

### Purpose
- Learn genre-specific acoustic patterns
- Improve genre clustering (already best performing task)
- Enable controlled generation (e.g., "generate rock music")

---

## 6️⃣ VaDE (Variational Deep Embedding)

**Architecture Type:** Fully-connected with GMM priors  
**Status:** ⏳ Not yet trained  
**Parameters:** 169,879,567 (~169.9M) - **LARGEST MODEL**

### Configuration
```yaml
Input: 165,376 (flattened)
Latent Dim: 128
Hidden Dims: [512, 256]  ← Much DEEPER than Basic VAE
N Clusters: 15 (4 languages × 3 genres + 3 extra)
Dropout: 0.2
```

### Architecture Flow
```
Input (165,376)
    ↓
Encoder: FC(165376→512) → BN → ReLU → Dropout
         FC(512→256) → BN → ReLU → Dropout
    ↓
Latent Space: FC(256→128) [μ, logσ²]
    ↓ Reparameterization
    ↓
GMM Component Assignment (15 clusters)
    ↓
Decoder: FC(128→256) → BN → ReLU → Dropout
         FC(256→512) → BN → ReLU → Dropout
         FC(512→165376)
    ↓
Output (165,376)
```

### GMM Parameters (Learnable)
```python
π_k: Mixture weights [15]           # Cluster probabilities
μ_k: Cluster means [15 × 128]       # Center of each cluster
σ²_k: Cluster variances [15 × 128]  # Spread of each cluster
```

### Key Features
✅ **Joint clustering** - Learns clusters during training (not post-hoc)  
✅ **GMM priors** - Gaussian Mixture Model in latent space  
✅ **15 clusters** - Matches 4 languages × 3 genres = 12 + extras  
✅ **Deepest FC architecture** - [512, 256] vs Basic VAE [16, 32, 64]  
✅ **End-to-end** - Clustering integrated into loss function

### Loss Function
```python
total_loss = recon_loss + kl_loss + gmm_loss

Where:
  recon_loss = MSE(reconstruction, original)
  kl_loss = KL(q(z|x) || p(z|c))  # Conditioned on cluster
  gmm_loss = -log p(c|z)          # Cluster assignment probability
```

### Why VaDE?
- **Purpose-built for clustering** - Not post-hoc like K-means
- **Probabilistic clusters** - Soft assignments via GMM
- **Pre-initialized** - Uses K-means or GMM for initialization
- **Largest capacity** - 169.9M parameters (31x Basic VAE)

---

## 7️⃣ Multimodal VAE (Audio + Lyrics)

**Architecture Type:** Hybrid (Conv + Transformer)  
**Status:** ⏳ Not yet trained  
**Parameters:** TBD (depends on fusion strategy)

### Configuration
```yaml
Audio Encoder: ConvVAE or BasicVAE
Lyrics Encoder: XLM-RoBERTa-base (multilingual)
Fusion Type: attention  # Options: early, late, attention
Audio Weight: 0.5
Lyrics Weight: 0.5
Latent Dim: 128 (combined)
```

### Fusion Strategies

#### Option 1: Early Fusion
```
Audio Features (165,376) + Lyrics Features (768)
    ↓ Concatenate
Combined (166,144)
    ↓ Joint Encoder
Latent Space (128)
```

#### Option 2: Late Fusion
```
Audio → Audio Encoder → Audio Latent (64)
Lyrics → Lyrics Encoder → Lyrics Latent (64)
    ↓ Concatenate
Combined Latent (128)
```

#### Option 3: Attention Fusion (Current)
```
Audio → Audio Encoder → Audio Latent
Lyrics → Lyrics Encoder → Lyrics Latent
    ↓ Cross-Modal Attention
    ↓ Weighted Fusion (0.5 audio, 0.5 lyrics)
Fused Latent (128)
```

### Components
- **Audio:** Mel spectrograms (128×1292)
- **Lyrics:** Multilingual text via XLM-RoBERTa-base (279M params)
- **Fusion:** Cross-modal attention with 8 heads

### Purpose
✅ **Multimodal learning** - Combines acoustic + linguistic features  
✅ **Language task** - Lyrics should help language classification  
✅ **Genre task** - Audio dominates, lyrics provide context  
✅ **Best of both** - Acoustic patterns + lyrical content

---

## 📈 Current Training Configuration

**From configs/config.yaml:**

### Data Settings
```yaml
Sample Rate: 22,050 Hz
Audio Duration: null (full songs, 3-5 minutes)
Windowing: true
  Window Size: 30s
  Hop Size: 20s (33% overlap)
Mel Bins: 128
N_FFT: 2048
Hop Length: 512
```

### Training Settings
```yaml
Batch Size: 32
Epochs: 100
Learning Rate: 0.0001
Weight Decay: 0.0001
Early Stopping Patience: 15
Gradient Clip: 0.5
Mixed Precision: true (FP16)
Optimizer: Adam
```

### Hardware Constraints
```yaml
GPU: RTX 3060 12GB
RAM: 16GB DDR5
CPU: Ryzen 7700 (8 cores)
Num Workers: 0 (keep PC responsive)
```

---

## 🎯 Architecture Comparison

### By Parameter Count
```
VaDE          169.9M  ████████████████████████████████████ (largest)
CVAE-Language  69.7M  ██████████████
CVAE-Genre     69.7M  ██████████████
Conv VAE       64.5M  █████████████
Beta-VAE       64.5M  █████████████
Basic VAE       5.5M  █ (baseline)
Multimodal      TBD
```

### By Architecture Type
```
Convolutional (CNN):
  - Conv VAE (64.5M)
  - Beta-VAE (64.5M)
  - CVAE-Language (69.7M)
  - CVAE-Genre (69.7M)

Fully-Connected (FC):
  - Basic VAE (5.5M)
  - VaDE (169.9M)

Hybrid:
  - Multimodal VAE (TBD)
```

### By Task Suitability

**Language Classification (4 classes):**
1. **CVAE-Language** - Explicitly conditioned on language
2. **Multimodal VAE** - Lyrics + audio
3. **VaDE** - Joint clustering with 15 components
4. Conv VAE - Spatial patterns
5. Basic VAE - Current: NMI=0.0314 (poor)

**Genre Classification (3 classes):**
1. **CVAE-Genre** - Explicitly conditioned on genre
2. **Conv VAE** - Best for acoustic patterns
3. **Beta-VAE** - Disentangled features
4. VaDE - Joint clustering
5. Basic VAE - Current: NMI=0.0651 (poor but better than language)

---

## 🔧 Recommendations

### Priority 1: Train Conv VAE
**Why?** 
- Spectrograms are 2D images → need spatial processing
- 12x more parameters than Basic VAE (64.5M vs 5.5M)
- Standard architecture for audio spectrograms
- Diagnostic script recommended this

**Expected Improvement:** 2-5x better clustering (NMI 0.1-0.3)

### Priority 2: Train Beta-VAE
**Why?**
- Disentanglement helps clustering
- Same architecture as Conv VAE but β=4.0
- Better feature separation

**Expected Improvement:** Similar to Conv VAE, possibly better clustering

### Priority 3: Train CVAE-Genre
**Why?**
- Genre already best performing task (NMI=0.0651)
- Conditioning on genre should amplify this
- 69.7M parameters

**Expected Improvement:** Significant genre clustering boost

### Priority 4: Train VaDE
**Why?**
- Purpose-built for clustering (GMM priors)
- Largest model (169.9M params)
- End-to-end clustering

**Risk:** Very large, may overfit or be slow

### Priority 5: Train CVAE-Language
**Why?**
- Language task currently weakest (NMI=0.0314)
- Conditioning might help

**Expected:** Moderate improvement

### Priority 6: Train Multimodal
**Why?**
- Lyrics should help language task
- Most complex to implement

**Risk:** High complexity, may not converge

### Priority 7: Avoid Deeper Basic VAE
**Evidence:** Already tried latent=288, hidden=[80,160,320] → **WORSE results**  
**Conclusion:** Fully-connected is wrong architecture for spectrograms

---

## 📊 Performance Baseline (Basic VAE)

**Dataset:** 2107 windowed samples (11.7x augmentation)

### Current Results
| Task | Best Method | NMI | ARI | Silhouette |
|------|-------------|-----|-----|------------|
| **Genre** | Agglomerative | 0.0651 | 0.0258 | 0.0793 |
| **Language** | Agglomerative | 0.0314 | 0.0051 | 0.0256 |

### Target Results (Good Clustering)
| Task | Target NMI | Target ARI | Target Silhouette |
|------|------------|------------|-------------------|
| **Genre** | > 0.3 | > 0.2 | > 0.3 |
| **Language** | > 0.3 | > 0.2 | > 0.3 |

**Gap to Close:** 3-5x improvement needed

---

## 🚀 Next Steps

1. **Train Conv VAE** (highest priority) - Expected: 2-5x improvement
2. **Train Beta-VAE** - Expected: Similar to Conv VAE with better disentanglement
3. **Train CVAE-Genre** - Expected: Significant genre boost
4. **Compare Results** - Identify best architecture
5. **Train VaDE** - If Conv VAE works well
6. **Train CVAE-Language** - If language task improves
7. **Train Multimodal** - Final experiment

**Estimated Training Time:** ~15 minutes per model (7 models × 15 min = 1.75 hours total)

---

**Report End**
