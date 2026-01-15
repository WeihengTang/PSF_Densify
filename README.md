# Conditional VAE for PSF Interpolation

## Overview

This document describes the implementation of a **Conditional Variational Autoencoder (CVAE)** for spatially-varying Point Spread Function (PSF) interpolation, added to the PSF_Densify repository.

The CVAE learns a continuous latent representation of PSFs across a 4D coordinate space, enabling:
- **PSF reconstruction** from noisy/incomplete observations
- **Smooth spatial interpolation** between measured PSF locations
- **PSF generation** at novel spatial coordinates
- **Learned prior** `p(z|c)` that captures spatial and optical structure

---

## Version History

### v3.0 (January 2026) - 4D Coordinate Sampling with Stratified Jitter

**Major upgrade to support realistic volumetric PSF sampling for smartphone cameras.**

#### New Features:
- **4D Coordinate Space**: Extended from 2D (x, y) to 4D (x, y, z, f)
  - `x, y`: Spatial field coordinates (100×100 grid, up from 20×20)
  - `z`: Object depth (50 steps, range 0.1m - 3.0m)
  - `f`: Focus distance (50 steps, range 0.1m - 3.0m)

- **Stratified Sampling with Jitter**: Prevents aliasing and overfitting
  - Divides 4D space into grid cells
  - Samples uniformly within each cell: `c = c_grid + Uniform(-δ, δ)`
  - Cell half-widths: δ_xy=0.01, δ_z=0.01, δ_f=0.01

- **Smartphone-Scale Physical Ranges**:
  - Depth range: 0.1m (macro) to 3.0m (far field)
  - Focus range: 0.1m to 3.0m
  - FOV: ~0.015m (typical smartphone sensor)

#### Files Modified:
- `models/archs/CVAE_PSF_arch.py`: Updated PositionalEncoding and config for 4D
- `models/PSFlatent/CVAE_PSF_model.py`: New sampling methods, 4D grid setup
- `checkpoints/PSF/CVAE_*.yml`: New configuration parameters

#### New Configuration Parameters:
```yaml
grid_size_xy: 100    # Spatial grid resolution
grid_size_z: 50      # Depth sampling steps
grid_size_f: 50      # Focus sampling steps
z_min: 0.1           # Min depth (meters)
z_max: 3.0           # Max depth (meters)
f_min: 0.1           # Min focus (meters)
f_max: 3.0           # Max focus (meters)
use_jitter: true     # Enable stratified jitter
samples_per_batch: 256
coord_dim: 4         # 4D coordinates
```

---

### v2.0 (December 2025) - Bug Fixes and Stability Improvements

**Critical fixes for training stability and posterior collapse prevention.**

#### Bug Fixes:

1. **Smoothness Loss Gradient Flow** (2025-12-03)
   - **Problem**: Grid prior computation wrapped in `torch.no_grad()`, preventing backpropagation
   - **Solution**: Removed wrapper to enable gradient flow
   - **Impact**: Smoothness loss now constrains prior network properly

2. **Posterior Collapse Prevention** (2025-12-03)
   - **Problem**: Reconstructed PSFs were simple blobs, encoder ignored input
   - **Root Causes**: KL weight too high, MSE-only loss, no per-dimension protection
   - **Solutions**:
     - Enhanced reconstruction: `MSE + 0.5·L1 + 0.1·Gradient`
     - Reduced KL weight: `1e-4 → 1e-5`
     - Added Free Bits technique (min 0.5 bits per dimension)

3. **Validation-Only Mode** (2025-12-02)
   - **Problem**: `-test` flag didn't skip training
   - **Solution**: Set `opt['is_train'] = False` when flag used

4. **Validation Image Logging** (2025-12-02)
   - **Problem**: `log_image()` received 3D tensor, expected 4D
   - **Solution**: Added batch dimension with `np.newaxis`

5. **Double Network Initialization** (2025-12-02)
   - **Problem**: `KeyError` from consuming `'type'` field twice
   - **Solution**: Call `BaseModel.__init__()` directly

#### Loss Component Toggles Added:
```yaml
use_l1_loss: true/false
use_gradient_loss: true/false
use_smooth_loss: true/false
use_free_bits: true/false
```

---

### v1.0 (December 2025) - Initial CVAE Implementation

**First release of Conditional VAE for PSF interpolation.**

#### Core Features:
- Conditional VAE architecture with learned prior `p(z|c)`
- Fourier positional encoding for coordinates
- MLP-based encoder, prior network, and decoder
- Physical constraints: non-negativity (softplus) and normalization
- 20×20 spatial grid sampling
- DeepLens integration for on-the-fly PSF generation
- Three-component loss: reconstruction + KL divergence + smoothness

#### Files Added:
- `models/PSFlatent/CVAE_PSF_model.py`
- `models/archs/CVAE_PSF_arch.py`
- `checkpoints/PSF/CVAE_v1.yml`

---

## Architecture

### Model Overview

```
Input: PSF K (15×15×3) + 4D Coordinate c (x, y, z, f)
       ↓
    Encoder q(z|K,c)
       ↓
    Latent z (128-dim)
       ↓
    Decoder p(K|z,c)
       ↓
Output: Reconstructed PSF K̂ (15×15×3)

Alongside:
    Prior Network p(z|c)
    (Learned conditional prior based on 4D coordinate)
```

### Network Components

#### 1. **Positional Encoding** (Fourier Features)

Transforms 4D coordinates into high-dimensional representation:

```python
PE(c) = [sin(2^0·π·c), cos(2^0·π·c),
         sin(2^1·π·c), cos(2^1·π·c),
         ...,
         sin(2^9·π·c), cos(2^9·π·c)]
```

- Input: `c ∈ ℝ⁴` (x, y in [-1,1], z, f in [0,1])
- Output: `PE(c) ∈ ℝ⁸⁰` (10 frequencies × 4 dims × 2 functions)
- Purpose: Captures high-frequency spatial and optical variations

#### 2. **Encoder** (Posterior Network) `q(z|K,c)`

Maps PSF and coordinate to latent distribution:

```
Input: [Flattened PSF, PE(c)] → ℝ^(675 + 80) = ℝ^755
  ↓
MLP (4 layers, 512 hidden units, ReLU)
  ↓
μ_q ∈ ℝ^128, log σ²_q ∈ ℝ^128
```

#### 3. **Prior Network** `p(z|c)`

Learned conditional prior based on 4D location:

```
Input: PE(c) → ℝ^80
  ↓
MLP (4 layers, 512 hidden units, ReLU)
  ↓
μ_p ∈ ℝ^128, log σ²_p ∈ ℝ^128
```

#### 4. **Decoder** `p(K|z,c)`

Reconstructs PSF from latent code and coordinate:

```
Input: [z, PE(c)] → ℝ^(128 + 80) = ℝ^208
  ↓
MLP (4 layers, 512 hidden units, ReLU)
  ↓
Physical Constraints:
  - Softplus activation (non-negativity)
  - Per-channel normalization (∑K = 1)
  ↓
Output: K̂ ∈ ℝ^(3×15×15)
```

### 4D Stratified Sampling Strategy

During training, PSFs are sampled from the 4D coordinate space with jittered stratification:

```
For each sample:
  1. Select random grid cell indices (i_x, i_y, i_z, i_f)
  2. Get cell center: c_center = (x_centers[i_x], y_centers[i_y], z_centers[i_z], f_centers[i_f])
  3. Add jitter: c = c_center + Uniform(-δ, δ) for each dimension
  4. Clamp to valid ranges
```

**Benefits**:
- Prevents aliasing from regular grid sampling
- Better coverage of continuous 4D space
- Reduces overfitting to specific coordinates

---

## Loss Function

The total loss is a weighted combination of components:

### 1. **Reconstruction Loss** (Primary objective)

```python
L_recon = MSE(K̂, K) + 0.5·L1(K̂, K) + 0.1·L_grad(K̂, K)
```

### 2. **KL Divergence** (Regularization)

```python
KL(q(z|K,c) || p(z|c)) = ½·∑_d [σ²_q/σ²_p + (μ_q - μ_p)²/σ²_p - 1 - log(σ²_q/σ²_p)]
```

With Free Bits: `KL_per_dim = max(KL_dim, 0.5)`

### 3. **Smoothness Regularization** (Spatial structure)

```python
L_smooth = (1/N_pairs) · ∑_{(i,j) neighbors} ||μ_p(c_i) - μ_p(c_j)||²
```

### Total Loss

```python
L_total = 1.0·L_recon + 1e-5·L_KL + 1e-3·L_smooth
```

---

## Training

### Command

```bash
# Single GPU training
accelerate launch trainer.py -opt checkpoints/PSF/CVAE_v1.yml

# Validation-only mode (requires trained checkpoint)
accelerate launch trainer.py -opt checkpoints/PSF/CVAE_v1.yml -test
```

### Hyperparameters (v3.0)

```yaml
# Network Architecture
latent_dim: 128
hidden_dim: 512
num_layers: 4
num_frequencies: 10
coord_dim: 4

# 4D Grid Sampling
grid_size_xy: 100
grid_size_z: 50
grid_size_f: 50
use_jitter: true
samples_per_batch: 256

# Physical Ranges (meters)
z_min: 0.1
z_max: 3.0
f_min: 0.1
f_max: 3.0

# Loss Weights
recon_weight: 1.0
kl_weight: 1e-5
smooth_weight: 1e-3

# Optimization
learning_rate: 1e-4
batch_size: 64
max_train_steps: 100000
```

---

## Integration with Existing Framework

### Model Discovery

```python
# models/__init__.py scans for *_model.py files
model_type: CVAE_PSF_model  # In YAML config
→ finds class CVAE_PSF_model in models/PSFlatent/CVAE_PSF_model.py
```

### Architecture Discovery

```python
# models/archs/__init__.py scans for *_arch.py files
network:
  type: CVAE_PSF
→ finds CVAE_PSF_config and CVAE_PSF_arch in models/archs/CVAE_PSF_arch.py
```

### Dataset Integration

Uses existing `DummyDataset` with on-the-fly PSF generation via DeepLens raytracing.

---

## Future Improvements

1. **Convolutional Decoder** - Better spatial PSF structure
2. **Hierarchical Latent Space** - Separate global/local aberrations
3. **Adaptive Grid Sampling** - Denser in high-variation regions
4. **Multi-scale Training** - Multiple PSF sizes
5. **Uncertainty Quantification** - Use posterior variance

---

## References

1. **Conditional VAE**: Sohn, K., et al. "Learning Structured Output Representation using Deep Conditional Generative Models." NeurIPS 2015.
2. **Free Bits**: Kingma, D. P., et al. "Improved Variational Inference with Inverse Autoregressive Flow." NeurIPS 2016.
3. **Positional Encoding**: Tancik, M., et al. "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains." NeurIPS 2020.

---

## Contact

For questions about this CVAE implementation, please open an issue on GitHub.

**Latest Version**: v3.0 (January 2026)
**Framework Version**: PSF_Densify (post-ICASSP submission)
