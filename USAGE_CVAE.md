# CVAE for PSF Interpolation - Usage Guide

This document explains the **Conditional Variational Autoencoder (CVAE)** implementation for Point Spread Function (PSF) interpolation with grid sampling and smoothness prior.

---

## Overview

The CVAE model learns to:
1. **Encode** sparse PSF kernels (15×15) into a latent space conditioned on spatial coordinates
2. **Generate** PSFs at arbitrary coordinates by sampling from a learned prior distribution
3. **Interpolate** smoothly across the field of view using spatial smoothness constraints

### Key Features

✅ **Uniform N×N Grid Sampling**: Ensures spatial consistency for neighbor-based smoothness loss
✅ **Learned Conditional Prior**: p(z|c) instead of standard normal prior
✅ **Fourier Positional Encoding**: Rich coordinate representation with L=10 frequency bands
✅ **Physical Constraints**: Non-negativity (softplus) and normalization (sum to 1)
✅ **Smoothness Regularization**: Penalizes sudden changes in latent space across neighbors
✅ **On-the-fly PSF Generation**: Uses DeepLens raytracing for synthetic data

---

## Architecture

### 1. Data Strategy: Grid Sampling

**Problem**: Random sampling is insufficient for spatial smoothness loss (need neighbors).

**Solution**: Generate PSFs on a uniform **N×N grid** (default N=20, yielding 400 points):

```python
Grid coordinates: [-1, 1] × [-1, 1]
Grid points: 20 × 20 = 400 PSFs per focal depth
Neighbor structure: 4-connected (right, bottom, diagonals)
```

**Data Flow**:
1. `setup_grid()`: Creates uniform grid and neighbor mapping
2. `generate_grid_psfs(foc_z)`: Ray-traces PSFs for all grid points using DeepLens
3. `feed_data()`: Samples batches from grid for training
4. Caching: Grid PSFs cached per epoch to avoid redundant raytracing

**File**: `models/PSFlatent/CVAE_PSF_model.py:setup_grid()`, `generate_grid_psfs()`

### 2. Network Architecture

#### **Inputs**
- **PSF Kernel K**: 15×15×3 (RGB), flattened to 675-dim
- **Coordinates c**: (x, y) normalized to [-1, 1]

#### **Positional Encoding**

Fourier features for coordinate representation:

```
γ(c) = [sin(2^0 π c), cos(2^0 π c), ..., sin(2^(L-1) π c), cos(2^(L-1) π c)]
```

- **Frequency bands L**: 10 (configurable)
- **Output dimension**: 2 × 10 × 2 = 40 features

**File**: `models/archs/CVAE_PSF_arch.py:PositionalEncoding`

#### **Encoder (Posterior Network)**

```
Input: [K_flat, PE(c)] → 675 + 40 = 715 dims
MLP: 715 → 512 → 512 → 512 → 512
Output: μ_q (128), log σ²_q (128)
```

Learns: **q(z | K, c)** (posterior distribution)

**File**: `models/archs/CVAE_PSF_arch.py:encode()`

#### **Prior Network**

```
Input: PE(c) → 40 dims
MLP: 40 → 512 → 512 → 512 → 512
Output: μ_p (128), log σ²_p (128)
```

Learns: **p(z | c)** (conditional prior, NOT standard normal!)

**File**: `models/archs/CVAE_PSF_arch.py:prior_network()`

#### **Decoder**

```
Input: [z, PE(c)] → 128 + 40 = 168 dims
MLP: 168 → 512 → 512 → 512 → 512 → 675
Physical Constraints:
  1. Softplus: K̂ = softplus(MLP_output)
  2. Normalize: K̂ = K̂ / sum(K̂) per channel
Output: K̂ (15×15×3)
```

Learns: **p(K | z, c)**

**File**: `models/archs/CVAE_PSF_arch.py:decode()`, `apply_physical_constraints()`

### 3. Loss Function

**Total Loss**:
```
L = λ_recon * L_recon + λ_kl * L_kl + λ_smooth * L_smooth
```

**Components**:

1. **Reconstruction Loss** (λ_recon = 1.0):
   ```
   L_recon = MSE(K̂, K) = ||K̂ - K||²
   ```

2. **KL Divergence** (λ_kl = 1e-4):
   ```
   L_kl = D_KL(q(z|K,c) || p(z|c))
        = -0.5 * Σ[1 + log σ²_q - log σ²_p - (σ²_q + (μ_q - μ_p)²) / σ²_p]
   ```
   Analytical form for two Gaussians.

3. **Smoothness Regularization** (λ_smooth = 1e-3):
   ```
   L_smooth = (1/N_pairs) * Σ_{neighbors} ||μ_p(c_i) - μ_p(c_neighbor)||²
   ```
   Penalizes sudden changes in prior mean across spatial neighbors.

**File**: `models/PSFlatent/CVAE_PSF_model.py:optimize_parameters()`

---

## File Structure

```
models/
├── PSFlatent/
│   └── CVAE_PSF_model.py          # Model class with grid sampling and training loop
models/archs/
└── CVAE_PSF_arch.py                # Network architecture (encoder/prior/decoder)
checkpoints/PSF/
└── CVAE_v1.yml                     # Training configuration
USAGE_CVAE.md                       # This file
```

---

## Configuration

### Key Hyperparameters (`checkpoints/PSF/CVAE_v1.yml`)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `grid_size` | 20 | N×N grid size (400 points) |
| `kernel_size_small` | 15 | Sparse PSF kernel (15×15) |
| `latent_dim` | 128 | Latent space dimension |
| `hidden_dim` | 512 | MLP hidden layer width |
| `num_layers` | 4 | MLP depth |
| `num_frequencies` | 10 | Positional encoding bands |
| `batch_size` | 64 | Training batch size |
| `learning_rate` | 1e-4 | Adam learning rate |
| `recon_weight` | 1.0 | Reconstruction loss weight |
| `kl_weight` | 1e-4 | KL divergence weight (β) |
| `smooth_weight` | 1e-3 | Smoothness loss weight |

### DeepLens Raytracing Settings

| Parameter | Value | Description |
|-----------|-------|-------------|
| `lens_file` | `.../acl12708u.json` | Thorlabs lens specification |
| `spp` | 100000 | Samples per pixel for raytracing |
| `kernel_size` | 64 | Full-res PSF (downsampled to 15) |
| `fov` | 0.015 m | Field of view |
| `depth_min/max` | 0.4-20 m | Depth range |

---

## Training

### Quick Start

```bash
# Test configuration (validation only)
accelerate launch trainer.py -opt checkpoints/PSF/CVAE_v1.yml -test

# Full training (single GPU)
accelerate launch trainer.py -opt checkpoints/PSF/CVAE_v1.yml

# Multi-GPU training (Accelerate auto-detects GPUs)
accelerate launch trainer.py -opt checkpoints/PSF/CVAE_v1.yml
```

### Training Process

1. **Grid Setup**: Creates 20×20 uniform grid and neighbor mapping
2. **Data Generation**: Ray-traces PSFs for grid points using DeepLens
3. **Batch Sampling**: Randomly samples 64 points from grid per iteration
4. **Forward Pass**: Encoder → Reparameterization → Decoder
5. **Loss Computation**: Reconstruction + KL + Smoothness
6. **Validation**: Visualizes reconstructed PSFs and prior samples every 500 steps

### Monitoring

**Comet ML Tracker**:
- Project: `PSF_CVAE`
- Experiment: `CVAE_PSF_GridSampling_v1`

**Logged Metrics**:
- `loss_all`: Total loss
- `loss_recon`: Reconstruction MSE
- `loss_kl`: KL divergence
- `loss_smooth`: Smoothness penalty
- `val_mse`: Validation MSE

**Logged Images**:
- `val_psf_gt`: Ground truth PSFs (grid visualization)
- `val_psf_recon`: Reconstructed PSFs from posterior
- `val_psf_prior`: Generated PSFs from prior (unconditional)

---

## Validation & Inference

### Validation Mode

During validation, the model:
1. Generates grid PSFs sequentially (no random sampling)
2. Reconstructs PSFs using encoder → decoder path
3. Generates PSFs using prior → decoder path (tests prior quality)
4. Visualizes results as N×N grids

### Inference (PSF Interpolation)

To generate PSFs at arbitrary coordinates:

```python
from models import create_model
import torch

# Load trained model
model = create_model(opt, logger)
model.prepare()

# Define query coordinates
coords = torch.tensor([[0.5, 0.5], [-0.3, 0.8], [0.0, 0.0]])  # (3, 2)
coords = coords.to(device)

# Generate PSFs from prior
with torch.no_grad():
    psf_flat = model.net_cvae.sample_from_prior(coords)
    psf = psf_flat.view(-1, 3, 15, 15)  # (3, 3, 15, 15)
```

---

## Expected Behavior

### Convergence

- **Reconstruction loss** should decrease rapidly (< 1e-3 after 10k steps)
- **KL divergence** should stabilize (balance between posterior and prior)
- **Smoothness loss** should decrease gradually (prior becomes spatially coherent)

### Failure Modes

❌ **KL Collapse**: KL → 0, prior ignored. Fix: Increase `kl_weight`
❌ **Posterior Collapse**: KL explodes, decoder ignores latent. Fix: Decrease `kl_weight`, increase `latent_dim`
❌ **Non-smooth Prior**: Smoothness loss doesn't decrease. Fix: Increase `smooth_weight`, check grid neighbors
❌ **Poor Reconstruction**: MSE high. Fix: Increase `hidden_dim`, `num_layers`, decrease `kl_weight`

---

## Hyperparameter Tuning

### Loss Weight Balancing

Start with defaults, then adjust based on training dynamics:

| Issue | Adjustment |
|-------|------------|
| Blurry reconstructions | Increase `recon_weight` or decrease `kl_weight` |
| Prior doesn't match data | Decrease `kl_weight` |
| Non-smooth interpolation | Increase `smooth_weight` |
| Training unstable | Decrease all weights by 10× |

### Network Capacity

| Issue | Adjustment |
|-------|------------|
| Underfitting (high MSE) | Increase `hidden_dim`, `num_layers`, `latent_dim` |
| Overfitting | Add dropout, decrease `hidden_dim` |
| Slow training | Decrease `hidden_dim`, `num_layers` |

---

## Advanced Usage

### Custom Grid Resolution

Change `grid_size` in config:
```yaml
grid_size: 30  # 30×30 = 900 points (denser sampling)
```

Trade-off:
- **Larger grid**: Better spatial coverage, slower raytracing
- **Smaller grid**: Faster training, coarser smoothness

### Variable Depth

Modify `feed_data()` to vary depth per grid point:
```python
z = torch.rand(num_points) * 0.6 + 0.2  # [0.2, 0.8] depth range
```

### Alternative Smoothness Loss

Current: L2 on prior mean. Alternatives:
- L2 on posterior mean (requires batch contains neighbors)
- L2 on decoded PSFs (stricter constraint)
- Total variation regularization

---

## Troubleshooting

### Issue: "Grid PSFs not generated"

**Cause**: DeepLens raytracing failed.
**Fix**: Check `lens_file` path, ensure DeepLens is installed.

### Issue: "Smoothness loss = 0"

**Cause**: Neighbor dictionary empty or prior network not updating.
**Fix**: Check `setup_grid()` output, verify `smooth_weight > 0`.

### Issue: "CUDA out of memory"

**Cause**: Grid too large or batch size too high.
**Fix**: Reduce `grid_size`, `batch_size`, or `hidden_dim`.

### Issue: "NaN losses"

**Cause**: Numerical instability in log variance or normalization.
**Fix**: Add epsilon to log/div operations, decrease learning rate.

---

## Citation & References

This implementation is based on:

1. **VAE**: Kingma & Welling, "Auto-Encoding Variational Bayes" (2014)
2. **Conditional VAE**: Sohn et al., "Learning Structured Output Representation using Deep Conditional Generative Models" (2015)
3. **Positional Encoding**: Mildenhall et al., "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis" (2020)
4. **Smoothness Prior**: Inspired by spatial coherence in optical PSFs

---

## Future Extensions

### Potential Improvements

1. **Hierarchical VAE**: Multi-scale latent representation for better interpolation
2. **Flow-based Prior**: Replace Gaussian prior with normalizing flow for flexibility
3. **Attention Mechanism**: Cross-attention between coordinates and latent codes
4. **Depth Conditioning**: Explicit depth input for depth-varying PSFs
5. **Adversarial Training**: Add discriminator for sharper PSFs

### Research Directions

- **Zero-shot Interpolation**: Test on real lens systems not seen during training
- **Uncertainty Quantification**: Use posterior variance for confidence estimation
- **Multi-wavelength Modeling**: Separate latents per color channel
- **Real Data Finetuning**: Adapt synthetic model to measured PSFs

---

## Contact & Support

For questions or issues:
1. Check this documentation
2. Review `VAE_INTEGRATION_ROADMAP.md` for framework details
3. Examine training logs in Comet ML
4. Inspect checkpoint files in `experiments/` directory

**Model Files**:
- Model: `models/PSFlatent/CVAE_PSF_model.py`
- Architecture: `models/archs/CVAE_PSF_arch.py`
- Config: `checkpoints/PSF/CVAE_v1.yml`

**Key Classes**:
- `CVAE_PSF_model`: Training loop, grid sampling, loss computation
- `CVAE_PSF_arch`: Network architecture
- `PositionalEncoding`: Fourier feature transformation

---

**Last Updated**: 2025-01-XX
**Version**: 1.0
**Status**: ✅ Ready for Training
