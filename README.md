# Conditional VAE for PSF Interpolation

## Overview

This document describes the implementation of a **Conditional Variational Autoencoder (CVAE)** for spatially-varying Point Spread Function (PSF) interpolation, added to the PSF_Densify repository.

The CVAE learns a continuous latent representation of PSFs across spatial coordinates, enabling:
- **PSF reconstruction** from noisy/incomplete observations
- **Smooth spatial interpolation** between measured PSF locations
- **PSF generation** at novel spatial coordinates
- **Learned prior** `p(z|c)` that captures spatial structure

---

## Architecture

### Model Overview

```
Input: PSF K (15×15×3) + Spatial Coordinate c (2D)
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
    (Learned conditional prior based on spatial coordinate)
```

### Network Components

#### 1. **Positional Encoding** (Fourier Features)

Transforms 2D coordinates into high-dimensional representation:

```python
PE(c) = [sin(2^0·π·c), cos(2^0·π·c),
         sin(2^1·π·c), cos(2^1·π·c),
         ...,
         sin(2^9·π·c), cos(2^9·π·c)]
```

- Input: `c ∈ ℝ²` (normalized coordinates in [-1, 1])
- Output: `PE(c) ∈ ℝ⁴⁰` (10 frequencies × 2 dims × 2 functions)
- Purpose: Captures high-frequency spatial variations

#### 2. **Encoder** (Posterior Network) `q(z|K,c)`

Maps PSF and coordinate to latent distribution:

```
Input: [Flattened PSF, PE(c)] → ℝ^(675 + 40) = ℝ^715
  ↓
MLP (4 layers, 512 hidden units, ReLU)
  ↓
μ_q ∈ ℝ^128, log σ²_q ∈ ℝ^128
```

- **Purpose**: Encode input PSF into latent space
- **Output**: Posterior distribution parameters

#### 3. **Prior Network** `p(z|c)`

Learned conditional prior based on spatial location:

```
Input: PE(c) → ℝ^40
  ↓
MLP (4 layers, 512 hidden units, ReLU)
  ↓
μ_p ∈ ℝ^128, log σ²_p ∈ ℝ^128
```

- **Purpose**: Learn spatial structure of PSF variations
- **Key difference from standard VAE**: Prior depends on coordinate, not just N(0, I)

#### 4. **Decoder** `p(K|z,c)`

Reconstructs PSF from latent code and coordinate:

```
Input: [z, PE(c)] → ℝ^(128 + 40) = ℝ^168
  ↓
MLP (4 layers, 512 hidden units, ReLU)
  ↓
Raw output ∈ ℝ^675
  ↓
Physical Constraints:
  - Softplus activation (non-negativity)
  - Per-channel normalization (∑K = 1)
  ↓
Output: K̂ ∈ ℝ^(3×15×15)
```

### Grid Sampling Strategy

During training, PSFs are sampled on a **uniform N×N grid** (default 20×20 = 400 points):

```
Grid coordinates: {(x_i, y_j) | i,j ∈ [0, N-1]}
x_i = -1 + 2i/(N-1)
y_j = -1 + 2j/(N-1)
```

**Purpose**:
- Spatial consistency for smoothness loss
- Efficient batched training
- Neighbor structure for regularization

---

## Loss Function

The total loss is a weighted combination of three terms:

### 1. **Reconstruction Loss** (Primary objective)

```python
L_recon = MSE(K̂, K) + 0.5·L1(K̂, K) + 0.1·L_grad(K̂, K)
```

**Components**:
- **MSE**: Overall intensity matching
- **L1**: Preserves sharp elliptical boundaries (robust to outliers)
- **Gradient Loss**: Preserves shape structure and orientation
  ```python
  L_grad = L1(∇_x K̂, ∇_x K) + L1(∇_y K̂, ∇_y K)
  ```

**Weight**: `recon_weight = 1.0`

### 2. **KL Divergence** (Regularization)

Analytical KL between posterior and learned prior:

```python
KL(q(z|K,c) || p(z|c)) = ½·∑_d [σ²_q/σ²_p + (μ_q - μ_p)²/σ²_p - 1 - log(σ²_q/σ²_p)]
```

**Free Bits Technique** (prevents posterior collapse):
```python
KL_per_dim = max(KL_dim, free_bits=0.5)
```

**Weight**: `kl_weight = 1e-5` (small to prioritize reconstruction)

### 3. **Smoothness Regularization** (Spatial structure)

Penalizes differences in prior means for neighboring grid points:

```python
L_smooth = (1/N_pairs) · ∑_{(i,j) neighbors} ||μ_p(c_i) - μ_p(c_j)||²
```

**Purpose**:
- Ensures spatially smooth latent mappings
- Better interpolation between measured points
- Physically realistic (optical blur varies smoothly)

**Weight**: `smooth_weight = 1e-3`

### Total Loss

```python
L_total = 1.0·L_recon + 1e-5·L_KL + 1e-3·L_smooth
```

---

## Key Implementation Details

### Physical Constraints

PSFs must satisfy physical properties:

1. **Non-negativity**: `K(x,y) ≥ 0`
   - Implemented via Softplus activation

2. **Normalization**: `∑_{x,y} K(x,y,λ) = 1` per wavelength
   - Implemented via per-channel division by sum

### Reparameterization Trick

Enables backpropagation through stochastic sampling:

```python
z = μ + σ · ε,  where ε ~ N(0, I)
```

This allows gradients to flow through the sampling operation.

### Grid Neighbor Structure

For 20×20 grid, each point has 2-4 neighbors (4-connected):
- Corner points: 2 neighbors
- Edge points: 3 neighbors
- Interior points: 4 neighbors

Total neighbor pairs: ~760 for smoothness loss computation

---

## Files Added/Modified

### **New Files**

1. **`models/PSFlatent/CVAE_PSF_model.py`** (280 lines)
   - Main model class inheriting from `BaseModel`
   - Grid sampling logic
   - Three-component loss computation
   - DeepLens integration for on-the-fly PSF generation

2. **`models/archs/CVAE_PSF_arch.py`** (320 lines)
   - `CVAE_PSF_config`: Configuration class
   - `CVAE_PSF_arch`: Network architecture with encoder/prior/decoder
   - `PositionalEncoding`: Fourier feature transformation
   - `MLP`: Multi-layer perceptron building block

3. **`checkpoints/PSF/CVAE_v1.yml`** (3.3 KB)
   - Complete training configuration
   - Hyperparameters and loss weights
   - DeepLens settings

4. **Documentation**:
   - `USAGE_CVAE.md`: Comprehensive usage guide
   - `CVAE_IMPLEMENTATION_SUMMARY.md`: Technical reference
   - `test_cvae_setup.py`: Validation script

### **Modified Files**

1. **`trainer.py`**
   - Added validation-only mode support via `-test` flag
   - Sets `is_train = False` to skip training loop

2. **`CLAUDE.md`**
   - Updated training commands to use `accelerate launch`
   - Added CVAE model documentation

3. **All checkpoint YAMLs** (60 files)
   - Updated experiment output path from `/scratch/gilbreth/wweligam/` to `/scratch/gilbreth/tang843/`

---

## Critical Bug Fixes Applied

### 1. **Double Network Initialization** (Fixed: 2025-12-02)

**Problem**: `CVAE_PSF_model` called `PSF_model.__init__()` which called `define_network(opt.network)`, consuming the `'type'` field via `pop()`. Then CVAE tried to call it again → `KeyError`.

**Solution**: Call `BaseModel.__init__()` directly, copy DeepLens setup from PSF_model manually.

### 2. **Validation Image Logging Shape Error** (Fixed: 2025-12-02)

**Problem**: `log_image()` expects 4D tensors `(n, c, h, w)` but received 3D after slicing.

**Solution**: Add batch dimension with `np.newaxis` before logging.

### 3. **Validation-Only Mode Bug** (Fixed: 2025-12-02)

**Problem**: `-test` flag didn't actually skip training, still executed full 100k steps.

**Solution**: Set `opt['is_train'] = False` when `-test` flag is used, triggering existing skip logic.

### 4. **Smoothness Loss Gradient Flow** (Fixed: 2025-12-03)

**Problem**: Grid prior computation wrapped in `torch.no_grad()`, preventing smoothness loss from backpropagating.

**Solution**: Remove `with torch.no_grad():` wrapper around grid prior computation.

**Impact**: Smoothness loss now actually constrains the prior network to learn spatially smooth mappings.

### 5. **Posterior Collapse** (Fixed: 2025-12-03)

**Problem**:
- Reconstructed PSFs were simple circular blobs vs complex elliptical GT
- Reconstruction looked identical to prior samples
- Encoder was ignoring input, producing generic latent codes

**Root Causes**:
1. KL weight too high (1e-4)
2. MSE-only reconstruction loss
3. No per-dimension collapse prevention

**Solutions**:
1. **Enhanced Reconstruction Loss**:
   - Added L1 loss (0.5× weight) for sharp features
   - Added gradient loss (0.1× weight) for shape structure
   - Combined: `MSE + 0.5·L1 + 0.1·Grad`

2. **Reduced KL Weight**:
   - Changed from `1e-4` to `1e-5` (10× reduction)
   - Prioritizes reconstruction over distribution matching

3. **Free Bits Technique**:
   - Minimum KL per dimension = 0.5 bits
   - Prevents encoder from collapsing any dimension to prior
   - Forces all latent dimensions to encode information

---

## Training

### Command

```bash
# Single GPU training
accelerate launch trainer.py -opt checkpoints/PSF/CVAE_v1.yml

# Validation-only mode (requires trained checkpoint)
accelerate launch trainer.py -opt checkpoints/PSF/CVAE_v1.yml -test
```

### Hyperparameters

```yaml
# Network Architecture
latent_dim: 128        # Latent space dimensionality
hidden_dim: 512        # MLP hidden layer width
num_layers: 4          # Layers in each MLP
num_frequencies: 10    # Positional encoding frequency bands

# Loss Weights
recon_weight: 1.0      # Reconstruction (MSE + L1 + Grad)
kl_weight: 1e-5        # KL divergence (low to prevent collapse)
smooth_weight: 1e-3    # Spatial smoothness
free_bits: 0.5         # Minimum KL per dimension

# Optimization
learning_rate: 1e-4
scheduler: cosine_with_restarts
lr_num_cycles: 3
batch_size: 64
max_train_steps: 100000

# Grid Sampling
grid_size: 20          # 20×20 = 400 spatial points
kernel_size_small: 15  # PSF size (15×15)
```

### Expected Training Behavior

**Healthy training shows**:
1. `loss_recon` decreases steadily
2. `loss_kl` stabilizes at moderate value (not 0)
3. `loss_smooth` decreases (spatial consistency improving)
4. `val_mse` tracks `loss_recon` without overfitting

**Validation images**:
- `val_psf_gt`: Ground truth from DeepLens
- `val_psf_recon`: Should match GT (elliptical shapes, orientations)
- `val_psf_prior`: Should differ from recon (shows learned prior)

---

## Integration with Existing Framework

### Model Discovery

The framework automatically discovers the CVAE model via:

```python
# models/__init__.py scans for *_model.py files
model_type: CVAE_PSF_model  # In YAML config
→ finds class CVAE_PSF_model in models/PSFlatent/CVAE_PSF_model.py
```

### Architecture Discovery

```python
# models/archs/__init__.py scans for *_arch.py files
network:
  type: CVAE_PSF  # In YAML config
→ finds CVAE_PSF_config and CVAE_PSF_arch in models/archs/CVAE_PSF_arch.py
```

### Dataset Integration

Uses existing `DummyDataset` with on-the-fly PSF generation via DeepLens raytracing.

---

## Validation Script

Test the implementation:

```bash
python test_cvae_setup.py
```

**Tests**:
1. ✓ Config loading
2. ✓ Model instantiation
3. ✓ Forward pass
4. ✓ Positional encoding
5. ✓ Grid neighbor structure

---

## Future Improvements

### Potential Enhancements

1. **Convolutional Decoder**
   - Current MLP decoder may struggle with complex aberrations
   - Conv layers could better capture spatial PSF structure

2. **Hierarchical Latent Space**
   - Separate latents for global blur vs local aberrations
   - Better disentanglement

3. **Adaptive Grid Sampling**
   - Denser sampling in regions with high PSF variation
   - Coarser sampling in uniform regions

4. **Multi-scale Training**
   - Train on multiple PSF sizes (15×15, 31×31, 65×65)
   - Better generalization

5. **Uncertainty Quantification**
   - Use posterior variance for uncertainty estimates
   - Identify regions needing more measurements

---

## References

### Theoretical Background

1. **Conditional VAE**:
   - Sohn, K., et al. "Learning Structured Output Representation using Deep Conditional Generative Models." NeurIPS 2015.

2. **Free Bits**:
   - Kingma, D. P., et al. "Improved Variational Inference with Inverse Autoregressive Flow." NeurIPS 2016.

3. **Positional Encoding**:
   - Tancik, M., et al. "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains." NeurIPS 2020.

4. **Posterior Collapse Prevention**:
   - Razavi, A., et al. "Preventing Posterior Collapse with delta-VAEs." ICLR 2019.

### Implementation Notes

- Built on HuggingFace Accelerate for distributed training
- Compatible with Comet.ml experiment tracking
- Follows existing PSF_Densify naming conventions
- DeepLens integration for physics-based PSF simulation

---

## Contact

For questions about this CVAE implementation, please open an issue on GitHub.

**Implementation Date**: December 2025
**Framework Version**: PSF_Densify (post-ICASSP submission)
