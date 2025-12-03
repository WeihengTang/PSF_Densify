# CVAE Ablation Study Guide

## Overview

The CVAE model supports **optional loss components** that can be toggled on/off via YAML configuration for systematic ablation studies.

## Loss Components

### Always Enabled (Core)
- **MSE Loss**: Mean Squared Error for reconstruction
- **KL Divergence**: Regularization term between posterior and prior

### Optional Components (Controllable via YAML)

| Component | Flag | Default | Weight Parameter | Purpose |
|-----------|------|---------|------------------|---------|
| **L1 Loss** | `use_l1_loss` | `false` | `l1_weight: 0.5` | Preserves sharp features, robust to outliers |
| **Gradient Loss** | `use_gradient_loss` | `false` | `grad_weight: 0.1` | Preserves PSF shape structure and orientation |
| **Smoothness Loss** | `use_smooth_loss` | `false` | `smooth_weight: 1e-3` | Spatial smoothness of latent codes across grid |
| **Free Bits** | `use_free_bits` | `false` | `free_bits: 0.5` | Prevents posterior collapse per latent dimension |

## Configuration Files

### Baseline (MSE + KL only)
```yaml
# checkpoints/PSF/CVAE_baseline.yml
use_l1_loss: false
use_gradient_loss: false
use_smooth_loss: false
use_free_bits: false
kl_weight: 1e-4  # Standard KL weight
```

### Full (All components enabled)
```yaml
# checkpoints/PSF/CVAE_full.yml
use_l1_loss: true
use_gradient_loss: true
use_smooth_loss: true
use_free_bits: true
kl_weight: 1e-5  # Reduced when using free bits
```

### Custom Ablations
Create your own configs by mixing components:

```yaml
# Example: Test L1 loss only
use_l1_loss: true
use_gradient_loss: false
use_smooth_loss: false
use_free_bits: false
```

## Ablation Study Workflow

### Step 1: Baseline Experiment
Train with only core components (MSE + KL):

```bash
accelerate launch trainer.py -opt checkpoints/PSF/CVAE_baseline.yml
```

**Expected results**:
- `loss_mse`: Should decrease
- `loss_l1`: 0 (disabled)
- `loss_grad`: 0 (disabled)
- `loss_smooth`: 0 (disabled)
- `loss_kl`: Standard KL divergence
- `loss_recon` = `loss_mse` (only MSE enabled)

### Step 2: Add L1 Loss
Test reconstruction improvement with L1:

```yaml
# CVAE_+L1.yml
use_l1_loss: true
use_gradient_loss: false
use_smooth_loss: false
use_free_bits: false
```

```bash
accelerate launch trainer.py -opt checkpoints/PSF/CVAE_+L1.yml
```

**Compare with baseline**:
- Does `loss_mse` improve?
- Are elliptical PSF shapes better preserved?
- Check validation images: `val_psf_recon`

### Step 3: Add Gradient Loss
Test shape preservation:

```yaml
# CVAE_+Grad.yml
use_l1_loss: false
use_gradient_loss: true
use_smooth_loss: false
use_free_bits: false
```

```bash
accelerate launch trainer.py -opt checkpoints/PSF/CVAE_+Grad.yml
```

**Compare with baseline**:
- Are PSF orientations better captured?
- Check gradient loss curve: `loss_grad`

### Step 4: Add Smoothness Loss
Test spatial consistency:

```yaml
# CVAE_+Smooth.yml
use_l1_loss: false
use_gradient_loss: false
use_smooth_loss: true
use_free_bits: false
```

```bash
accelerate launch trainer.py -opt checkpoints/PSF/CVAE_+Smooth.yml
```

**Compare with baseline**:
- Is the learned prior spatially smooth?
- Check smoothness loss curve: `loss_smooth` (should decrease)
- Better interpolation between grid points?

### Step 5: Add Free Bits
Test posterior collapse prevention:

```yaml
# CVAE_+FreeBits.yml
use_l1_loss: false
use_gradient_loss: false
use_smooth_loss: false
use_free_bits: true
kl_weight: 1e-5  # Reduce KL weight with free bits
```

```bash
accelerate launch trainer.py -opt checkpoints/PSF/CVAE_+FreeBits.yml
```

**Compare with baseline**:
- Is posterior collapse prevented?
- Compare `val_psf_recon` vs `val_psf_prior` (should differ!)
- `loss_kl` should be higher (not collapse to 0)

### Step 6: Combine Components
Test combinations:

```yaml
# CVAE_+L1+Grad.yml
use_l1_loss: true
use_gradient_loss: true
use_smooth_loss: false
use_free_bits: false
```

```yaml
# CVAE_+Smooth+FreeBits.yml
use_l1_loss: false
use_gradient_loss: false
use_smooth_loss: true
use_free_bits: true
kl_weight: 1e-5
```

### Step 7: Full Model
All components enabled:

```bash
accelerate launch trainer.py -opt checkpoints/PSF/CVAE_full.yml
```

## Metrics to Track

### Training Losses (Comet.ml)

Monitor these curves for each experiment:

| Metric | Always Logged | Meaning |
|--------|---------------|---------|
| `loss_all` | ✓ | Total weighted loss (optimization target) |
| `loss_recon` | ✓ | Total reconstruction loss |
| `loss_mse` | ✓ | MSE component (always enabled) |
| `loss_l1` | ✓ | L1 component (0 if disabled) |
| `loss_grad` | ✓ | Gradient component (0 if disabled) |
| `loss_kl` | ✓ | KL divergence (always enabled) |
| `loss_smooth` | ✓ | Smoothness regularization (0 if disabled) |
| `val_mse` | ✓ | Validation reconstruction error |

**Note**: All components are logged even if disabled (will be 0).

### Validation Images

Compare these across experiments:

1. **`val_psf_gt`**: Ground truth PSFs
   - Shows target elliptical aberrations

2. **`val_psf_recon`**: Reconstructed PSFs
   - Should match GT shape/orientation
   - Key metric for reconstruction quality

3. **`val_psf_prior`**: Prior samples
   - Should differ from recon (indicates encoder works)
   - Should be spatially smooth if `use_smooth_loss=true`

## Analysis Checklist

For each ablation experiment, check:

### Quantitative Metrics
- [ ] Final `val_mse` (lower is better)
- [ ] `loss_kl` behavior (stable, not collapsed to 0)
- [ ] Training stability (no divergence/NaN)
- [ ] Convergence speed (steps to reach target MSE)

### Qualitative Assessment
- [ ] PSF shape fidelity (circles vs ellipses)
- [ ] Orientation preservation (GT vs recon)
- [ ] Spatial smoothness (neighboring PSFs)
- [ ] Prior-recon difference (collapse check)

### Component-Specific Checks

**L1 Loss**:
- [ ] Sharper elliptical boundaries in `val_psf_recon`
- [ ] Better preservation of PSF peak intensity

**Gradient Loss**:
- [ ] Improved orientation matching
- [ ] Better edge structure in PSFs

**Smoothness Loss**:
- [ ] `loss_smooth` decreases over training
- [ ] Spatially coherent `val_psf_prior` samples
- [ ] Better interpolation quality

**Free Bits**:
- [ ] `val_psf_recon` ≠ `val_psf_prior` (no collapse)
- [ ] `loss_kl` > 0 (not collapsed)
- [ ] All latent dimensions used (check latent statistics)

## Example Ablation Table

Create a table summarizing results:

| Config | L1 | Grad | Smooth | FreeBits | val_mse ↓ | KL | Shape Quality | Notes |
|--------|----|----|--------|----------|-----------|----|--------------|----|
| Baseline | ✗ | ✗ | ✗ | ✗ | 0.00015 | 0.5 | Poor (circular) | Posterior collapse |
| +L1 | ✓ | ✗ | ✗ | ✗ | 0.00012 | 0.5 | Better | Sharper edges |
| +Grad | ✗ | ✓ | ✗ | ✗ | 0.00014 | 0.5 | Better | Good orientation |
| +L1+Grad | ✓ | ✓ | ✗ | ✗ | 0.00010 | 0.5 | Best recon | Combined benefit |
| +FreeBits | ✗ | ✗ | ✗ | ✓ | 0.00013 | 5.0 | Good | No collapse |
| Full | ✓ | ✓ | ✓ | ✓ | 0.00008 | 5.2 | Excellent | Best overall |

## Tips

1. **Use consistent seeds**: Set `seed: 42` in all configs for fair comparison
2. **Same training steps**: Use `max_train_steps: 100000` for all experiments
3. **Log to separate projects**: Use different `experiment_key` for each ablation
4. **Save validation images**: Set `val.save_images: true`
5. **Monitor early**: Check first 1000 steps for divergence/NaN issues
6. **Tune weights**: If a component doesn't help, try adjusting its weight

## Common Issues

### Posterior Collapse (recon = prior)
**Symptom**: `val_psf_recon` looks identical to `val_psf_prior`, both are circular blobs

**Solutions**:
1. Enable `use_free_bits: true`
2. Reduce `kl_weight` (try 1e-5 or 1e-6)
3. Enable `use_l1_loss` or `use_gradient_loss` for stronger reconstruction signal

### Training Instability
**Symptom**: Losses spike or diverge, NaN values

**Solutions**:
1. Disable `use_gradient_loss` (can be unstable with high gradients)
2. Reduce `grad_weight` (try 0.01 instead of 0.1)
3. Reduce learning rate
4. Check gradient clipping is enabled (`max_grad_norm: 1.0`)

### Smoothness Loss Doesn't Decrease
**Symptom**: `loss_smooth` stays high or increases

**Solutions**:
1. Check gradients are flowing (not wrapped in `no_grad`)
2. Increase `smooth_weight` (try 1e-2)
3. May be conflicting with other objectives - check other losses

### KL Divergence Too High
**Symptom**: `loss_kl` >> 10, reconstruction suffers

**Solutions**:
1. Reduce `kl_weight` (try 1e-6)
2. Disable `use_free_bits` temporarily
3. Increase reconstruction weights (`recon_weight`, `l1_weight`, `grad_weight`)

## Recommended Ablation Order

1. Baseline (MSE + KL)
2. + Free Bits (prevent collapse)
3. + L1 (sharper features)
4. + Gradient (shape preservation)
5. + L1 + Gradient (combined reconstruction)
6. + Smoothness (spatial structure)
7. Full (all components)

This order builds up from simplest to most complex, making it easier to attribute improvements to specific components.

---

**Note**: All configuration flags default to `false`, so the baseline model uses only MSE + KL unless explicitly enabled.
