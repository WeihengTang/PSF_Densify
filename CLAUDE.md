# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project for Spatially Varying Deblurring (SVD) with Point Spread Function (PSF) simulation and learning. The codebase supports both synthetic and real-world PSF generation, dataset creation, model training, and evaluation for image deblurring tasks.

## Core Workflow

The typical workflow involves four main stages:

1. **PSF Generation**: `1.generate_psfs_synthetic.py` or `generate_psfs_real.py`
2. **Basis PSF Creation**: `2.generate_basis_psfs.py` (generates PCA/Zernike decomposition)
3. **Dataset Generation**: `3.generate_dataset_synthetic.py` (creates blurred/ground truth pairs)
4. **Dataset Upload**: `4.push_dataset_to_hf.py` (pushes to HuggingFace Hub)

## Training and Testing

### Training Models

```bash
# Single GPU training
accelerate launch trainer.py -opt checkpoints/NAFNet.yml

# Multi-GPU training with Accelerate (same command, Accelerate handles GPU detection)
accelerate launch trainer.py -opt checkpoints/NAFNet.yml

# Test mode (saves to 'test' directory)
accelerate launch trainer.py -opt checkpoints/NAFNet.yml -test

# PSF model example
accelerate launch trainer.py -opt checkpoints/PSF/DeepLens_v1.yml
```

### SLURM Training Scripts

Training scripts for HPC clusters are in `scripts/`:
- `scripts/icassp/sjob_*` - ICASSP experiments
- `scripts/metazoom/` - MetaZoom experiments
- `scripts/depth/` - Depth estimation experiments
- `scripts/qis/` - QIS (Quanta Image Sensor) experiments

Example SLURM submission:
```bash
cd scripts/icassp
sbatch sjob_ex0.1
```

### Calculating Metrics

```bash
python calculate_metrics.py --image_size 512 \
  --gt_folder /path/to/gt/ \
  --pred_folder /path/to/predictions/
```

## Architecture Overview

### Model System

The codebase uses a dynamic model loading system based on naming conventions:

- **Base Class**: `models/base_model.py` - `BaseModel` class handles training loop, dataloading, optimization
- **Model Factory**: `models/__init__.py` - `create_model(opt, logger)` dynamically instantiates models by:
  1. Scanning all `*_model.py` files recursively in `models/`
  2. Using `find_attr()` to match `model_type` from YAML config to class name
  3. Example: `model_type: PSF_model` → finds class `PSF_model` in `models/PSFlatent/PSF_model.py`
- **Model Types**: All files ending with `_model.py` are automatically discovered:
  - `Simple_model.py` - Basic image-to-image models (NAFNet, Restormer, U-Net)
  - `Sparse_model.py` - Sparse representation models
  - `LD_model.py` - Latent diffusion models
  - `HF_model.py` - HuggingFace integration models
  - `PSFlatent/PSF_model.py` - Base PSF prediction model
  - `PSFlatent/PSF_Emb_model.py` - VAE-based PSF model with latent interpolation
  - `DFlatModels/` - DFlat metasurface optics models
  - `DeepLensModels/` - DeepLens optical models
  - `Diffusion/` - Diffusion-based restoration models
  - `OneStepDiffusion/` - One-step diffusion models (OSEDiff)

### Network Architectures

Located in `models/archs/`:
- **Architecture Factory**: `models/archs/__init__.py` - `define_network(opt)` instantiates networks by:
  1. Scanning all `*_arch.py` files recursively
  2. Looking for `{type}_config` and `{type}_arch` classes
  3. Example: `type: PSF_Emb` → finds `PSF_Emb_config` and `PSF_Emb_arch`
  4. All architectures must inherit from `PreTrainedModel` with a `config_class` attribute
- Standard architectures in `related/`: NAFNet, Restormer, U-Net
- Building blocks in `blocks/`: CAB, RDB, common blocks
- Custom architectures: LKPN, STN, PSF_arch, PSF_Emb_arch, Wiener deconvolution

### Dataset System

All dataset loaders in `dataset/` inherit from `torch.utils.data.Dataset`:
- **Dataset Factory**: `dataset/__init__.py` - `create_dataset(opt)` dynamically instantiates datasets by:
  1. Scanning all `*_dataset.py` files in `dataset/`
  2. Using `find_attr()` to match `type` from YAML config to class name
  3. Example: `type: DummyDataset` → finds class `DummyDataset` in `dataset/dummy_dataset.py`
- `hf_dataset.py` - HuggingFace datasets (primary loader)
- `dummy_dataset.py` - Dummy dataset that returns only indices (used for on-the-fly generation)
- `deeplens_dataset.py` - DeepLens optical simulations
- `svb_dataset.py` - Spatially varying blur datasets
- `qis_dataset.py` - Quanta image sensor datasets

### Configuration System

Training/evaluation configs in `checkpoints/*.yml`:
- Standard fields: `model_type`, `network`, `datasets`, `train`, `val`, `path`
- Model type determines which `*_model.py` class is loaded
- Network type determines which architecture from `models/archs/` is used
- Uses OmegaConf for YAML parsing with variable resolution

## PSF Simulation

### PSF Types

The `psf/svpsf.py` module (`PSFSimulator` class) supports:
- **Gaussian**: Circular/elliptical PSFs with chromatic aberration
- **Motion**: Motion blur kernels
- **DeepLens**: Physics-based lens simulation from JSON profiles
- **DFlat**: Metasurface/diffractive optics simulation (see `DFlat/` directory)

### PSF Data Format

PSFs are stored in HDF5 files with:
- `psfs`: Full dense PSF array [out_channels, in_channels, n_psfs, psf_h, psf_w]
- `basis_psfs`: PCA/decomposition basis
- `basis_coef`: Spatially-varying coefficients
- `H_obj`, `H_img`: Homography matrices for coordinate mapping
- `metadata`: Generation parameters

## DFlat Integration

`DFlat/` contains metasurface optics simulation:
- `generate_psfs.py` - Generate DFlat PSFs
- `generate_image.py` - Simulate images through DFlat optics
- `depth_array.py`, `cutoff_depth*.py` - Depth-dependent PSF simulation
- `two_lens*.py` - Multi-lens system simulation
- Uses the DFlat library for electromagnetic wave propagation

## Utilities

`utils/` contains:
- `dataset_utils.py` - Transform wrappers, patching, cropping
- `pca_utils.py` - PCA decomposition for PSF compression
- `sv_renderer.py` - Spatially-varying convolution rendering
- `dflat_utils.py` - DFlat-specific utilities
- `lens_profiles.py` - Lens parameter loading
- `mtf.py` - Modulation Transfer Function analysis
- `zernike.py` - Zernike polynomial utilities
- `misc.py` - General utilities (`DictAsMember`, `find_attr`, `scandir`)

## Experiment Organization

Experiments are tracked with:
- **Comet ML**: Set `report_to: comet_ml` in config
- **HuggingFace Hub**: Set `push_to_hub: true` for automatic model upload
- **Checkpointing**: Uses Accelerate's save/load hooks for model state

Experiment outputs saved to `path.root/{experiment_key} {timestamp}/`:
- `checkpoints/` - Model checkpoints
- `logs/` - Training logs
- `results/` - Validation images

## Key Dependencies

- PyTorch with CUDA support
- Accelerate (distributed training)
- Diffusers (diffusion models)
- HuggingFace Datasets and Hub
- DFlat (for metasurface simulation)
- DeepLens (for lens simulation)
- OpenCV, einops, h5py, comet-ml

## Important Notes

- PSF simulation coordinates use homography matrices to map between object space (mm), image space (pixels), and index space
- The `normalize` parameter in PSFSimulator affects whether PSFs sum to 1
- Basis PSF decomposition (PCA) enables efficient spatially-varying rendering
- Model configs support patched inference for large images (set `patched: true`)
- Training uses mixed precision by default (`mixed_precision: no` to disable)
- Accelerate handles multi-GPU automatically based on `num_gpu` in config
