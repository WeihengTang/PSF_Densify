# **Integration Roadmap for New VAE Model**

Based on deep-dive analysis of the PSF_Densify codebase, here's your complete integration guide for implementing a custom VAE model within this framework.

---

## **Summary of Execution Flow**

### **Step 1: Entry Point & Config Parsing** (`trainer.py:33-54`)

1. **YAML Loading**: Uses `OmegaConf` to load and resolve YAML config from `-opt` argument
2. **Config Object**: Converts to `DictAsMember` object for attribute-style access
3. **Model Instantiation**: Calls `create_model(opt, logger)` from `models/__init__.py`

### **Step 2: Model Factory Pattern** (`models/__init__.py:14-29`)

The framework uses **automatic discovery**:
```python
# Scans models/ recursively for *_model.py files
model_filenames = [v for v in scandir(model_folder, recursive=True) if v.endswith('_model.py')]
# Imports all model modules
_model_modules = [importlib.import_module(f'models.{file_name}') for file_name in model_filenames]
# Finds class matching model_type from YAML
model_cls = find_attr(_model_modules, model_type)
model = model_cls(opt, logger)
```

**Key Discovery**: The `model_type` field in your YAML **must exactly match** the class name in your model file.

---

## **Step 3: PSFLatent Reference Analysis**

### **Directory Structure**: `models/PSFlatent/`
```
PSFlatent/
├── PSF_model.py          # Base PSF prediction model
└── PSF_Emb_model.py      # VAE extension (YOUR REFERENCE!)
```

### **PSF_Emb_model Architecture** (`models/PSFlatent/PSF_Emb_model.py`)

**Inheritance**: `PSF_Emb_model` → `PSF_model` → `BaseModel`

**Required Methods from BaseModel**:
1. `__init__(self, opt, logger)` - Initialize model, networks, loss functions
2. `setup_dataloaders(self)` - Create train/val DataLoaders (inherited from BaseModel by default)
3. `feed_data(self, data, is_train=True)` - Process batch and prepare `self.sample` dict
4. `optimize_parameters(self)` - Forward pass, loss computation, backward pass
5. `validate_step(self, batch, idx, lq_key, gt_key)` - Validation logic, logging

**Critical Implementation Details**:

```python
class PSF_Emb_model(PSF_model):
    def __init__(self, opt, logger):
        super().__init__(opt, logger)  # Calls PSF_model.__init__ which calls BaseModel.__init__

        # Initialize your network (automatically discovered from models/archs/)
        self.net_g = define_network(opt.network)
        self.net_g.train()
        self.models.append(self.net_g)  # CRITICAL: Add to self.models list for Accelerate

        # Setup loss
        self.criterion = Loss(self.opt['train'].loss).to(self.accelerator.device)

    def feed_data(self, data, is_train=True):
        # data comes from DummyDataset: {'x': idx}
        # Generate training data on-the-fly using lens raytracing
        # Store in self.sample dict
        self.sample = {
            'inp': inp.to(self.accelerator.device),
            'psf': psf.to(self.accelerator.device),
        }

    def optimize_parameters(self):
        # Zero gradients
        for optimizer in self.optimizers:
            optimizer.zero_grad()

        # Forward pass through network
        pred_psf, latent = self.net_g(self.sample['inp'])

        # Compute loss
        loss = self.criterion(pred_psf, self.sample['psf'])

        # Backward and step
        self.accelerator.backward(loss['all'])
        for optimizer in self.optimizers:
            optimizer.step()

        return {'all': loss['all']}  # Must return dict with 'all' key

    def validate_step(self, batch, idx, lq_key, gt_key):
        # Run inference
        # Log images using log_image()
        # Return updated idx
        return idx + 1
```

---

## **Step 4: Data Flow with DummyDataset**

### **DummyDataset Implementation** (`dataset/dummy_dataset.py:8-19`)

```python
class DummyDataset(data.Dataset):
    def __init__(self, opt):
        self.len = opt['size']  # e.g., 1024

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return {'x': idx}  # Just returns the index!
```

**How PSF_Emb_model uses it**:
- Dataset returns `{'x': idx}` where `idx` is batch index
- In `feed_data()`, the model uses `idx` to sample random PSF parameters
- **On-the-fly generation**: PSFs are computed using DeepLens raytracing during training
- This is **memory efficient** - no need to store millions of PSFs on disk

---

## **Your Integration Roadmap**

### **1. Where to Create Your VAE Model File**

**Location**: `models/PSFlatent/YourVAE_model.py`

**Alternative**: `models/YourVAE_model.py` (at top level, but PSFlatent is better for organization)

**Naming Convention**:
- File must end with `_model.py`
- Class name must match `model_type` in your YAML exactly

### **2. VAE Class Skeleton**

```python
# File: models/PSFlatent/YourVAE_model.py

import torch
import torch.nn.functional as F
from models.PSFlatent.PSF_model import PSF_model
from models.archs import define_network
from utils.loss import Loss
from utils import log_image, log_metrics
from utils.misc import log_metric

class YourVAE_model(PSF_model):  # Inherit from PSF_model to reuse lens setup
    """
    Your custom VAE model for PSF prediction with latent space manipulation.
    """

    def __init__(self, opt, logger):
        super(YourVAE_model, self).__init__(opt, logger)

        # Your VAE network will be instantiated from models/archs/
        # The network type is specified in opt.network.type
        self.net_vae = define_network(opt.network)
        self.net_vae.train()

        # CRITICAL: Append to self.models for Accelerate to handle
        self.models.append(self.net_vae)

        # Setup your loss functions
        self.reconstruction_loss = Loss('1*MSE').to(self.accelerator.device)

        # VAE-specific hyperparameters
        self.kl_weight = opt.train.get('kl_weight', 1e-4)

    def feed_data(self, data, is_train=True):
        """
        Process a batch of data.
        For PSF tasks with DummyDataset, generate PSFs on-the-fly.
        """
        # You can reuse parent's feed_data for PSF generation
        super().feed_data(data, is_train)

        # self.sample now contains 'inp' and 'psf' from parent class
        # Add any additional preprocessing here

    def encode(self, x):
        """Your encoding logic"""
        mu, logvar = self.net_vae.encode(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, condition=None):
        """Your decoding logic"""
        return self.net_vae.decode(z, condition)

    def optimize_parameters(self):
        """
        Training step: forward pass, loss computation, backward pass.
        """
        for optimizer in self.optimizers:
            optimizer.zero_grad()

        # Get input from self.sample (prepared by feed_data)
        inp = self.sample['inp']
        target_psf = self.sample['psf']

        # Forward pass through VAE
        mu, logvar = self.encode(inp)
        z = self.reparameterize(mu, logvar)
        reconstructed_psf = self.decode(z, condition=inp)  # Use inp as condition

        # Compute losses
        recon_loss = self.reconstruction_loss(reconstructed_psf, target_psf)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / inp.size(0)  # Normalize by batch size

        total_loss = recon_loss['all'] + self.kl_weight * kl_loss

        # Backward pass
        self.accelerator.backward(total_loss)

        # Optional: gradient clipping
        torch.nn.utils.clip_grad_norm_(self.net_vae.parameters(), 1.0)

        # Optimizer step
        for optimizer in self.optimizers:
            optimizer.step()

        # Return losses for logging (must include 'all' key)
        return {
            'all': total_loss,
            'recon': recon_loss['all'],
            'kl': kl_loss
        }

    def validate_step(self, batch, idx, lq_key, gt_key):
        """
        Validation step: inference and visualization.
        """
        self.feed_data(batch, is_train=False)

        with torch.no_grad():
            inp = self.sample['inp']
            target_psf = self.sample['psf']

            # Encode and decode
            mu, logvar = self.encode(inp)
            z = self.reparameterize(mu, logvar)
            pred_psf = self.decode(z, condition=inp)

        # Convert to numpy for visualization
        pred_psf_np = pred_psf.cpu().numpy()
        target_psf_np = target_psf.cpu().numpy()

        # Log images (framework handles tracker routing)
        log_image(self.opt, self.accelerator, pred_psf_np, f"pred_psf", self.global_step)
        log_image(self.opt, self.accelerator, target_psf_np, f"target_psf", self.global_step)

        # Optional: Log metrics
        if hasattr(self.opt.val, 'metrics'):
            log_metrics(pred_psf_np, target_psf_np, self.opt.val.metrics,
                       self.accelerator, self.global_step, comment="val_")

        return idx + 1
```

### **3. Create Your VAE Network Architecture**

**Location**: `models/archs/YourVAE_arch.py`

**Naming Convention**:
- Must define `{NetworkName}_config` class inheriting from `PretrainedConfig`
- Must define `{NetworkName}_arch` class inheriting from `PreTrainedModel`
- `{NetworkName}` must match `network.type` in your YAML

**Skeleton**:

```python
# File: models/archs/YourVAE_arch.py

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig

class YourVAE_config(PretrainedConfig):
    """
    Configuration class for YourVAE.
    All hyperparameters from YAML network section will be passed here.
    """
    model_type = "YourVAE"

    def __init__(
        self,
        in_channels=3,
        latent_dim=256,
        hidden_dims=[32, 64, 128, 256],
        kernel_size=64,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.kernel_size = kernel_size


class YourVAE_arch(PreTrainedModel):
    """
    Your VAE architecture.
    Must inherit from PreTrainedModel for Accelerate compatibility.
    """
    config_class = YourVAE_config  # CRITICAL: Link to config class

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # Build encoder
        self.encoder = self._build_encoder()

        # Latent space
        self.fc_mu = nn.Linear(config.hidden_dims[-1], config.latent_dim)
        self.fc_logvar = nn.Linear(config.hidden_dims[-1], config.latent_dim)

        # Build decoder
        self.decoder = self._build_decoder()

    def _build_encoder(self):
        """Build your encoder architecture"""
        modules = []
        in_channels = self.config.in_channels
        for h_dim in self.config.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()
                )
            )
            in_channels = h_dim
        return nn.Sequential(*modules)

    def _build_decoder(self):
        """Build your decoder architecture"""
        modules = []
        hidden_dims = self.config.hidden_dims[::-1]
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i+1],
                                     kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i+1]),
                    nn.LeakyReLU()
                )
            )
        return nn.Sequential(*modules)

    def encode(self, x):
        """Encode input to latent distribution parameters"""
        h = self.encoder(x)
        h = torch.flatten(h, start_dim=1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def decode(self, z, condition=None):
        """Decode latent vector to output"""
        # Reshape z to spatial dimensions
        # ... your decoding logic
        return self.decoder(z)

    def forward(self, x, condition=None):
        """
        Full forward pass (for compatibility).
        Returns reconstruction and latent parameters.
        """
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        recon = self.decode(z, condition)
        return recon, mu, logvar
```

### **4. YAML Configuration Template**

**Location**: `checkpoints/YourVAE/your_experiment.yml`

```yaml
name: YourVAE
model_type: YourVAE_model  # Must match class name exactly!
is_train: true
report_to: comet_ml
push_to_hub: false
num_gpu: 1  # Start with 1 for testing
seed: 42
mixed_precision: !!str no  # Start without mixed precision
allow_tf32: false
tracker_project_name: YourVAE_Project
experiment_key: YourVAE_Experiment_v1

# If using DeepLens for PSF generation (from PSF_model)
deeplens:
  lens_file: "/path/to/your/lens.json"
  spp: 100000
  kernel_size: 64
  wavelength_set_m: [!!float 450e-9, !!float 550e-9, !!float 650e-9]
  fov: !!float 0.015
  depth_min: !!float 400e-3
  depth_max: !!float 20000e-3

# Datasets
datasets:
  train:
    type: DummyDataset  # Uses DummyDataset for on-the-fly generation
    size: 10000  # Number of training samples per epoch
    use_shuffle: true

  val:
    type: DummyDataset
    size: 100
    use_shuffle: false

# Network architecture
network:
  type: YourVAE  # Must match YourVAE_config/YourVAE_arch
  in_channels: 3
  latent_dim: 256
  hidden_dims: [32, 64, 128, 256]
  kernel_size: 64

# Paths
path:
  root: /path/to/experiments  # Where to save checkpoints
  logging_dir: logs
  resume_from_path: null  # Set to checkpoint path if resuming
  resume_from_checkpoint: null  # Or 'latest'

# Training settings
train:
  kl_weight: !!float 1e-4  # Custom VAE hyperparameter

  optim:
    scale_lr: false
    use_8bit_adam: false
    learning_rate: !!float 1e-4
    adam_beta1: 0.9
    adam_beta2: 0.999
    adam_weight_decay: 0.01
    adam_epsilon: !!float 1e-8
    max_grad_norm: 1.0

  scheduler:
    type: cosine_with_restarts
    lr_warmup_steps: 500
    lr_num_cycles: 3
    lr_power: 1.0

  loss: 1*MSE  # Base reconstruction loss
  gradient_accumulation_steps: 1
  num_train_epochs: 10
  batch_size: 16

  max_train_steps: 50000
  validation_steps: 500
  checkpointing_steps: 1000
  checkpoints_total_limit: 3

  patched: false  # Set true if images are large
  check_speed: false

# Validation settings
val:
  save_images: true
  batch_size: 4

  metrics:
    psnr:
      crop: 0
    mse:
      crop: 0
```

---

## **Integration Checklist**

### **Before Coding**:
- [ ] Read `models/PSFlatent/PSF_Emb_model.py` thoroughly
- [ ] Read `models/archs/PSF_Emb_arch.py` to see VAE architecture pattern
- [ ] Understand `models/base_model.py` training loop (lines 361-500)

### **Implementation**:
- [ ] Create `models/PSFlatent/YourVAE_model.py` with all 5 required methods
- [ ] Create `models/archs/YourVAE_arch.py` with config and arch classes
- [ ] Ensure `config_class` attribute is set in architecture
- [ ] Append network to `self.models` list in `__init__`
- [ ] `optimize_parameters()` must return dict with `'all'` key

### **Configuration**:
- [ ] Create `checkpoints/YourVAE/experiment.yml`
- [ ] Set `model_type` to exact class name
- [ ] Set `network.type` to exact architecture name (without `_arch` suffix)
- [ ] Configure dataset (`DummyDataset` for on-the-fly generation)

### **Testing**:
```bash
# Test with small config first
python trainer.py -opt checkpoints/YourVAE/experiment.yml -test

# If successful, full training
accelerate launch trainer.py -opt checkpoints/YourVAE/experiment.yml
```

---

## **Common Pitfalls**

1. **Name Mismatch**: `model_type` must EXACTLY match class name (case-sensitive)
2. **Forgot `self.models.append()`**: Networks won't be moved to GPU/prepared by Accelerate
3. **Missing `config_class`**: Architecture won't be discoverable
4. **Wrong loss dict format**: Must return `{'all': total_loss, ...}`
5. **Not calling `super().__init__()`**: Will miss BaseModel initialization
6. **Dataloader issues**: DummyDataset is perfect for PSF - don't overthink it

---

## **Key Framework Patterns**

### **Model Discovery Pattern**
```
YAML config → model_type: "MyModel_model"
    ↓
models/__init__.py scans for *_model.py files
    ↓
Imports all model modules
    ↓
find_attr(_model_modules, "MyModel_model") → finds class MyModel_model
    ↓
Instantiates MyModel_model(opt, logger)
```

### **Architecture Discovery Pattern**
```
YAML config → network.type: "MyVAE"
    ↓
models/archs/__init__.py scans for *_arch.py files
    ↓
find_attr(_arch_modules, "MyVAE_config") → creates config object
    ↓
find_attr(_arch_modules, "MyVAE_arch") → creates network with config
```

### **Dataset Discovery Pattern**
```
YAML config → datasets.train.type: "DummyDataset"
    ↓
dataset/__init__.py scans for *_dataset.py files
    ↓
find_attr(_dataset_modules, "DummyDataset") → finds class DummyDataset
    ↓
Instantiates DummyDataset(opt.datasets.train)
```

---

## **Reference Files**

**Must Read**:
- `models/PSFlatent/PSF_Emb_model.py` - Your reference implementation
- `models/archs/PSF_Emb_arch.py` - VAE architecture with latent interpolation
- `models/base_model.py` - Base class with training loop
- `checkpoints/PSF/DeepLens_v2.yml` - Reference YAML config

**Helper Utilities**:
- `utils/loss.py` - Loss function parsing
- `utils/misc.py` - Logging utilities (log_image, log_metric, log_metrics)
- `dataset/dummy_dataset.py` - Simple dummy dataset

---

## **Next Steps**

1. **Start Simple**: Copy `PSF_Emb_model.py` and modify incrementally
2. **Test Immediately**: Use `-test` flag to skip training loop and test validation only
3. **Use Comet ML**: Check `tracker_project_name` in YAML for experiment tracking
4. **Iterate Quickly**: Start with small batch size and few steps to debug faster
5. **Read DeepLens docs**: If using lens raytracing in `feed_data()`

---

## **Quick Start Commands**

```bash
# 1. Create your model files
touch models/PSFlatent/YourVAE_model.py
touch models/archs/YourVAE_arch.py
mkdir -p checkpoints/YourVAE
touch checkpoints/YourVAE/experiment.yml

# 2. Test configuration loading
python trainer.py -opt checkpoints/YourVAE/experiment.yml -test

# 3. Run training (single GPU)
python trainer.py -opt checkpoints/YourVAE/experiment.yml

# 4. Run training (multi-GPU with Accelerate)
accelerate launch trainer.py -opt checkpoints/YourVAE/experiment.yml
```

---

This roadmap provides everything you need to integrate your VAE successfully into the PSF_Densify framework. The key insight is that this framework is **convention-based** with automatic discovery, so follow the naming patterns exactly and you'll be up and running quickly!
