"""
Conditional VAE for PSF Interpolation with Grid Sampling and Smoothness Prior

This model implements a CVAE that:
1. Samples PSFs on a uniform NxN grid for spatial smoothness
2. Uses learned conditional prior p(z|c) instead of standard normal
3. Applies physical constraints (non-negativity, normalization)
4. Includes smoothness regularization on the prior network
"""

import os
import einops
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader

from models.PSFlatent.PSF_model import PSF_model
from models.archs import define_network
from utils import log_image
from utils.misc import log_metric


class CVAE_PSF_model(PSF_model):
    """
    Conditional VAE for PSF prediction with grid sampling and smoothness prior.

    Key Features:
    - NxN grid sampling for spatial consistency
    - Learned conditional prior network
    - Physical constraints in decoder
    - Smoothness regularization loss
    """

    def __init__(self, opt, logger):
        # Call BaseModel.__init__ instead of PSF_model.__init__
        # to avoid double network initialization
        from models.base_model import BaseModel
        BaseModel.__init__(self, opt, logger)

        # Copy DeepLens setup from PSF_model
        self.m_scale = 1e3
        self.psf_rescale_factor = 1

        settings = opt.deeplens
        self.single_wavelength = settings.get("single_wavelength", False)
        self.spp = settings.get("spp", 100000)
        self.kernel_size = settings.get("kernel_size", 65)

        self.depth_min = -settings.depth_min * self.m_scale
        self.depth_max = -settings.depth_max * self.m_scale
        self.fov = settings.fov
        self.foc_d_arr = np.array(
            [
                -400, -425, -450, -500, -550, -600, -650, -700, -800,
                -900, -1000, -1250, -1500, -1750, -2000, -2500, -3000,
                -4000, -5000, -6000, -8000, -10000, -12000, -15000, -20000,
            ]
        )
        self.foc_z_arr = (self.foc_d_arr - self.depth_min) / (self.depth_max - self.depth_min)

        self.wavelength_set_m = np.array(settings.wavelength_set_m)
        from deeplens.geolens import GeoLens
        self.lens = GeoLens(filename=settings.lens_file, device=self.accelerator.device)

        # Grid configuration
        self.grid_size = opt.get('grid_size', 20)  # NxN grid
        self.kernel_size_small = opt.get('kernel_size_small', 15)  # Sparse PSF size (15x15)

        # Initialize grid coordinates (will be populated in setup_grid)
        self.grid_coords = None  # (N*N, 2) normalized coordinates
        self.grid_neighbors = None  # Dictionary mapping index to neighbor indices

        # Setup the grid
        self.setup_grid()

        # Network (CVAE instead of PSF_model's net_g)
        self.net_cvae = define_network(opt.network)
        self.net_cvae.train()
        self.models.append(self.net_cvae)

        # Loss component toggles (for ablation studies)
        self.use_l1_loss = opt.train.get('use_l1_loss', False)
        self.use_gradient_loss = opt.train.get('use_gradient_loss', False)
        self.use_smooth_loss = opt.train.get('use_smooth_loss', False)
        self.use_free_bits = opt.train.get('use_free_bits', False)

        # Loss weights
        self.recon_weight = opt.train.get('recon_weight', 1.0)
        self.kl_weight = opt.train.get('kl_weight', 1e-4)
        self.smooth_weight = opt.train.get('smooth_weight', 1e-3)
        self.l1_weight = opt.train.get('l1_weight', 0.5)
        self.grad_weight = opt.train.get('grad_weight', 0.1)
        self.free_bits = opt.train.get('free_bits', 0.5)

        # Log which loss components are enabled
        self.logger.info(f"Loss components enabled:")
        self.logger.info(f"  MSE: True (always enabled)")
        self.logger.info(f"  L1: {self.use_l1_loss}")
        self.logger.info(f"  Gradient: {self.use_gradient_loss}")
        self.logger.info(f"  KL: True (always enabled)")
        self.logger.info(f"  Free bits: {self.use_free_bits}")
        self.logger.info(f"  Smoothness: {self.use_smooth_loss}")

        # Cache for grid PSFs (computed once per epoch)
        self.cached_grid_psfs = None
        self.cache_valid = False

    def setup_grid(self):
        """
        Setup NxN uniform grid covering the field of view.
        Creates neighbor mapping for smoothness loss.
        """
        N = self.grid_size

        # Create uniform grid in [-1, 1] x [-1, 1]
        y_coords, x_coords = torch.meshgrid(
            torch.linspace(-1, 1, N),
            torch.linspace(-1, 1, N),
            indexing='xy'
        )

        # Flatten to (N*N, 2)
        x_flat = x_coords.flatten()
        y_flat = y_coords.flatten()
        self.grid_coords = torch.stack([x_flat, y_flat], dim=1)  # (N*N, 2)

        # Build neighbor dictionary for smoothness loss
        self.grid_neighbors = {}
        for i in range(N):
            for j in range(N):
                idx = i * N + j
                neighbors = []

                # Right neighbor
                if j < N - 1:
                    neighbors.append(i * N + (j + 1))
                # Bottom neighbor
                if i < N - 1:
                    neighbors.append((i + 1) * N + j)
                # Diagonal neighbors (optional, for stronger smoothness)
                if i < N - 1 and j < N - 1:
                    neighbors.append((i + 1) * N + (j + 1))
                if i < N - 1 and j > 0:
                    neighbors.append((i + 1) * N + (j - 1))

                self.grid_neighbors[idx] = neighbors

        self.logger.info(f"Grid setup: {N}x{N} = {N*N} points")
        self.logger.info(f"Grid coordinates range: x=[{x_flat.min():.2f}, {x_flat.max():.2f}], "
                        f"y=[{y_flat.min():.2f}, {y_flat.max():.2f}]")

    def z2depth(self, z):
        """Convert normalized depth [0,1] to actual depth in mm."""
        depth = z * (self.depth_max - self.depth_min) + self.depth_min
        return depth

    def generate_grid_psfs(self, foc_z):
        """
        Generate PSFs for all grid points using DeepLens raytracing.

        Args:
            foc_z: Normalized focal depth [0, 1]

        Returns:
            grid_psfs: (N*N, 3, kernel_size, kernel_size) PSF kernels
            grid_coords: (N*N, 2) normalized coordinates
        """
        lens = self.lens
        num_points = len(self.grid_coords)

        with torch.no_grad():
            # Use fixed depth for all points (can be extended to depth-varying)
            z = torch.full((num_points,), 0.5, dtype=torch.float32)  # Middle depth

            # Refocus lens
            foc_dist = self.z2depth(foc_z)
            lens.refocus(foc_dist)

            # Extract x, y from grid
            x = self.grid_coords[:, 0]
            y = self.grid_coords[:, 1]

            # Compute depth
            depth = self.z2depth(z)

            # Ray tracing to compute PSFs
            points = torch.stack((x, y, depth), dim=-1)
            psf = lens.psf_rgb(points=points, ks=self.kernel_size, spp=self.spp)

            # Downsample to sparse kernel size (15x15) if needed
            if self.kernel_size != self.kernel_size_small:
                psf = self.downsample_psf(psf, self.kernel_size_small)

        return psf, self.grid_coords

    def downsample_psf(self, psf, target_size):
        """
        Downsample PSF from kernel_size to target_size while preserving energy.

        Args:
            psf: (N, 3, kernel_size, kernel_size)
            target_size: Target kernel size (e.g., 15)

        Returns:
            psf_small: (N, 3, target_size, target_size)
        """
        # Use adaptive average pooling to downsample
        psf_small = F.adaptive_avg_pool2d(psf, (target_size, target_size))

        # Renormalize to preserve total energy
        psf_small = psf_small / (psf_small.sum(dim=(-2, -1), keepdim=True) + 1e-8)

        return psf_small

    def feed_data(self, data, is_train=True):
        """
        Generate grid PSFs for training/validation.
        Uses caching to avoid redundant computation within an epoch.

        Args:
            data: Dictionary from DummyDataset with 'x': batch_index
            is_train: Training or validation mode
        """
        # Sample focal depth
        if is_train:
            foc_z = float(np.random.choice(self.foc_z_arr))
        else:
            batch_idx = data['x'][0] // len(data['x'])
            total_batches = len(self.test_dataloader)
            foc_z = float(self.foc_z_arr[int(len(self.foc_z_arr) * batch_idx / total_batches)])

        # Generate or retrieve cached grid PSFs
        if not self.cache_valid or not is_train:
            grid_psfs, grid_coords = self.generate_grid_psfs(foc_z)
            if is_train:
                self.cached_grid_psfs = grid_psfs
                self.cache_valid = True
        else:
            grid_psfs = self.cached_grid_psfs
            grid_coords = self.grid_coords

        # Sample batch from grid
        num_grid = len(grid_coords)
        batch_size = min(len(data['x']), num_grid)

        if is_train:
            # Random sampling from grid
            indices = torch.randperm(num_grid)[:batch_size]
        else:
            # Sequential sampling for validation
            indices = torch.arange(min(batch_size, num_grid))

        # Prepare batch
        psf_batch = grid_psfs[indices].to(self.accelerator.device)
        coords_batch = grid_coords[indices].to(self.accelerator.device)

        # Store in sample dict
        self.sample = {
            'psf': psf_batch,  # (B, 3, 15, 15)
            'coords': coords_batch,  # (B, 2)
            'foc_z': foc_z,
            'grid_psfs': grid_psfs.to(self.accelerator.device),  # Full grid for smoothness loss
            'grid_coords': grid_coords.to(self.accelerator.device),  # Full grid coords
        }

    def optimize_parameters(self):
        """
        Training step: CVAE forward pass with reconstruction, KL, and smoothness losses.
        """
        for optimizer in self.optimizers:
            optimizer.zero_grad()

        # Get batch data
        psf = self.sample['psf']  # (B, 3, 15, 15)
        coords = self.sample['coords']  # (B, 2)

        # Flatten PSF for input to encoder
        B, C, H, W = psf.shape
        psf_flat = psf.view(B, -1)  # (B, 3*15*15)

        # Forward pass through CVAE
        recon_psf, mu_q, logvar_q, mu_p, logvar_p = self.net_cvae(psf_flat, coords)

        # Reshape reconstruction back to image
        recon_psf = recon_psf.view(B, C, H, W)

        # 1. Reconstruction Loss (MSE always, optionally + L1 + Gradient)
        # MSE for overall intensity matching (always enabled)
        mse_loss = F.mse_loss(recon_psf, psf)
        recon_loss = mse_loss

        # L1 for robustness to outliers (preserves sharp features) - OPTIONAL
        if self.use_l1_loss:
            l1_loss = F.l1_loss(recon_psf, psf)
            recon_loss = recon_loss + self.l1_weight * l1_loss
        else:
            l1_loss = torch.tensor(0.0, device=psf.device)

        # Gradient loss to preserve PSF shape structure - OPTIONAL
        if self.use_gradient_loss:
            def gradient_loss(pred, target):
                # Compute gradients in x and y directions
                pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
                pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
                target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
                target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]

                loss_dx = F.l1_loss(pred_dx, target_dx)
                loss_dy = F.l1_loss(pred_dy, target_dy)
                return loss_dx + loss_dy

            grad_loss = gradient_loss(recon_psf, psf)
            recon_loss = recon_loss + self.grad_weight * grad_loss
        else:
            grad_loss = torch.tensor(0.0, device=psf.device)

        # 2. KL Divergence: KL(q(z|K,c) || p(z|c)) - ALWAYS ENABLED
        # Analytical KL between two Gaussians (per dimension)
        kl_per_dim = -0.5 * (
            1 + logvar_q - logvar_p
            - (logvar_q.exp() + (mu_q - mu_p).pow(2)) / logvar_p.exp()
        )  # (B, latent_dim)

        # Free bits: prevent posterior collapse by allowing minimum KL per dimension - OPTIONAL
        # This prevents encoder from collapsing to prior for any dimension
        if self.use_free_bits:
            kl_per_dim_clamped = torch.clamp(kl_per_dim, min=self.free_bits)
        else:
            kl_per_dim_clamped = kl_per_dim

        # Average over batch and sum over latent dimensions
        kl_loss = torch.sum(kl_per_dim_clamped) / B

        # 3. Smoothness Loss on Prior Network - OPTIONAL
        if self.use_smooth_loss:
            # Compute prior for all grid points (WITH gradients for smoothness loss)
            grid_coords = self.sample['grid_coords']  # (N*N, 2)
            # Split into smaller batches if grid is large
            grid_mu_list = []
            grid_batch_size = 256
            for i in range(0, len(grid_coords), grid_batch_size):
                batch_coords = grid_coords[i:i+grid_batch_size]
                _, mu_grid, _ = self.net_cvae.prior_network(batch_coords)
                grid_mu_list.append(mu_grid)
            grid_mu = torch.cat(grid_mu_list, dim=0)  # (N*N, latent_dim)

            # Compute smoothness loss using neighbor pairs
            smooth_loss = 0.0
            num_pairs = 0
            for idx, neighbors in self.grid_neighbors.items():
                if len(neighbors) > 0:
                    mu_i = grid_mu[idx]  # (latent_dim,)
                    for neighbor_idx in neighbors:
                        mu_j = grid_mu[neighbor_idx]
                        smooth_loss += F.mse_loss(mu_i, mu_j)
                        num_pairs += 1

            if num_pairs > 0:
                smooth_loss = smooth_loss / num_pairs
            else:
                smooth_loss = torch.tensor(0.0, device=psf.device)
        else:
            smooth_loss = torch.tensor(0.0, device=psf.device)

        # Total loss (only include enabled components)
        total_loss = self.recon_weight * recon_loss + self.kl_weight * kl_loss

        if self.use_smooth_loss:
            total_loss = total_loss + self.smooth_weight * smooth_loss

        # Backward pass
        self.accelerator.backward(total_loss)

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.net_cvae.parameters(), 1.0)

        # Optimizer step
        for optimizer in self.optimizers:
            optimizer.step()

        # Return losses for logging (all components, even if disabled they'll be 0)
        return {
            'all': total_loss.detach(),
            'recon': recon_loss.detach(),
            'mse': mse_loss.detach(),
            'l1': l1_loss.detach() if isinstance(l1_loss, torch.Tensor) else torch.tensor(0.0),
            'grad': grad_loss.detach() if isinstance(grad_loss, torch.Tensor) else torch.tensor(0.0),
            'kl': kl_loss.detach(),
            'smooth': smooth_loss.detach() if isinstance(smooth_loss, torch.Tensor) else torch.tensor(0.0),
        }

    def validate_step(self, batch, idx, lq_key, gt_key):
        """
        Validation: visualize reconstructed PSFs across the grid.
        """
        self.feed_data(batch, is_train=False)

        with torch.no_grad():
            psf = self.sample['psf']
            coords = self.sample['coords']

            # Flatten PSF
            B, C, H, W = psf.shape
            psf_flat = psf.view(B, -1)

            # Forward pass
            recon_psf, mu_q, logvar_q, mu_p, logvar_p = self.net_cvae(psf_flat, coords)
            recon_psf = recon_psf.view(B, C, H, W)

            # Sample from prior (unconditional generation)
            with torch.no_grad():
                z_prior = mu_p + torch.randn_like(mu_p) * torch.exp(0.5 * logvar_p)
                prior_psf = self.net_cvae.decode(z_prior, coords)
                prior_psf = prior_psf.view(B, C, H, W)

        # Convert to numpy for logging
        psf_np = psf.cpu().numpy()
        recon_np = recon_psf.cpu().numpy()
        prior_np = prior_psf.cpu().numpy()

        # Visualize a grid of PSFs
        N = int(np.sqrt(min(B, 64)))
        if B >= N * N:
            # Create grid visualization
            psf_grid = einops.rearrange(psf_np[:N*N], '(h w) c H W -> c (h H) (w W)', h=N, w=N)
            recon_grid = einops.rearrange(recon_np[:N*N], '(h w) c H W -> c (h H) (w W)', h=N, w=N)
            prior_grid = einops.rearrange(prior_np[:N*N], '(h w) c H W -> c (h H) (w W)', h=N, w=N)

            # Log images (show green channel as grayscale)
            # Add batch dimension: (1, H, W) -> (1, 1, H, W) for log_image
            log_image(self.opt, self.accelerator,
                     (1 - np.clip(psf_grid[1:2] / psf_grid.max(), 0, 1))[np.newaxis, :, :, :],
                     f"val_psf_gt", self.global_step)
            log_image(self.opt, self.accelerator,
                     (1 - np.clip(recon_grid[1:2] / recon_grid.max(), 0, 1))[np.newaxis, :, :, :],
                     f"val_psf_recon", self.global_step)
            log_image(self.opt, self.accelerator,
                     (1 - np.clip(prior_grid[1:2] / prior_grid.max(), 0, 1))[np.newaxis, :, :, :],
                     f"val_psf_prior", self.global_step)

        # Log metrics
        mse = F.mse_loss(recon_psf, psf).item()
        log_metric(self.accelerator, {'val_mse': mse}, self.global_step)

        return idx + 1
