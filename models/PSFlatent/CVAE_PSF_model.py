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

        # Grid configuration for 4D coordinate sampling (x, y, z, f)
        # Spatial field resolution (x, y)
        self.grid_size_xy = opt.get('grid_size_xy', 100)  # 100x100 spatial grid
        # Depth and focus resolution
        self.grid_size_z = opt.get('grid_size_z', 50)  # 50 depth steps
        self.grid_size_f = opt.get('grid_size_f', 50)  # 50 focus distance steps

        # Physical ranges for smartphone-scale optics
        self.z_min = opt.get('z_min', 0.1)  # Minimum object depth in meters
        self.z_max = opt.get('z_max', 3.0)  # Maximum object depth in meters
        self.f_min = opt.get('f_min', 0.1)  # Minimum focus distance in meters
        self.f_max = opt.get('f_max', 3.0)  # Maximum focus distance in meters

        # Legacy support - kernel size
        self.kernel_size_small = opt.get('kernel_size_small', 15)  # Sparse PSF size (15x15)

        # Jittered sampling configuration
        self.use_jitter = opt.get('use_jitter', True)  # Enable stratified sampling with jitter

        # Number of samples per training batch (sampled from 4D space)
        self.samples_per_batch = opt.get('samples_per_batch', 256)

        # Initialize grid parameters (cell sizes for jittering)
        self.grid_coords = None  # Will be populated dynamically during sampling
        self.grid_neighbors = None  # Dictionary mapping index to neighbor indices

        # Compute cell sizes for jittered sampling (half-width of each cell)
        self._setup_grid_parameters()

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

    def _setup_grid_parameters(self):
        """
        Setup grid parameters for 4D stratified sampling with jitter.

        The 4D coordinate space is:
        - x, y: Spatial field coordinates in [-1, 1]
        - z: Object depth, normalized to [0, 1] (maps to z_min..z_max meters)
        - f: Focus distance, normalized to [0, 1] (maps to f_min..f_max meters)

        Cell sizes are computed for jittered sampling within each grid cell.
        """
        Nxy = self.grid_size_xy
        Nz = self.grid_size_z
        Nf = self.grid_size_f

        # Compute cell half-widths for jittering
        # For x, y: range is [-1, 1], so cell width = 2 / N
        self.delta_xy = 1.0 / Nxy  # Half-width of each cell in x, y

        # For z, f: range is [0, 1], so cell width = 1 / N
        self.delta_z = 0.5 / Nz  # Half-width of each cell in z
        self.delta_f = 0.5 / Nf  # Half-width of each cell in f

        # Create grid centers for each dimension
        # x, y: uniform in [-1, 1]
        self.x_centers = torch.linspace(-1 + self.delta_xy, 1 - self.delta_xy, Nxy)
        self.y_centers = torch.linspace(-1 + self.delta_xy, 1 - self.delta_xy, Nxy)

        # z, f: uniform in [0, 1]
        self.z_centers = torch.linspace(self.delta_z, 1 - self.delta_z, Nz)
        self.f_centers = torch.linspace(self.delta_f, 1 - self.delta_f, Nf)

        # For smoothness loss, we use a fixed 2D spatial grid (subset of full grid)
        # This is evaluated on a coarser grid for efficiency
        smooth_grid_size = min(20, Nxy)  # Use at most 20x20 for smoothness
        self._setup_smoothness_grid(smooth_grid_size)

        self.logger.info(f"4D Grid setup:")
        self.logger.info(f"  Spatial (x, y): {Nxy}x{Nxy} = {Nxy*Nxy} cells")
        self.logger.info(f"  Depth (z): {Nz} cells, range [{self.z_min:.2f}m, {self.z_max:.2f}m]")
        self.logger.info(f"  Focus (f): {Nf} cells, range [{self.f_min:.2f}m, {self.f_max:.2f}m]")
        self.logger.info(f"  Jittered sampling: {self.use_jitter}")
        self.logger.info(f"  Cell half-widths: δ_xy={self.delta_xy:.4f}, δ_z={self.delta_z:.4f}, δ_f={self.delta_f:.4f}")

    def _setup_smoothness_grid(self, N):
        """
        Setup a fixed 2D spatial grid for smoothness loss computation.

        Args:
            N: Grid size (NxN points)
        """
        # Create uniform grid in [-1, 1] x [-1, 1]
        y_coords, x_coords = torch.meshgrid(
            torch.linspace(-1, 1, N),
            torch.linspace(-1, 1, N),
            indexing='xy'
        )

        # Flatten to (N*N, 2)
        x_flat = x_coords.flatten()
        y_flat = y_coords.flatten()
        self.smooth_grid_xy = torch.stack([x_flat, y_flat], dim=1)  # (N*N, 2)
        self.smooth_grid_size = N

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

    def sample_4d_coordinates(self, num_samples, use_jitter=True):
        """
        Sample 4D coordinates using stratified sampling with optional jitter.

        Stratified Sampling: Divide the 4D space into grid cells and sample
        one point per cell with uniform jitter within the cell.

        Args:
            num_samples: Number of coordinate samples to generate
            use_jitter: If True, add uniform random jitter within each cell

        Returns:
            coords: (num_samples, 4) tensor of (x, y, z, f) coordinates
                - x, y: in [-1, 1] (spatial field)
                - z: in [0, 1] (normalized object depth)
                - f: in [0, 1] (normalized focus distance)
        """
        device = self.accelerator.device

        # Sample indices for each dimension
        x_idx = torch.randint(0, self.grid_size_xy, (num_samples,))
        y_idx = torch.randint(0, self.grid_size_xy, (num_samples,))
        z_idx = torch.randint(0, self.grid_size_z, (num_samples,))
        f_idx = torch.randint(0, self.grid_size_f, (num_samples,))

        # Get cell centers
        x = self.x_centers[x_idx]
        y = self.y_centers[y_idx]
        z = self.z_centers[z_idx]
        f = self.f_centers[f_idx]

        if use_jitter and self.use_jitter:
            # Add uniform jitter within each cell: c = c_grid + Uniform(-δ, δ)
            x = x + (torch.rand(num_samples) * 2 - 1) * self.delta_xy
            y = y + (torch.rand(num_samples) * 2 - 1) * self.delta_xy
            z = z + (torch.rand(num_samples) * 2 - 1) * self.delta_z
            f = f + (torch.rand(num_samples) * 2 - 1) * self.delta_f

            # Clamp to valid ranges
            x = torch.clamp(x, -1, 1)
            y = torch.clamp(y, -1, 1)
            z = torch.clamp(z, 0, 1)
            f = torch.clamp(f, 0, 1)

        # Stack into (num_samples, 4) tensor
        coords = torch.stack([x, y, z, f], dim=1).to(device)

        return coords

    def sample_validation_grid(self, grid_size_xy=7, grid_size_z=5, grid_size_f=5):
        """
        Create a fixed validation grid for consistent evaluation.

        Args:
            grid_size_xy: Spatial grid size (default 7x7)
            grid_size_z: Depth grid size (default 5)
            grid_size_f: Focus grid size (default 5)

        Returns:
            coords: (N, 4) tensor of validation coordinates
        """
        device = self.accelerator.device

        # Create uniform grids
        x = torch.linspace(-1, 1, grid_size_xy)
        y = torch.linspace(-1, 1, grid_size_xy)
        z = torch.linspace(0, 1, grid_size_z)
        f = torch.linspace(0, 1, grid_size_f)

        # Create meshgrid for all combinations
        # For efficiency, we sample a subset: fix z and f, vary x and y
        # Or create a sparse grid
        coords_list = []

        # Sample across z and f, with spatial grid
        for zi in z:
            for fi in f:
                yy, xx = torch.meshgrid(y, x, indexing='xy')
                xx_flat = xx.flatten()
                yy_flat = yy.flatten()
                zz = torch.full_like(xx_flat, zi.item())
                ff = torch.full_like(xx_flat, fi.item())
                coords_list.append(torch.stack([xx_flat, yy_flat, zz, ff], dim=1))

        coords = torch.cat(coords_list, dim=0).to(device)
        return coords

    def normalized_z_to_depth_mm(self, z_normalized):
        """
        Convert normalized depth [0, 1] to physical depth in mm.

        Args:
            z_normalized: Depth in [0, 1] range

        Returns:
            depth_mm: Physical depth in mm (negative for DeepLens convention)
        """
        # z_normalized: 0 -> z_min meters, 1 -> z_max meters
        depth_m = z_normalized * (self.z_max - self.z_min) + self.z_min
        depth_mm = -depth_m * self.m_scale  # Convert to mm and negate for DeepLens
        return depth_mm

    def normalized_f_to_focus_mm(self, f_normalized):
        """
        Convert normalized focus distance [0, 1] to physical distance in mm.

        Args:
            f_normalized: Focus distance in [0, 1] range

        Returns:
            focus_mm: Physical focus distance in mm (negative for DeepLens convention)
        """
        # f_normalized: 0 -> f_min meters, 1 -> f_max meters
        focus_m = f_normalized * (self.f_max - self.f_min) + self.f_min
        focus_mm = -focus_m * self.m_scale  # Convert to mm and negate for DeepLens
        return focus_mm

    def z2depth(self, z):
        """Convert normalized depth [0,1] to actual depth in mm."""
        depth = z * (self.depth_max - self.depth_min) + self.depth_min
        return depth

    def generate_psfs_for_coords(self, coords):
        """
        Generate PSFs for given 4D coordinates using DeepLens raytracing.

        Args:
            coords: (N, 4) tensor of (x, y, z, f) coordinates
                - x, y: Spatial field coordinates in [-1, 1]
                - z: Normalized object depth in [0, 1]
                - f: Normalized focus distance in [0, 1]

        Returns:
            psfs: (N, 3, kernel_size_small, kernel_size_small) PSF kernels
        """
        lens = self.lens
        num_points = coords.shape[0]

        with torch.no_grad():
            # Extract coordinates
            x = coords[:, 0]  # Spatial x in [-1, 1]
            y = coords[:, 1]  # Spatial y in [-1, 1]
            z_norm = coords[:, 2]  # Normalized depth in [0, 1]
            f_norm = coords[:, 3]  # Normalized focus in [0, 1]

            # Convert normalized depth to physical depth (mm)
            depth_mm = self.normalized_z_to_depth_mm(z_norm)

            # For focus distance, we need to handle per-point focusing
            # DeepLens refocus() sets a single focus distance for all points
            # For efficiency, we group points by similar focus distances
            psf_list = []

            # Quantize focus distances for batching (reduces raytrace calls)
            num_focus_bins = min(10, num_points)  # At most 10 different focus settings
            f_bins = torch.linspace(0, 1, num_focus_bins + 1)

            for i in range(num_focus_bins):
                # Find points in this focus bin
                mask = (f_norm >= f_bins[i]) & (f_norm < f_bins[i + 1])
                if i == num_focus_bins - 1:  # Include upper bound for last bin
                    mask = mask | (f_norm == f_bins[i + 1])

                if mask.sum() == 0:
                    continue

                # Get points in this bin
                x_bin = x[mask]
                y_bin = y[mask]
                depth_bin = depth_mm[mask]

                # Use bin center as focus distance
                f_bin_center = (f_bins[i] + f_bins[i + 1]) / 2
                foc_dist_mm = self.normalized_f_to_focus_mm(f_bin_center)

                # Refocus lens
                lens.refocus(foc_dist_mm)

                # Ray tracing to compute PSFs
                points = torch.stack((x_bin, y_bin, depth_bin), dim=-1)
                psf_bin = lens.psf_rgb(points=points, ks=self.kernel_size, spp=self.spp)

                # Store with original indices
                psf_list.append((mask, psf_bin))

            # Reassemble PSFs in original order
            psfs = torch.zeros(num_points, 3, self.kernel_size, self.kernel_size,
                             device=self.accelerator.device)
            for mask, psf_bin in psf_list:
                psfs[mask] = psf_bin

            # Downsample to sparse kernel size (15x15) if needed
            if self.kernel_size != self.kernel_size_small:
                psfs = self.downsample_psf(psfs, self.kernel_size_small)

        return psfs

    def generate_grid_psfs(self, foc_z):
        """
        Legacy method: Generate PSFs for a 2D spatial grid at fixed depth and focus.
        Used for backward compatibility and smoothness loss computation.

        Args:
            foc_z: Normalized focal depth [0, 1]

        Returns:
            grid_psfs: (N*N, 3, kernel_size_small, kernel_size_small) PSF kernels
            grid_coords: (N*N, 4) normalized coordinates (x, y, z, f)
        """
        lens = self.lens
        N = self.smooth_grid_size

        with torch.no_grad():
            # Use fixed depth and focus for smoothness grid
            z_fixed = 0.5  # Middle of depth range
            f_fixed = foc_z  # Use provided focus

            # Create 4D coordinates from 2D spatial grid
            x = self.smooth_grid_xy[:, 0]
            y = self.smooth_grid_xy[:, 1]
            z = torch.full((N * N,), z_fixed)
            f = torch.full((N * N,), f_fixed)

            coords_4d = torch.stack([x, y, z, f], dim=1).to(self.accelerator.device)

            # Convert to physical units
            depth_mm = self.normalized_z_to_depth_mm(torch.tensor(z_fixed))
            foc_dist_mm = self.normalized_f_to_focus_mm(foc_z)

            # Refocus lens
            lens.refocus(foc_dist_mm)

            # Ray tracing to compute PSFs
            points = torch.stack((x, y, depth_mm.expand(N * N)), dim=-1)
            psf = lens.psf_rgb(points=points, ks=self.kernel_size, spp=self.spp)

            # Downsample to sparse kernel size (15x15) if needed
            if self.kernel_size != self.kernel_size_small:
                psf = self.downsample_psf(psf, self.kernel_size_small)

        # Store grid_coords for compatibility
        self.grid_coords = coords_4d

        return psf, coords_4d

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
        Generate PSFs for 4D coordinates using stratified sampling with jitter.

        For training: Sample random 4D coordinates with jitter
        For validation: Use a fixed validation grid

        Args:
            data: Dictionary from DummyDataset with 'x': batch_index
            is_train: Training or validation mode
        """
        batch_size = len(data['x'])

        if is_train:
            # Sample 4D coordinates with stratified jitter
            num_samples = min(batch_size, self.samples_per_batch)
            coords_4d = self.sample_4d_coordinates(num_samples, use_jitter=True)

            # Generate PSFs for sampled coordinates
            psf_batch = self.generate_psfs_for_coords(coords_4d)

            # For smoothness loss, also generate a fixed 2D spatial grid
            # at a random focus distance
            foc_z = float(np.random.choice(self.foc_z_arr)) if len(self.foc_z_arr) > 0 else 0.5
            smooth_grid_psfs, smooth_grid_coords = self.generate_grid_psfs(foc_z)

        else:
            # Validation: use a fixed grid for consistent evaluation
            # Sample a subset of the full validation grid
            batch_idx = data['x'][0].item() if hasattr(data['x'][0], 'item') else data['x'][0]
            total_batches = len(self.test_dataloader) if hasattr(self, 'test_dataloader') else 10

            # Create validation coordinates
            val_coords = self.sample_validation_grid(
                grid_size_xy=7,
                grid_size_z=3,
                grid_size_f=3
            )

            # Subsample if too many
            max_val_samples = min(batch_size, 64)
            if len(val_coords) > max_val_samples:
                indices = torch.linspace(0, len(val_coords) - 1, max_val_samples).long()
                val_coords = val_coords[indices]

            coords_4d = val_coords
            psf_batch = self.generate_psfs_for_coords(coords_4d)

            # Use middle focus for smoothness grid during validation
            foc_z = 0.5
            smooth_grid_psfs, smooth_grid_coords = self.generate_grid_psfs(foc_z)

        # Store in sample dict
        self.sample = {
            'psf': psf_batch,  # (B, 3, kernel_size_small, kernel_size_small)
            'coords': coords_4d,  # (B, 4) - now 4D coordinates!
            'foc_z': foc_z if 'foc_z' in dir() else 0.5,
            'grid_psfs': smooth_grid_psfs.to(self.accelerator.device),  # For smoothness loss
            'grid_coords': smooth_grid_coords.to(self.accelerator.device),  # For smoothness loss
        }

    def optimize_parameters(self):
        """
        Training step: CVAE forward pass with reconstruction, KL, and smoothness losses.
        """
        for optimizer in self.optimizers:
            optimizer.zero_grad()

        # Get batch data
        psf = self.sample['psf']  # (B, 3, 15, 15)
        coords = self.sample['coords']  # (B, 4) - (x, y, z, f) coordinates

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
            grid_coords = self.sample['grid_coords']  # (N*N, 4) - 4D coordinates
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
