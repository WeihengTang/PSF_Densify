"""
Conditional VAE Architecture for PSF Interpolation

Features:
1. Fourier positional encoding for coordinates
2. MLP-based encoder (posterior network)
3. MLP-based prior network (learned conditional prior)
4. MLP-based decoder with physical constraints
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig


class PositionalEncoding(nn.Module):
    """
    Fourier Positional Encoding for coordinates.

    Applies: γ(c) = [sin(2^0 π c), cos(2^0 π c), ..., sin(2^(L-1) π c), cos(2^(L-1) π c)]

    Args:
        num_frequencies: Number of frequency bands (L)
        input_dim: Dimension of input coordinates (default 2 for x,y)
    """

    def __init__(self, num_frequencies=10, input_dim=2):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.input_dim = input_dim

        # Create frequency bands: 2^0, 2^1, ..., 2^(L-1)
        freq_bands = 2.0 ** torch.linspace(0, num_frequencies - 1, num_frequencies)
        self.register_buffer('freq_bands', freq_bands)

    def forward(self, coords):
        """
        Args:
            coords: (B, input_dim) normalized coordinates

        Returns:
            pe: (B, input_dim * num_frequencies * 2) positional encoding
        """
        B = coords.shape[0]

        # Expand coordinates for each frequency band
        # coords: (B, input_dim) -> (B, input_dim, 1)
        # freq_bands: (num_frequencies,) -> (1, 1, num_frequencies)
        coords_expanded = coords.unsqueeze(-1)  # (B, input_dim, 1)
        freqs_expanded = self.freq_bands.view(1, 1, -1)  # (1, 1, num_frequencies)

        # Compute angles: coords * 2π * freq_bands
        angles = coords_expanded * math.pi * freqs_expanded  # (B, input_dim, num_frequencies)

        # Apply sin and cos
        sin_features = torch.sin(angles)  # (B, input_dim, num_frequencies)
        cos_features = torch.cos(angles)  # (B, input_dim, num_frequencies)

        # Interleave sin and cos: [sin, cos, sin, cos, ...]
        pe = torch.stack([sin_features, cos_features], dim=-1)  # (B, input_dim, num_frequencies, 2)
        pe = pe.view(B, -1)  # (B, input_dim * num_frequencies * 2)

        return pe


class MLP(nn.Module):
    """
    Multi-layer Perceptron with configurable depth and width.

    Args:
        input_dim: Input dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension
        num_layers: Number of hidden layers
        activation: Activation function ('relu', 'leaky_relu', 'silu')
        output_activation: Optional activation for output layer
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3,
                 activation='relu', output_activation=None):
        super().__init__()

        # Build layers
        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(self._get_activation(activation))

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(self._get_activation(activation))

        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))

        if output_activation is not None:
            layers.append(self._get_activation(output_activation))

        self.network = nn.Sequential(*layers)

    def _get_activation(self, activation):
        """Get activation function by name."""
        if activation == 'relu':
            return nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            return nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'silu':
            return nn.SiLU(inplace=True)
        elif activation == 'tanh':
            return nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, x):
        return self.network(x)


class CVAE_PSF_config(PretrainedConfig):
    """
    Configuration for CVAE_PSF architecture.

    Args:
        kernel_size: Size of PSF kernel (e.g., 15 for 15x15)
        in_channels: Number of color channels (3 for RGB)
        latent_dim: Dimension of latent space
        hidden_dim: Hidden dimension for MLPs
        num_layers: Number of layers in MLPs
        num_frequencies: Number of frequency bands for positional encoding
        activation: Activation function name
    """
    model_type = "CVAE_PSF"

    def __init__(
        self,
        kernel_size=15,
        in_channels=3,
        latent_dim=128,
        hidden_dim=256,
        num_layers=4,
        num_frequencies=10,
        activation='relu',
        **kwargs
    ):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_frequencies = num_frequencies
        self.activation = activation

        # Derived dimensions
        self.psf_dim = in_channels * kernel_size * kernel_size
        self.coord_dim = 2  # (x, y)
        self.pe_dim = self.coord_dim * num_frequencies * 2  # Positional encoding dim


class CVAE_PSF_arch(PreTrainedModel):
    """
    Conditional VAE for PSF Interpolation.

    Architecture:
    1. Encoder (Posterior): q(z | K, c) -> μ_q, log σ²_q
    2. Prior Network: p(z | c) -> μ_p, log σ²_p
    3. Decoder: p(K | z, c) -> K̂

    Physical Constraints:
    - Non-negativity: softplus activation
    - Normalization: sum to 1
    """
    config_class = CVAE_PSF_config

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # Positional encoding for coordinates
        self.pe = PositionalEncoding(
            num_frequencies=config.num_frequencies,
            input_dim=config.coord_dim
        )

        # Encoder (Posterior Network): q(z | K, c)
        # Input: [K_flat, PE(c)]
        encoder_input_dim = config.psf_dim + config.pe_dim
        self.encoder = MLP(
            input_dim=encoder_input_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.hidden_dim,
            num_layers=config.num_layers,
            activation=config.activation
        )
        self.encoder_mu = nn.Linear(config.hidden_dim, config.latent_dim)
        self.encoder_logvar = nn.Linear(config.hidden_dim, config.latent_dim)

        # Prior Network: p(z | c)
        # Input: PE(c)
        self.prior = MLP(
            input_dim=config.pe_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.hidden_dim,
            num_layers=config.num_layers,
            activation=config.activation
        )
        self.prior_mu = nn.Linear(config.hidden_dim, config.latent_dim)
        self.prior_logvar = nn.Linear(config.hidden_dim, config.latent_dim)

        # Decoder: p(K | z, c)
        # Input: [z, PE(c)]
        decoder_input_dim = config.latent_dim + config.pe_dim
        self.decoder = MLP(
            input_dim=decoder_input_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.psf_dim,
            num_layers=config.num_layers,
            activation=config.activation,
            output_activation=None  # We apply constraints manually
        )

    def positional_encoding(self, coords):
        """
        Apply positional encoding to coordinates.

        Args:
            coords: (B, 2) normalized coordinates

        Returns:
            pe: (B, pe_dim) positional encoding
        """
        return self.pe(coords)

    def encode(self, psf_flat, coords):
        """
        Encode PSF and coordinates to posterior parameters.

        Args:
            psf_flat: (B, psf_dim) flattened PSF
            coords: (B, 2) coordinates

        Returns:
            mu_q: (B, latent_dim) posterior mean
            logvar_q: (B, latent_dim) posterior log variance
        """
        # Positional encoding
        pe = self.positional_encoding(coords)

        # Concatenate PSF and PE
        x = torch.cat([psf_flat, pe], dim=-1)

        # Encoder network
        h = self.encoder(x)
        mu_q = self.encoder_mu(h)
        logvar_q = self.encoder_logvar(h)

        return mu_q, logvar_q

    def prior_network(self, coords):
        """
        Compute prior parameters conditioned on coordinates.

        Args:
            coords: (B, 2) coordinates

        Returns:
            mu_p: (B, latent_dim) prior mean
            logvar_p: (B, latent_dim) prior log variance
        """
        # Positional encoding
        pe = self.positional_encoding(coords)

        # Prior network
        h = self.prior(pe)
        mu_p = self.prior_mu(h)
        logvar_p = self.prior_logvar(h)

        return pe, mu_p, logvar_p

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = μ + σ * ε, where ε ~ N(0, I)

        Args:
            mu: (B, latent_dim) mean
            logvar: (B, latent_dim) log variance

        Returns:
            z: (B, latent_dim) sampled latent
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z, coords):
        """
        Decode latent code to PSF with physical constraints.

        Args:
            z: (B, latent_dim) latent code
            coords: (B, 2) coordinates

        Returns:
            psf_flat: (B, psf_dim) reconstructed PSF (flattened)
        """
        # Positional encoding
        pe = self.positional_encoding(coords)

        # Concatenate z and PE
        x = torch.cat([z, pe], dim=-1)

        # Decoder network
        psf_raw = self.decoder(x)

        # Apply physical constraints
        psf_constrained = self.apply_physical_constraints(psf_raw)

        return psf_constrained

    def apply_physical_constraints(self, psf_raw):
        """
        Apply physical constraints to PSF:
        1. Non-negativity: softplus
        2. Normalization: sum to 1 per channel

        Args:
            psf_raw: (B, psf_dim) raw PSF output

        Returns:
            psf: (B, psf_dim) constrained PSF
        """
        B = psf_raw.shape[0]
        C = self.config.in_channels
        HW = self.config.kernel_size * self.config.kernel_size

        # Reshape to (B, C, HW)
        psf = psf_raw.view(B, C, HW)

        # 1. Non-negativity: softplus
        psf = F.softplus(psf, beta=1.0)

        # 2. Normalization: sum to 1 per channel
        psf = psf / (psf.sum(dim=-1, keepdim=True) + 1e-8)

        # Flatten back to (B, psf_dim)
        psf = psf.view(B, -1)

        return psf

    def forward(self, psf_flat, coords):
        """
        Full forward pass of CVAE.

        Args:
            psf_flat: (B, psf_dim) flattened input PSF
            coords: (B, 2) coordinates

        Returns:
            recon_psf: (B, psf_dim) reconstructed PSF
            mu_q: (B, latent_dim) posterior mean
            logvar_q: (B, latent_dim) posterior log variance
            mu_p: (B, latent_dim) prior mean
            logvar_p: (B, latent_dim) prior log variance
        """
        # Encode to posterior
        mu_q, logvar_q = self.encode(psf_flat, coords)

        # Compute prior
        _, mu_p, logvar_p = self.prior_network(coords)

        # Sample from posterior
        z = self.reparameterize(mu_q, logvar_q)

        # Decode
        recon_psf = self.decode(z, coords)

        return recon_psf, mu_q, logvar_q, mu_p, logvar_p

    def sample_from_prior(self, coords):
        """
        Sample PSF from prior distribution (generation mode).

        Args:
            coords: (B, 2) coordinates

        Returns:
            psf_flat: (B, psf_dim) generated PSF
        """
        with torch.no_grad():
            # Compute prior
            _, mu_p, logvar_p = self.prior_network(coords)

            # Sample from prior
            z = self.reparameterize(mu_p, logvar_p)

            # Decode
            psf = self.decode(z, coords)

        return psf
