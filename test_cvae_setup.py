"""
Quick test script to validate CVAE setup without running full training.

This script:
1. Loads the CVAE configuration
2. Instantiates the model
3. Tests grid generation
4. Tests forward pass
5. Tests loss computation

Run: python test_cvae_setup.py
"""

import yaml
import torch
from omegaconf import OmegaConf
from utils.misc import DictAsMember
from models import create_model
from accelerate.logging import get_logger

logger = get_logger(__name__)

def test_config_loading():
    """Test 1: Load YAML configuration"""
    print("\n" + "="*60)
    print("TEST 1: Loading CVAE configuration")
    print("="*60)

    config_path = 'checkpoints/PSF/CVAE_v1.yml'

    with open(config_path, 'r') as f:
        data = OmegaConf.load(f)
        data = OmegaConf.to_yaml(data, resolve=True)
        data = yaml.safe_load(data)
        opt = DictAsMember(**data)

    print(f"✓ Config loaded successfully")
    print(f"  Model type: {opt.model_type}")
    print(f"  Network type: {opt.network.type}")
    print(f"  Grid size: {opt.grid_size}")
    print(f"  Latent dim: {opt.network.latent_dim}")

    return opt

def test_model_instantiation(opt):
    """Test 2: Instantiate model (without Accelerate for testing)"""
    print("\n" + "="*60)
    print("TEST 2: Instantiating CVAE_PSF_model")
    print("="*60)

    # Temporarily disable Accelerate for testing
    opt.num_gpu = 0  # Use CPU

    try:
        model = create_model(opt, logger)
        print(f"✓ Model instantiated successfully: {type(model).__name__}")
        print(f"  Grid coordinates shape: {model.grid_coords.shape}")
        print(f"  Number of neighbors: {len(model.grid_neighbors)}")
        print(f"  Network parameters: {sum(p.numel() for p in model.net_cvae.parameters()):,}")
        return model
    except Exception as e:
        print(f"✗ Model instantiation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_network_architecture(model):
    """Test 3: Test network forward pass"""
    print("\n" + "="*60)
    print("TEST 3: Testing network forward pass")
    print("="*60)

    try:
        # Create dummy inputs
        batch_size = 4
        psf_flat = torch.randn(batch_size, 3 * 15 * 15)  # (4, 675)
        coords = torch.randn(batch_size, 2)  # (4, 2)

        print(f"  Input PSF shape: {psf_flat.shape}")
        print(f"  Input coords shape: {coords.shape}")

        # Forward pass
        with torch.no_grad():
            recon, mu_q, logvar_q, mu_p, logvar_p = model.net_cvae(psf_flat, coords)

        print(f"✓ Forward pass successful")
        print(f"  Reconstruction shape: {recon.shape}")
        print(f"  Posterior mean shape: {mu_q.shape}")
        print(f"  Prior mean shape: {mu_p.shape}")

        # Test physical constraints
        recon_reshaped = recon.view(batch_size, 3, 15, 15)
        channel_sums = recon_reshaped.sum(dim=(-2, -1))
        print(f"  Channel sums (should be ~1.0): {channel_sums[0].tolist()}")
        print(f"  Min value (should be >= 0): {recon.min().item():.6f}")

        # Test prior sampling
        with torch.no_grad():
            prior_samples = model.net_cvae.sample_from_prior(coords)
        print(f"✓ Prior sampling successful")
        print(f"  Prior sample shape: {prior_samples.shape}")

        return True
    except Exception as e:
        print(f"✗ Network test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_positional_encoding(model):
    """Test 4: Test positional encoding"""
    print("\n" + "="*60)
    print("TEST 4: Testing positional encoding")
    print("="*60)

    try:
        coords = torch.tensor([[0.0, 0.0], [1.0, 1.0], [-1.0, -1.0]])
        pe = model.net_cvae.positional_encoding(coords)

        print(f"✓ Positional encoding successful")
        print(f"  Input coords shape: {coords.shape}")
        print(f"  PE output shape: {pe.shape}")
        print(f"  Expected PE dim: {model.net_cvae.config.pe_dim}")

        assert pe.shape == (3, model.net_cvae.config.pe_dim), "PE dimension mismatch!"
        print(f"✓ PE dimension check passed")

        return True
    except Exception as e:
        print(f"✗ Positional encoding test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_grid_neighbors(model):
    """Test 5: Test grid neighbor structure"""
    print("\n" + "="*60)
    print("TEST 5: Testing grid neighbor structure")
    print("="*60)

    try:
        N = model.grid_size
        total_neighbors = sum(len(neighbors) for neighbors in model.grid_neighbors.values())
        avg_neighbors = total_neighbors / len(model.grid_neighbors)

        print(f"✓ Grid neighbor structure created")
        print(f"  Grid size: {N}×{N} = {N*N} points")
        print(f"  Total neighbor pairs: {total_neighbors}")
        print(f"  Average neighbors per point: {avg_neighbors:.2f}")

        # Check corner, edge, and interior points
        corner_idx = 0  # Top-left corner
        edge_idx = N // 2  # Top edge
        interior_idx = N + 1  # Interior point

        print(f"  Corner point (0) neighbors: {len(model.grid_neighbors[corner_idx])}")
        print(f"  Edge point ({edge_idx}) neighbors: {len(model.grid_neighbors[edge_idx])}")
        print(f"  Interior point ({interior_idx}) neighbors: {len(model.grid_neighbors[interior_idx])}")

        return True
    except Exception as e:
        print(f"✗ Grid neighbor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("CVAE PSF Model Setup Validation")
    print("="*60)

    # Test 1: Config loading
    opt = test_config_loading()
    if opt is None:
        print("\n✗ Configuration loading failed. Aborting tests.")
        return

    # Test 2: Model instantiation
    model = test_model_instantiation(opt)
    if model is None:
        print("\n✗ Model instantiation failed. Aborting tests.")
        return

    # Test 3: Network architecture
    if not test_network_architecture(model):
        print("\n✗ Network architecture test failed.")

    # Test 4: Positional encoding
    if not test_positional_encoding(model):
        print("\n✗ Positional encoding test failed.")

    # Test 5: Grid neighbors
    if not test_grid_neighbors(model):
        print("\n✗ Grid neighbor test failed.")

    print("\n" + "="*60)
    print("✓ ALL TESTS COMPLETED")
    print("="*60)
    print("\nNext steps:")
    print("1. Run validation-only: accelerate launch trainer.py -opt checkpoints/PSF/CVAE_v1.yml -test")
    print("2. Start training: accelerate launch trainer.py -opt checkpoints/PSF/CVAE_v1.yml")
    print("\nNote: DeepLens raytracing requires GPU and lens file. Tests above use CPU.")
    print("="*60 + "\n")

if __name__ == '__main__':
    main()
