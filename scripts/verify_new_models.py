print("Verification script starting...")
import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from ml_mobility_ns3.models.vq_vae import ConditionalTrajectoryVQVAE
from ml_mobility_ns3.models.diffusion import TrajectoryDiffusionModel

def test_vq_vae():
    print("Testing VQ-VAE...")
    model = ConditionalTrajectoryVQVAE(
        input_dim=3,
        base_channels=32,
        num_embeddings=128,
        embedding_dim=32,
        sequence_length=100
    )
    
    x = torch.randn(2, 100, 3)
    transport_mode = torch.tensor([0, 1])
    length = torch.tensor([50, 80])
    mask = torch.ones(2, 100)
    
    output = model(x, transport_mode, length, mask)
    print(f"VQ-VAE output keys: {output.keys()}")
    print(f"Recon shape: {output['recon'].shape}")
    assert output['recon'].shape == (2, 100, 3)
    print("VQ-VAE test passed!")

def test_diffusion():
    print("\nTesting Diffusion...")
    model = TrajectoryDiffusionModel(
        input_dim=3,
        base_channels=32,
        timesteps=10, # small for testing
        sequence_length=100
    )
    
    x = torch.randn(2, 100, 3)
    transport_mode = torch.tensor([0, 1])
    length = torch.tensor([50, 80])
    mask = torch.ones(2, 100)
    
    output = model(x, transport_mode, length, mask)
    print(f"Diffusion output keys: {output.keys()}")
    print(f"Predicted noise shape: {output['predicted_noise'].shape}")
    assert output['predicted_noise'].shape == (2, 3, 100) # U-Net returns (B, C, L)
    
    print("Testing Diffusion generation...")
    gen = model.generate({'transport_mode': transport_mode, 'length': length}, n_samples=2, target_length=100)
    print(f"Generated shape: {gen.shape}")
    assert gen.shape == (2, 100, 3)
    print("Diffusion test passed!")

if __name__ == "__main__":
    try:
        test_vq_vae()
        test_diffusion()
        print("\nAll tests passed successfully!")
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
