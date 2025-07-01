import json
import torch
import shutil
import pickle
from pathlib import Path
from jinja2 import Environment, FileSystemLoader

# Répertoires
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "cpp" / "utils" / "templates"
CPP_DIR = BASE_DIR / "cpp"
RESULTS_DIR = BASE_DIR / "results" / "efficient_run"
PREPROCESSING_DIR = BASE_DIR / "preprocessing"

# Fichiers de modèle
MODEL_PATH = BASE_DIR / "traced_vae_model.pt"
METADATA_PATH = BASE_DIR / "metadata.json"
SCALERS_PATH = BASE_DIR / "scalers.json"

# Jinja2 : setup
env = Environment(loader=FileSystemLoader(TEMPLATES_DIR))


def render_template(template_name, output_path, context=None):
    context = context or {}
    template = env.get_template(template_name)
    with open(output_path, 'w') as f:
        f.write(template.render(context))


def copy_placeholder(filename: str):
    src = TEMPLATES_DIR / filename
    dst = CPP_DIR / filename.replace(".placeholder", "")
    shutil.copyfile(src, dst)


def load_model_metadata():
    """Load model configuration from the saved model."""
    try:
        best_model_path = RESULTS_DIR / "best_model.pt"
        checkpoint = torch.load(best_model_path, map_location='cpu')
        
        # Handle different save formats
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                model = checkpoint['model']
            elif 'model_state_dict' in checkpoint:
                # Need to reconstruct model from config
                if 'config' in checkpoint:
                    config = checkpoint['config']
                    print(f"📋 Model configuration loaded from checkpoint: {config}")
                    return None, config  # Return config without model object
                else:
                    print("❌ Checkpoint contains state_dict but no config")
                    return None, None
            else:
                print("❌ Unknown checkpoint format")
                return None, None
        else:
            # Direct model object
            model = checkpoint
        
        # Get model configuration
        if model and hasattr(model, 'get_config'):
            config = model.get_config()
        elif model:
            # Fallback: try to infer from model structure
            config = {
                'input_dim': getattr(model, 'input_dim', 3),
                'sequence_length': getattr(model, 'sequence_length', 2000),
                'hidden_dim': getattr(model, 'hidden_dim', 128),
                'latent_dim': getattr(model, 'latent_dim', 32),
                'num_layers': getattr(model, 'num_layers', 2),
                'num_transport_modes': getattr(model, 'num_transport_modes', 10),
                'condition_dim': getattr(model, 'condition_dim', 32),
                'architecture': getattr(model, 'architecture', 'lstm'),
            }
        else:
            return None, None
        
        print(f"📋 Model configuration loaded: {config}")
        return model, config
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None


def create_traced_model():
    """Create traced model from best_model.pt using actual model configuration."""
    print("🔄 Creating traced model...")
    
    # First, let's check what's in the checkpoint
    best_model_path = RESULTS_DIR / "best_model.pt"
    checkpoint = torch.load(best_model_path, map_location='cpu')
    
    try:
        # Try to reconstruct the model
        if isinstance(checkpoint, dict) and 'config' in checkpoint:
            config = checkpoint['config']
            
            # Import and recreate the model
            from models.vae import ConditionalTrajectoryVAE
            model = ConditionalTrajectoryVAE(**config)
            
            # Load the state dict
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'model' in checkpoint and hasattr(checkpoint['model'], 'state_dict'):
                model.load_state_dict(checkpoint['model'].state_dict())
            else:
                print("❌ Could not find model state dict in checkpoint")
                return False
                
        elif hasattr(checkpoint, 'eval'):
            # Direct model object
            model = checkpoint
            config = model.get_config() if hasattr(model, 'get_config') else {}
        else:
            print("❌ Could not reconstruct model from checkpoint")
            print(f"   Checkpoint keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else type(checkpoint)}")
            return False
        
        model.eval()
        
        # Create dummy inputs based on actual model configuration
        batch_size = 1
        seq_len = config.get('sequence_length', 2000)
        input_dim = config.get('input_dim', 3)
        num_transport_modes = config.get('num_transport_modes', 10)
        
        # Create dummy inputs for VAE forward pass
        dummy_trajectory = torch.randn(batch_size, seq_len, input_dim)
        dummy_transport_mode = torch.randint(0, num_transport_modes, (batch_size,))
        dummy_trip_length = torch.randint(50, seq_len, (batch_size,))
        dummy_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        
        print(f"🔍 Model architecture: {config.get('architecture', 'unknown')}")
        print(f"📏 Input shape: ({batch_size}, {seq_len}, {input_dim})")
        
        # Trace the model
        traced_model = torch.jit.trace(
            model, 
            (dummy_trajectory, dummy_transport_mode, dummy_trip_length, dummy_mask)
        )
        
        # Save traced model
        torch.jit.save(traced_model, MODEL_PATH)
        print(f"✅ Traced model saved: {MODEL_PATH}")
        return True
        
    except Exception as e:
        print(f"❌ Error during tracing: {e}")
        print("💡 Tip: The model might not be compatible with TorchScript tracing")
        print("    Alternative: Use the regular PyTorch model file with LibTorch")
        
        # Alternative: Save the model in a different format
        try:
            print("🔄 Attempting to save model without tracing...")
            if isinstance(checkpoint, dict) and 'config' in checkpoint:
                # Save just the state dict and config for C++
                torch.save({
                    'state_dict': model.state_dict(),
                    'config': config
                }, MODEL_PATH.with_suffix('.pth'))
                print(f"✅ Model saved as .pth file: {MODEL_PATH.with_suffix('.pth')}")
                return True
        except Exception as e2:
            print(f"❌ Alternative save also failed: {e2}")
            
        return False


def load_preprocessing_metadata():
    """Load metadata from preprocessing directory."""
    metadata_pkl_path = PREPROCESSING_DIR / "metadata.pkl"
    
    if not metadata_pkl_path.exists():
        print(f"❌ Preprocessing metadata not found: {metadata_pkl_path}")
        return None
    
    try:
        with open(metadata_pkl_path, 'rb') as f:
            metadata = pickle.load(f)
        print(f"📊 Preprocessing metadata loaded: {list(metadata.keys())}")
        return metadata
    except Exception as e:
        print(f"❌ Error loading preprocessing metadata: {e}")
        return None


def load_scalers():
    """Load actual scalers from preprocessing directory."""
    scalers_pkl_path = PREPROCESSING_DIR / "scalers.pkl"
    
    if not scalers_pkl_path.exists():
        print(f"❌ Scalers not found: {scalers_pkl_path}")
        return None
    
    try:
        with open(scalers_pkl_path, 'rb') as f:
            scalers = pickle.load(f)
        print(f"⚖️  Scalers loaded: {list(scalers.keys())}")
        return scalers
    except Exception as e:
        print(f"❌ Error loading scalers: {e}")
        return None


def scalers_to_json(scalers):
    """Convert sklearn scalers to JSON-serializable format."""
    scalers_json = {}
    
    for name, scaler in scalers.items():
        if hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_'):
            # StandardScaler
            scalers_json[name] = {
                'type': 'StandardScaler',
                'mean': scaler.mean_.tolist(),
                'scale': scaler.scale_.tolist(),
                'var': scaler.var_.tolist() if hasattr(scaler, 'var_') else None
            }
        elif hasattr(scaler, 'min_') and hasattr(scaler, 'scale_'):
            # MinMaxScaler
            scalers_json[name] = {
                'type': 'MinMaxScaler',
                'min': scaler.min_.tolist(),
                'scale': scaler.scale_.tolist(),
                'data_min': scaler.data_min_.tolist(),
                'data_max': scaler.data_max_.tolist(),
                'data_range': scaler.data_range_.tolist()
            }
        else:
            # Fallback for unknown scaler types
            scalers_json[name] = {
                'type': str(type(scaler).__name__),
                'attributes': {k: v.tolist() if hasattr(v, 'tolist') else v 
                             for k, v in scaler.__dict__.items() 
                             if not k.startswith('_')}
            }
    
    return scalers_json


def create_metadata():
    """Create metadata.json from model config, preprocessing metadata, and training history."""
    print("📄 Creating metadata...")
    
    # Load model configuration
    _, model_config = load_model_metadata()
    if model_config is None:
        return False
    
    # Load preprocessing metadata
    prep_metadata = load_preprocessing_metadata()
    
    # Load training history
    history_path = RESULTS_DIR / "history.json"
    history = {}
    if history_path.exists():
        try:
            with open(history_path, 'r') as f:
                history = json.load(f)
        except Exception as e:
            print(f"⚠️  Could not load training history: {e}")
    
    # Load sequence length
    seq_len_path = PREPROCESSING_DIR / "sequence_length.txt"
    sequence_length = model_config.get('sequence_length', 2000)
    if seq_len_path.exists():
        try:
            with open(seq_len_path, 'r') as f:
                sequence_length = int(f.read().strip())
        except Exception as e:
            print(f"⚠️  Could not read sequence_length.txt: {e}")
    
    # Create comprehensive metadata
    metadata = {
        "model": {
            **model_config,
            "sequence_length": sequence_length
        },
        "training": {
            "epochs": len(history.get("train_loss", [])),
            "final_train_loss": history.get("train_loss", [])[-1] if history.get("train_loss") else None,
            "final_val_loss": history.get("val_loss", [])[-1] if history.get("val_loss") else None,
            "best_epoch": history.get("best_epoch", None) if "best_epoch" in history else None
        },
        "preprocessing": prep_metadata if prep_metadata else {},
        "created_at": "2025-07-01",
        "files": {
            "model": "traced_vae_model.pt",
            "scalers": "scalers.json"
        }
    }
    
    try:
        with open(METADATA_PATH, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Metadata created: {METADATA_PATH}")
        return True
        
    except Exception as e:
        print(f"❌ Error creating metadata: {e}")
        return False


def create_scalers_json():
    """Create scalers.json from actual saved scalers."""
    print("⚖️  Creating scalers JSON file...")
    
    scalers = load_scalers()
    if scalers is None:
        return False
    
    try:
        scalers_json = scalers_to_json(scalers)
        
        with open(SCALERS_PATH, 'w') as f:
            json.dump(scalers_json, f, indent=2)
        
        print(f"✅ Scalers JSON created: {SCALERS_PATH}")
        return True
        
    except Exception as e:
        print(f"❌ Error creating scalers JSON: {e}")
        return False


def generate_cpp_project():
    print("📦 Generating C++ project...")

    # Create missing files from actual data
    traced_ok = create_traced_model()
    metadata_ok = create_metadata()
    scalers_ok = create_scalers_json()
    
    if not all([traced_ok, metadata_ok, scalers_ok]):
        print("⚠️  Some files could not be created, but continuing...")

    # Clean old files (except utils directory)
    if CPP_DIR.exists():
        for f in CPP_DIR.iterdir():
            if f.is_file():
                f.unlink()
            elif f.is_dir() and f.name != "utils":
                shutil.rmtree(f)

    # Context for Jinja2 templates
    context = {
        "model_path": MODEL_PATH.name,
        "metadata_path": METADATA_PATH.name,
        "scalers_path": SCALERS_PATH.name,
    }

    templates = [
        "main.cc.jinja",
        "trajectory_generator.h.jinja", 
        "trajectory_generator.cc.jinja",
        "CMakeLists.txt.jinja",
        "README.md.jinja"
    ]
    
    for tpl in templates:
        try:
            output_file = CPP_DIR / tpl.replace(".jinja", "")
            render_template(tpl, output_file, context)
            print(f"✅ Template generated: {output_file.name}")
        except Exception as e:
            print(f"❌ Error with template {tpl}: {e}")

    # Copy placeholder files
    try:
        copy_placeholder("json.hpp.placeholder")
        print("✅ json.hpp copied")
    except Exception as e:
        print(f"❌ Error copying json.hpp: {e}")

    # Copy the generated files to cpp directory
    for file in [MODEL_PATH, METADATA_PATH, SCALERS_PATH]:
        if file.exists():
            try:
                shutil.copyfile(file, CPP_DIR / file.name)
                print(f"✅ File copied: {file.name}")
            except Exception as e:
                print(f"❌ Error copying {file.name}: {e}")
        else:
            print(f"⚠️  Missing file: {file}")

    print(f"\n🎉 C++ project generated in: {CPP_DIR}")
    print("\n📋 Next steps:")
    print("   1. cd cpp")
    print("   2. mkdir build && cd build")
    print("   3. cmake ..")
    print("   4. make")


if __name__ == "__main__":
    generate_cpp_project()