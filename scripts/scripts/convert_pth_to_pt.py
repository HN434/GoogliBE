"""
Utility script to convert PyTorch checkpoint (.pth) to Ultralytics format (.pt)

This script helps convert RT-DETR model checkpoints from .pth (PyTorch) format
to .pt (Ultralytics) format for use with the bat detection system.

Usage:
    python scripts/convert_pth_to_pt.py --input checkpoint_best_regular.pth --output rtdetr_bat.pt --base-model rtdetr-l
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from ultralytics import RTDETR
    RTDETR_AVAILABLE = True
except ImportError:
    RTDETR_AVAILABLE = False
    print("❌ Ultralytics not available. Install: pip install ultralytics>=8.1.0")
    sys.exit(1)


def load_checkpoint_state_dict(checkpoint_path: Path) -> dict:
    """
    Load state dict from various checkpoint formats
    
    Args:
        checkpoint_path: Path to .pth checkpoint file
        
    Returns:
        State dictionary
    """
    print(f"📦 Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(str(checkpoint_path), map_location='cpu')
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
            print("   Found state_dict in checkpoint['model']")
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print("   Found state_dict in checkpoint['state_dict']")
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print("   Found state_dict in checkpoint['model_state_dict']")
        else:
            # Assume entire dict is the state_dict
            state_dict = checkpoint
            print("   Using entire checkpoint as state_dict")
            
        # Print some info about the checkpoint
        if 'epoch' in checkpoint:
            print(f"   Checkpoint epoch: {checkpoint['epoch']}")
        if 'best_fitness' in checkpoint or 'best_score' in checkpoint:
            score = checkpoint.get('best_fitness') or checkpoint.get('best_score')
            print(f"   Best score: {score}")
    else:
        # Checkpoint might be the model itself or state dict directly
        if hasattr(checkpoint, 'state_dict'):
            state_dict = checkpoint.state_dict()
            print("   Extracted state_dict from model object")
        else:
            state_dict = checkpoint
            print("   Using checkpoint directly as state_dict")
    
    # Clean state dict keys (remove common prefixes)
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        # Remove common prefixes
        clean_key = key
        for prefix in ['model.', 'module.', '_orig_mod.']:
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix):]
        cleaned_state_dict[clean_key] = value
    
    print(f"   State dict contains {len(cleaned_state_dict)} keys")
    
    # Print first few keys for debugging
    if cleaned_state_dict:
        sample_keys = list(cleaned_state_dict.keys())[:5]
        print(f"   Sample keys: {sample_keys}")
    
    return cleaned_state_dict


def convert_checkpoint_to_ultralytics(
    checkpoint_path: Path,
    output_path: Path,
    base_model: str = "rtdetr-l.pt",
) -> bool:
    """
    Convert PyTorch checkpoint to Ultralytics format
    
    Args:
        checkpoint_path: Path to input .pth checkpoint
        output_path: Path to output .pt model
        base_model: Base RT-DETR model architecture to use
        
    Returns:
        True if conversion successful
    """
    try:
        # Load checkpoint state dict
        state_dict = load_checkpoint_state_dict(checkpoint_path)
        
        # Load base RT-DETR model
        print(f"\n🏗️  Loading base model: {base_model}")
        model = RTDETR(base_model)
        
        # Load checkpoint weights into model
        print(f"🔧 Loading checkpoint weights into model...")
        
        if hasattr(model, 'model'):
            # Try to load state dict
            try:
                # First try strict loading
                incompatible_keys = model.model.load_state_dict(state_dict, strict=True)
                print(f"   ✅ Loaded all weights successfully (strict=True)")
            except Exception as e:
                # Fall back to non-strict loading
                print(f"   ⚠️  Strict loading failed: {e}")
                print(f"   Attempting non-strict loading...")
                incompatible_keys = model.model.load_state_dict(state_dict, strict=False)
                print(f"   ✅ Loaded weights with strict=False")
                
                # Report any missing or unexpected keys
                if hasattr(incompatible_keys, 'missing_keys') and incompatible_keys.missing_keys:
                    print(f"   Missing keys: {len(incompatible_keys.missing_keys)}")
                    if len(incompatible_keys.missing_keys) <= 10:
                        for key in incompatible_keys.missing_keys:
                            print(f"      - {key}")
                
                if hasattr(incompatible_keys, 'unexpected_keys') and incompatible_keys.unexpected_keys:
                    print(f"   Unexpected keys: {len(incompatible_keys.unexpected_keys)}")
                    if len(incompatible_keys.unexpected_keys) <= 10:
                        for key in incompatible_keys.unexpected_keys:
                            print(f"      - {key}")
        else:
            print(f"   ⚠️  Model structure unexpected")
            return False
        
        # Override model class names for fine-tuned bat detection model
        print(f"\n🏷️  Updating model metadata for bat detection...")
        if hasattr(model, 'names'):
            print(f"   Original classes: {model.names}")
            # Override with bat class only
            model.names = {0: 'bat'}
            print(f"   Updated classes: {model.names}")
        
        # Update number of classes in model
        if hasattr(model, 'model') and hasattr(model.model, 'nc'):
            print(f"   Original nc (num classes): {model.model.nc}")
            model.model.nc = 1  # Only 1 class: bat
            print(f"   Updated nc: {model.model.nc}")
        
        # Save in Ultralytics format
        print(f"\n💾 Saving model in Ultralytics format to: {output_path}")
        model.save(str(output_path))
        
        print(f"\n✅ Conversion successful!")
        print(f"   Input:  {checkpoint_path}")
        print(f"   Output: {output_path}")
        print(f"\nYou can now use this model with:")
        print(f"   RTDETR_MODEL_PATH={output_path}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Convert PyTorch checkpoint (.pth) to Ultralytics format (.pt)"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to input .pth checkpoint file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to output .pt model file",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="rtdetr-l.pt",
        choices=["rtdetr-l.pt", "rtdetr-x.pt"],
        help="Base RT-DETR model architecture (default: rtdetr-l.pt)",
    )
    
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"❌ Input checkpoint not found: {args.input}")
        sys.exit(1)
    
    if args.output.exists():
        response = input(f"⚠️  Output file already exists: {args.output}\nOverwrite? (y/N): ")
        if response.lower() != 'y':
            print("Aborted.")
            sys.exit(0)
    
    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # Perform conversion
    success = convert_checkpoint_to_ultralytics(
        checkpoint_path=args.input,
        output_path=args.output,
        base_model=args.base_model,
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

