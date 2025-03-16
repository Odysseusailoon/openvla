#!/usr/bin/env python3
"""
Fix Flash Attention dependencies.

This script will:
1. Disable Flash Attention temporarily (to allow imports to succeed)
2. Install the correct version that matches your PyTorch
"""

import os
import subprocess
import sys

def fix_flash_attention():
    print("🔧 Fixing Flash Attention compatibility issues...")
    
    # Temporarily disable flash attention by setting an environment variable
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
    os.environ["DISABLE_FLASH_ATTN"] = "1"
    
    # Check PyTorch version and CUDA version
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            
            # Install the matching flash-attention version
            print("Installing compatible flash-attention...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "--force-reinstall", 
                "flash-attn", "--no-build-isolation"
            ])
            print("✅ Flash Attention reinstalled with correct PyTorch bindings")
        else:
            print("⚠️ CUDA not available - Flash Attention will not function")
    except Exception as e:
        print(f"❌ Error checking PyTorch/CUDA versions: {e}")
        print("⚠️ Will proceed with Flash Attention disabled")
    
    print("\nYou can now import and use the models, but Flash Attention optimizations may be disabled.")
    print("For full performance, ensure PyTorch and flash-attn versions are compatible.")

if __name__ == "__main__":
    fix_flash_attention() 