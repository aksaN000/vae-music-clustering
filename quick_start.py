"""
Quick Start - Train a Single Model for Testing
This script trains Conv VAE on audio data as a quick test
"""

import subprocess
import sys
from pathlib import Path

def main():
    print("="*80)
    print("QUICK START - CONV VAE TRAINING")
    print("="*80)
    print("\nThis will train a Convolutional VAE on audio-only data")
    print("Optimized for 15-core CPU + GPU with mixed precision")
    print("\nConfiguration:")
    print("  - Model: Conv VAE")
    print("  - Data: Audio-only (1,119 files)")
    print("  - Epochs: 50 (quick training)")
    print("  - Batch size: 32")
    print("  - Hardware: 15 workers + GPU FP16")
    print("="*80)
    
    response = input("\nStart training? (y/n): ").strip().lower()
    if response not in ['y', 'yes']:
        print("Training cancelled.")
        return
    
    # Train Conv VAE
    cmd = [
        sys.executable,
        'experiments/train_vae.py',
        '--model', 'conv_vae',
        '--modality', 'audio',
        '--epochs', '50',
        '--batch_size', '32',
        '--learning_rate', '0.0001'
    ]
    
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80)
    print(f"Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    
    if result.returncode == 0:
        print("\n" + "="*80)
        print("✓ TRAINING COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\nModel checkpoint saved in: checkpoints/")
        print("\nNext steps:")
        print("  1. Run clustering: python experiments/run_clustering.py --checkpoint checkpoints/best_model.pt --model conv")
        print("  2. Try other models: python train_all.py")
        print("  3. Train with lyrics: python experiments/train_vae.py --model conv_vae --modality multimodal")
    else:
        print("\n" + "="*80)
        print("✗ TRAINING FAILED")
        print("="*80)
        print("\nPlease check the error messages above.")

if __name__ == "__main__":
    main()
