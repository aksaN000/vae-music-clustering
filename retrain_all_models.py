"""
Comprehensive Retraining Script for All 7 VAE Models
Trains each model with appropriate epochs for proper convergence
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path

def run_training(model_type, model_name, epochs, additional_args=""):
    """Run training for a specific model"""
    
    print(f"\n{'='*80}")
    print(f"TRAINING: {model_name}")
    print(f"Model Type: {model_type}")
    print(f"Epochs: {epochs}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print('='*80 + "\n")
    
    cmd = [sys.executable, "experiments/train_vae.py", "--model", model_type, "--epochs", str(epochs)]
    if additional_args:
        cmd.extend(additional_args.split())
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=False,
            text=True,
            check=True
        )
        print(f"\n* {model_name} training completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nX Error training {model_name}")
        return False


def main():
    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL RETRAINING")
    print("="*80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nModels to train:")
    print("  1. Basic VAE (50 epochs)")
    print("  2. Conv VAE (50 epochs)")
    print("  3. Beta-VAE (50 epochs)")
    print("  4. CVAE-Language (100 epochs)")
    print("  5. CVAE-Genre (100 epochs)")
    print("  6. VaDE (60 epochs)")
    print("  7. Multimodal VAE (50 epochs)")
    print("="*80 + "\n")
    
    results = {}
    
    # 1. Basic VAE - Baseline fully connected
    results['Basic VAE'] = run_training('basic', 'Basic VAE', 50)
    
    # 2. Conv VAE - Convolutional architecture
    results['Conv VAE'] = run_training('conv', 'Conv VAE', 50)
    
    # 3. Beta-VAE - Disentanglement with beta weighting
    results['Beta-VAE'] = run_training('beta', 'Beta-VAE', 50)
    
    # 4. CVAE-Language - Conditional on language
    results['CVAE-Language'] = run_training('cvae', 'CVAE-Language', 100, '--condition language')
    
    # 5. CVAE-Genre - Conditional on genre
    results['CVAE-Genre'] = run_training('cvae', 'CVAE-Genre', 100, '--condition genre')
    
    # 6. VaDE - Clustering VAE
    results['VaDE'] = run_training('vade', 'VaDE', 60)
    
    # 7. Multimodal VAE - Audio + Lyrics
    results['Multimodal VAE'] = run_training('multimodal', 'Multimodal VAE', 50)
    
    # Summary
    print(f"\n{'='*80}")
    print("TRAINING SUMMARY")
    print('='*80)
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nResults:")
    
    success_count = 0
    for model_name, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        symbol = "*" if success else "X"
        print(f"  {symbol} {model_name}: {status}")
        if success:
            success_count += 1
    
    print(f"\n{success_count}/{len(results)} models trained successfully")
    print('='*80 + "\n")


if __name__ == "__main__":
    main()
