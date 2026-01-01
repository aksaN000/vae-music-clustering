"""
Complete Pipeline Runner
Runs data processing, training, clustering, and generates all results
"""

import subprocess
import sys
from pathlib import Path
import time


def run_command(command, description):
    """Run a command and print status"""
    print("\n" + "="*80)
    print(f"STEP: {description}")
    print("="*80)
    print(f"Command: {command}\n")
    
    start_time = time.time()
    result = subprocess.run(command, shell=True)
    elapsed = time.time() - start_time
    
    if result.returncode != 0:
        print(f"\nERROR: {description} failed!")
        return False
    
    print(f"\nCompleted in {elapsed/60:.1f} minutes")
    return True


def main():
    print("\n" + "="*80)
    print("VAE MUSIC CLUSTERING - COMPLETE PIPELINE")
    print("="*80)
    
    # Check if data is already processed
    features_dir = Path("data/features")
    if not features_dir.exists() or len(list(features_dir.glob("*.pkl"))) == 0:
        # Step 1: Process Audio
        if not run_command(
            "python src/data/audio_processor.py --audio_dir data/audio --n_workers 15",
            "Audio Feature Extraction (All 1120 files)"
        ):
            return
        
        # Step 2: Process Lyrics
        if not run_command(
            "python src/data/lyrics_processor.py --lyrics_dir data/processed_lyrics",
            "Lyrics Processing with Multilingual Transformers"
        ):
            return
        
        # Step 3: Match Audio-Lyrics Pairs
        if not run_command(
            "python src/data/data_matcher.py",
            "Matching Audio-Lyrics Pairs"
        ):
            return
    else:
        print("\nData already processed, skipping to training...")
    
    # Step 4: Train Models
    models_to_train = [
        ("basic", "audio", "Basic VAE (Easy Task)"),
        ("conv", "audio", "Convolutional VAE (Medium Task)"),
        ("beta", "audio", "Beta-VAE (Hard Task)"),
        ("cvae", "audio", "Conditional VAE (Hard Task)"),
        ("vade", "audio", "VaDE with GMM Priors")
    ]
    
    for model, modality, description in models_to_train:
        checkpoint_dir = Path(f"results/checkpoints/{model}")
        if checkpoint_dir.exists() and (checkpoint_dir / "best_model.pt").exists():
            print(f"\n{description} already trained, skipping...")
            continue
        
        if not run_command(
            f"python experiments/train_vae.py --model {model} --modality {modality}",
            f"Training: {description}"
        ):
            print(f"WARNING: Training {model} failed, continuing...")
            continue
    
    # Step 5: Run Clustering Experiments
    for model in ["basic", "conv", "beta", "cvae", "vade"]:
        checkpoint_path = f"results/checkpoints/{model}/best_model.pt"
        if not Path(checkpoint_path).exists():
            print(f"\nSkipping clustering for {model} (no checkpoint)")
            continue
        
        if not run_command(
            f"python experiments/run_clustering.py --model {model} --checkpoint {checkpoint_path}",
            f"Clustering with {model.upper()}"
        ):
            print(f"WARNING: Clustering with {model} failed, continuing...")
            continue
    
    # Step 6: Run Baseline Comparisons
    if not run_command(
        "python experiments/baseline.py",
        "Baseline Comparisons (PCA, Autoencoder)"
    ):
        print("WARNING: Baseline comparison failed, continuing...")
    
    # Step 7: Generate Final Report
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)
    print("\nResults saved in:")
    print("  - results/checkpoints/     : Trained model weights")
    print("  - results/visualizations/  : All plots and figures")
    print("  - results/metrics/         : CSV files with all metrics")
    print("\nNext steps:")
    print("  1. Review visualizations in results/visualizations/")
    print("  2. Compare metrics in results/metrics/*.csv")
    print("  3. Write NeurIPS-style paper using results/")
    print("="*80)


if __name__ == "__main__":
    main()
