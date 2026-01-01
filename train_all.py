"""
Comprehensive Training Launcher
Trains all VAE models with optimal settings for 15-core CPU + GPU
"""

import subprocess
import sys
from pathlib import Path

def run_training(model_type, modality='audio', condition='language', extra_args=None):
    """Run training with specified configuration"""
    
    cmd = [
        sys.executable,
        'experiments/train_vae.py',
        '--model', model_type,
        '--modality', modality,
        '--epochs', '100',
        '--batch_size', '32',
        '--learning_rate', '0.0001'
    ]
    
    if model_type == 'cvae':
        cmd.extend(['--condition', condition])
    
    if extra_args:
        cmd.extend(extra_args)
    
    print("\n" + "="*80)
    print(f"Training: {model_type.upper()} ({modality}) {'- Condition: ' + condition if model_type == 'cvae' else ''}")
    print("="*80)
    print(f"Command: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    
    if result.returncode != 0:
        print(f"WARNING: Training failed for {model_type}")
        return False
    
    return True

def main():
    print("="*80)
    print("MULTI-LINGUAL MUSIC CLUSTERING - FULL TRAINING PIPELINE")
    print("="*80)
    print("\nDataset:")
    print("  - 5 Languages: Arabic, Bangla, English, Hindi, Spanish")
    print("  - 45 Genres")
    print("  - 1,119 audio files")
    print("  - 677 audio-lyrics pairs")
    print("\nHardware Optimization:")
    print("  - 15-core CPU parallel processing")
    print("  - GPU with mixed precision (FP16)")
    print("\nTraining Schedule:")
    print("  Phase 1: Audio-only models (Basic VAE, Conv VAE, Beta-VAE)")
    print("  Phase 2: Conditional models (CVAE for language and genre)")
    print("  Phase 3: Joint clustering (VaDE)")
    print("  Phase 4: Multimodal models (audio + lyrics)")
    
    input("\nPress Enter to start training or Ctrl+C to cancel...")
    
    # Phase 1: Basic Audio-Only Models
    print("\n" + "#"*80)
    print("# PHASE 1: BASIC AUDIO-ONLY MODELS")
    print("#"*80)
    
    models_phase1 = [
        ('basic_vae', 'audio', None),
        ('conv_vae', 'audio', None),
        ('beta_vae', 'audio', None),
    ]
    
    for model, modality, _ in models_phase1:
        success = run_training(model, modality)
        if not success:
            print(f"Skipping remaining models due to failure in {model}")
            break
    
    # Phase 2: Conditional Models
    print("\n" + "#"*80)
    print("# PHASE 2: CONDITIONAL MODELS (LANGUAGE & GENRE)")
    print("#"*80)
    
    models_phase2 = [
        ('cvae', 'audio', 'language'),  # 5 language classes
        ('cvae', 'audio', 'genre'),     # 45 genre classes
    ]
    
    for model, modality, condition in models_phase2:
        success = run_training(model, modality, condition)
        if not success:
            print(f"Skipping remaining models due to failure in {model}")
            break
    
    # Phase 3: VaDE (Joint Clustering)
    print("\n" + "#"*80)
    print("# PHASE 3: VARIATIONAL DEEP EMBEDDING (VaDE)")
    print("#"*80)
    
    run_training('vade', 'audio')
    
    # Phase 4: Multimodal Models (if user confirms)
    print("\n" + "#"*80)
    print("# PHASE 4: MULTIMODAL MODELS (AUDIO + LYRICS)")
    print("#"*80)
    print("\nMultimodal training uses 677 matched audio-lyrics pairs.")
    
    response = input("Train multimodal models? (y/n, default=y): ").strip().lower()
    if response in ['', 'y', 'yes']:
        models_phase4 = [
            ('basic_vae', 'multimodal', None),
            ('conv_vae', 'multimodal', None),
            ('beta_vae', 'multimodal', None),
        ]
        
        for model, modality, _ in models_phase4:
            run_training(model, modality)
    
    # Summary
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print("\nNext Steps:")
    print("  1. Run clustering: python experiments/run_clustering.py")
    print("  2. Evaluate baselines: python experiments/baseline.py")
    print("  3. Generate visualizations in results/ directory")
    print("\nModel checkpoints saved in: checkpoints/")
    print("="*80)

if __name__ == "__main__":
    main()
