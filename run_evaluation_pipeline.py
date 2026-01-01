"""
Master Pipeline: Complete Evaluation, Visualization & Comparison
Executes all analysis steps in sequence
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime


def run_script(script_name, description):
    """Run a Python script and handle errors"""
    
    print(f"\n{'='*80}")
    print(f"{description}")
    print('='*80)
    print(f"Running: {script_name}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Print output
        if result.stdout:
            print(result.stdout)
        
        print(f"\n✓ {description} completed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error in {script_name}:")
        print(e.stderr)
        return False


def check_prerequisites():
    """Check if models exist"""
    
    checkpoints_dir = Path("results/checkpoints")
    
    if not checkpoints_dir.exists():
        print(f"✗ Checkpoints directory not found: {checkpoints_dir}")
        return False
    
    # Check for model files (actual directory names and .pt extension)
    models = {
        "basic": "basic",
        "conv": "conv",
        "beta": "beta",
        "cvae_language": "cvae_language",
        "cvae_genre": "cvae_genre",
        "vade": "vade",
        "conv_multimodal": "multimodal"
    }
    
    missing = []
    for dir_name, display_name in models.items():
        model_path = checkpoints_dir / dir_name / "best_model.pt"
        if not model_path.exists():
            missing.append(display_name)
    
    if missing:
        print(f"✗ Missing model checkpoints: {', '.join(missing)}")
        return False
    
    print(f"✓ All {len(models)} model checkpoints found")
    return True


def main():
    print("\n" + "="*80)
    print("EVALUATION PIPELINE - Q1 STANDARD")
    print("="*80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nSteps:")
    print("  1. Evaluate all models (clustering metrics)")
    print("  2. Generate visualizations (publication-quality)")
    print("  3. Statistical comparisons & reports")
    print("="*80)
    
    # Check prerequisites
    print("\nChecking prerequisites...")
    if not check_prerequisites():
        print("\n✗ Prerequisites not met. Please ensure all models are trained.")
        return
    
    # Step 1: Evaluate
    success = run_script(
        "evaluate_all_models.py",
        "STEP 1: Comprehensive Model Evaluation"
    )
    
    if not success:
        print("\n✗ Evaluation failed. Stopping pipeline.")
        return
    
    # Step 2: Visualize
    success = run_script(
        "generate_visualizations.py",
        "STEP 2: Generate Visualizations"
    )
    
    if not success:
        print("\n✗ Visualization failed. Stopping pipeline.")
        return
    
    # Step 3: Compare
    success = run_script(
        "compare_all_models.py",
        "STEP 3: Statistical Comparisons & Reports"
    )
    
    if not success:
        print("\n✗ Comparison failed. Stopping pipeline.")
        return
    
    # Summary
    print(f"\n{'='*80}")
    print("PIPELINE COMPLETE")
    print(f"{'='*80}")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nGenerated Outputs:")
    print("  📊 Evaluation metrics: results/evaluations/")
    print("  📈 Visualizations: results/visualizations/")
    print("  📝 Comparisons: results/comparisons/")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
