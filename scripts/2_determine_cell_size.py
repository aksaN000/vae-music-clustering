"""
Step 2: Determine Sample Size Per Cell
Builds 5×5 matrix and finds minimum cell size
"""

import json
import pandas as pd
from collections import defaultdict
from pathlib import Path

def main():
    print("="*80)
    print("STEP 2: Determine Sample Size Per Cell (5×5 Matrix)")
    print("="*80)
    
    # Load metadata
    metadata_path = Path('data/lyrics_metadata.csv')
    df = pd.read_csv(metadata_path)
    
    # Load selected genres
    config_path = Path('balancedData/selected_genres.json')
    if not config_path.exists():
        print("❌ Error: Run step 1 first (1_select_balanced_genres.py)")
        return
    
    with open(config_path, 'r') as f:
        selected_genres = json.load(f)
    
    languages = ['arabic', 'bangla', 'english', 'hindi', 'spanish']
    
    print(f"\nLanguages: {languages}")
    print(f"Genres: {selected_genres}")
    
    # Build 5×5 matrix
    cell_samples = defaultdict(lambda: defaultdict(list))
    
    for idx, row in df.iterrows():
        language = row['language']
        genre = row['genre']
        song_id = row['song_id']
        
        if language in languages and genre in selected_genres:
            cell_samples[language][genre].append(song_id)
    
    # Display matrix
    print("\n" + "="*80)
    print("5×5 MATRIX: Available Samples Per Cell")
    print("="*80)
    
    # Header
    print(f"\n{'Language':<12}", end="")
    for genre in selected_genres:
        print(f"{genre[:10]:>12s}", end="")
    print()
    print("-" * (12 + 12 * len(selected_genres)))
    
    # Data
    min_cell_size = float('inf')
    matrix = {}
    
    for language in languages:
        print(f"{language:<12}", end="")
        matrix[language] = {}
        for genre in selected_genres:
            count = len(cell_samples[language].get(genre, []))
            matrix[language][genre] = count
            print(f"{count:>12d}", end="")
            if count > 0:  # Only count non-zero cells
                min_cell_size = min(min_cell_size, count)
        print()
    
    print("-" * (12 + 12 * len(selected_genres)))
    
    # Column totals
    print(f"{'Total':<12}", end="")
    for genre in selected_genres:
        col_total = sum(matrix[lang].get(genre, 0) for lang in languages)
        print(f"{col_total:>12d}", end="")
    print()
    
    print(f"\n{'='*80}")
    print(f"Minimum non-zero cell size: {min_cell_size} songs")
    print(f"{'='*80}")
    
    # Recommend sample size
    if min_cell_size >= 20:
        recommended = 20
        print(f"\n✅ EXCELLENT: Can use n=20 per cell")
        print(f"   Total dataset: {20 * 25} songs (100% balanced)")
    elif min_cell_size >= 15:
        recommended = 15
        print(f"\n✅ GOOD: Can use n=15 per cell")
        print(f"   Total dataset: {15 * 25} songs (100% balanced)")
    elif min_cell_size >= 10:
        recommended = 10
        print(f"\n⚠️  OK: Can use n=10 per cell")
        print(f"   Total dataset: {10 * 25} songs (100% balanced)")
    elif min_cell_size >= 5:
        recommended = 5
        print(f"\n⚠️  LIMITED: Can use n=5 per cell")
        print(f"   Total dataset: {5 * 25} songs (100% balanced)")
    else:
        recommended = min_cell_size
        print(f"\n❌ CRITICAL: Only n={min_cell_size} per cell available")
        print(f"   Total dataset: {min_cell_size * 25} songs")
    
    # Check for zero cells
    zero_cells = []
    for language in languages:
        for genre in selected_genres:
            if matrix[language][genre] == 0:
                zero_cells.append(f"{language}/{genre}")
    
    if zero_cells:
        print(f"\n⚠️  WARNING: {len(zero_cells)} cells have ZERO samples:")
        for cell in zero_cells:
            print(f"   - {cell}")
        print("\n   Consider selecting different genres or excluding these combinations")
    
    # Save configuration
    config = {
        'languages': languages,
        'genres': selected_genres,
        'samples_per_cell': recommended,
        'total_samples': recommended * 25,
        'matrix': matrix,
        'zero_cells': zero_cells
    }
    
    output_path = Path('balancedData/balanced_config.json')
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✅ Configuration saved to: {output_path}")
    print(f"\n📊 RECOMMENDED: n = {recommended} samples per cell")
    print(f"   Total balanced dataset: {recommended * 25} songs")
    print(f"   Language balance: 20% each (5 languages)")
    print(f"   Genre balance: 20% each (5 genres)")

if __name__ == '__main__':
    main()
