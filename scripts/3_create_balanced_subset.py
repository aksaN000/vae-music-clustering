"""
Step 3: Create Balanced Subset
Samples exactly n from each cell to create perfectly balanced dataset
"""

import json
import numpy as np
import shutil
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict

def main():
    print("="*80)
    print("STEP 3: Creating Balanced Dataset")
    print("="*80)
    
    # Load configuration
    config_path = Path('balancedData/balanced_config.json')
    if not config_path.exists():
        print("❌ Error: Run step 2 first (2_determine_cell_size.py)")
        return
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load metadata
    metadata_path = Path('data/lyrics_metadata.csv')
    df = pd.read_csv(metadata_path)
    
    languages = config['languages']
    genres = config['genres']
    n_per_cell = config['samples_per_cell']
    
    print(f"\nConfiguration:")
    print(f"  Languages: {languages}")
    print(f"  Genres: {genres}")
    print(f"  Samples per cell: {n_per_cell}")
    print(f"  Target total: {n_per_cell * 25} songs")
    
    # Collect samples for each cell
    cell_samples = defaultdict(lambda: defaultdict(list))
    metadata_dict = {}
    
    for idx, row in df.iterrows():
        language = row['language']
        genre = row['genre']
        song_id = row['song_id']
        
        if language in languages and genre in genres:
            cell_samples[language][genre].append(song_id)
            metadata_dict[song_id] = row.to_dict()
    
    # Sample exactly n from each cell
    np.random.seed(42)  # Reproducibility
    
    balanced_dataset = []
    balanced_metadata = {}
    skipped_cells = []
    
    print("\n" + "="*80)
    print("Sampling from each cell...")
    print("="*80 + "\n")
    
    for language in languages:
        for genre in genres:
            available = cell_samples[language].get(genre, [])
            
            if len(available) == 0:
                print(f"⚠️  SKIP: {language:8s} × {genre:12s} - No samples available")
                skipped_cells.append(f"{language}/{genre}")
                continue
            elif len(available) < n_per_cell:
                print(f"⚠️  PARTIAL: {language:8s} × {genre:12s} - Only {len(available)}/{n_per_cell} available")
                sampled = available
            else:
                sampled = np.random.choice(available, size=n_per_cell, replace=False).tolist()
                print(f"✅ {language:8s} × {genre:12s} - Sampled {len(sampled)}/{n_per_cell}")
            
            for song_id in sampled:
                balanced_dataset.append(song_id)
                balanced_metadata[song_id] = metadata_dict[song_id]
    
    print(f"\n{'='*80}")
    print(f"Total samples collected: {len(balanced_dataset)}")
    print(f"{'='*80}")
    
    # Verify balance
    print("\n" + "="*80)
    print("VERIFICATION: Dataset Balance")
    print("="*80)
    
    languages_count = Counter([balanced_metadata[tid]['language'] for tid in balanced_dataset])
    genres_count = Counter([balanced_metadata[tid]['genre'] for tid in balanced_dataset])
    
    print("\n📊 Language distribution:")
    for lang in languages:
        count = languages_count.get(lang, 0)
        pct = count/len(balanced_dataset)*100 if len(balanced_dataset) > 0 else 0
        print(f"  {lang:10s}: {count:3d} ({pct:5.1f}%)")
    
    print("\n📊 Genre distribution:")
    for genre in genres:
        count = genres_count.get(genre, 0)
        pct = count/len(balanced_dataset)*100 if len(balanced_dataset) > 0 else 0
        print(f"  {genre:12s}: {count:3d} ({pct:5.1f}%)")
    
    print("\n📋 Cell-by-cell breakdown:")
    print(f"{'Language':<12}", end="")
    for genre in genres:
        print(f"{genre[:10]:>12s}", end="")
    print(" │ Total")
    print("─" * (12 + 12 * len(genres) + 8))
    
    cell_matrix = defaultdict(dict)
    for language in languages:
        print(f"{language:<12}", end="")
        row_total = 0
        for genre in genres:
            count = sum(
                1 for tid in balanced_dataset
                if balanced_metadata[tid]['language'] == language
                and balanced_metadata[tid]['genre'] == genre
            )
            cell_matrix[language][genre] = count
            print(f"{count:>12d}", end="")
            row_total += count
        print(f" │ {row_total:>5d}")
    
    print("─" * (12 + 12 * len(genres) + 8))
    print(f"{'Total':<12}", end="")
    for genre in genres:
        col_total = sum(
            1 for tid in balanced_dataset
            if balanced_metadata[tid]['genre'] == genre
        )
        print(f"{col_total:>12d}", end="")
    print(f" │ {len(balanced_dataset):>5d}")
    
    # Save balanced dataset
    output_dir = Path('balancedData/dataset')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save track IDs
    with open(output_dir / 'track_ids.json', 'w') as f:
        json.dump(balanced_dataset, f, indent=2)
    
    # Save metadata as JSON
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(balanced_metadata, f, indent=2)
    
    # Save metadata as CSV
    df_balanced = pd.DataFrame([balanced_metadata[tid] for tid in balanced_dataset])
    df_balanced.to_csv(output_dir / 'metadata.csv', index=False)
    
    # Copy audio files
    print("\n" + "="*80)
    print("Copying audio files...")
    print("="*80)
    
    audio_output_dir = output_dir / 'audio'
    audio_output_dir.mkdir(exist_ok=True)
    
    copied_count = 0
    missing_count = 0
    
    for song_id in balanced_dataset:
        language = balanced_metadata[song_id]['language']
        genre = balanced_metadata[song_id]['genre']
        
        # Check multiple possible audio locations
        source_paths = [
            Path('data/audio') / language / genre / f"{song_id}.mp3",
            Path('data/audio') / language / genre / f"{song_id}.wav",
            Path('data/audio') / f"{song_id}.mp3",
            Path('data/audio') / f"{song_id}.wav",
        ]
        
        copied = False
        for source in source_paths:
            if source.exists():
                dest = audio_output_dir / f"{song_id}{source.suffix}"
                shutil.copy2(source, dest)
                copied_count += 1
                copied = True
                break
        
        if not copied:
            missing_count += 1
            if missing_count <= 5:  # Only show first 5 missing files
                print(f"⚠️  Missing audio: {song_id}")
    
    if missing_count > 5:
        print(f"⚠️  ... and {missing_count - 5} more missing files")
    
    # Copy lyrics files
    print("\nCopying lyrics files...")
    
    lyrics_output_dir = output_dir / 'lyrics'
    lyrics_output_dir.mkdir(exist_ok=True)
    
    lyrics_copied = 0
    lyrics_missing = 0
    
    for song_id in balanced_dataset:
        language = balanced_metadata[song_id]['language']
        
        source_paths = [
            Path('data/processed_lyrics') / language / f"{song_id}.txt",
            Path('data/lyrics') / language / f"{song_id}.txt",
        ]
        
        copied = False
        for source in source_paths:
            if source.exists():
                dest = lyrics_output_dir / f"{song_id}.txt"
                shutil.copy2(source, dest)
                lyrics_copied += 1
                copied = True
                break
        
        if not copied:
            lyrics_missing += 1
    
    # Summary
    print("\n" + "="*80)
    print("✅ BALANCED DATASET CREATED")
    print("="*80)
    print(f"\n📁 Output directory: {output_dir}")
    print(f"\n📊 Statistics:")
    print(f"  Total songs: {len(balanced_dataset)}")
    print(f"  Languages: {len(languages)} (20% each)")
    print(f"  Genres: {len(genres)} (20% each)")
    print(f"  Samples per cell: {n_per_cell} (target)")
    print(f"\n📂 Files:")
    print(f"  ✅ track_ids.json: {len(balanced_dataset)} IDs")
    print(f"  ✅ metadata.json: {len(balanced_metadata)} entries")
    print(f"  ✅ metadata.csv: {len(df_balanced)} rows")
    print(f"  ✅ audio/: {copied_count} files copied")
    if missing_count > 0:
        print(f"  ⚠️  audio/: {missing_count} files missing")
    print(f"  ✅ lyrics/: {lyrics_copied} files copied")
    if lyrics_missing > 0:
        print(f"  ⚠️  lyrics/: {lyrics_missing} files missing")
    
    if skipped_cells:
        print(f"\n⚠️  Skipped {len(skipped_cells)} empty cells:")
        for cell in skipped_cells[:10]:
            print(f"  - {cell}")
        if len(skipped_cells) > 10:
            print(f"  ... and {len(skipped_cells) - 10} more")
    
    print(f"\n{'='*80}")
    print("✅ Ready for training!")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()
