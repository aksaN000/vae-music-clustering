"""
Step 1: Select 5 Genres Present in ALL Languages
Analyzes the dataset and finds genres that exist across all 5 languages
"""

import json
import pandas as pd
from collections import Counter, defaultdict
from pathlib import Path

def load_metadata():
    """Load metadata from CSV"""
    metadata_path = Path('data/lyrics_metadata.csv')
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    
    df = pd.read_csv(metadata_path)
    print(f"Loaded {len(df)} samples from metadata")
    return df

def analyze_genre_distribution(df):
    """Analyze genre distribution per language"""
    
    genre_by_language = defaultdict(lambda: defaultdict(list))
    
    for idx, row in df.iterrows():
        language = row['language']
        genre = row['genre']
        song_id = row['song_id']
        
        genre_by_language[language][genre].append(song_id)
    
    return genre_by_language

def main():
    print("="*80)
    print("STEP 1: Select 5 Genres Present in ALL Languages")
    print("="*80)
    
    # Load data
    df = load_metadata()
    
    # Define target languages
    all_languages = ['arabic', 'bangla', 'english', 'hindi', 'spanish']
    
    print(f"\nTarget languages: {all_languages}")
    print(f"Total samples in dataset: {len(df)}")
    
    # Analyze distribution
    genre_by_language = analyze_genre_distribution(df)
    
    # Find all unique genres
    candidate_genres = set()
    for language in all_languages:
        if language in genre_by_language:
            candidate_genres.update(genre_by_language[language].keys())
    
    print(f"\nTotal unique genres across all languages: {len(candidate_genres)}")
    
    # For each genre, check how many languages it appears in
    genre_language_count = {}
    for genre in candidate_genres:
        languages_with_genre = sum(
            1 for lang in all_languages 
            if lang in genre_by_language and genre in genre_by_language[lang]
        )
        genre_language_count[genre] = languages_with_genre
    
    # Filter to genres in all 5 languages
    universal_genres = [
        genre for genre, count in genre_language_count.items() 
        if count == 5
    ]
    
    print(f"\nGenres present in ALL 5 languages: {len(universal_genres)}")
    if len(universal_genres) == 0:
        print("\n⚠️  WARNING: No genres found in all 5 languages!")
        print("Finding genres in at least 4 languages...")
        universal_genres = [
            genre for genre, count in genre_language_count.items() 
            if count >= 4
        ]
        print(f"Genres in 4+ languages: {len(universal_genres)}")
    
    # Calculate total sample count for each universal genre
    genre_totals = {}
    for genre in universal_genres:
        total = sum(
            len(genre_by_language.get(lang, {}).get(genre, [])) 
            for lang in all_languages
        )
        genre_totals[genre] = total
    
    # Select top 5 by total sample count
    selected_genres = sorted(
        genre_totals.items(), 
        key=lambda x: x[1], 
        reverse=True
    )[:5]
    
    print("\n" + "="*80)
    print("SELECTED 5 GENRES:")
    print("="*80)
    
    for genre, count in selected_genres:
        print(f"\n{genre.upper()} ({count} total songs)")
        for lang in all_languages:
            lang_count = len(genre_by_language.get(lang, {}).get(genre, []))
            print(f"  {lang:10s}: {lang_count:3d} songs")
    
    selected_genre_names = [g for g, _ in selected_genres]
    print(f"\n{'='*80}")
    print(f"Final selected genres: {selected_genre_names}")
    print(f"{'='*80}")
    
    # Save for next step
    output_dir = Path('balancedData')
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / 'selected_genres.json', 'w') as f:
        json.dump(selected_genre_names, f, indent=2)
    
    print(f"\n✅ Saved to: {output_dir / 'selected_genres.json'}")

if __name__ == '__main__':
    main()
