"""
Verify Dataset Structure and Audio-Lyrics Matching
This script checks:
1. Total number of audio files per language and genre
2. Total number of lyrics files per language
3. Audio-lyrics matching statistics
4. Dataset readiness for 5 languages and 45 genres
"""

import os
from pathlib import Path
from collections import defaultdict
import re

def count_audio_files():
    """Count audio files by language and genre"""
    audio_dir = Path("data/audio")
    stats = defaultdict(lambda: defaultdict(int))
    total = 0
    all_genres = set()
    
    for lang_dir in audio_dir.iterdir():
        if not lang_dir.is_dir():
            continue
        language = lang_dir.name
        
        for genre_dir in lang_dir.iterdir():
            if not genre_dir.is_dir():
                continue
            genre = genre_dir.name
            all_genres.add(genre)
            
            # Count audio files
            audio_files = list(genre_dir.glob("*.mp3")) + list(genre_dir.glob("*.wav"))
            count = len(audio_files)
            stats[language][genre] = count
            total += count
    
    return stats, total, all_genres

def count_lyrics_files():
    """Count lyrics files by language"""
    lyrics_dir = Path("data/processed_lyrics")
    stats = {}
    total = 0
    
    for lang_dir in lyrics_dir.iterdir():
        if not lang_dir.is_dir():
            continue
        language = lang_dir.name
        
        # Count lyrics files
        lyrics_files = list(lang_dir.glob("*.txt"))
        count = len(lyrics_files)
        stats[language] = count
        total += count
    
    return stats, total

def extract_id_from_filename(filename):
    """Extract ID from filename (e.g., ar_0016 from ar_0016_Title.mp3)"""
    stem = Path(filename).stem
    parts = stem.split('_')
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return stem

def check_matching():
    """Check audio-lyrics matching"""
    audio_dir = Path("data/audio")
    lyrics_dir = Path("data/processed_lyrics")
    
    # Collect all audio IDs
    audio_ids = set()
    audio_by_id = {}
    
    for lang_dir in audio_dir.iterdir():
        if not lang_dir.is_dir():
            continue
        language = lang_dir.name
        
        for genre_dir in lang_dir.iterdir():
            if not genre_dir.is_dir():
                continue
            genre = genre_dir.name
            
            for audio_file in genre_dir.glob("*.mp3"):
                audio_id = extract_id_from_filename(audio_file.name)
                audio_ids.add(audio_id)
                audio_by_id[audio_id] = {
                    'language': language,
                    'genre': genre,
                    'file': audio_file.name
                }
    
    # Collect all lyrics IDs
    lyrics_ids = set()
    lyrics_by_id = {}
    
    for lang_dir in lyrics_dir.iterdir():
        if not lang_dir.is_dir():
            continue
        language = lang_dir.name
        
        for lyrics_file in lang_dir.glob("*.txt"):
            lyrics_id = extract_id_from_filename(lyrics_file.name)
            lyrics_ids.add(lyrics_id)
            lyrics_by_id[lyrics_id] = {
                'language': language,
                'file': lyrics_file.name
            }
    
    # Calculate matches
    matched_ids = audio_ids & lyrics_ids
    audio_only = audio_ids - lyrics_ids
    lyrics_only = lyrics_ids - audio_ids
    
    return {
        'matched': len(matched_ids),
        'audio_only': len(audio_only),
        'lyrics_only': len(lyrics_only),
        'total_audio': len(audio_ids),
        'total_lyrics': len(lyrics_ids),
        'matched_ids': matched_ids,
        'audio_only_ids': audio_only,
        'lyrics_only_ids': lyrics_only
    }

def main():
    print("=" * 80)
    print("DATASET VERIFICATION REPORT")
    print("=" * 80)
    
    # Audio files statistics
    print("\n1. AUDIO FILES STATISTICS")
    print("-" * 80)
    audio_stats, audio_total, all_genres = count_audio_files()
    
    print(f"\nTotal audio files: {audio_total}")
    print(f"Number of languages: {len(audio_stats)}")
    print(f"Number of unique genres: {len(all_genres)}")
    
    print("\nBreakdown by language:")
    for language, genre_counts in sorted(audio_stats.items()):
        lang_total = sum(genre_counts.values())
        print(f"\n  {language.upper()}: {lang_total} files")
        for genre, count in sorted(genre_counts.items()):
            print(f"    - {genre}: {count}")
    
    print("\n\nAll genres across languages:")
    for i, genre in enumerate(sorted(all_genres), 1):
        print(f"  {i:2d}. {genre}")
    
    # Lyrics files statistics
    print("\n\n2. LYRICS FILES STATISTICS")
    print("-" * 80)
    lyrics_stats, lyrics_total = count_lyrics_files()
    
    print(f"\nTotal lyrics files: {lyrics_total}")
    print(f"Number of languages: {len(lyrics_stats)}")
    
    print("\nBreakdown by language:")
    for language, count in sorted(lyrics_stats.items()):
        print(f"  {language}: {count}")
    
    # Matching statistics
    print("\n\n3. AUDIO-LYRICS MATCHING")
    print("-" * 80)
    matching = check_matching()
    
    print(f"\nTotal audio files: {matching['total_audio']}")
    print(f"Total lyrics files: {matching['total_lyrics']}")
    print(f"Matched pairs: {matching['matched']} ({matching['matched']/matching['total_audio']*100:.1f}%)")
    print(f"Audio-only (no lyrics): {matching['audio_only']}")
    print(f"Lyrics-only (no audio): {matching['lyrics_only']}")
    
    # Dataset readiness check
    print("\n\n4. DATASET READINESS CHECK")
    print("-" * 80)
    
    checks = []
    checks.append(("5 languages required", len(audio_stats) == 5, f"Found {len(audio_stats)} languages"))
    checks.append(("45 genres required", len(all_genres) == 45, f"Found {len(all_genres)} genres"))
    checks.append(("Audio files present", audio_total > 0, f"Found {audio_total} files"))
    checks.append(("Lyrics files present", lyrics_total > 0, f"Found {lyrics_total} files"))
    checks.append(("Matched pairs exist", matching['matched'] > 0, f"Found {matching['matched']} pairs"))
    
    print("\nReadiness checks:")
    all_passed = True
    for check_name, passed, detail in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {check_name} - {detail}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✓ DATASET IS READY FOR TRAINING!")
        print("  - Audio-only tasks: Use all {} audio files".format(matching['total_audio']))
        print("  - Multimodal tasks: Use {} audio-lyrics pairs".format(matching['matched']))
    else:
        print("✗ DATASET HAS ISSUES - Please review the checks above")
    print("=" * 80)

if __name__ == "__main__":
    main()
