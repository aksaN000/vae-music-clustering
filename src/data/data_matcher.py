"""
Data Matcher: Match audio files with their corresponding lyrics
Creates audio-lyrics pairs for multi-modal training
"""

import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from collections import defaultdict


class DataMatcher:
    """Match audio files with lyrics files"""
    
    def __init__(
        self,
        audio_features_path: str = "data/features/audio_features_summary.pkl",
        lyrics_features_path: str = "data/features/lyrics_features_summary.pkl",
        output_dir: str = "data/features"
    ):
        self.audio_features_path = Path(audio_features_path)
        self.lyrics_features_path = Path(lyrics_features_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_features(self) -> Tuple[List[Dict], List[Dict]]:
        """Load audio and lyrics features"""
        print("Loading audio features...")
        with open(self.audio_features_path, 'rb') as f:
            audio_features = pickle.load(f)
        
        print("Loading lyrics features...")
        with open(self.lyrics_features_path, 'rb') as f:
            lyrics_features = pickle.load(f)
        
        return audio_features, lyrics_features
    
    def extract_id_from_path(self, path: str) -> Optional[str]:
        """
        Extract ID from file path
        Handles multiple naming conventions:
        - lang_XXXX.txt/mp3 (e.g., ar_0001.txt, en_0001.mp3)
        - Full path-based matching using filename stem
        """
        filename = Path(path).stem
        
        # Try to extract language code and number
        parts = filename.split('_')
        if len(parts) >= 2:
            lang_code = parts[0]
            id_num = parts[1]
            return f"{lang_code}_{id_num}"
        
        # Fallback: use full filename as ID
        return filename
    
    def match_audio_lyrics(
        self,
        audio_features: List[Dict],
        lyrics_features: List[Dict]
    ) -> Tuple[List[Dict], List[Dict]]:
        """Match audio with lyrics based on filename/ID"""
        
        # Create lookup dictionaries
        audio_dict = {}
        for audio in audio_features:
            audio_path = audio.get('audio_path', '')
            audio_id = self.extract_id_from_path(audio_path)
            audio_dict[audio_id] = audio
        
        lyrics_dict = {}
        for lyrics in lyrics_features:
            lyrics_path = lyrics.get('lyrics_path', '')
            lyrics_id = self.extract_id_from_path(lyrics_path)
            lyrics_dict[lyrics_id] = lyrics
        
        # Find matches
        matched_pairs = []
        unmatched_audio = []
        
        print(f"\nMatching {len(audio_dict)} audio files with {len(lyrics_dict)} lyrics files...")
        
        for audio_id, audio_data in audio_dict.items():
            if audio_id in lyrics_dict:
                # Found match - ensure both have same language and genre
                lyrics_data = lyrics_dict[audio_id]
                
                # Verify language match for data consistency
                audio_lang = audio_data.get('language')
                lyrics_lang = lyrics_data.get('language')
                
                matched_pair = {
                    'id': audio_id,
                    'audio_data': audio_data,
                    'lyrics_data': lyrics_data,
                    'language': audio_lang,  # Use audio language as primary
                    'genre': audio_data.get('genre'),
                    'filename': audio_data.get('filename')
                }
                
                # Verify consistency
                if lyrics_lang and audio_lang and lyrics_lang != audio_lang:
                    print(f"Warning: Language mismatch for {audio_id}: audio={audio_lang}, lyrics={lyrics_lang}")
                
                matched_pairs.append(matched_pair)
            else:
                # No lyrics found
                unmatched_audio.append(audio_data)
        
        print(f"\nMatching results:")
        print(f"  Matched pairs: {len(matched_pairs)}")
        print(f"  Audio without lyrics: {len(unmatched_audio)}")
        print(f"  Lyrics without audio: {len(lyrics_dict) - len(matched_pairs)}")
        
        return matched_pairs, unmatched_audio
    
    def create_datasets(
        self,
        matched_pairs: List[Dict],
        unmatched_audio: List[Dict]
    ):
        """Create and save different datasets"""
        
        # Dataset 1: Matched audio-lyrics pairs (for multi-modal tasks)
        multimodal_dataset = matched_pairs
        
        # Dataset 2: All audio (for audio-only tasks)
        audio_only_dataset = unmatched_audio + [p['audio_data'] for p in matched_pairs]
        
        # Save datasets
        multimodal_path = self.output_dir / "multimodal_dataset.pkl"
        with open(multimodal_path, 'wb') as f:
            pickle.dump(multimodal_dataset, f)
        print(f"\nSaved multimodal dataset: {len(multimodal_dataset)} samples")
        
        audio_only_path = self.output_dir / "audio_only_dataset.pkl"
        with open(audio_only_path, 'wb') as f:
            pickle.dump(audio_only_dataset, f)
        print(f"Saved audio-only dataset: {len(audio_only_dataset)} samples")
        
        # Generate statistics
        self.print_statistics(matched_pairs, unmatched_audio)
        
        return multimodal_dataset, audio_only_dataset
    
    def print_statistics(
        self,
        matched_pairs: List[Dict],
        unmatched_audio: List[Dict]
    ):
        """Print dataset statistics"""
        
        print("\n" + "="*60)
        print("DATASET STATISTICS")
        print("="*60)
        
        # Language distribution for matched pairs
        print("\nMulti-modal (Audio + Lyrics) Dataset:")
        lang_counts = defaultdict(int)
        genre_counts = defaultdict(int)
        
        for pair in matched_pairs:
            lang = pair.get('language', 'unknown')
            genre = pair.get('genre', 'unknown')
            lang_counts[lang] += 1
            genre_counts[genre] += 1
        
        print(f"  Total: {len(matched_pairs)} samples")
        print("\n  By Language:")
        for lang, count in sorted(lang_counts.items()):
            print(f"    {lang}: {count}")
        
        print("\n  By Genre (showing all):")
        for genre, count in sorted(genre_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"    {genre}: {count}")
        
        print(f"\n  Total unique genres: {len(genre_counts)}")
        print(f"  Total unique languages: {len(lang_counts)}")
        
        # Audio-only distribution
        print("\n\nAudio-Only Dataset:")
        all_audio = unmatched_audio + [p['audio_data'] for p in matched_pairs]
        lang_counts_audio = defaultdict(int)
        
        for audio in all_audio:
            lang = audio.get('language', 'unknown')
            lang_counts_audio[lang] += 1
        
        print(f"  Total: {len(all_audio)} samples")
        print("\n  By Language:")
        for lang, count in sorted(lang_counts_audio.items()):
            print(f"    {lang}: {count}")
        
        print("\n" + "="*60)


def main():
    """Main matching pipeline"""
    
    # Initialize matcher
    matcher = DataMatcher()
    
    # Load features
    audio_features, lyrics_features = matcher.load_features()
    
    # Match audio with lyrics
    matched_pairs, unmatched_audio = matcher.match_audio_lyrics(
        audio_features,
        lyrics_features
    )
    
    # Create and save datasets
    multimodal_dataset, audio_only_dataset = matcher.create_datasets(
        matched_pairs,
        unmatched_audio
    )
    
    print("\nData matching complete!")
    print(f"Datasets saved in: {matcher.output_dir}")


if __name__ == "__main__":
    main()
