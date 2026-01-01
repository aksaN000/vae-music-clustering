"""
Audio Feature Extraction Pipeline
Extracts mel-spectrograms and MFCC features from audio files
Optimized for parallel processing with 15 CPU cores
"""

import os
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Dict, Tuple, Optional
import pickle
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')


class AudioProcessor:
    """Extract and cache audio features"""
    
    def __init__(
        self,
        sample_rate: int = 22050,
        n_mels: int = 128,
        n_mfcc: int = 40,
        n_fft: int = 2048,
        hop_length: int = 512,
        duration: int = 30,
        cache_dir: str = "data/features"
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.duration = duration
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def load_audio(self, audio_path: str) -> np.ndarray:
        """Load and normalize audio file"""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.sample_rate, duration=self.duration)
            
            # Normalize
            y = librosa.util.normalize(y)
            
            # Pad or trim to exact duration
            target_length = self.sample_rate * self.duration
            if len(y) < target_length:
                y = np.pad(y, (0, target_length - len(y)), mode='constant')
            else:
                y = y[:target_length]
                
            return y
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return None
    
    def extract_melspectrogram(self, y: np.ndarray) -> np.ndarray:
        """Extract mel-spectrogram"""
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        # Convert to log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db
    
    def extract_mfcc(self, y: np.ndarray) -> np.ndarray:
        """Extract MFCC features"""
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        # Add delta and delta-delta
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        # Concatenate all features
        mfcc_combined = np.vstack([mfcc, mfcc_delta, mfcc_delta2])
        return mfcc_combined
    
    def extract_all_features(self, audio_path: str) -> Optional[Dict[str, np.ndarray]]:
        """Extract all features from audio file"""
        # Check cache
        cache_path = self.cache_dir / f"{Path(audio_path).stem}_features.pkl"
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        
        # Load audio
        y = self.load_audio(audio_path)
        if y is None:
            return None
        
        # Extract features
        features = {
            'mel_spectrogram': self.extract_melspectrogram(y),
            'mfcc': self.extract_mfcc(y),
            'audio_path': audio_path
        }
        
        # Cache features
        with open(cache_path, 'wb') as f:
            pickle.dump(features, f)
        
        return features
    
    def get_audio_metadata(self, audio_path: str) -> Dict[str, str]:
        """Extract metadata from file path"""
        path_parts = Path(audio_path).parts
        
        # Find language and genre from path
        language = None
        genre = None
        
        for i, part in enumerate(path_parts):
            if part in ['arabic', 'bangla', 'english', 'hindi', 'spanish']:
                language = part
                if i + 1 < len(path_parts):
                    genre = path_parts[i + 1]
                break
        
        return {
            'language': language,
            'genre': genre,
            'filename': Path(audio_path).stem
        }


def process_single_audio(args: Tuple[str, AudioProcessor]) -> Optional[Dict]:
    """Process a single audio file (for parallel processing)"""
    audio_path, processor = args
    
    try:
        features = processor.extract_all_features(audio_path)
        if features is None:
            return None
        
        metadata = processor.get_audio_metadata(audio_path)
        features.update(metadata)
        
        return features
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None


def process_all_audio(
    audio_dir: str,
    n_workers: int = 15,
    sample_rate: int = 22050,
    n_mels: int = 128,
    n_mfcc: int = 40
) -> list:
    """Process all audio files in parallel"""
    
    # Initialize processor
    processor = AudioProcessor(
        sample_rate=sample_rate,
        n_mels=n_mels,
        n_mfcc=n_mfcc
    )
    
    # Find all audio files
    audio_dir = Path(audio_dir)
    audio_files = []
    for ext in ['*.mp3', '*.wav', '*.flac', '*.m4a', '*.ogg']:
        audio_files.extend(list(audio_dir.rglob(ext)))
    
    print(f"Found {len(audio_files)} audio files")
    
    # Process in parallel
    all_features = []
    args_list = [(str(f), processor) for f in audio_files]
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_single_audio, args): args[0] 
                  for args in args_list}
        
        for future in tqdm(as_completed(futures), total=len(futures), 
                          desc="Processing audio"):
            result = future.result()
            if result is not None:
                all_features.append(result)
    
    print(f"Successfully processed {len(all_features)} audio files")
    
    # Save summary
    summary_path = processor.cache_dir / "audio_features_summary.pkl"
    with open(summary_path, 'wb') as f:
        pickle.dump(all_features, f)
    
    return all_features


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract audio features")
    parser.add_argument("--audio_dir", type=str, default="data/audio",
                       help="Directory containing audio files")
    parser.add_argument("--n_workers", type=int, default=15,
                       help="Number of parallel workers")
    parser.add_argument("--sample_rate", type=int, default=22050,
                       help="Audio sample rate")
    parser.add_argument("--n_mels", type=int, default=128,
                       help="Number of mel bands")
    parser.add_argument("--n_mfcc", type=int, default=40,
                       help="Number of MFCC coefficients")
    
    args = parser.parse_args()
    
    features = process_all_audio(
        audio_dir=args.audio_dir,
        n_workers=args.n_workers,
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
        n_mfcc=args.n_mfcc
    )
    
    print(f"\nFeature extraction complete!")
    print(f"Total files: {len(features)}")
    
    # Print language distribution
    languages = {}
    for f in features:
        lang = f.get('language', 'unknown')
        languages[lang] = languages.get(lang, 0) + 1
    
    print("\nLanguage distribution:")
    for lang, count in sorted(languages.items()):
        print(f"  {lang}: {count}")
