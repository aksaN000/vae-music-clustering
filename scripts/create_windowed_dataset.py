"""
Create Windowed Dataset for Data Augmentation
Applies sliding window to each audio file to increase dataset size
"""

import sys
sys.path.append('.')

import yaml
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
from src.data.audio_processor import AudioProcessor


def create_windowed_dataset(config_path='configs/config.yaml'):
    """Create windowed audio dataset"""
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    data_config = config['data']
    
    # Initialize audio processor with windowing
    processor = AudioProcessor(
        sample_rate=data_config['sample_rate'],
        n_mels=data_config['n_mels'],
        n_mfcc=data_config['n_mfcc'],
        n_fft=data_config['n_fft'],
        hop_length=data_config['hop_length'],
        duration=data_config['audio_duration'],
        cache_dir=data_config['features_dir'],
        use_windowing=data_config.get('use_windowing', False),
        window_size=data_config.get('window_size', 15),
        hop_size=data_config.get('hop_size', 7.5)
    )
    
    print(f"Windowing: {processor.use_windowing}")
    print(f"Window size: {processor.window_size}s, Hop: {processor.hop_size}s")
    
    # Get all audio files
    audio_dir = Path(data_config['audio_dir'])
    audio_files = list(audio_dir.rglob('*.mp3')) + list(audio_dir.rglob('*.wav'))
    
    print(f"\nFound {len(audio_files)} audio files")
    
    dataset = []
    
    for audio_path in tqdm(audio_files, desc="Processing audio files"):
        # Extract language and genre from path
        # Path format: data/audio/{language}/{genre}/{filename}
        parts = audio_path.parts
        language = parts[-3]
        genre = parts[-2]
        file_id = audio_path.stem
        
        # Load audio
        y = processor.load_audio(str(audio_path))
        if y is None:
            continue
        
        # Create windows
        windows = processor.create_windows(y)
        
        # Process each window
        for window_idx, window_audio in enumerate(windows):
            # Extract mel spectrogram
            mel_spec = processor.extract_melspectrogram(window_audio)
            
            # Extract MFCC
            mfcc = processor.extract_mfcc(window_audio)
            
            # Create sample
            sample = {
                'id': f"{file_id}_w{window_idx}",
                'original_id': file_id,
                'window_idx': window_idx,
                'language': language,
                'genre': genre,
                'mel_spectrogram': mel_spec,
                'mfcc': mfcc,
                'audio_path': str(audio_path)
            }
            
            dataset.append(sample)
    
    print(f"\nCreated {len(dataset)} windowed samples from {len(audio_files)} files")
    print(f"Augmentation factor: {len(dataset) / len(audio_files):.1f}x")
    
    # Save dataset
    output_path = Path(data_config['features_dir']) / 'audio_windowed_dataset.pkl'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)
    
    print(f"\nSaved dataset to: {output_path}")
    
    # Print statistics
    languages = {}
    genres = {}
    for sample in dataset:
        languages[sample['language']] = languages.get(sample['language'], 0) + 1
        genres[sample['genre']] = genres.get(sample['genre'], 0) + 1
    
    print("\nDataset Statistics:")
    print(f"Languages: {dict(sorted(languages.items()))}")
    print(f"Genres: {dict(sorted(genres.items()))}")
    
    # Check mel spec shape
    print(f"\nFeature shapes:")
    print(f"  Mel spectrogram: {dataset[0]['mel_spectrogram'].shape}")
    print(f"  MFCC: {dataset[0]['mfcc'].shape}")


if __name__ == '__main__':
    create_windowed_dataset()
