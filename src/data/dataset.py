"""
PyTorch Datasets for Music Clustering
Supports audio-only and multi-modal (audio+lyrics) data
"""

import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, Optional, Tuple
import torchvision.transforms as transforms


class MusicDataset(Dataset):
    """Base dataset for music data"""
    
    # Define language and genre mappings (class variables)
    LANGUAGE_MAP = {
        'arabic': 0, 'bangla': 1, 'english': 2,
        'hindi': 3, 'spanish': 4
    }
    
    GENRE_MAP = {
        'adhunik': 0, 'arabic_classic': 1, 'arabic_dabke': 2,
        'arabic_electronic': 3, 'arabic_folk': 4, 'arabic_indie': 5,
        'arabic_pop': 6, 'arabic_rap': 7, 'arabic_rock': 8,
        'bachata': 9, 'banda': 10, 'bengali_rock': 11,
        'bolero': 12, 'bollywood_classic': 13, 'bollywood_modern': 14,
        'bollywood_patriotic': 15, 'bollywood_pop': 16, 'corrido': 17,
        'country': 18, 'electronic': 19, 'flamenco': 20,
        'flamenco_pop': 21, 'folk_baul': 22, 'hiphop': 23,
        'indipop': 24, 'jazz': 25, 'khaleeji': 26,
        'latin_folk': 27, 'latin_indie': 28, 'latin_pop': 29,
        'latin_rock': 30, 'latin_trap': 31, 'mahraganat': 32,
        'mariachi': 33, 'nazrul_geeti': 34, 'norteno': 35,
        'pop': 36, 'rabindra_sangeet': 37, 'rai': 38,
        'reggaeton': 39, 'rnb': 40, 'rock': 41,
        'salsa': 42, 'tejano': 43, 'vallenato': 44
    }
    
    def __init__(
        self,
        data_path: str,
        modality: str = 'audio',  # 'audio', 'lyrics', or 'multimodal'
        feature_type: str = 'melspec',  # 'melspec', 'mfcc', or 'both'
        transform: Optional[callable] = None
    ):
        self.data_path = Path(data_path)
        self.modality = modality
        self.feature_type = feature_type
        self.transform = transform
        
        # Load data
        with open(self.data_path, 'rb') as f:
            self.data = pickle.load(f)
        
        print(f"Loaded {len(self.data)} samples")
        
    def __len__(self) -> int:
        return len(self.data)
    
    def _get_audio_features(self, sample: Dict) -> torch.Tensor:
        """Extract audio features based on feature_type"""
        # For multimodal dataset, get from audio_data
        if 'audio_data' in sample:
            audio_data = sample['audio_data']
        else:
            audio_data = sample
        
        if self.feature_type == 'melspec':
            features = audio_data['mel_spectrogram']
        elif self.feature_type == 'mfcc':
            features = audio_data['mfcc']
        elif self.feature_type == 'both':
            melspec = audio_data['mel_spectrogram']
            mfcc = audio_data['mfcc']
            features = np.concatenate([melspec, mfcc], axis=0)
        else:
            raise ValueError(f"Unknown feature_type: {self.feature_type}")
        
        # Convert to tensor
        features = torch.FloatTensor(features)
        
        # Normalize features (mean=0, std=1)
        features = (features - features.mean()) / (features.std() + 1e-8)
        
        # Apply transforms if any
        if self.transform:
            features = self.transform(features)
        
        return features
    
    def _get_lyrics_features(self, sample: Dict) -> torch.Tensor:
        """Extract lyrics embeddings"""
        # For multimodal dataset, get from lyrics_data
        if 'lyrics_data' in sample:
            lyrics_data = sample['lyrics_data']
        else:
            lyrics_data = sample
        
        # Use combined embedding (average of original, romanized, translated)
        embedding = lyrics_data['combined_embedding']
        
        return torch.FloatTensor(embedding)
    
    def _get_labels(self, sample: Dict) -> Dict[str, int]:
        """Get labels for evaluation"""
        labels = {}
        
        if 'language' in sample:
            # Map language to integer (0-4)
            labels['language'] = self.LANGUAGE_MAP.get(sample['language'], -1)
        
        if 'genre' in sample:
            # Map genre to integer (0-44)
            genre_str = sample['genre']
            labels['genre'] = self.GENRE_MAP.get(genre_str, -1)
            labels['genre_name'] = genre_str  # Keep string name for logging
        
        if 'id' in sample:
            labels['id'] = sample['id']
        
        return labels
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample"""
        sample = self.data[idx]
        
        output = {}
        
        # Get features based on modality
        if self.modality == 'audio':
            output['features'] = self._get_audio_features(sample)
        
        elif self.modality == 'lyrics':
            output['features'] = self._get_lyrics_features(sample)
        
        elif self.modality == 'multimodal':
            output['audio_features'] = self._get_audio_features(sample)
            output['lyrics_features'] = self._get_lyrics_features(sample)
        
        # Get labels
        labels = self._get_labels(sample)
        output.update(labels)
        
        return output


class AudioOnlyDataset(MusicDataset):
    """Dataset for audio-only experiments (all 1120 audio files)"""
    
    def __init__(
        self,
        data_path: str = "data/features/audio_only_dataset.pkl",
        feature_type: str = 'melspec',
        transform: Optional[callable] = None
    ):
        super().__init__(
            data_path=data_path,
            modality='audio',
            feature_type=feature_type,
            transform=transform
        )


class MultimodalDataset(MusicDataset):
    """Dataset for multi-modal experiments (677 paired audio+lyrics)"""
    
    def __init__(
        self,
        data_path: str = "data/features/multimodal_dataset.pkl",
        feature_type: str = 'melspec',
        transform: Optional[callable] = None
    ):
        super().__init__(
            data_path=data_path,
            modality='multimodal',
            feature_type=feature_type,
            transform=transform
        )


def get_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 15,
    pin_memory: bool = True
) -> DataLoader:
    """Create DataLoader with optimizations for 15 cores"""
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )


class SpecAugment:
    """SpecAugment for audio spectrograms"""
    
    def __init__(
        self,
        freq_mask_param: int = 30,
        time_mask_param: int = 40,
        num_masks: int = 2
    ):
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_masks = num_masks
    
    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        """Apply SpecAugment"""
        # spec shape: (freq, time) or (channels, freq, time)
        
        if spec.dim() == 2:
            spec = spec.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        
        _, freq, time = spec.shape
        
        # Frequency masking
        for _ in range(self.num_masks):
            f = np.random.randint(0, self.freq_mask_param)
            f0 = np.random.randint(0, max(1, freq - f))
            spec[:, f0:f0+f, :] = 0
        
        # Time masking
        for _ in range(self.num_masks):
            t = np.random.randint(0, self.time_mask_param)
            t0 = np.random.randint(0, max(1, time - t))
            spec[:, :, t0:t0+t] = 0
        
        if squeeze:
            spec = spec.squeeze(0)
        
        return spec


def get_transforms(augment: bool = True):
    """Get data transforms"""
    if augment:
        return SpecAugment(
            freq_mask_param=30,
            time_mask_param=40,
            num_masks=2
        )
    else:
        return None


if __name__ == "__main__":
    # Test dataset loading
    print("Testing Audio-Only Dataset:")
    audio_dataset = AudioOnlyDataset(
        data_path="data/features/audio_only_dataset.pkl",
        feature_type='melspec'
    )
    print(f"Dataset size: {len(audio_dataset)}")
    
    # Get a sample
    sample = audio_dataset[0]
    print(f"Sample keys: {sample.keys()}")
    if 'features' in sample:
        print(f"Features shape: {sample['features'].shape}")
    
    print("\n" + "="*60)
    print("Testing Multimodal Dataset:")
    multimodal_dataset = MultimodalDataset(
        data_path="data/features/multimodal_dataset.pkl",
        feature_type='melspec'
    )
    print(f"Dataset size: {len(multimodal_dataset)}")
    
    # Get a sample
    sample = multimodal_dataset[0]
    print(f"Sample keys: {sample.keys()}")
    if 'audio_features' in sample:
        print(f"Audio features shape: {sample['audio_features'].shape}")
    if 'lyrics_features' in sample:
        print(f"Lyrics features shape: {sample['lyrics_features'].shape}")
    
    print("\n" + "="*60)
    print("Testing DataLoader:")
    dataloader = get_dataloader(
        audio_dataset,
        batch_size=8,
        num_workers=4
    )
    
    batch = next(iter(dataloader))
    print(f"Batch keys: {batch.keys()}")
    if 'features' in batch:
        print(f"Batch features shape: {batch['features'].shape}")
