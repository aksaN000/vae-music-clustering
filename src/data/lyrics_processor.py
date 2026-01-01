"""
Lyrics Processing Pipeline with Multilingual Transformers
Extracts original, romanized, and translated lyrics
Uses XLM-RoBERTa for multilingual embeddings
"""

import os
import re
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from transformers import (
    AutoTokenizer, 
    AutoModel,
    MarianMTModel,
    MarianTokenizer
)
from langdetect import detect, LangDetectException
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class LyricsProcessor:
    """Process and embed lyrics using multilingual transformers"""
    
    def __init__(
        self,
        model_name: str = "xlm-roberta-base",
        max_length: int = 512,
        cache_dir: str = "data/features",
        device: str = None
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Loading multilingual model: {model_name}")
        print(f"Using device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        # Language mapping
        self.lang_map = {
            'ar': 'arabic',
            'bn': 'bangla',
            'en': 'english',
            'hi': 'hindi',
            'es': 'spanish'
        }
        
        # Romanization dictionaries (basic transliteration)
        self.arabic_romanization = self._load_arabic_romanization()
        self.bangla_romanization = self._load_bangla_romanization()
        self.hindi_romanization = self._load_hindi_romanization()
        
    def _load_arabic_romanization(self) -> Dict[str, str]:
        """Basic Arabic to Latin transliteration"""
        return {
            'ا': 'a', 'ب': 'b', 'ت': 't', 'ث': 'th', 'ج': 'j', 'ح': 'h',
            'خ': 'kh', 'د': 'd', 'ذ': 'dh', 'ر': 'r', 'ز': 'z', 'س': 's',
            'ش': 'sh', 'ص': 's', 'ض': 'd', 'ط': 't', 'ظ': 'z', 'ع': 'a',
            'غ': 'gh', 'ف': 'f', 'ق': 'q', 'ك': 'k', 'ل': 'l', 'م': 'm',
            'ن': 'n', 'ه': 'h', 'و': 'w', 'ي': 'y'
        }
    
    def _load_bangla_romanization(self) -> Dict[str, str]:
        """Basic Bangla to Latin transliteration"""
        return {
            'অ': 'o', 'আ': 'a', 'ই': 'i', 'ঈ': 'i', 'উ': 'u', 'ঊ': 'u',
            'এ': 'e', 'ঐ': 'oi', 'ও': 'o', 'ঔ': 'ou',
            'ক': 'k', 'খ': 'kh', 'গ': 'g', 'ঘ': 'gh', 'ঙ': 'ng',
            'চ': 'ch', 'ছ': 'chh', 'জ': 'j', 'ঝ': 'jh', 'ঞ': 'n',
            'ট': 't', 'ঠ': 'th', 'ড': 'd', 'ঢ': 'dh', 'ণ': 'n',
            'ত': 't', 'থ': 'th', 'দ': 'd', 'ধ': 'dh', 'ন': 'n',
            'প': 'p', 'ফ': 'ph', 'ব': 'b', 'ভ': 'bh', 'ম': 'm',
            'য': 'j', 'র': 'r', 'ল': 'l', 'শ': 'sh', 'ষ': 'sh',
            'স': 's', 'হ': 'h'
        }
    
    def _load_hindi_romanization(self) -> Dict[str, str]:
        """Basic Hindi/Devanagari to Latin transliteration"""
        return {
            'अ': 'a', 'आ': 'a', 'इ': 'i', 'ई': 'i', 'उ': 'u', 'ऊ': 'u',
            'ए': 'e', 'ऐ': 'ai', 'ओ': 'o', 'औ': 'au',
            'क': 'k', 'ख': 'kh', 'ग': 'g', 'घ': 'gh', 'ङ': 'ng',
            'च': 'ch', 'छ': 'chh', 'ज': 'j', 'झ': 'jh', 'ञ': 'n',
            'ट': 't', 'ठ': 'th', 'ड': 'd', 'ढ': 'dh', 'ण': 'n',
            'त': 't', 'थ': 'th', 'द': 'd', 'ध': 'dh', 'न': 'n',
            'प': 'p', 'फ': 'ph', 'ब': 'b', 'भ': 'bh', 'म': 'm',
            'य': 'y', 'र': 'r', 'ल': 'l', 'व': 'v', 'श': 'sh',
            'ष': 'sh', 'स': 's', 'ह': 'h'
        }
    
    def load_lyrics(self, lyrics_path: str) -> Optional[str]:
        """Load lyrics from file"""
        try:
            with open(lyrics_path, 'r', encoding='utf-8') as f:
                lyrics = f.read().strip()
            
            # Remove structural markers
            lyrics = re.sub(r'\[.*?\]', '', lyrics)
            lyrics = re.sub(r'\(.*?\)', '', lyrics)
            
            # Clean up whitespace
            lyrics = re.sub(r'\n+', '\n', lyrics)
            lyrics = re.sub(r' +', ' ', lyrics)
            
            return lyrics if lyrics else None
        except Exception as e:
            print(f"Error loading {lyrics_path}: {e}")
            return None
    
    def detect_language(self, text: str) -> Optional[str]:
        """Detect language of text"""
        try:
            lang_code = detect(text)
            return self.lang_map.get(lang_code, lang_code)
        except LangDetectException:
            return None
    
    def romanize_text(self, text: str, language: str) -> str:
        """Romanize non-Latin script text"""
        if language == 'arabic':
            romanization_dict = self.arabic_romanization
        elif language == 'bangla':
            romanization_dict = self.bangla_romanization
        elif language == 'hindi':
            romanization_dict = self.hindi_romanization
        else:
            return text  # Already Latin script
        
        # Simple character-by-character replacement
        romanized = []
        for char in text:
            if char in romanization_dict:
                romanized.append(romanization_dict[char])
            else:
                romanized.append(char)
        
        return ''.join(romanized)
    
    def translate_text(self, text: str, source_lang: str, target_lang: str = 'en') -> str:
        """
        Translate text using transformers
        Note: For production, use proper translation APIs (Google Translate, DeepL)
        This is a simplified version
        """
        # For now, return original text with note
        # In production, use MarianMT or translation API
        return f"[Translation needed from {source_lang} to {target_lang}]\n{text}"
    
    @torch.no_grad()
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for text using transformer model"""
        # Tokenize
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        # Get embeddings
        outputs = self.model(**inputs)
        
        # Use [CLS] token embedding (first token)
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return embedding.squeeze()
    
    def process_lyrics(self, lyrics_path: str) -> Optional[Dict]:
        """Process lyrics: load, romanize, translate, and embed"""
        # Check cache
        cache_path = self.cache_dir / f"{Path(lyrics_path).stem}_lyrics_features.pkl"
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        
        # Load original lyrics
        original_lyrics = self.load_lyrics(lyrics_path)
        if original_lyrics is None:
            return None
        
        # Detect language
        language = self.detect_language(original_lyrics)
        
        # Romanize if non-Latin script
        romanized_lyrics = self.romanize_text(original_lyrics, language)
        
        # Translate to English (placeholder)
        translated_lyrics = self.translate_text(original_lyrics, language, 'en')
        
        # Generate embeddings
        original_embedding = self.embed_text(original_lyrics)
        romanized_embedding = self.embed_text(romanized_lyrics)
        translated_embedding = self.embed_text(translated_lyrics)
        
        # Combine features
        features = {
            'lyrics_path': lyrics_path,
            'language': language,
            'original_lyrics': original_lyrics,
            'romanized_lyrics': romanized_lyrics,
            'translated_lyrics': translated_lyrics,
            'original_embedding': original_embedding,
            'romanized_embedding': romanized_embedding,
            'translated_embedding': translated_embedding,
            'combined_embedding': np.mean([
                original_embedding,
                romanized_embedding,
                translated_embedding
            ], axis=0)
        }
        
        # Cache features
        with open(cache_path, 'wb') as f:
            pickle.dump(features, f)
        
        return features
    
    def get_lyrics_metadata(self, lyrics_path: str) -> Dict[str, str]:
        """Extract metadata from lyrics file path"""
        path_parts = Path(lyrics_path).parts
        
        # Find language from path
        language = None
        for part in path_parts:
            if part in ['arabic', 'bangla', 'english', 'hindi', 'spanish']:
                language = part
                break
        
        return {
            'language': language,
            'filename': Path(lyrics_path).stem
        }


def process_all_lyrics(
    lyrics_dir: str,
    model_name: str = "xlm-roberta-base",
    device: str = None
) -> list:
    """Process all lyrics files"""
    
    # Initialize processor
    processor = LyricsProcessor(
        model_name=model_name,
        device=device
    )
    
    # Find all lyrics files
    lyrics_dir = Path(lyrics_dir)
    lyrics_files = list(lyrics_dir.rglob("*.txt"))
    
    print(f"Found {len(lyrics_files)} lyrics files")
    
    # Process all files
    all_features = []
    for lyrics_file in tqdm(lyrics_files, desc="Processing lyrics"):
        features = processor.process_lyrics(str(lyrics_file))
        if features is not None:
            metadata = processor.get_lyrics_metadata(str(lyrics_file))
            features.update(metadata)
            all_features.append(features)
    
    print(f"Successfully processed {len(all_features)} lyrics files")
    
    # Save summary
    summary_path = processor.cache_dir / "lyrics_features_summary.pkl"
    with open(summary_path, 'wb') as f:
        pickle.dump(all_features, f)
    
    return all_features


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process lyrics with multilingual transformers")
    parser.add_argument("--lyrics_dir", type=str, default="data/processed_lyrics",
                       help="Directory containing lyrics files")
    parser.add_argument("--model", type=str, default="xlm-roberta-base",
                       help="Multilingual transformer model")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    features = process_all_lyrics(
        lyrics_dir=args.lyrics_dir,
        model_name=args.model,
        device=args.device
    )
    
    print(f"\nLyrics processing complete!")
    print(f"Total files: {len(features)}")
    
    # Print language distribution
    languages = {}
    for f in features:
        lang = f.get('language', 'unknown')
        languages[lang] = languages.get(lang, 0) + 1
    
    print("\nLanguage distribution:")
    for lang, count in sorted(languages.items()):
        print(f"  {lang}: {count}")
