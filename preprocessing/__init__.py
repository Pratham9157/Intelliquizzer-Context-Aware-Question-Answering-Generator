"""
Preprocessing module for AQG system.
Handles text extraction, cleaning, and preparation.
"""

from .file_extractor import TextExtractor
from .preprocessor import (
    preprocess,
    clean_text,
    tokenize_sentences,
    filter_sentences,
    is_noisy_sentence,
    normalize_unicode,
    get_preprocessing_stats,
)

__all__ = [
    "TextExtractor",
    "preprocess",
    "clean_text",
    "tokenize_sentences",
    "filter_sentences",
    "is_noisy_sentence",
    "normalize_unicode",
    "get_preprocessing_stats",
]
