"""
Preprocessing module for AQG system.
Handles text extraction, cleaning, tokenization, and sentence ranking.
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
from .sentence_ranker import (
    rank_sentences,
    compute_centroid,
    score_sentences,
    ner_boost,
    remove_similar_sentences,
    get_ranking_stats,
)

__all__ = [
    # Module 1: Text Extraction
    "TextExtractor",
    # Module 2: Text Preprocessing
    "preprocess",
    "clean_text",
    "tokenize_sentences",
    "filter_sentences",
    "is_noisy_sentence",
    "normalize_unicode",
    "get_preprocessing_stats",
    # Module 3: Sentence Ranking
    "rank_sentences",
    "compute_centroid",
    "score_sentences",
    "ner_boost",
    "remove_similar_sentences",
    "get_ranking_stats",
]
