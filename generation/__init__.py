"""
Generation module for AQG system.
Handles answer extraction and question generation.
"""

from .answer_extractor import (
    extract_named_entities,
    extract_noun_phrases,
    filter_answers,
    classify_answer_type,
    score_answer,
    extract_answers,
    get_answer_stats,
)

__all__ = [
    # Module 4: Answer Extraction
    "extract_named_entities",
    "extract_noun_phrases",
    "filter_answers",
    "classify_answer_type",
    "score_answer",
    "extract_answers",
    "get_answer_stats",
]
