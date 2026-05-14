"""
MODULE 3 — Sentence Ranker  (v2)
=================================
Ranks sentences by informational density using TF-IDF centrality + heuristics.

Improvements over v1:
  - Penalises sentences that are mostly stopwords (low information density)
  - Boosts sentences containing named-entity cues (proper nouns, numbers, dates)
  - Boosts sentences containing definition patterns ("is", "refers to", "is defined as")
  - Filters out sentences that are headings / single phrases masquerading as sentences
"""

from __future__ import annotations

import re
import logging
from typing import List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

_POSITION_DECAY   = 0.03
_IDEAL_WORD_COUNT = (8, 40)
_LENGTH_PENALTY   = 0.15

_STOPWORDS = {
    'a','an','the','and','or','but','in','on','at','to','for','of','with','by',
    'from','is','are','was','were','be','been','being','have','has','had',
    'do','does','did','will','would','could','should','may','might','must','can',
    'it','its','this','that','these','those','they','we','he','she','you','i',
    'also','just','very','so','all','any','some','more','most','no','not',
    'such','each','both','when','where','who','which','what','how','why',
    'then','than','now','up','out','if','into','as',
}

_DEFINITION_CUES = re.compile(
    r'\b(is defined as|is known as|refers to|is called|stands for|'
    r'can be defined|is described as|is the process|is a field|is a type|'
    r'is an example)\b', re.I
)
_ENTITY_CUES = re.compile(
    r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b|'  # Multi-word proper nouns
    r'\b\d{4}\b|'                              # Years
    r'\b\d+(?:\.\d+)?\s*%|'                   # Percentages
    r'\b(?:such as|including|for example)\b',  # List introducers
    re.I
)


def _info_density(sentence: str) -> float:
    """Fraction of non-stopword tokens — higher = more informative."""
    words = sentence.lower().split()
    if not words:
        return 0.0
    content = sum(1 for w in words if w not in _STOPWORDS and w.isalpha())
    return content / len(words)


def _definition_bonus(sentence: str) -> float:
    return 0.15 if _DEFINITION_CUES.search(sentence) else 0.0


def _entity_bonus(sentence: str) -> float:
    matches = len(_ENTITY_CUES.findall(sentence))
    return min(0.20, matches * 0.05)


def _position_bonus(idx: int, total: int) -> float:
    return max(0.0, 1.0 - idx * _POSITION_DECAY)


def _length_score(sentence: str) -> float:
    n = len(sentence.split())
    lo, hi = _IDEAL_WORD_COUNT
    if lo <= n <= hi:
        return 1.0
    if n < lo:
        return 0.5 + 0.5 * (n / lo)
    return max(0.4, 1.0 - (n - hi) * 0.02)


def rank_sentences(
    sentences: List[str],
    top_k: int | None = None,
    score_threshold: float = 0.0,
) -> List[Tuple[str, float]]:
    if not sentences:
        return []
    if len(sentences) == 1:
        return [(sentences[0], 1.0)]

    n = len(sentences)
    try:
        vec = TfidfVectorizer(ngram_range=(1, 2), stop_words='english',
                              min_df=1, sublinear_tf=True)
        tfidf = vec.fit_transform(sentences)
    except Exception as e:
        logger.warning(f"TF-IDF failed: {e}")
        return [(s, 0.5) for s in sentences][:top_k or n]

    sim = cosine_similarity(tfidf)
    centrality = []
    for i in range(n):
        row = np.concatenate([sim[i, :i], sim[i, i+1:]])
        centrality.append(float(row.mean()) if len(row) else 0.5)

    combined = []
    for i, sent in enumerate(sentences):
        score = (
            0.45 * centrality[i]
            + 0.15 * _position_bonus(i, n)
            + 0.15 * _length_score(sent)
            + 0.10 * _info_density(sent)
            + 0.10 * _entity_bonus(sent)
            + 0.05 * _definition_bonus(sent)
        )
        combined.append(score)

    max_s = max(combined) if combined else 1.0
    if max_s > 0:
        combined = [s / max_s for s in combined]

    ranked_idx = sorted(range(n), key=lambda i: combined[i], reverse=True)
    if top_k:
        ranked_idx = ranked_idx[:top_k]
    selected = sorted(ranked_idx)

    result = [(sentences[i], combined[i]) for i in selected
              if combined[i] >= score_threshold]
    logger.info(f"Ranked {n} sentences → kept {len(result)}")
    return result


def select_sentences(
    sentences: List[str],
    max_sentences: int = 20,
    score_threshold: float = 0.1,
) -> List[str]:
    ranked = rank_sentences(sentences, top_k=max_sentences,
                            score_threshold=score_threshold)
    return [s for s, _ in ranked]
