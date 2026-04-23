"""
MODULE 3: Sentence Ranking / Importance Module
==============================================
Ranks sentences based on their importance within a document using
TF-IDF vectors, cosine similarity, and Named Entity Recognition.

Purpose:
    Identify and rank the most informative sentences in a document
    so that only high-importance sentences are used for question generation.

Key Features:
    - TF-IDF vectorization for semantic representation
    - Document centroid calculation (mean of all vectors)
    - Cosine similarity-based importance scoring
    - Named Entity Recognition (NER) boosting
    - Similar sentence deduplication
    - Length normalization to avoid bias
    - Configurable logging and top-K selection

Architecture:
    Input: List of cleaned sentences (from Module 2)
    → TF-IDF vectorization
    → Centroid computation
    → Cosine similarity scoring
    → NER boosting
    → Similarity deduplication
    → Rank and select top-K
    Output: Ranked list of (sentence, score) tuples
"""

import numpy as np
from typing import List, Tuple, Dict, Set
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings

# Try to import spaCy for NER, with fallback
try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
        SPACY_AVAILABLE = True
    except OSError:
        nlp = None
        SPACY_AVAILABLE = False
        warnings.warn(
            "spaCy model 'en_core_web_sm' not found. "
            "Install with: python -m spacy download en_core_web_sm\n"
            "NER boosting will be disabled."
        )
except ImportError:
    nlp = None
    SPACY_AVAILABLE = False
    warnings.warn(
        "spaCy not installed. Install with: pip install spacy\n"
        "NER boosting will be disabled."
    )

# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

DEFAULT_LOG_LEVEL = logging.INFO
DEFAULT_TOP_K = 10
DEFAULT_SIMILARITY_THRESHOLD = 0.9  # For duplicate detection
NER_BOOST_FACTOR = 0.05  # Boost per entity (max 0.05 * 3 = 0.15)
NER_BOOST_CAP = 3  # Cap NER entities to avoid over-boosting
POSITION_BOOST_FACTOR = 0.02  # Small boost for earlier sentences
MIN_SENTENCES_FOR_RANKING = 2  # Need at least 2 sentences
TFIDF_MAX_FEATURES = 1000  # Increased from 500 to preserve vocabulary

# ============================================================================
# LOGGING SETUP
# ============================================================================

logger = logging.getLogger(__name__)


# ============================================================================
# CORE RANKING FUNCTIONS
# ============================================================================

def compute_centroid(tfidf_matrix: np.ndarray) -> np.ndarray:
    """
    Compute the document centroid from TF-IDF matrix.
    
    The centroid represents the "average" semantic content of the document.
    It's calculated as the mean vector of all sentence TF-IDF vectors.
    
    Args:
        tfidf_matrix: Sparse or dense matrix of shape (num_sentences, vocab_size)
                     Each row is a sentence's TF-IDF vector
        
    Returns:
        1D array of shape (vocab_size,) representing the centroid vector
        
    Example:
        >>> tfidf_matrix = np.array([[0.5, 0.3], [0.4, 0.6]])
        >>> centroid = compute_centroid(tfidf_matrix)
        >>> print(centroid)
        [0.45 0.45]  # Mean of all rows
        
    Mathematical:
        centroid = mean(tfidf_matrix) along axis 0
        For sparse matrices, convert to dense array first
    """
    logger.debug(f"Computing centroid from {tfidf_matrix.shape[0]} sentences")
    
    # Handle sparse matrices (convert to dense for mean calculation)
    if hasattr(tfidf_matrix, 'toarray'):
        # Sparse matrix case
        centroid = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
    else:
        # Dense array case
        centroid = tfidf_matrix.mean(axis=0)
    
    # Normalize centroid to unit length for consistent similarity scores
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm
    
    logger.debug(f"Centroid computed: shape {centroid.shape}, norm {np.linalg.norm(centroid):.4f}")
    return centroid


def score_sentences(tfidf_matrix: np.ndarray, centroid: np.ndarray) -> List[float]:
    """
    Score each sentence based on cosine similarity to document centroid.
    
    The idea: Sentences that are more "central" to the document's content
    (i.e., closer to the average document vector) are more important.
    
    Args:
        tfidf_matrix: Matrix of shape (num_sentences, vocab_size)
                     Each row is a sentence's TF-IDF vector
        centroid: 1D array of shape (vocab_size,) - the document centroid
        
    Returns:
        List of floats in range [0, 1] - cosine similarity scores
        Each index corresponds to a sentence
        
    Example:
        >>> tfidf_matrix = np.array([[0.5, 0.3], [0.4, 0.6]])
        >>> centroid = np.array([0.45, 0.45])
        >>> scores = score_sentences(tfidf_matrix, centroid)
        >>> print(scores)
        [0.98, 0.99]  # Cosine similarity values
        
    Mathematical:
        score[i] = cosine_similarity(sentence_vector[i], centroid)
                 = dot_product(v1, v2) / (||v1|| * ||v2||)
    """
    logger.debug(f"Scoring {tfidf_matrix.shape[0]} sentences against centroid")
    
    # Reshape centroid to 2D for sklearn function
    centroid_2d = centroid.reshape(1, -1)
    
    # Compute cosine similarity (handles sparse matrices)
    similarities = cosine_similarity(tfidf_matrix, centroid_2d)
    
    # Flatten to 1D list and clip to [0, 1] range (in case of floating point errors)
    scores = np.clip(similarities.flatten().tolist(), 0.0, 1.0)
    
    logger.debug(f"Scores computed: min={min(scores):.4f}, max={max(scores):.4f}, "
                f"mean={np.mean(scores):.4f}")
    return scores


def ner_boost(sentence: str, base_score: float, boost_factor: float = NER_BOOST_FACTOR) -> float:
    """
    Boost sentence score based on number of named entities (NER).
    
    Rationale: Sentences with named entities (people, places, organizations)
    often contain important factual information and should be boosted.
    
    IMPROVEMENT: Cap entities at 3 to avoid over-boosting
    (prevents sentences with many entities from dominating TF-IDF scores)
    
    Args:
        sentence: Input sentence text
        base_score: Original TF-IDF similarity score (0-1)
        boost_factor: How much to boost per entity (default: 0.05)
        
    Returns:
        Boosted score (max boost: 0.05 * 3 = 0.15)
        
    Example:
        >>> score = ner_boost("Barack Obama visited Paris yesterday.", 0.7)
        >>> # Sentence has 2 entities (PERSON: Obama, GPE: Paris)
        >>> # boost = 0.05 * min(2, 3) = 0.10
        >>> print(score)
        0.8  # 0.7 + 0.1
        
        >>> # Sentence with 5 entities still gets capped boost
        >>> score = ner_boost("Obama, Putin, Xi, Macron, Modi met.", 0.7)
        >>> # boost = 0.05 * min(5, 3) = 0.15 (capped)
        >>> print(score)
        0.85  # 0.7 + 0.15 (not 0.7 + 0.25)
        
    Feature:
        Returns original score if spaCy not available
    """
    if not SPACY_AVAILABLE:
        return base_score
    
    # Extract entities using spaCy
    doc = nlp(sentence)
    num_entities = len(doc.ents)
    
    # Cap entities at NER_BOOST_CAP to avoid over-boosting
    # More entities provide diminishing returns
    capped_entities = min(num_entities, NER_BOOST_CAP)
    
    # Apply capped boost
    boost = boost_factor * capped_entities
    boosted_score = base_score + boost
    
    logger.debug(f"NER boost: {num_entities} entities (capped at {capped_entities}), "
                f"boost={boost:.4f}, score: {base_score:.4f} -> {boosted_score:.4f}")
    
    return boosted_score


def remove_similar_sentences(sentences: List[str],
                            tfidf_matrix: np.ndarray,
                            threshold: float = DEFAULT_SIMILARITY_THRESHOLD) -> Tuple[List[str], List[int]]:
    """
    Remove duplicate or highly similar sentences.
    
    Strategy: Calculate pairwise cosine similarity between all sentences.
    If two sentences are very similar (above threshold), keep only the first one.
    
    Args:
        sentences: List of sentences to deduplicate
        tfidf_matrix: TF-IDF vectors corresponding to sentences
        threshold: Similarity threshold (0-1). Pairs above this are considered duplicates.
                  Default 0.9 = 90% similar
        
    Returns:
        Tuple of:
        - deduplicated_sentences: List of unique sentences (in original order)
        - kept_indices: List of indices of kept sentences (for tracking)
        
    Example:
        >>> sentences = ["Paris is the capital of France.",
        ...              "The capital of France is Paris.",
        ...              "Machine learning is useful."]
        >>> # First two are very similar
        >>> unique_sents, indices = remove_similar_sentences(sentences, tfidf_matrix)
        >>> print(len(unique_sents))
        2  # Third sentence kept, first duplicate removed
        
    Algorithm:
        1. Calculate pairwise cosine similarity matrix
        2. For each sentence, check if it's similar to any earlier sentence
        3. If yes, mark it as duplicate
        4. Return non-duplicate sentences in original order
    """
    logger.debug(f"Removing similar sentences (threshold={threshold})")
    
    if len(sentences) < 2:
        return sentences, list(range(len(sentences)))
    
    # Calculate pairwise cosine similarity
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    # Track which sentences to keep (use set to avoid duplicates)
    kept_indices = set()
    kept_indices.add(0)  # Always keep first sentence
    
    num_duplicates = 0
    
    for i in range(1, len(sentences)):
        is_duplicate = False
        
        # Check similarity with all earlier kept sentences
        for j in kept_indices:
            if similarity_matrix[i, j] >= threshold:
                is_duplicate = True
                logger.debug(f"Duplicate detected: sentence {i} is {similarity_matrix[i, j]:.4f} "
                           f"similar to sentence {j}")
                num_duplicates += 1
                break
        
        if not is_duplicate:
            kept_indices.add(i)
    
    # Preserve original order
    kept_indices = sorted(list(kept_indices))
    deduplicated_sentences = [sentences[i] for i in kept_indices]
    
    logger.debug(f"Deduplication complete: {len(sentences)} → {len(deduplicated_sentences)} "
                f"sentences ({num_duplicates} duplicates removed)")
    
    return deduplicated_sentences, kept_indices


def rank_sentences(sentences: List[str],
                  top_k: int = DEFAULT_TOP_K,
                  similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
                  normalize_by_length: bool = False,
                  apply_position_boost: bool = True,
                  log_level: int = DEFAULT_LOG_LEVEL) -> List[Tuple[str, float]]:
    """
    Main ranking function: Rank sentences by importance.
    
    Complete pipeline (IMPROVED):
    1. Validate input
    2. TF-IDF vectorization (max_features=1000 for better vocab coverage)
    3. Compute document centroid
    4. Score sentences by cosine similarity
    5. Apply NER boost (CAPPED at 3 entities to avoid over-boosting)
    6. Apply position boost (earlier sentences slightly boosted)
    7. Normalize scores to [0, 1.0] (CLAMPED)
    8. Sort by score and select top-K
    9. Remove duplicates from top-K results only (MOVED to end)
    10. Return ranked results
    
    Args:
        sentences: List of cleaned sentences (from Module 2)
        top_k: Number of top sentences to return (default: 10)
        similarity_threshold: Threshold for duplicate detection (0-1, default: 0.9)
        normalize_by_length: If True, divide score by sentence length
                           (reduces bias toward long sentences)
        apply_position_boost: If True, boost earlier sentences slightly (default: True)
        log_level: Logging verbosity (default: INFO)
        
    Returns:
        List of tuples: [(sentence, score), ...]
        - Sorted by score descending (highest importance first)
        - Up to top_k sentences
        - Scores normalized to [0, 1.0] range
        
    Example:
        >>> sentences = [
        ...     "Paris is the capital of France.",
        ...     "France is a country in Europe.",
        ...     "Machine learning requires data.",
        ... ]
        >>> ranked = rank_sentences(sentences, top_k=2)
        >>> for sent, score in ranked:
        ...     print(f"{score:.3f}: {sent}")
        0.823: Paris is the capital of France.
        0.756: France is a country in Europe.
        
    Improvements:
        - Position boost: First sentences get +0.02, tapering to 0 at end
        - NER boost capped: Max 0.15 (0.05 * 3 entities) to avoid domination
        - Score normalization: All final scores clamped to [0, 1.0]
        - Deduplication moved to end: Preserves original TF-IDF context
        - Increased max_features: 1000 instead of 500 for better vocabulary
    """
    # Configure logging
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    logger.setLevel(log_level)
    logger.info(f"Starting sentence ranking pipeline ({len(sentences)} input sentences)")
    
    # Step 0: Validate input
    if not sentences:
        logger.warning("Empty sentence list provided")
        return []
    
    if len(sentences) < MIN_SENTENCES_FOR_RANKING:
        logger.warning(f"Too few sentences ({len(sentences)} < {MIN_SENTENCES_FOR_RANKING})")
        # Return all with equal score
        score = 0.5
        return [(sent, score) for sent in sentences]
    
    # Step 1: TF-IDF vectorization (IMPROVED: max_features=1000)
    logger.info("Step 1: TF-IDF vectorization...")
    vectorizer = TfidfVectorizer(stop_words='english', max_features=TFIDF_MAX_FEATURES)
    tfidf_matrix = vectorizer.fit_transform(sentences)
    logger.debug(f"TF-IDF matrix shape: {tfidf_matrix.shape} (improved: max_features={TFIDF_MAX_FEATURES})")
    
    # Step 2: Compute centroid
    logger.info("Step 2: Computing document centroid...")
    centroid = compute_centroid(tfidf_matrix)
    
    # Step 3: Score sentences
    logger.info("Step 3: Scoring sentences by TF-IDF similarity...")
    base_scores = score_sentences(tfidf_matrix, centroid)
    
    # Step 4: Apply NER boost (IMPROVED: capped at 3 entities)
    logger.info("Step 4: Applying NER boosting (capped at 3 entities)...")
    if SPACY_AVAILABLE:
        ner_boosted_scores = [ner_boost(sent, base_score) for sent, base_score in zip(sentences, base_scores)]
    else:
        ner_boosted_scores = base_scores
        logger.debug("spaCy not available, skipping NER boost")
    
    # Step 5: Apply position boost (IMPROVEMENT: added)
    logger.info("Step 5: Applying position boost (early sentences favored)...")
    if apply_position_boost:
        num_sentences = len(sentences)
        position_boosted_scores = [
            score + (POSITION_BOOST_FACTOR * (1 - idx / num_sentences))
            for idx, score in enumerate(ner_boosted_scores)
        ]
        logger.debug(f"Position boost: first={POSITION_BOOST_FACTOR:.4f}, last=0.0")
    else:
        position_boosted_scores = ner_boosted_scores
    
    # Step 6: Optional length normalization
    if normalize_by_length:
        logger.info("Step 6: Normalizing by sentence length...")
        normalized_scores = [
            score / (1 + 0.01 * len(sent.split()))  # Penalize very long sentences
            for score, sent in zip(position_boosted_scores, sentences)
        ]
    else:
        normalized_scores = position_boosted_scores
    
    # Step 7: Clamp scores to [0, 1.0] (IMPROVEMENT: normalize)
    logger.info("Step 7: Normalizing scores to [0, 1.0]...")
    clamped_scores = [min(max(score, 0.0), 1.0) for score in normalized_scores]
    logger.debug(f"Score range before clamping: {min(normalized_scores):.4f} - {max(normalized_scores):.4f}")
    logger.debug(f"Score range after clamping: {min(clamped_scores):.4f} - {max(clamped_scores):.4f}")
    
    # Step 8: Rank and select top-K
    logger.info(f"Step 8: Selecting top-{top_k} sentences...")
    
    # Create list of (sentence, score, original_index)
    ranked_with_idx = [
        (sent, score, idx)
        for idx, (sent, score) in enumerate(zip(sentences, clamped_scores))
    ]
    
    # Sort by score descending
    ranked_with_idx.sort(key=lambda x: x[1], reverse=True)
    
    # Select top-K
    top_k_with_idx = ranked_with_idx[:top_k]
    
    # Step 9: Remove duplicates from top-K results (IMPROVEMENT: moved to end)
    logger.info(f"Step 9: Removing duplicates from top-{top_k} results...")
    
    # Extract top-K sentences for deduplication
    top_k_sentences = [sent for sent, _, _ in top_k_with_idx]
    
    # Only deduplicate if we have multiple top-K results
    if len(top_k_sentences) > 1:
        # Re-vectorize top-K sentences for deduplication
        top_k_vectorizer = TfidfVectorizer(stop_words='english', max_features=TFIDF_MAX_FEATURES)
        top_k_tfidf = top_k_vectorizer.fit_transform(top_k_sentences)
        
        # Remove duplicates from top-K
        deduplicated_sentences, kept_indices = remove_similar_sentences(
            top_k_sentences, top_k_tfidf, threshold=similarity_threshold
        )
        
        # Filter top_k_with_idx to keep only non-duplicates
        top_k_with_idx = [top_k_with_idx[i] for i in kept_indices]
    
    # Return as (sentence, score) tuples
    result = [(sent, score) for sent, score, _ in top_k_with_idx]
    
    logger.info(f"Ranking complete: returned {len(result)} top sentences "
               f"(after deduplication, original top-K was {len(top_k_sentences)})")
    return result


def get_ranking_stats(ranked_sentences: List[Tuple[str, float]]) -> Dict:
    """
    Get statistics about ranked sentences.
    
    Args:
        ranked_sentences: Output from rank_sentences()
        
    Returns:
        Dictionary with statistics:
        - count: Number of sentences
        - avg_score: Average importance score
        - max_score: Highest score
        - min_score: Lowest score
        - score_range: (min, max)
        - avg_length: Average sentence length in words
    """
    if not ranked_sentences:
        return {
            'count': 0,
            'avg_score': 0.0,
            'max_score': 0.0,
            'min_score': 0.0,
            'score_range': (0.0, 0.0),
            'avg_length': 0.0,
        }
    
    scores = [score for _, score in ranked_sentences]
    lengths = [len(sent.split()) for sent, _ in ranked_sentences]
    
    return {
        'count': len(ranked_sentences),
        'avg_score': np.mean(scores),
        'max_score': max(scores),
        'min_score': min(scores),
        'score_range': (min(scores), max(scores)),
        'avg_length': np.mean(lengths),
    }


# ============================================================================
# TEST SUITE
# ============================================================================

def test_sentence_ranker():
    """
    Comprehensive test suite for sentence ranking module.
    
    Tests:
    1. Normal paragraph - Standard ranking
    2. Repeated sentences - Deduplication
    3. Named entities - NER boosting
    4. Mixed importance - Varied content
    5. Edge cases - Empty, single sentence
    """
    print("="*80)
    print("SENTENCE RANKING MODULE - TEST SUITE")
    print("="*80 + "\n")
    
    # Test 1: Normal paragraph
    print("[TEST 1] Normal paragraph with varied importance...")
    text1 = [
        "Artificial Intelligence is transforming industries.",
        "Machine learning models learn from data.",
        "Deep learning uses neural networks.",
        "Python is widely used for AI.",
        "Data preprocessing is crucial for model performance.",
    ]
    
    try:
        ranked = rank_sentences(text1, top_k=3, log_level=logging.WARNING)
        print(f"[OK] Ranked {len(ranked)} sentences\n")
        for i, (sent, score) in enumerate(ranked, 1):
            print(f"  {i}. ({score:.4f}) {sent}")
        print()
    except Exception as e:
        print(f"[ERROR] {e}\n")
    
    # Test 2: Duplicate sentences
    print("[TEST 2] Handling duplicate and similar sentences...")
    text2 = [
        "Paris is the capital of France.",
        "The capital of France is Paris.",  # Very similar to first
        "France is located in Europe.",
        "Machine learning is powerful.",
        "Powerful machine learning algorithms exist.",  # Similar to previous
    ]
    
    try:
        ranked = rank_sentences(text2, top_k=3, similarity_threshold=0.85, log_level=logging.WARNING)
        print(f"[OK] Ranked {len(ranked)} sentences (after deduplication)\n")
        for i, (sent, score) in enumerate(ranked, 1):
            print(f"  {i}. ({score:.4f}) {sent}")
        print()
    except Exception as e:
        print(f"[ERROR] {e}\n")
    
    # Test 3: Named entities
    print("[TEST 3] Named Entity Recognition boosting...")
    text3 = [
        "Albert Einstein developed the theory of relativity.",
        "Physics studies the laws of nature.",
        "Marie Curie discovered polonium and radium.",
        "Chemistry involves reactions between atoms.",
        "Isaac Newton formulated the laws of motion.",
    ]
    
    try:
        ranked = rank_sentences(text3, top_k=3, log_level=logging.WARNING)
        print(f"[OK] Ranked {len(ranked)} sentences (with NER boosting)\n")
        print("  [Note: Sentences with named entities should rank higher]\n")
        for i, (sent, score) in enumerate(ranked, 1):
            print(f"  {i}. ({score:.4f}) {sent}")
        print()
    except Exception as e:
        print(f"[ERROR] {e}\n")
    
    # Test 4: Mixed importance
    print("[TEST 4] Mixed importance sentences...")
    text4 = [
        "Global warming is a serious environmental challenge.",
        "Climate change affects weather patterns.",
        "Rising temperatures cause ice melting.",
        "Renewable energy reduces carbon emissions.",
        "Technology innovations are important.",
    ]
    
    try:
        ranked = rank_sentences(text4, top_k=2, log_level=logging.WARNING)
        stats = get_ranking_stats(ranked)
        print(f"[OK] Ranked {len(ranked)} sentences\n")
        print(f"  Statistics:")
        print(f"    - Average score: {stats['avg_score']:.4f}")
        print(f"    - Score range: {stats['score_range'][0]:.4f} - {stats['score_range'][1]:.4f}")
        print(f"    - Avg sentence length: {stats['avg_length']:.1f} words\n")
        for i, (sent, score) in enumerate(ranked, 1):
            print(f"  {i}. ({score:.4f}) {sent}")
        print()
    except Exception as e:
        print(f"[ERROR] {e}\n")
    
    # Test 5: Edge cases
    print("[TEST 5] Edge cases...")
    
    # Empty input
    try:
        ranked = rank_sentences([], log_level=logging.WARNING)
        print(f"[OK] Empty input handled: {len(ranked)} sentences returned")
    except Exception as e:
        print(f"[ERROR] Empty input: {e}")
    
    # Single sentence
    try:
        ranked = rank_sentences(["Only one sentence here."], log_level=logging.WARNING)
        print(f"[OK] Single sentence handled: {len(ranked)} sentences returned")
    except Exception as e:
        print(f"[ERROR] Single sentence: {e}")
    
    print()
    print("="*80)
    print("TEST SUITE COMPLETED")
    print("="*80 + "\n")


if __name__ == "__main__":
    test_sentence_ranker()
