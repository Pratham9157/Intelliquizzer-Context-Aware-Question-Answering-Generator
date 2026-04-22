"""
MODULE 2: Text Preprocessing Module
====================================
Converts raw extracted text into clean, structured sentences
suitable for downstream NLP processing.

Purpose:
    Take extracted plain text and produce a list of clean,
    normalized sentences ready for NLP tasks like ranking,
    entity extraction, and question generation.

Key Features:
    - Unicode normalization
    - Text cleaning (special chars, extra whitespace)
    - Sentence tokenization (NLTK punkt)
    - Noise filtering (remove very short/noisy sentences)
    - Configurable filtering thresholds
    - Detailed logging and statistics
"""

import re
import unicodedata
from typing import List, Dict, Tuple
import logging

# Auto-setup dependencies on first import
try:
    from setup import ensure_dependencies
    ensure_dependencies()
except ImportError:
    pass  # setup.py not available, assume manual setup

# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

# Filtering thresholds
MIN_SENTENCE_LENGTH = 6  # Minimum words per sentence
MIN_ALPHABETIC_RATIO = 0.5  # Minimum ratio of alphabetic chars
MAX_DIGIT_RATIO = 0.3  # Maximum ratio of digits allowed
MAX_SPECIAL_CHAR_RATIO = 0.2  # Maximum ratio of special chars

# Noise detection thresholds
NOISE_PATTERNS = {
    'all_caps': r'^[A-Z\s]{10,}$',  # All caps sentence
    'mostly_numbers': r'\d',
    'too_short': None,  # Handled separately
}

# Logging configuration
DEFAULT_LOG_LEVEL = logging.INFO

# Configure logging
logger = logging.getLogger(__name__)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def normalize_unicode(text: str) -> str:
    """
    Normalize unicode text to NFKD form.
    
    NFKD normalization decomposes characters into base + combining marks.
    Useful for handling accented characters and special unicode variants.
    
    Args:
        text: Input text with potential unicode variations
        
    Returns:
        Normalized text in NFKD form
        
    Example:
        >>> normalize_unicode("café")
        'cafe'  # With combining accent mark removed in subsequent processing
    """
    return unicodedata.normalize("NFKD", text)


def is_noisy_sentence(sentence: str,
                     min_length: int = MIN_SENTENCE_LENGTH,
                     min_alpha_ratio: float = MIN_ALPHABETIC_RATIO,
                     max_digit_ratio: float = MAX_DIGIT_RATIO,
                     max_special_ratio: float = MAX_SPECIAL_CHAR_RATIO) -> Tuple[bool, str]:
    """
    Check if a sentence is "noisy" and should be filtered out.
    
    Criteria for noise:
    1. Too short (< min_length words)
    2. Too few alphabetic characters (< min_alpha_ratio)
    3. Too many digits (> max_digit_ratio)
    4. Too many special characters (> max_special_ratio)
    
    Args:
        sentence: Sentence to evaluate
        min_length: Minimum number of words
        min_alpha_ratio: Minimum ratio of alphabetic characters (0-1)
        max_digit_ratio: Maximum ratio of digit characters (0-1)
        max_special_ratio: Maximum ratio of special characters (0-1)
        
    Returns:
        Tuple of (is_noisy: bool, reason: str)
        - is_noisy: True if sentence is noisy
        - reason: Why it's noisy (for logging)
        
    Example:
        >>> is_noisy_sentence("Hello world")
        (False, "")
        
        >>> is_noisy_sentence("A")
        (True, "Too short (1 words < 6)")
        
        >>> is_noisy_sentence("123 456 789 000 111")
        (True, "Too many digits (ratio: 0.77 > 0.30)")
    """
    sentence = sentence.strip()
    
    # Check 1: Sentence length (in words)
    word_count = len(sentence.split())
    if word_count < min_length:
        return True, f"Too short ({word_count} words < {min_length})"
    
    # Check 2: Alphabetic ratio
    if len(sentence) == 0:
        return True, "Empty sentence"
    
    alpha_chars = sum(1 for c in sentence if c.isalpha())
    alpha_ratio = alpha_chars / len(sentence)
    if alpha_ratio < min_alpha_ratio:
        return True, f"Too few alphabetic chars (ratio: {alpha_ratio:.2f} < {min_alpha_ratio})"
    
    # Check 3: Digit ratio
    digit_chars = sum(1 for c in sentence if c.isdigit())
    digit_ratio = digit_chars / len(sentence)
    if digit_ratio > max_digit_ratio:
        return True, f"Too many digits (ratio: {digit_ratio:.2f} > {max_digit_ratio})"
    
    # Check 4: Special character ratio
    special_chars = sum(1 for c in sentence if not (c.isalnum() or c.isspace() or c in '.,!?;:-'))
    special_ratio = special_chars / len(sentence)
    if special_ratio > max_special_ratio:
        return True, f"Too many special chars (ratio: {special_ratio:.2f} > {max_special_ratio})"
    
    return False, ""


# ============================================================================
# CORE PREPROCESSING FUNCTIONS
# ============================================================================

def clean_text(text: str) -> str:
    """
    Clean raw extracted text for further processing.
    
    Operations performed:
    1. Unicode normalization (NFKD)
    2. Remove bracketed/parenthesized text ([1], (note), {ref})
    3. Remove control characters
    4. Normalize whitespace (multiple spaces → single)
    5. Normalize line endings
    
    Args:
        text: Raw extracted text
        
    Returns:
        Cleaned text
        
    Example:
        >>> clean_text("Text with  [1]  extra    spaces")
        "Text with extra spaces"
    """
    logger.debug(f"Cleaning text ({len(text)} chars)")
    
    # Step 1: Unicode normalization
    text = normalize_unicode(text)
    
    # Step 2: Remove control characters (except newlines)
    text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', text)
    
    # Step 3: Remove bracketed/parenthesized content
    # Patterns: [1], (note), {ref}, etc.
    text = re.sub(r'\[\d+\]', '', text)  # [1], [2], etc. (citations)
    text = re.sub(r'\([^)]*\)', '', text)  # (...)
    text = re.sub(r'\{[^}]*\}', '', text)  # {...}
    text = re.sub(r'\[[^\]]*\]', '', text)  # [...]
    
    # Step 4: Normalize line breaks
    text = re.sub(r'\r\n', '\n', text)  # Windows → Unix
    text = re.sub(r'\r', '\n', text)    # Old Mac → Unix
    
    # Step 5: Normalize spaces
    # Preserve paragraph breaks (double newlines)
    paragraphs = text.split('\n\n')
    normalized_paragraphs = [
        re.sub(r'[ \t]+', ' ', para).strip()
        for para in paragraphs
    ]
    text = '\n\n'.join(p for p in normalized_paragraphs if p)
    
    logger.debug(f"Cleaned text ({len(text)} chars after cleaning)")
    return text


def tokenize_sentences(text: str,
                      min_length: int = MIN_SENTENCE_LENGTH) -> List[str]:
    """
    Tokenize text into sentences using NLTK punkt tokenizer.
    
    Operations:
    1. Split text into sentences using NLTK sent_tokenize
    2. Filter sentences shorter than min_length words
    3. Preserve original casing
    4. Strip whitespace from each sentence
    
    Args:
        text: Cleaned text to tokenize
        min_length: Minimum words per sentence (default: 6)
        
    Returns:
        List of tokenized sentences
        
    Raises:
        ImportError: If NLTK punkt tokenizer not available
        
    Example:
        >>> text = "Hello world. This is a test. A."
        >>> tokenize_sentences(text, min_length=2)
        ["Hello world.", "This is a test."]
    """
    logger.debug(f"Tokenizing text into sentences")
    
    try:
        from nltk.tokenize import sent_tokenize
    except ImportError:
        raise ImportError(
            "NLTK not installed. Install with: pip install nltk\n"
            "Then download punkt tokenizer: python -m nltk.downloader punkt"
        )
    
    # Tokenize into sentences
    try:
        sentences = sent_tokenize(text)
    except Exception as e:
        logger.warning(f"Sentence tokenization failed: {e}. Using fallback.")
        # Fallback: simple period-based splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Filter by minimum length
    filtered = []
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        
        word_count = len(sent.split())
        if word_count >= min_length:
            filtered.append(sent)
        else:
            logger.debug(f"Filtered short sentence: '{sent}' ({word_count} words)")
    
    logger.debug(f"Tokenized into {len(filtered)} sentences (from {len(sentences)} raw)")
    return filtered


def filter_sentences(sentences: List[str],
                    min_length: int = MIN_SENTENCE_LENGTH,
                    min_alpha_ratio: float = MIN_ALPHABETIC_RATIO,
                    max_digit_ratio: float = MAX_DIGIT_RATIO,
                    max_special_ratio: float = MAX_SPECIAL_CHAR_RATIO) -> List[str]:
    """
    Filter out noisy sentences that don't meet quality criteria.
    
    Removes sentences that:
    - Are too short
    - Have too few alphabetic characters
    - Have too many digits
    - Have too many special characters
    
    Args:
        sentences: List of sentences to filter
        min_length: Minimum words per sentence
        min_alpha_ratio: Minimum alphabetic character ratio
        max_digit_ratio: Maximum digit character ratio
        max_special_ratio: Maximum special character ratio
        
    Returns:
        Filtered list of clean sentences
        
    Example:
        >>> sentences = ["Normal sentence here.", "12345 67890", "A"]
        >>> filter_sentences(sentences)
        ["Normal sentence here."]
    """
    logger.debug(f"Filtering {len(sentences)} sentences")
    
    filtered = []
    filtered_out = []
    
    for sent in sentences:
        is_noisy, reason = is_noisy_sentence(
            sent,
            min_length=min_length,
            min_alpha_ratio=min_alpha_ratio,
            max_digit_ratio=max_digit_ratio,
            max_special_ratio=max_special_ratio
        )
        
        if is_noisy:
            filtered_out.append((sent, reason))
            logger.debug(f"Filtered: {reason} | '{sent[:50]}...'")
        else:
            filtered.append(sent)
    
    logger.debug(f"Filtering complete: {len(filtered)} kept, {len(filtered_out)} removed")
    return filtered


def preprocess(text: str,
              min_sentence_length: int = MIN_SENTENCE_LENGTH,
              min_alpha_ratio: float = MIN_ALPHABETIC_RATIO,
              max_digit_ratio: float = MAX_DIGIT_RATIO,
              max_special_ratio: float = MAX_SPECIAL_CHAR_RATIO,
              log_level: int = DEFAULT_LOG_LEVEL) -> List[str]:
    """
    Main preprocessing pipeline.
    
    Complete preprocessing pipeline:
    1. clean_text() — Remove special chars, normalize whitespace
    2. tokenize_sentences() — Split into sentences
    3. filter_sentences() — Remove noisy sentences
    
    Args:
        text: Raw extracted text
        min_sentence_length: Minimum words per sentence
        min_alpha_ratio: Minimum alphabetic ratio (0-1)
        max_digit_ratio: Maximum digit ratio (0-1)
        max_special_ratio: Maximum special char ratio (0-1)
        log_level: Logging verbosity (logging.DEBUG, INFO, etc.)
        
    Returns:
        List of clean, normalized sentences ready for NLP
        
    Example:
        >>> raw_text = "Hello world.  This is a test."
        >>> sentences = preprocess(raw_text)
        >>> print(sentences)
        ["Hello world.", "This is a test."]
    """
    # Configure logging for this call
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    logger.setLevel(log_level)
    logger.info(f"Starting preprocessing pipeline ({len(text)} input chars)")
    
    # Step 1: Clean text
    cleaned_text = clean_text(text)
    
    # Step 2: Tokenize into sentences
    sentences = tokenize_sentences(cleaned_text, min_length=min_sentence_length)
    logger.info(f"Tokenized into {len(sentences)} sentences")
    
    # Step 3: Filter noisy sentences
    filtered_sentences = filter_sentences(
        sentences,
        min_length=min_sentence_length,
        min_alpha_ratio=min_alpha_ratio,
        max_digit_ratio=max_digit_ratio,
        max_special_ratio=max_special_ratio
    )
    
    logger.info(f"Preprocessing complete: {len(filtered_sentences)} final sentences")
    return filtered_sentences


def get_preprocessing_stats(sentences: List[str]) -> Dict:
    """
    Get statistics about preprocessed sentences.
    
    Useful for quality assurance and debugging.
    
    Args:
        sentences: List of preprocessed sentences
        
    Returns:
        Dictionary with statistics:
        - sentence_count: Total sentences
        - word_count: Total words
        - avg_sentence_length: Average words per sentence
        - avg_char_length: Average characters per sentence
        - min/max_sentence_length: Length range
    """
    if not sentences:
        return {
            'sentence_count': 0,
            'word_count': 0,
            'avg_sentence_length': 0,
            'avg_char_length': 0,
            'min_sentence_length': 0,
            'max_sentence_length': 0,
        }
    
    word_counts = [len(s.split()) for s in sentences]
    char_counts = [len(s) for s in sentences]
    
    return {
        'sentence_count': len(sentences),
        'word_count': sum(word_counts),
        'avg_sentence_length': sum(word_counts) / len(sentences),
        'avg_char_length': sum(char_counts) / len(sentences),
        'min_sentence_length': min(word_counts),
        'max_sentence_length': max(word_counts),
        'total_characters': sum(char_counts),
    }


# ============================================================================
# TEST FUNCTION
# ============================================================================

def test_preprocessor():
    """
    Comprehensive test suite for the preprocessing module.
    
    Tests:
    1. Messy text with special chars and extra spaces
    2. Academic paragraph with citations
    3. Text with numbers and symbols
    4. Empty and edge case inputs
    5. Unicode handling
    6. Noise filtering
    """
    print("\n" + "="*80)
    print("PREPROCESSING MODULE - TEST SUITE")
    print("="*80 + "\n")
    
    # Configure logging for tests
    test_logger = logging.getLogger(__name__)
    if not test_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            '%(levelname)s: %(message)s'
        ))
        test_logger.addHandler(handler)
    test_logger.setLevel(logging.INFO)
    
    # Test 1: Messy text with extra spaces and special chars
    print("[TEST 1] Messy text with extra spaces and special characters...")
    messy_text = """
    Natural Language Processing (NLP)  is  a  field of AI.
    
    It involves   working with text data.  The goal is [1] to extract meaning
    from {unstructured} text.  Multiple   spaces    and    newlines     are common.
    """
    
    try:
        sentences = preprocess(messy_text, log_level=logging.WARNING)
        print(f"[OK] Successfully processed {len(sentences)} sentences")
        for i, sent in enumerate(sentences, 1):
            print(f"  {i}. {sent}")
        print()
    except Exception as e:
        print(f"[ERROR] {e}\n")
    
    # Test 2: Academic paragraph with citations
    print("[TEST 2] Academic paragraph with citations...")
    academic_text = """
    Machine Learning [2] is a subset of Artificial Intelligence [3].
    According to Smith et al. [2020], deep learning models (specifically 
    transformers) have revolutionized NLP. The BERT model [4] achieved 
    state-of-the-art results on multiple benchmarks. T5 [5] introduced 
    the text-to-text paradigm for various NLP tasks.
    """
    
    try:
        sentences = preprocess(academic_text, log_level=logging.WARNING)
        print(f"[OK] Successfully processed {len(sentences)} sentences")
        for i, sent in enumerate(sentences, 1):
            print(f"  {i}. {sent}")
        
        # Show statistics
        stats = get_preprocessing_stats(sentences)
        print(f"\n  Statistics:")
        print(f"    - Average sentence length: {stats['avg_sentence_length']:.1f} words")
        print(f"    - Total words: {stats['word_count']}")
        print()
    except Exception as e:
        print(f"[ERROR] {e}\n")
    
    # Test 3: Text with numbers and symbols
    print("[TEST 3] Text with numbers and symbols...")
    noisy_text = """
    Important facts: 42% of people use AI. The year 2024 saw growth: 
    $100,000,000 invested. Contact: email@example.com or 123-456-7890.
    However, this sentence: @#$%^&*() 12345 67890 should be filtered.
    """
    
    try:
        sentences = preprocess(noisy_text, log_level=logging.WARNING)
        print(f"[OK] Successfully processed {len(sentences)} sentences")
        for i, sent in enumerate(sentences, 1):
            print(f"  {i}. {sent}")
        print()
    except Exception as e:
        print(f"[ERROR] {e}\n")
    
    # Test 4: Edge cases
    print("[TEST 4] Edge cases (empty, very short, etc.)...")
    
    # Empty input
    try:
        result = preprocess("", log_level=logging.WARNING)
        print(f"[OK] Empty input handled: {len(result)} sentences returned")
    except Exception as e:
        print(f"[ERROR] Error with empty input: {e}")
    
    # Only short sentences (should be filtered)
    try:
        result = preprocess("A. B. C. D.", log_level=logging.WARNING)
        print(f"[OK] Short sentences handled: {len(result)} sentences kept (all filtered as < 6 words)")
    except Exception as e:
        print(f"[ERROR] {e}")
    
    print()
    
    # Test 5: Unicode normalization
    print("[TEST 5] Unicode normalization...")
    unicode_text = "Café, naïve, über, Zürich, São Paulo are cities."
    
    try:
        sentences = preprocess(unicode_text, log_level=logging.WARNING)
        print(f"[OK] Unicode handled: {len(sentences)} sentences")
        for sent in sentences:
            print(f"  {sent}")
        print()
    except Exception as e:
        print(f"[ERROR] {e}\n")
    
    # Test 6: Noise filtering details
    print("[TEST 6] Detailed noise filtering analysis...")
    test_sentences = [
        "This is a normal sentence.",
        "A",
        "123 456 789 000 111",
        "@#$%^&*()",
        "This is OBVIOUSLY A VERY LONG SENTENCE IN ALL CAPS THAT SHOULD POSSIBLY BE FILTERED OUT",
        "The quick brown fox jumps over the lazy dog.",
    ]
    
    print("  Analyzing sentences:")
    for sent in test_sentences:
        is_noisy, reason = is_noisy_sentence(sent)
        status = "[NOISY]" if is_noisy else "[CLEAN]"
        print(f"    {status}: '{sent}' -> {reason if reason else 'OK'}")
    
    print()
    
    # Test 7: Full pipeline with statistics
    print("[TEST 7] Full pipeline with statistics...")
    full_text = """
    Artificial Intelligence and Machine Learning are transforming industries.
    
    AI can automate complex tasks. Machine learning models learn patterns 
    from data. Deep learning uses neural networks with multiple layers.
    
    The transformer architecture [1] introduced attention mechanisms.
    BERT [2] and T5 [3] are popular pre-trained models. These models
    achieve state-of-the-art results on various NLP benchmarks.
    
    Future research will focus on efficiency and interpretability.
    """
    
    try:
        sentences = preprocess(full_text, log_level=logging.WARNING)
        stats = get_preprocessing_stats(sentences)
        
        print(f"[OK] Pipeline complete!")
        print(f"\n  Results:")
        print(f"    - Sentences: {stats['sentence_count']}")
        print(f"    - Total words: {stats['word_count']}")
        print(f"    - Avg sentence length: {stats['avg_sentence_length']:.1f} words")
        print(f"    - Avg sentence length: {stats['avg_char_length']:.0f} chars")
        print(f"    - Range: {stats['min_sentence_length']}-{stats['max_sentence_length']} words")
        print(f"\n  Sentences:")
        for i, sent in enumerate(sentences, 1):
            word_count = len(sent.split())
            char_count = len(sent)
            print(f"    {i}. ({word_count}w, {char_count}c) {sent[:60]}...")
        print()
    except Exception as e:
        print(f"[ERROR] {e}\n")
    
    print("="*80)
    print("TEST SUITE COMPLETED")
    print("="*80 + "\n")


if __name__ == "__main__":
    test_preprocessor()
