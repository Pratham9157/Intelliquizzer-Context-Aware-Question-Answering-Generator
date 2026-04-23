"""
MODULE 4: Answer Extraction Module
===================================
Extracts meaningful answer spans from sentences using a hybrid approach
combining Named Entity Recognition (NER), noun phrases, and rule-based filtering.

Purpose:
    Identify and extract meaningful answer candidates from sentences.
    These answers are used as targets for question generation in Module 5.

Key Features:
    - Named Entity Extraction (who, where, organization)
    - Noun Phrase Extraction (important noun chunks)
    - Rule-based answer filtering (remove noise, trivial content)
    - Answer type classification (PERSON, LOCATION, DATE, etc.)
    - Answer importance scoring (length, NER status, position, frequency)
    - Hybrid extraction pipeline (combines multiple signals)
    - Configurable thresholds and limits
    - Comprehensive logging

Architecture:
    Input: List of sentences (from Module 3)
    → For each sentence:
        1. Extract NER entities
        2. Extract noun phrases
        3. Combine candidates
        4. Filter by quality rules
        5. Score by importance
        6. Classify answer types
        7. Return top-N answers
    Output: Dict[sentence → List[answer_objects]]
"""

import re
import string
import logging
from typing import List, Tuple, Dict, Set, Optional
import warnings

# Try to import spaCy
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
            "Answer extraction will be disabled."
        )
except ImportError:
    nlp = None
    SPACY_AVAILABLE = False
    warnings.warn(
        "spaCy not installed. Install with: pip install spacy\n"
        "Answer extraction will be disabled."
    )

# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

DEFAULT_LOG_LEVEL = logging.INFO
DEFAULT_MAX_ANSWERS_PER_SENTENCE = 3
MIN_ANSWER_LENGTH = 3  # Minimum characters for valid answer
MIN_ANSWER_WORDS = 1  # Minimum words for valid answer
MAX_ANSWER_LENGTH = 100  # Maximum characters (avoid verbose answers)
MIN_SCORE_THRESHOLD = 0.3  # IMPROVEMENT 6: Minimum score to keep an answer; filter below this
NER_SOURCE_BOOST = 0.1  # IMPROVEMENT 4: Bonus score for NER-sourced answers (ensures ~10% advantage)
GENERIC_WORDS = {
    'thing', 'something', 'someone', 'way', 'fact', 'one',
    'it', 'they', 'he', 'she', 'we', 'you', 'i',
    'this', 'that', 'these', 'those', 'what', 'which',
    'here', 'there', 'where', 'when', 'why', 'how',
}  # Words to filter out

STOPWORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'must', 'can',
}

# Answer type patterns
YEAR_PATTERN = re.compile(r'\b(19|20)\d{2}\b')
DATE_PATTERN = re.compile(r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b', re.IGNORECASE)
NUMBER_PATTERN = re.compile(r'^-?\d+(?:\.\d+)?$')
PERCENT_PATTERN = re.compile(r'^\d+(?:\.\d+)?%$')

# ============================================================================
# LOGGING SETUP
# ============================================================================

logger = logging.getLogger(__name__)


# ============================================================================
# CORE EXTRACTION FUNCTIONS
# ============================================================================

def extract_named_entities(sentence: str) -> List[Tuple[str, str]]:
    """
    Extract named entities from a sentence using spaCy.
    
    Uses spaCy's trained NER model to identify and classify named entities
    such as people, locations, organizations, dates, etc.
    
    Args:
        sentence: Input sentence text
        
    Returns:
        List of tuples: [(entity_text, entity_label), ...]
        Labels: PERSON, GPE (location), ORG, DATE, MONEY, FACILITY, PRODUCT, etc.
        
    Example:
        >>> entities = extract_named_entities("Barack Obama visited Paris in 2022.")
        >>> print(entities)
        [("Barack Obama", "PERSON"), ("Paris", "GPE"), ("2022", "DATE")]
        
    Features:
        - Empty list if spaCy not available
        - Filters out entities with only punctuation
        - Returns entities in order of appearance
    """
    if not SPACY_AVAILABLE:
        logger.warning("spaCy not available, skipping NER extraction")
        return []
    
    try:
        doc = nlp(sentence)
        entities = []
        
        for ent in doc.ents:
            # Skip entities that are purely punctuation
            if not ent.text.strip(string.punctuation):
                continue
            
            entities.append((ent.text, ent.label_))
        
        logger.debug(f"Extracted {len(entities)} NER entities from sentence")
        return entities
    
    except Exception as e:
        logger.warning(f"Error extracting NER entities: {e}")
        return []


def extract_noun_phrases(sentence: str) -> List[str]:
    """
    Extract noun phrases (noun chunks) from a sentence.
    
    Uses spaCy's noun_chunks to identify complete noun phrases including
    adjectives, determiners, and other modifiers.
    
    Args:
        sentence: Input sentence text
        
    Returns:
        List of noun phrase strings
        
    Example:
        >>> phrases = extract_noun_phrases("The quick brown fox jumps.")
        >>> print(phrases)
        ["The quick brown fox"]
        
    Filtering Rules:
        1. Minimum 2 words (to avoid single determiners/articles)
        2. Not purely stopwords
        3. Must contain at least one meaningful word
        4. Maximum 5 words (to avoid over-long phrases)
    """
    if not SPACY_AVAILABLE:
        logger.warning("spaCy not available, skipping noun phrase extraction")
        return []
    
    try:
        doc = nlp(sentence)
        phrases = []
        
        for chunk in doc.noun_chunks:
            phrase_text = chunk.text.strip()
            words = phrase_text.split()
            
            # Filter 1: Must be at least 2 words (or very meaningful single word)
            if len(words) == 1:
                # Accept single-word nouns, but skip determiners/pronouns
                if chunk[0].pos_ not in ('DET', 'PRON'):
                    phrases.append(phrase_text)
                continue
            
            # Filter 2: Not purely stopwords
            non_stop_words = [w for w in words if w.lower() not in STOPWORDS]
            if not non_stop_words:
                continue
            
            # Filter 3: Maximum 5 words
            if len(words) > 5:
                continue
            
            phrases.append(phrase_text)
        
        logger.debug(f"Extracted {len(phrases)} noun phrases from sentence")
        return phrases
    
    except Exception as e:
        logger.warning(f"Error extracting noun phrases: {e}")
        return []


def remove_overlapping_answers(answers: List[str]) -> List[str]:
    """
    IMPROVEMENT 3: Remove overlapping/redundant answers where one is substring of another.
    
    Keeps the longer, more informative span and removes shorter duplicates.
    Important for deduplication after combining NER entities and noun phrases.
    
    Example:
        >>> answers = ["Einstein", "Albert Einstein", "learning", "machine learning"]
        >>> result = remove_overlapping_answers(answers)
        >>> print(result)
        ["Albert Einstein", "machine learning"]
    
    Logic:
        - For each answer, check if it's a substring of another
        - Keep only the longest non-overlapping form to avoid redundancy
        - Preserves order of first occurrence of non-overlapped answers
    """
    if not answers or len(answers) <= 1:
        return answers
    
    # Sort by length descending to process longer strings first
    sorted_answers = sorted(set(answers), key=len, reverse=True)
    kept = []
    
    for answer in sorted_answers:
        answer_lower = answer.lower().strip()
        
        # Check if this answer is substring of any already kept answer
        is_substring = any(answer_lower in k.lower() for k in kept)
        
        if not is_substring:
            kept.append(answer)
            logger.debug(f"Kept answer: '{answer}' (overlapping check)")
        else:
            logger.debug(f"Removed overlapping: '{answer}' (substring of kept answer)")
    
    # Restore original order
    result = [ans for ans in answers if ans in kept]
    logger.debug(f"After overlapping removal: {len(answers)} → {len(result)} answers")
    return result


def filter_answers(candidates: List[str]) -> List[str]:
    """
    Filter and normalize answer candidates based on quality rules.
    
    Removes trivial, duplicate, or low-quality answers to ensure
    meaningful answers suitable for question generation.
    
    Args:
        candidates: List of candidate answer strings
        
    Returns:
        Filtered and normalized list of high-quality answers
        
    Filtering Rules:
        1. Remove duplicates (case-insensitive)
        2. Remove very short answers (< MIN_ANSWER_LENGTH chars)
        3. Remove generic/trivial words
        4. Remove purely numeric values (except years, percentages)
        5. Remove purely punctuation
        6. Remove answers longer than MAX_ANSWER_LENGTH
        7. Normalize whitespace
        
    Example:
        >>> candidates = ["the", "Apple Inc", "Apple Inc", "123", "2022"]
        >>> filtered = filter_answers(candidates)
        >>> print(filtered)
        ["Apple Inc", "2022"]
    """
    if not candidates:
        return []
    
    filtered = []
    seen = set()  # Track normalized versions to avoid duplicates
    
    for answer in candidates:
        # Normalize whitespace
        answer = ' '.join(answer.split())
        normalized = answer.lower()
        
        # Skip if already seen (case-insensitive)
        if normalized in seen:
            logger.debug(f"Skipping duplicate: '{answer}'")
            continue
        
        # Rule 1: Length checks
        if len(answer) < MIN_ANSWER_LENGTH:
            logger.debug(f"Skipping too short: '{answer}' ({len(answer)} < {MIN_ANSWER_LENGTH})")
            continue
        
        if len(answer) > MAX_ANSWER_LENGTH:
            logger.debug(f"Skipping too long: '{answer}' ({len(answer)} > {MAX_ANSWER_LENGTH})")
            continue
        
        # Rule 2: Not purely punctuation
        if not answer.strip(string.punctuation):
            logger.debug(f"Skipping pure punctuation: '{answer}'")
            continue
        
        # Rule 3: Skip generic words
        if normalized in GENERIC_WORDS:
            logger.debug(f"Skipping generic word: '{answer}'")
            continue
        
        # Rule 4: Handle numeric values
        if NUMBER_PATTERN.match(normalized):
            # Only keep years or percentages
            if not (YEAR_PATTERN.match(answer) or PERCENT_PATTERN.match(normalized)):
                logger.debug(f"Skipping generic number: '{answer}'")
                continue
        
        # Rule 5: Check word count
        word_count = len(answer.split())
        if word_count < MIN_ANSWER_WORDS:
            logger.debug(f"Skipping too few words: '{answer}'")
            continue
        
        # All checks passed
        filtered.append(answer)
        seen.add(normalized)
    
    logger.debug(f"Filtered {len(candidates)} → {len(filtered)} candidates")
    return filtered


def classify_answer_type(answer: str, entities_dict: Optional[Dict[str, str]] = None) -> str:
    """
    Classify the type of an answer (PERSON, LOCATION, DATE, etc.).
    
    Uses NER labels when available, otherwise applies heuristic rules
    to categorize the answer.
    
    Args:
        answer: Answer text to classify
        entities_dict: Optional dict mapping entity text → NER label
                      (for quick lookup if entity is pre-extracted)
        
    Returns:
        Answer type: "PERSON", "LOCATION", "ORGANIZATION", "DATE", 
                    "CONCEPT", "NUMBER", or "OTHER"
        
    Example:
        >>> classify_answer_type("Albert Einstein")
        "PERSON"
        
        >>> classify_answer_type("machine learning")
        "CONCEPT"
        
        >>> classify_answer_type("1956")
        "DATE"
    """
    # Check if answer is in entities dict with known label
    if entities_dict and any(answer.lower() == ent.lower() for ent in entities_dict):
        ner_label = entities_dict[answer]
        
        # Map spaCy NER labels to our types
        if ner_label == "PERSON":
            return "PERSON"
        elif ner_label in ("GPE", "FAC", "LOC"):
            return "LOCATION"
        elif ner_label == "ORG":
            return "ORGANIZATION"
        elif ner_label in ("DATE", "TIME"):
            return "DATE"
    
    # Heuristic-based classification
    answer_lower = answer.lower()
    
    # Check for date patterns
    if YEAR_PATTERN.search(answer):
        return "DATE"
    if DATE_PATTERN.search(answer):
        return "DATE"
    
    # Check for numeric patterns
    if PERCENT_PATTERN.match(answer_lower):
        return "NUMBER"
    if NUMBER_PATTERN.match(answer_lower):
        return "NUMBER"
    
    # Check for location keywords
    location_keywords = {'city', 'country', 'region', 'state', 'province', 'nation', 'ocean', 'river', 'mountain'}
    if any(keyword in answer_lower for keyword in location_keywords):
        return "LOCATION"
    
    # Check for organization keywords
    org_keywords = {'company', 'corporation', 'university', 'institute', 'organization', 'agency', 'bank', 'hospital'}
    if any(keyword in answer_lower for keyword in org_keywords):
        return "ORGANIZATION"
    
    # Check for person indicators (title + name pattern)
    person_titles = {'mr', 'ms', 'dr', 'prof', 'sir', 'lady', 'king', 'queen', 'president', 'minister'}
    words = answer_lower.split()
    if words and words[0] in person_titles:
        return "PERSON"
    
    # Multi-word phrases are likely CONCEPTS
    if len(answer.split()) >= 2:
        return "CONCEPT"
    
    # Default to OTHER
    return "OTHER"


def score_answer(answer: str, 
                 sentence: str, 
                 entities_dict: Dict[str, str],
                 position: int = 0, 
                 frequency: int = 1,
                 source: str = "NOUN_PHRASE",
                 answer_position_in_sentence: int = 0) -> float:
    """
    Score the importance of an answer based on multiple heuristics.
    
    Combines several signals to compute an importance score [0, 1]:
    - Answer length (longer = more informative)
    - Whether it's a named entity (entities more important)
    - Position in document (early = more important)
    - Frequency (repeated answers more important)
    - Position within sentence (earlier = more important) [IMPROVEMENT 5]
    - Source priority (NER answers get slight boost) [IMPROVEMENT 4]
    
    IMPROVEMENT 1: Accepts entities_dict instead of recomputing NER.
    Eliminates redundant spaCy processing for EVERY answer in EVERY sentence.
    
    Args:
        answer: Answer text to score
        sentence: Original sentence containing the answer
        entities_dict: Dict[entity_text → NER_label] (pre-computed once per sentence)
        position: Position of answer in document (0 = first)
        frequency: How many times this answer appears across all sentences
        source: Source of answer ("NER" or "NOUN_PHRASE") - NER gets boost
        answer_position_in_sentence: Character position where answer starts in sentence
        
    Returns:
        Importance score in range [0, 1]
        
    Scoring Formula:
        score = length_score * 0.3 + 
                ner_score * 0.4 + 
                position_score * 0.15 +
                position_in_sentence_score * 0.15 +
                frequency_score * 0.1 +
                source_boost
        
    Example:
        >>> entities_dict = {"Albert Einstein": "PERSON"}
        >>> score = score_answer("Albert Einstein", 
        ...                      "Albert Einstein was born...",
        ...                      entities_dict, 0, 2, "NER", 0)
        >>> print(f"{score:.3f}")  # Will be higher due to frequency=2 and NER source
        0.87
    """
    # Component 1: Length score (30% weight)
    # Longer answers are more informative, but avoid extremes
    answer_length = len(answer)
    if answer_length < 10:
        length_score = 0.3 + (answer_length / 10) * 0.4
    elif answer_length < 50:
        length_score = 0.7 + ((answer_length - 10) / 40) * 0.3
    else:
        length_score = 1.0  # Cap at 1.0
    
    # Component 2: NER presence (40% weight - HIGHEST)
    # IMPROVEMENT 1: Check pre-computed entities_dict instead of recomputing NER
    # This is the CRITICAL FIX - eliminates redundant spaCy calls
    ner_score = 0.7 if any(answer.lower() == ent.lower() for ent in entities_dict) else 0.3
    
    # Component 3: Position in document score (15% weight, reduced from 20%)
    # Earlier answers/sentences more important
    position_score = max(0.3, 1.0 - (position / 1000))
    
    # IMPROVEMENT 5: Component 4 - Position within sentence (15% weight)
    # Answers appearing earlier in sentence are usually more prominent
    # Formula: 1.0 at start, linearly decays toward 0.3 at end
    sentence_length = len(sentence)
    if sentence_length > 0:
        position_in_sent_score = max(0.3, 1.0 - (answer_position_in_sentence / sentence_length))
    else:
        position_in_sent_score = 0.7
    
    # Component 5: Frequency score (10% weight)
    # IMPROVEMENT 2: Now uses actual frequency from global frequency_map
    # # (instead of always being 1)
    # frequency_score = min(1.0, 0.3 + (frequency * 0.2))
    frequency_score = min(1.0, 0.3 + (min(frequency, 3) * 0.2)) # Cap frequency influence at 3 occurrences for diminishing returns

    # IMPROVEMENT 4: Component 6 - Source priority boost
    # NER-sourced answers get slight advantage since they're pre-extracted entities
    source_boost = NER_SOURCE_BOOST if source == "NER" else 0.0
    
    # Weighted combination (note: base weights sum to 1.0)
    score = (
        length_score * 0.3 +
        ner_score * 0.4 +
        position_score * 0.15 +
        position_in_sent_score * 0.15 +
        frequency_score * 0.1
    )
    
    # Apply source boost (small addition, doesn't break normalization)
    score = score + source_boost
    
    # Clamp to [0, 1]
    score = min(max(score, 0.0), 1.0)
    
    logger.debug(f"Answer score for '{answer}': {score:.4f} "
                f"(len={length_score:.2f}, ner={ner_score:.2f}, pos={position_score:.2f}, "
                f"sent_pos={position_in_sent_score:.2f}, freq={frequency_score:.2f}, "
                f"source_boost={source_boost:.2f})")
    
    return score


def extract_answers(sentences: List[str],
                   max_answers_per_sentence: int = DEFAULT_MAX_ANSWERS_PER_SENTENCE,
                   log_level: int = DEFAULT_LOG_LEVEL) -> Dict[str, List[Dict]]:
    """
    Main answer extraction pipeline.
    
    For each sentence, extracts meaningful answer spans using a hybrid approach:
    1. Named Entity Recognition (NER)
    2. Noun Phrase Extraction
    3. Candidate combination and deduplication
    4. Quality filtering
    5. Importance scoring
    6. Answer type classification
    7. Selection of top-N answers per sentence
    
    Args:
        sentences: List of sentences (from Module 3)
        max_answers_per_sentence: Maximum answers to extract per sentence (default: 3)
        log_level: Logging verbosity (default: INFO)
        
    Returns:
        Dictionary: {sentence → List[answer_dicts]}
        Each answer_dict contains:
        {
            "answer": "answer text",
            "type": "PERSON|LOCATION|ORGANIZATION|DATE|CONCEPT|NUMBER|OTHER",
            "score": 0.0-1.0,
            "source": "NER|NOUN_PHRASE",  # Extraction method
        }
        
    Example:
        >>> sentences = ["Albert Einstein was born in Germany.", "He developed relativity."]
        >>> results = extract_answers(sentences, max_answers_per_sentence=2)
        >>> for sent, answers in results.items():
        ...     print(f"{sent}")
        ...     for ans in answers:
        ...         print(f"  - {ans['answer']} ({ans['type']}, score={ans['score']:.3f})")
        Albert Einstein was born in Germany.
          - Albert Einstein (PERSON, score=0.850)
          - Germany (LOCATION, score=0.720)
    """
    # Configure logging
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    logger.setLevel(log_level)
    logger.info(f"Starting answer extraction pipeline ({len(sentences)} sentences)")
    
    if not sentences:
        logger.warning("Empty sentence list provided")
        return {}
    
    if not SPACY_AVAILABLE:
        logger.error("spaCy not available - answer extraction disabled")
        return {}
    
    if not sentences:
        logger.warning("Empty sentence list provided")
        return {}
    
    if not SPACY_AVAILABLE:
        logger.error("spaCy not available - answer extraction disabled")
        return {}
    
    # IMPROVEMENT 1: Compute frequency map globally across ALL sentences BEFORE scoring
    # This enables TRUE frequency-based scoring instead of always using frequency=1
    logger.debug("Computing answer frequencies across all sentences...")
    from collections import Counter
    
    all_candidates = []
    for sentence in sentences:
        ner_entities = extract_named_entities(sentence)
        ner_texts = [ent[0] for ent in ner_entities]
        noun_phrases = extract_noun_phrases(sentence)
        candidates = ner_texts + noun_phrases
        filtered = filter_answers(candidates)
        all_candidates.extend(filtered)
    
    frequency_map = Counter(all_candidates)
    logger.debug(f"Built frequency map for {len(frequency_map)} unique answers")
    
    results = {}
    total_answers_extracted = 0
    
    for sent_idx, sentence in enumerate(sentences):
        logger.debug(f"Processing sentence {sent_idx + 1}/{len(sentences)}: {sentence[:60]}...")
        
        # OPTIMIZATION: Extract NER once, reuse for all answers in this sentence
        # This avoids redundant spaCy processing
        ner_entities = extract_named_entities(sentence)
        ner_texts = [ent[0] for ent in ner_entities]
        ner_dict = {ent[0]: ent[1] for ent in ner_entities}  # Dict for quick lookup
        
        # Step 2: Extract noun phrases
        noun_phrases = extract_noun_phrases(sentence)
        
        # Step 3: Combine candidates
        candidates = ner_texts + noun_phrases
        logger.debug(f"Combined candidates: {len(ner_texts)} NER + {len(noun_phrases)} noun phrases = {len(candidates)} total")
        
        # Step 4: Filter answers
        filtered_answers = filter_answers(candidates)
        
        # IMPROVEMENT 3: Remove overlapping answers (keep longer forms)
        filtered_answers = remove_overlapping_answers(filtered_answers)
        
        if not filtered_answers:
            logger.debug(f"No valid answers extracted from: {sentence[:60]}...")
            results[sentence] = []
            continue
        
        # Step 5: Score and create answer objects
        answer_objects = []
        
        for answer_text in filtered_answers:
            # Determine source (NER vs NOUN_PHRASE)
            source = "NER" if answer_text in ner_texts else "NOUN_PHRASE"
            
            # IMPROVEMENT 2: Use TRUE frequency from global frequency_map
            frequency = frequency_map.get(answer_text, 1)
            
            # IMPROVEMENT 5: Find answer position within sentence
            answer_position_in_sentence = sentence.lower().find(answer_text.lower())
            if answer_position_in_sentence == -1:
                answer_position_in_sentence = 0  # Fallback if not found
            
            # Score the answer with all improvements applied
            # IMPROVEMENT 1: Pass pre-computed entities_dict (avoids redundant NER calls)
            # IMPROVEMENT 4: Pass source for NER boost
            # IMPROVEMENT 5: Pass answer position in sentence
            score = score_answer(
                answer_text, 
                sentence, 
                ner_dict,
                sent_idx, 
                frequency,
                source,
                answer_position_in_sentence
            )
            
            # IMPROVEMENT 6: Filter by minimum score threshold
            if score < MIN_SCORE_THRESHOLD:
                logger.debug(f"Skipping below threshold: '{answer_text}' (score={score:.3f} < {MIN_SCORE_THRESHOLD})")
                continue
            
            # Classify answer type
            answer_type = classify_answer_type(answer_text, ner_dict)
            
            # Create answer object
            answer_obj = {
                "answer": answer_text,
                "type": answer_type,
                "score": score,
                "source": source,
            }
            
            answer_objects.append(answer_obj)
        
        # Step 6: Sort by score and limit to top-N
        answer_objects.sort(key=lambda x: x['score'], reverse=True)
        top_answers = answer_objects[:max_answers_per_sentence]
        
        results[sentence] = top_answers
        total_answers_extracted += len(top_answers)
        
        logger.debug(f"Extracted {len(top_answers)} top answers from sentence")
    
    logger.info(f"Answer extraction complete: {total_answers_extracted} answers from {len(sentences)} sentences")
    return results


def get_answer_stats(results: Dict[str, List[Dict]]) -> Dict:
    """
    Get statistics about extracted answers.
    
    Args:
        results: Output from extract_answers()
        
    Returns:
        Dictionary with statistics:
        - total_sentences: Number of sentences processed
        - total_answers: Total answers extracted
        - avg_answers_per_sentence: Average answers per sentence
        - answer_types_distribution: Count by type
        - avg_answer_length: Average answer text length
        - source_distribution: Count by extraction method
    """
    if not results:
        return {
            'total_sentences': 0,
            'total_answers': 0,
            'avg_answers_per_sentence': 0.0,
            'answer_types_distribution': {},
            'avg_answer_length': 0.0,
            'source_distribution': {},
        }
    
    total_answers = sum(len(answers) for answers in results.values())
    
    # Count by type
    type_counts = {}
    source_counts = {}
    lengths = []
    
    for answers in results.values():
        for ans in answers:
            ans_type = ans['type']
            type_counts[ans_type] = type_counts.get(ans_type, 0) + 1
            
            source = ans['source']
            source_counts[source] = source_counts.get(source, 0) + 1
            
            lengths.append(len(ans['answer']))
    
    return {
        'total_sentences': len(results),
        'total_answers': total_answers,
        'avg_answers_per_sentence': total_answers / len(results) if results else 0.0,
        'answer_types_distribution': type_counts,
        'avg_answer_length': sum(lengths) / len(lengths) if lengths else 0.0,
        'source_distribution': source_counts,
    }


# ============================================================================
# TEST SUITE
# ============================================================================

def test_answer_extractor():
    """
    Comprehensive test suite for answer extraction module.
    """
    print("="*80)
    print("ANSWER EXTRACTION MODULE - TEST SUITE")
    print("="*80 + "\n")
    
    # Test 1: Named entities
    print("[TEST 1] Named Entity Extraction...")
    sent1 = "Albert Einstein was born in Germany in 1879."
    
    try:
        entities = extract_named_entities(sent1)
        print(f"[OK] Sentence: {sent1}")
        print(f"     Extracted {len(entities)} entities:")
        for ent, label in entities:
            print(f"       - '{ent}' ({label})")
        print()
    except Exception as e:
        print(f"[ERROR] {e}\n")
    
    # Test 2: Noun phrases
    print("[TEST 2] Noun Phrase Extraction...")
    sent2 = "The quick brown fox jumps over the lazy dog in the forest."
    
    try:
        phrases = extract_noun_phrases(sent2)
        print(f"[OK] Sentence: {sent2}")
        print(f"     Extracted {len(phrases)} noun phrases:")
        for phrase in phrases:
            print(f"       - '{phrase}'")
        print()
    except Exception as e:
        print(f"[ERROR] {e}\n")
    
    # Test 3: Filtering
    print("[TEST 3] Answer Filtering...")
    candidates = ["the", "machine learning", "machine learning", "123", "2022", "it", "Python programming"]
    
    try:
        filtered = filter_answers(candidates)
        print(f"[OK] Input candidates: {candidates}")
        print(f"     After filtering: {filtered}\n")
    except Exception as e:
        print(f"[ERROR] {e}\n")
    
    # Test 4: Answer type classification
    print("[TEST 4] Answer Type Classification...")
    test_answers = [
        "Albert Einstein",
        "Paris",
        "Apple Inc",
        "2022",
        "machine learning",
    ]
    
    try:
        print(f"[OK] Answer types:")
        for ans in test_answers:
            ans_type = classify_answer_type(ans)
            print(f"     '{ans}' → {ans_type}")
        print()
    except Exception as e:
        print(f"[ERROR] {e}\n")
    
    # Test 5: Full extraction pipeline
    print("[TEST 5] Full Answer Extraction Pipeline...")
    sentences = [
        "Albert Einstein developed the theory of relativity in 1905.",
        "Machine learning is a subset of artificial intelligence.",
        "The company Google was founded by Larry Page and Sergey Brin.",
    ]
    
    try:
        results = extract_answers(sentences, max_answers_per_sentence=3, log_level=logging.WARNING)
        stats = get_answer_stats(results)
        
        print(f"[OK] Pipeline complete\n")
        
        for sent, answers in results.items():
            print(f"Sentence: {sent}")
            for i, ans in enumerate(answers, 1):
                print(f"  {i}. '{ans['answer']}' ({ans['type']}, score={ans['score']:.3f}, source={ans['source']})")
            print()
        
        print(f"Statistics:")
        print(f"  - Total sentences: {stats['total_sentences']}")
        print(f"  - Total answers: {stats['total_answers']}")
        print(f"  - Avg answers/sentence: {stats['avg_answers_per_sentence']:.1f}")
        print(f"  - Answer types: {stats['answer_types_distribution']}")
        print(f"  - Source distribution: {stats['source_distribution']}")
        print(f"  - Avg answer length: {stats['avg_answer_length']:.1f} chars\n")
    except Exception as e:
        print(f"[ERROR] {e}\n")
    
    # Test 6: Edge cases
    print("[TEST 6] Edge Cases...")
    
    # Empty list
    try:
        results = extract_answers([], log_level=logging.WARNING)
        print(f"[OK] Empty list handled: {len(results)} results")
    except Exception as e:
        print(f"[ERROR] Empty list: {e}")
    
    # Single word sentence
    try:
        results = extract_answers(["Hello"], log_level=logging.WARNING)
        print(f"[OK] Single word sentence handled: {len(results)} results")
    except Exception as e:
        print(f"[ERROR] Single word: {e}")
    
    # All stopwords/generic
    try:
        results = extract_answers(["The and or but"], log_level=logging.WARNING)
        print(f"[OK] All stopwords handled: {len(results)} results\n")
    except Exception as e:
        print(f"[ERROR] All stopwords: {e}\n")
    
    print("="*80)
    print("TEST SUITE COMPLETED")
    print("="*80 + "\n")


if __name__ == "__main__":
    test_answer_extractor()
