"""
MODULE 2 — Preprocessor  (v3 — Slide-Aware)
=============================================
v3 adds slide-deck awareness:

  Problem (from PDF lecture slides):
    • Entire bullet-point blocks were kept as one "sentence":
      "What about text? • Can we use CNN? • Deep learning has achieved..."
    • Math/diagram content passed through:
      "𝐽 = 𝑖 𝑖 ෍ ℒ( 𝑦 ො , 𝑦 ) 𝑚 𝑖=1 Fully Connected Cat /dog/.."
    • Slide headings became answers:
      "Creating Vectors from Text •"

  Fixes:
    1. _expand_bullets()    — split on • to extract individual bullet sentences
    2. _is_math_heavy()     — filter Unicode math / diagram / dimension content
    3. _is_slide_heading()  — filter short heading-only lines (no verb, title-case)
    4. _clean_bullet_text() — strip leading bullet chars and numbering artifacts
    5. _is_numbered_fragment() — filter "BoW, which stands for Bag of Words 2."
"""

from __future__ import annotations

import re
import unicodedata
import logging
from typing import List

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_ABBREVS = {
    'mr','mrs','ms','dr','prof','sr','jr','rev','gen','pvt','sgt','cpl',
    'cpt','lt','col','maj','brig','capt','vs','etc','approx','dept','est',
    'govt','univ','corp','inc','ltd','co','st','ave','blvd','rd',
    'jan','feb','mar','apr','jun','jul','aug','sep','oct','nov','dec',
    'e.g','i.e','cf','al','fig','vol','no','pp','ref',
}

MIN_WORDS       = 6
MAX_WORDS       = 70
MIN_ALPHA_RATIO = 0.52
MIN_UNIQUE      = 4

# Unicode blocks that indicate math/special content
_MATH_CATEGORIES = {'Sm', 'So', 'Mn'}   # Math symbol, Other symbol, Nonspacing mark

# Common English verbs — a sentence without any of these is likely a heading
_COMMON_VERBS = {
    'is','are','was','were','be','been','being','have','has','had',
    'do','does','did','will','would','could','should','can','may','might',
    'include','use','uses','create','creates','achieve','achieve',
    'appear','allow','help','make','take','give','get','put','set',
    'stand','mean','refer','define','describe','show','know','think',
    'become','develop','provide','process','represent','require','perform',
    'classify','generate','detect','predict','train','learn','solve',
    'capture','convert','contain','result','avoid','understand','need',
    'break','build','call','come','differ','focus','improve','involve',
    'measure','occur','place','reduce','relate','retain','see','tend',
    'understand','work','turn','run','find','tell','read','put',
}


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Unicode & encoding normalisation
# ─────────────────────────────────────────────────────────────────────────────

def _fix_encoding(text: str) -> str:
    text = unicodedata.normalize('NFC', text)
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2014', ' — ').replace('\u2013', '–')
    text = text.replace('\u00a0', ' ')
    return text


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Noise removal
# ─────────────────────────────────────────────────────────────────────────────

_BRACKET_RE   = re.compile(r'\[[^\]]{0,60}\]')
_PAREN_NUM_RE = re.compile(r'\(\d+\)')
_SLIDE_TAG_RE = re.compile(r'\[Slide \d+[^\]]*\]')
_HEADING_RE   = re.compile(r'^#+\s+', re.M)
_URL_RE       = re.compile(r'https?://\S+')
# Dimension strings like "13×13 ×256 6×6 ×256 9216 4096"
_DIMENSION_RE = re.compile(r'\d+\s*[×x]\s*\d+[\d\s×x×]+')
# AlexNet-style strings
_ARCH_RE      = re.compile(r'\b(?:MAX-POOL|POOL|ReLU|softmax|FC)\b', re.I)


def _strip_noise(text: str) -> str:
    text = _SLIDE_TAG_RE.sub(' ', text)
    text = _URL_RE.sub('', text)
    text = _HEADING_RE.sub(' ', text)
    text = _BRACKET_RE.sub('', text)
    text = _PAREN_NUM_RE.sub('', text)
    text = _DIMENSION_RE.sub(' ', text)
    text = re.sub(r'\.{3,}', '.', text)
    text = re.sub(r'-{2,}', ' ', text)
    text = re.sub(r':(?!\s)', ': ', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{2,}', '\n\n', text)
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Step 2b — Bullet expansion  (KEY FIX FOR SLIDES)
# ─────────────────────────────────────────────────────────────────────────────

_BULLET_CHARS = '•·▪▸◦‣⁃●○◉▶►★☆–—'
_BULLET_SPLIT_RE = re.compile(
    r'(?<=[^•])\s*[•·▪▸◦‣⁃●○◉▶►]\s*'
)

def _clean_bullet_text(text: str) -> str:
    """Strip leading bullet chars, list numbers (1. 2. a.) from a fragment."""
    text = text.strip()
    # Strip leading bullet chars
    text = text.lstrip(_BULLET_CHARS + ' ')
    # Strip leading list numbers: "1. ", "a) ", "(2) "
    text = re.sub(r'^[\(\[]?[a-zA-Z0-9]{1,2}[\.\)\]]\s+', '', text)
    # Strip trailing bullet or number artifacts like "2." at end
    text = re.sub(r'\s+\d+\.\s*$', '', text)
    text = text.strip()
    return text


def _expand_bullets(text: str) -> List[str]:
    """
    Split a bullet-containing block into individual sentences.
    "Heading • Point A • Point B • Point C"
    → ["Heading", "Point A", "Point B", "Point C"]

    If no bullets, return the text as-is.
    """
    # Does this block have bullets?
    if not any(c in text for c in _BULLET_CHARS):
        return [text]

    # Split on bullet markers
    parts = _BULLET_SPLIT_RE.split(text)
    cleaned: List[str] = []
    for part in parts:
        part = _clean_bullet_text(part)
        if part:
            cleaned.append(part)
    return cleaned if cleaned else [text]


def _expand_all_bullets(paragraphs: List[str]) -> List[str]:
    """Apply bullet expansion to every paragraph block."""
    result: List[str] = []
    for para in paragraphs:
        result.extend(_expand_bullets(para))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Sentence boundary detection
# ─────────────────────────────────────────────────────────────────────────────

def _protect_abbreviations(text: str) -> str:
    for abbrev in _ABBREVS:
        pattern = re.compile(r'\b' + re.escape(abbrev) + r'\.', re.IGNORECASE)
        text = pattern.sub(abbrev.rstrip('.') + '<DOT>', text)
    text = re.sub(r'\b([A-Z])\.\s', r'\1<DOT> ', text)
    return text


def _restore_abbreviations(text: str) -> str:
    return text.replace('<DOT>', '.')


def sent_tokenize(text: str) -> List[str]:
    paragraphs = re.split(r'\n\n+', text)
    sentences: List[str] = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        # First expand any bullets in this paragraph
        chunks = _expand_bullets(para)
        for chunk in chunks:
            chunk = _protect_abbreviations(chunk)
            parts = re.split(r'(?<=[.!?])\s+(?=[A-Z\"\(\[])', chunk)
            for part in parts:
                part = _restore_abbreviations(part).strip()
                part = _clean_bullet_text(part)
                if part:
                    sentences.append(part)
    return sentences


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Quality filtering
# ─────────────────────────────────────────────────────────────────────────────

def _alpha_ratio(text: str) -> float:
    letters = sum(c.isalpha() for c in text)
    return letters / max(len(text), 1)


def _math_char_ratio(text: str) -> float:
    """Fraction of characters that are Unicode math/symbol/nonspacing-mark."""
    math_count = 0
    for ch in text:
        try:
            cat = unicodedata.category(ch)
        except Exception:
            cat = ''
        if cat in _MATH_CATEGORIES:
            math_count += 1
        # Also count Unicode math block chars (U+1D400–U+1D7FF)
        elif 0x1D400 <= ord(ch) <= 0x1D7FF:
            math_count += 1
        # Sinhala script (was used for math in some PDFs)
        elif 0x0D80 <= ord(ch) <= 0x0DFF:
            math_count += 1
    return math_count / max(len(text), 1)


def _is_math_heavy(text: str) -> bool:
    """True if the sentence contains significant math/Unicode content."""
    if _math_char_ratio(text) > 0.015:    # Lowered from 0.04
        return True
    # Any single Unicode math char (𝐽, 𝑥, 𝑦, ŷ, etc.)
    for ch in text:
        cp = ord(ch)
        if 0x1D400 <= cp <= 0x1D7FF:      # Mathematical Alphanumeric Symbols block
            return True
        if 0x0D80 <= cp <= 0x0DFF:        # Sinhala (appears in some slide PDFs)
            return True
        if ch in 'ŷĉ𝑥𝑦𝑧𝑎𝑏𝑐𝑤θφψ∑∏∫∂∇≈≤≥≠∈∉⊂⊃∪∩':
            return True
    # Equation-like patterns
    if re.search(r'[\d]+\s*[=+\-×÷/]\s*[\d]+', text):
        eq_density = len(re.findall(r'[=+×÷]', text)) / max(len(text.split()), 1)
        if eq_density > 0.3:
            return True
    return False


def _is_slide_heading(text: str) -> bool:
    """
    True if the text looks like a slide heading rather than a proper sentence.
    Handles both short headings AND long comma-joined title strings.
    """
    words = text.split()
    words_lower = [w.lower().strip('.,;:!?\"') for w in words]
    has_verb = any(w in _COMMON_VERBS for w in words_lower)

    # Short headings with no verb → definitely a heading
    if len(words) <= 15 and not has_verb:
        capitalised = sum(1 for w in words if w and w[0].isupper())
        if capitalised / max(len(words), 1) > 0.45:
            return True

    # Long comma-joined title strings: "Text Processing, CNN for Text, Intro to..."
    # Pattern: mostly Title-Cased words with commas, no sentence-ending period
    if len(words) > 8 and not text.rstrip().endswith(('.', '!', '?')):
        capitalised = sum(1 for w in words if w and w[0].isupper())
        comma_count  = text.count(',')
        if (capitalised / max(len(words), 1) > 0.55 and comma_count >= 2
                and not has_verb):
            return True

    # Slide title with a quoted phrase glued on: heading + opening quote
    if not has_verb and '"' in text and len(words) <= 12:
        return True

    # Ends with "]" (citation ref without surrounding sentence)
    if text.rstrip().endswith(']') and not has_verb:
        return True

    return False


def _is_numbered_fragment(text: str) -> bool:
    """
    True for fragments that are numbered list continuations:
    "BoW, which stands for Bag of Words 2."
    "It should not result in a sparse matrix since... 2."
    """
    # Ends with a standalone number (list continuation artifact)
    if re.search(r'\s+\d+\.\s*$', text):
        return True
    # Starts with a standalone number that's a list item
    if re.match(r'^\d+\.\s+[a-z]', text):
        return True
    return False


def _is_architecture_string(text: str) -> bool:
    """Detect pure architecture/dimension notation like AlexNet layer specs."""
    words = text.split()
    # High proportion of numeric/dimension tokens
    numeric_tokens = sum(
        1 for w in words
        if re.match(r'^[\d×x×\-]+$', w) or w in {'MAX-POOL','POOL','ReLU','FC','Softmax'}
    )
    if len(words) > 0 and numeric_tokens / len(words) > 0.35:
        return True
    return False


def _is_quality_sentence(sent: str) -> bool:
    words = sent.split()
    n = len(words)

    if n < MIN_WORDS or n > MAX_WORDS:
        return False
    if _alpha_ratio(sent) < MIN_ALPHA_RATIO:
        return False
    if len(set(w.lower() for w in words)) < MIN_UNIQUE:
        return False

    # New slide-specific filters
    if _is_math_heavy(sent):
        return False
    if _is_slide_heading(sent):
        return False
    if _is_numbered_fragment(sent):
        return False
    if _is_architecture_string(sent):
        return False

    # Skip sentences that are mostly numbers
    digit_tokens = sum(1 for w in words if re.sub(r'[.,]', '', w).isdigit())
    if digit_tokens > n * 0.4:
        return False

    return True


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Deduplication
# ─────────────────────────────────────────────────────────────────────────────

def _deduplicate(sentences: List[str]) -> List[str]:
    seen_exact: set  = set()
    seen_prefix: set = set()
    out: List[str]   = []
    for s in sentences:
        key_exact  = s.lower().strip()
        key_prefix = key_exact[:70]
        if key_exact in seen_exact or key_prefix in seen_prefix:
            continue
        seen_exact.add(key_exact)
        seen_prefix.add(key_prefix)
        out.append(s)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(raw_text: str) -> List[str]:
    if not raw_text or not raw_text.strip():
        return []

    text = _fix_encoding(raw_text)
    text = _strip_noise(text)
    sentences = sent_tokenize(text)
    sentences = [s for s in sentences if _is_quality_sentence(s)]
    sentences = _deduplicate(sentences)

    logger.info(f"Preprocessed {len(sentences)} sentences from {len(raw_text)} chars")
    return sentences
