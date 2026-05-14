"""
MODULE 4 — Answer Extractor  (v5 — Pattern-First Extraction)
=============================================================
Root cause of all bad answers ("AI research include learning",
"2020s generative AI has"):  TF-IDF / n-gram extraction treated
every n-gram equally, including verb-containing clause fragments.

NEW ARCHITECTURE — three clean layers, priority order:

  Layer 1 — Pattern regex (highest quality)
    Targets specific syntactic constructions that reliably contain
    the answer:
      "such as X, Y, and Z"     → [X, Y, Z]
      "including X and Y"       → [X, Y]
      "(X)"                     → [X]  (parenthetical definitions)
      "called/known as X"       → [X]
      "from X to Y"             → [X, Y]
      "X, Y, and Z"             at end of sentence

  Layer 2 — BERT NER (dslim/bert-base-NER)
    Named entities: PERSON, ORG, LOCATION, MISC
    These are always clean — BERT won't label a verb phrase as an entity.

  Layer 3 — Subject noun extraction
    Only the SUBJECT of each sentence (tokens before the first verb).
    This avoids any verb-containing span by construction.

  Layer 4 — Rule-based dates / numbers / percentages

Hard invariant: NO answer span may contain a finite verb.
               Enforced by _contains_verb() as a final gate.
"""

from __future__ import annotations

import re
import string
import logging
import threading
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Verb list — used as a hard gate, not a splitter
# ─────────────────────────────────────────────────────────────────────────────

_VERBS: Set[str] = {
    'is','are','was','were','be','been','being',
    'have','has','had','do','does','did',
    'will','would','could','should','may','might','must','can','shall',
    'include','includes','included','use','uses','used',
    'aim','aims','aimed','draw','draws','drew','drawn',
    'create','creates','created','enable','enables','enabled',
    'allow','allows','allowed','become','becomes','became',
    'develop','develops','developed','provide','provides','provided',
    'make','makes','made','take','takes','took','give','gives','gave',
    'get','gets','got','help','helps','helped','reach','reaches',
    'achieve','achieves','refer','refers','mean','means',
    'define','defines','describe','describes','involve','involves',
    'represent','represents','require','requires','perform','performs',
    'apply','applies','find','finds','show','shows','call','calls',
    'run','runs','lead','leads','train','trains','learn','learns',
    'solve','solves','process','processes','generate','generates',
    'detect','detects','predict','predicts','classify','classifies',
    'optimize','optimizes','differ','differs','focus','focuses',
    'rely','relies','improve','improves','work','works','come','comes',
    'go','goes','see','sees','know','knows','think','thinks','say','says',
    'want','wants','need','needs','build','builds','built','set','sets',
}

_STOP: Set[str] = {
    'a','an','the','and','or','but','in','on','at','to','for','of','with',
    'by','from','as','so','yet','nor','than','that','which','who','whose',
    'when','where','why','how','although','because','since','while',
    'if','unless','until','after','before','during','through','about',
    'into','upon','over','under','above','below','between','among',
    'its','their','our','your','his','her','my',
    'this','these','those','such','both','each','every','all','any',
}

_BAD_LEAD: Set[str] = _STOP | {
    'also','just','only','even','still','already','yet','again',
    'very','quite','rather','however','therefore','thus','hence',
    'moreover','furthermore','additionally','consequently',
}

# ─────────────────────────────────────────────────────────────────────────────
# Layer 1 — Pattern extraction
# ─────────────────────────────────────────────────────────────────────────────

# Patterns that capture answer spans:
_SUCH_AS_RE   = re.compile(r'such as\s+(.+?)(?:\.|,\s+(?:and|but|or)\s+\w+|\s+–|\s+—|\s*$)', re.I)
_INCLUDING_RE = re.compile(r'including\s+(.+?)(?:\.|,\s+(?:and|but)\s+[a-z]|\s*$)', re.I)
_PAREN_RE     = re.compile(r'\(([^)]{3,60})\)')           # parenthetical "(AGI)"
_KNOWN_AS_RE  = re.compile(r'(?:known|referred to|called)\s+as\s+([^,\.;]+)', re.I)
_DASH_DEF_RE  = re.compile(r'[–—-]\s+([^–—\.,;]{5,80})')  # "- definition here"
_COLON_RE     = re.compile(r':\s+([A-Z][^\.;]{5,80})')    # ": Named thing"


def _split_list(text: str) -> List[str]:
    """Split "X, Y, and Z" → ["X", "Y", "Z"]."""
    text = re.sub(r',?\s+and\s+', ', ', text, flags=re.I)
    text = re.sub(r',?\s+or\s+',  ', ', text, flags=re.I)
    parts = [p.strip().rstrip('.,;') for p in text.split(',')]
    return [p for p in parts if len(p) >= 2]


def _pattern_extract(sentence: str) -> List[Tuple[str, str]]:
    """Returns [(span, type)] using regex patterns."""
    results: List[Tuple[str, str]] = []

    # "such as OpenAI, DeepMind and Meta" → ["OpenAI", "DeepMind", "Meta"]
    m = _SUCH_AS_RE.search(sentence)
    if m:
        for item in _split_list(m.group(1)):
            results.append((item, 'PROPER_NOUN'))

    # "including learning, reasoning, problem-solving" → each item
    m = _INCLUDING_RE.search(sentence)
    if m:
        items = _split_list(m.group(1))
        if len(items) >= 2:
            # Add both individual and combined
            for item in items[:3]:
                results.append((item, 'CONCEPT'))
            combined = ', '.join(items[:3])
            if len(combined) <= 80:
                results.append((combined, 'CONCEPT'))

    # Parenthetical "(AGI)" or "(AI)"
    for m in _PAREN_RE.finditer(sentence):
        results.append((m.group(1).strip(), 'PROPER_NOUN'))

    # "known as artificial general intelligence"
    m = _KNOWN_AS_RE.search(sentence)
    if m:
        results.append((m.group(1).strip(), 'CONCEPT'))

    # Dash definitions
    m = _DASH_DEF_RE.search(sentence)
    if m:
        results.append((m.group(1).strip(), 'CONCEPT'))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2 — BERT NER
# ─────────────────────────────────────────────────────────────────────────────

_NER_MODEL = "dslim/bert-base-NER"
_ner_pipe   = None
_ner_lock   = threading.Lock()

_NER_TYPE_MAP = {
    'PER': 'PERSON', 'ORG': 'ORGANIZATION',
    'LOC': 'LOCATION', 'MISC': 'PROPER_NOUN',
}


def _get_ner():
    global _ner_pipe
    if _ner_pipe is not None:
        return _ner_pipe
    with _ner_lock:
        if _ner_pipe is not None:
            return _ner_pipe
        try:
            from transformers import pipeline as hf_pipeline
            logger.info(f"Loading NER model '{_NER_MODEL}' ...")
            _ner_pipe = hf_pipeline(
                "ner", model=_NER_MODEL, aggregation_strategy="simple"
            )
            logger.info("NER model ready.")
        except Exception as e:
            logger.error(f"NER load failed: {e}")
            _ner_pipe = None
    return _ner_pipe


def _ner_extract(sentence: str) -> List[Tuple[str, str, float]]:
    ner = _get_ner()
    if ner is None:
        return []
    try:
        out = []
        for ent in ner(sentence):
            span  = re.sub(r'\s*##', '', ent.get('word', '')).strip()
            label = re.sub(r'^[BI]-', '', ent.get('entity_group', ent.get('entity', '')))
            score = float(ent.get('score', 0.0))
            etype = _NER_TYPE_MAP.get(label, 'PROPER_NOUN')
            span  = span.strip('.,;:!? ')
            if len(span) >= 2 and score >= 0.70:
                out.append((span, etype, score))
        return out
    except Exception as e:
        logger.debug(f"NER error: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Layer 3 — Subject noun (tokens before first verb)
# ─────────────────────────────────────────────────────────────────────────────

def _subject_extract(sentence: str) -> Optional[str]:
    """
    Extract the subject noun phrase — tokens before the first finite verb.
    E.g. "Artificial intelligence is..." → "Artificial intelligence"
    """
    tokens = sentence.split()
    subject_tokens: List[str] = []
    for tok in tokens:
        clean = tok.strip(string.punctuation).lower()
        if clean in _VERBS:
            break
        t = tok.strip(string.punctuation)
        if t:
            subject_tokens.append(t)
    if not subject_tokens:
        return None
    # Trim leading articles/stopwords
    while subject_tokens and subject_tokens[0].lower() in _BAD_LEAD:
        subject_tokens = subject_tokens[1:]
    if not subject_tokens:
        return None
    phrase = ' '.join(subject_tokens)
    return phrase if len(phrase) >= 3 else None


# ─────────────────────────────────────────────────────────────────────────────
# Layer 4 — Rule-based dates / numbers
# ─────────────────────────────────────────────────────────────────────────────

_DECADE_RE  = re.compile(r'\bthe\s+\d{4}s\b', re.I)
_YEAR_RE    = re.compile(r'\b(1[5-9]\d{2}|20[0-3]\d)\b')
_DATE_RE    = re.compile(
    r'\b(?:January|February|March|April|May|June|July|August|'
    r'September|October|November|December)(?:\s+\d{1,2})?(?:,\s*\d{4})?\b', re.I
)
_PERCENT_RE = re.compile(r'\b\d+(?:\.\d+)?\s*%')


def _rule_extract(sentence: str) -> List[Tuple[str, str]]:
    results: List[Tuple[str, str]] = []
    seen: Set[str] = set()
    for m in _DECADE_RE.finditer(sentence):
        s = m.group().strip()
        results.append((s, 'DATE')); seen.add(s)
    for m in _DATE_RE.finditer(sentence):
        s = m.group().strip()
        if s not in seen:
            results.append((s, 'DATE')); seen.add(s)
    for m in _YEAR_RE.finditer(sentence):
        s = m.group()
        if not any(s in ex for ex in seen):
            results.append((s, 'DATE')); seen.add(s)
    for m in _PERCENT_RE.finditer(sentence):
        results.append((m.group().strip(), 'NUMBER'))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Quality filtering
# ─────────────────────────────────────────────────────────────────────────────

def _contains_verb(text: str) -> bool:
    return any(w.strip(string.punctuation).lower() in _VERBS for w in text.split())


def _trim(span: str) -> str:
    words = span.strip().split()
    while words and words[0].lower() in _BAD_LEAD:
        words = words[1:]
    tail_bad = {'and','or','but','of','in','on','at','to','for','with',
                'by','from','a','an','the','also','such','as','including'}
    while words and words[-1].lower() in tail_bad:
        words = words[:-1]
    return ' '.join(words).strip('.,;:!?()-– ')


def _is_valid(span: str) -> bool:
    if not span or len(span) < 2:
        return False
    if _contains_verb(span):                       # hard gate
        return False
    words = span.split()
    if words[0].lower() in _BAD_LEAD:
        return False
    if all(w.lower() in _STOP for w in words):
        return False
    # Must have ≥1 alphabetic content word ≥ 3 chars
    if not any(w.isalpha() and len(w) >= 3 and w.lower() not in _STOP for w in words):
        return False
    # Reject obvious junk
    _junk = {'thing','way','fact','type','kind','form','part','case',
             'area','role','set','group','level','scope','use','goal',
             'task','result','effect','method','approach'}
    if len(words) == 1 and words[0].lower() in _junk:
        return False
    return True


def _remove_overlapping(candidates: List[str]) -> List[str]:
    by_len = sorted(set(candidates), key=len, reverse=True)
    kept: List[str] = []
    for c in by_len:
        if not any(c.lower() in k.lower() and c.lower() != k.lower() for k in kept):
            kept.append(c)
    return kept


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────

def _score(span: str, etype: Optional[str], ner_conf: float, sentence: str) -> float:
    s = 0.35
    if etype in ('PERSON', 'ORGANIZATION', 'LOCATION'):
        s += 0.40 * min(1.0, ner_conf)
    elif etype in ('DATE', 'NUMBER'):
        s += 0.30
    elif etype in ('PROPER_NOUN',):
        s += 0.25 * min(1.0, ner_conf)
    elif etype == 'CONCEPT':
        s += 0.20
    wc = len(span.split())
    if 2 <= wc <= 6:   s += 0.20
    elif wc == 1:      s += 0.08 if len(span) >= 5 else 0.0
    elif wc > 8:       s -= 0.10
    if span[0].isupper() and not sentence.startswith(span):
        s += 0.05
    return max(0.0, min(1.0, s))


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

MAX_PER_SENT = 3

def extract_answers(
    sentences: List[str],
    max_per_sentence: int = MAX_PER_SENT,
) -> Dict[str, List[dict]]:
    _get_ner()
    results: Dict[str, List[dict]] = {}

    for sentence in sentences:
        candidates: Dict[str, dict] = {}   # span → obj

        def _add(span: str, etype: str, source: str, ner_conf: float = 0.0):
            span = _trim(span)
            if not span or not _is_valid(span):
                return
            if span in candidates:
                return
            s = _score(span, etype, ner_conf, sentence)
            if s >= 0.25:
                candidates[span] = {
                    'answer': span, 'type': etype,
                    'score': round(s, 4), 'source': source,
                }

        # Layer 1: pattern-based
        for span, etype in _pattern_extract(sentence):
            _add(span, etype, 'PATTERN')

        # Layer 2: BERT NER
        for span, etype, conf in _ner_extract(sentence):
            _add(span, etype, 'NER', conf)

        # Layer 3: subject noun
        subj = _subject_extract(sentence)
        if subj:
            _add(subj, 'CONCEPT', 'SUBJECT')

        # Layer 4: rules
        for span, etype in _rule_extract(sentence):
            _add(span, etype, 'RULE')

        # Deduplicate overlapping spans
        clean_spans = _remove_overlapping(list(candidates.keys()))

        answer_objects = sorted(
            [candidates[s] for s in clean_spans if s in candidates],
            key=lambda x: -x['score']
        )
        results[sentence] = answer_objects[:max_per_sentence]

    total = sum(len(v) for v in results.values())
    logger.info(f"Extracted {total} answer candidates from {len(sentences)} sentences")
    return results
