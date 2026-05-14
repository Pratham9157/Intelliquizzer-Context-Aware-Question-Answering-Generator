"""
MODULE 6 — Distractor Generator  (v5 — Cluster-Based Parallelism)
==================================================================
Problem with previous versions: options from completely different semantic
fields — "the scope of AI" next to "OpenAI Google DeepMind and Meta".

Root solution: ALL four options must come from the SAME semantic cluster.

Architecture:
  Step 1 — Cluster the full answer pool by type + semantic similarity
  Step 2 — Find the cluster containing the correct answer
  Step 3 — Pull the 3 best distractors from that SAME cluster
  Step 4 — If cluster too small → type-matched curated bank (domain-aware)
  Step 5 — Last resort → morphological variants

This guarantees structural parallelism because items in the same cluster
have the same type, similar length, and similar semantic content.
"""

from __future__ import annotations

import re
import random
import string
import logging
import threading
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_embed_model = None
_embed_lock  = threading.Lock()

# ─────────────────────────────────────────────────────────────────────────────
# Verb gate (same list as M4 — no span with a verb should ever be a distractor)
# ─────────────────────────────────────────────────────────────────────────────

_VERBS: Set[str] = {
    'is','are','was','were','be','been','being','have','has','had',
    'do','does','did','will','would','could','should','may','might','must','can',
    'include','includes','use','uses','aim','aims','draw','draws',
    'create','creates','enable','enables','allow','allows','become','becomes',
    'develop','develops','provide','provides','make','makes','get','gets',
    'give','gives','take','takes','help','helps','achieve','achieves',
    'refer','refers','mean','means','define','defines','describe','describes',
    'involve','involves','represent','represents','require','requires',
    'perform','performs','apply','applies','find','finds','show','shows',
    'train','trains','learn','learns','solve','solves','generate','generates',
    'detect','detects','predict','predicts','classify','classifies',
}

def _has_verb(text: str) -> bool:
    return any(w.strip(string.punctuation).lower() in _VERBS for w in text.split())


# ─────────────────────────────────────────────────────────────────────────────
# Type-specific curated banks  (domain-aware for CONCEPT type)
# ─────────────────────────────────────────────────────────────────────────────

_TYPED_BANKS: Dict[str, List[str]] = {
    'PERSON': [
        'Alan Turing', 'Ada Lovelace', 'John McCarthy', 'Marvin Minsky',
        'Geoffrey Hinton', 'Yann LeCun', 'Yoshua Bengio', 'Andrew Ng',
        'Demis Hassabis', 'Sam Altman', 'Fei-Fei Li', 'Claude Shannon',
        'Norbert Wiener', 'Herbert Simon', 'Allen Newell', 'Frank Rosenblatt',
    ],
    'ORGANIZATION': [
        'Google DeepMind', 'OpenAI', 'Anthropic', 'Meta AI Research',
        'Microsoft Research', 'IBM Research', 'NVIDIA', 'Hugging Face',
        'MIT CSAIL', 'Stanford HAI', 'Carnegie Mellon University',
        'Berkeley AI Research', 'Allen Institute for AI', 'Turing Institute',
    ],
    'LOCATION': [
        'Silicon Valley', 'United Kingdom', 'Japan', 'Germany',
        'Canada', 'France', 'China', 'Australia', 'Israel', 'Singapore',
        'Netherlands', 'South Korea', 'India', 'Sweden',
    ],
    'DATE': [
        '1956', '1969', '1986', '1997', '2006', '2012', '2017',
        '2019', '2021', '2023', 'the early 1990s', 'the mid-2010s',
        'the 1980s', 'the 1970s', 'the late 2000s',
    ],
    'NUMBER': [
        '42%', '17.3%', '63%', '88%', '2.5 billion', '500 million',
        'approximately 100', 'over 1,000', 'fewer than 50', 'three-fold',
    ],
}

# Domain-detected concept banks
_CONCEPT_BANKS: Dict[str, List[str]] = {
    'ai_general': [
        'symbolic AI', 'expert systems', 'knowledge representation',
        'automated planning', 'constraint satisfaction', 'heuristic search',
        'game theory', 'multi-agent systems', 'cognitive architectures',
        'natural language processing', 'computer vision', 'robotics',
        'speech recognition', 'recommender systems', 'autonomous vehicles',
    ],
    'machine_learning': [
        'supervised learning', 'unsupervised learning', 'reinforcement learning',
        'transfer learning', 'semi-supervised learning', 'self-supervised learning',
        'few-shot learning', 'zero-shot learning', 'meta-learning',
        'contrastive learning', 'curriculum learning', 'online learning',
        'active learning', 'federated learning', 'multi-task learning',
    ],
    'deep_learning': [
        'convolutional neural networks', 'recurrent neural networks',
        'transformer architecture', 'attention mechanism',
        'generative adversarial networks', 'variational autoencoders',
        'ResNet', 'BERT', 'GPT-4', 'diffusion models',
        'graph neural networks', 'long short-term memory',
        'encoder-decoder architecture', 'gated recurrent units',
    ],
    'nlp': [
        'named entity recognition', 'part-of-speech tagging',
        'dependency parsing', 'coreference resolution',
        'sentiment analysis', 'machine translation',
        'text summarisation', 'question answering',
        'word embeddings', 'language modelling',
        'text classification', 'information extraction',
        'semantic role labelling', 'discourse analysis',
    ],
    'computer_vision': [
        'object detection', 'image segmentation', 'image classification',
        'face recognition', 'optical flow', 'depth estimation',
        'image generation', 'super-resolution', 'pose estimation',
        'scene understanding', 'visual question answering', 'image captioning',
    ],
}

_DOMAIN_KW: Dict[str, Set[str]] = {
    'machine_learning': {'train', 'dataset', 'model', 'predict', 'classif', 'regress', 'feature'},
    'deep_learning':    {'neural', 'network', 'layer', 'deep', 'weight', 'gradient', 'backprop'},
    'nlp':              {'language', 'text', 'word', 'token', 'sentence', 'nlp', 'semantics', 'parsing'},
    'computer_vision':  {'image', 'pixel', 'visual', 'vision', 'detection', 'recognition', 'photo'},
    'ai_general':       {'artificial', 'intelligence', 'reasoning', 'knowledge', 'logic', 'agent'},
}


def _best_concept_bank(context: str) -> List[str]:
    ctx = context.lower()
    best, n = 'ai_general', 0
    for domain, kws in _DOMAIN_KW.items():
        score = sum(1 for kw in kws if kw in ctx)
        if score > n:
            n, best = score, domain
    return _CONCEPT_BANKS.get(best, _CONCEPT_BANKS['ai_general'])


# ─────────────────────────────────────────────────────────────────────────────
# Shape helpers for structural parallelism
# ─────────────────────────────────────────────────────────────────────────────

def _length_bucket(text: str) -> str:
    wc = len(text.split())
    if wc <= 2:  return 'short'
    if wc <= 5:  return 'medium'
    return 'long'


def _is_proper(text: str) -> bool:
    """True if text looks like a proper noun (starts uppercase, not sentence-start)."""
    words = text.split()
    return bool(words) and words[0][0].isupper() if words else False


def _has_digits(text: str) -> bool:
    return any(c.isdigit() for c in text)


def _same_shape(candidate: str, answer: str) -> bool:
    """All four options should share shape: length bucket + capitalisation + digits."""
    if _has_verb(candidate):
        return False
    lb_c, lb_a = _length_bucket(candidate), _length_bucket(answer)
    # Allow one bucket of difference
    buckets = ['short', 'medium', 'long']
    if abs(buckets.index(lb_c) - buckets.index(lb_a)) > 1:
        return False
    if _is_proper(candidate) != _is_proper(answer):
        return False
    if _has_digits(candidate) != _has_digits(answer):
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Dedup helpers
# ─────────────────────────────────────────────────────────────────────────────

def _seq_sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def _jaccard(a: str, b: str) -> float:
    s1, s2 = set(a.lower().split()), set(b.lower().split())
    if not s1 and not s2: return 1.0
    return len(s1 & s2) / len(s1 | s2)

def _is_fresh(cand: str, answer: str, existing: List[str]) -> bool:
    if cand.lower().strip() == answer.lower().strip(): return False
    if _seq_sim(cand, answer) > 0.75: return False
    for ex in existing:
        if _jaccard(cand, ex) > 0.70 or _seq_sim(cand, ex) > 0.80:
            return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Lazy embedder
# ─────────────────────────────────────────────────────────────────────────────

def _get_embedder():
    global _embed_model
    if _embed_model is not None:
        return _embed_model
    with _embed_lock:
        if _embed_model is not None:
            return _embed_model
        try:
            from sentence_transformers import SentenceTransformer
            _embed_model = SentenceTransformer(_EMBED_MODEL)
            logger.info("Distractor embedding model ready.")
        except Exception as e:
            logger.warning(f"Embedding model unavailable: {e}")
            _embed_model = None
    return _embed_model


# ─────────────────────────────────────────────────────────────────────────────
# Step 1-3 — Cluster-based selection
# ─────────────────────────────────────────────────────────────────────────────

def _cluster_distractors(
    answer: str,
    answer_type: str,
    all_answer_dicts: List[dict],
    existing: List[str],
    num: int,
) -> List[str]:
    """
    Find corpus spans in the same type-cluster as the answer, ranked by
    semantic similarity, filtered by shape match.

    Only uses NER/RULE/PATTERN sourced spans — never NP fragments.
    """
    embedder = _get_embedder()

    # Pool: same type (or a related type), clean source, verb-free
    type_compat = {answer_type}
    if answer_type == 'PERSON':       type_compat |= {'PROPER_NOUN'}
    elif answer_type == 'ORGANIZATION': type_compat |= {'PROPER_NOUN'}
    elif answer_type == 'CONCEPT':    type_compat |= {'PROPER_NOUN', 'OTHER'}

    pool = [
        obj['answer'] for obj in all_answer_dicts
        if obj.get('type') in type_compat
        and obj['answer'].lower() != answer.lower()
        and not _has_verb(obj['answer'])
        and obj.get('source') in ('NER', 'RULE', 'PATTERN', 'SUBJECT')
        and len(obj['answer']) >= 3
    ]

    if not pool:
        return []

    if embedder is None:
        # Fallback without embeddings: just use type-matched pool
        out: List[str] = []
        for span in pool:
            if _is_fresh(span, answer, existing + out) and _same_shape(span, answer):
                out.append(span)
                if len(out) >= num:
                    break
        return out

    try:
        from sklearn.metrics.pairwise import cosine_similarity as cos_sim
        ans_emb  = embedder.encode([answer])
        pool_emb = embedder.encode(pool)
        sims     = cos_sim(ans_emb, pool_emb).flatten()

        ranked = sorted(zip(pool, sims.tolist()), key=lambda x: -x[1])
        out: List[str] = []
        for span, sim in ranked:
            # Sweet spot: similar domain but different enough to be wrong
            if 0.20 <= sim <= 0.82:
                if (_is_fresh(span, answer, existing + out)
                        and _same_shape(span, answer)):
                    out.append(span)
                    if len(out) >= num:
                        break
        return out
    except Exception as e:
        logger.debug(f"Cluster distractor error: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Curated bank (type + domain aware)
# ─────────────────────────────────────────────────────────────────────────────

def _bank_distractors(
    answer: str, answer_type: str, context: str, existing: List[str], num: int,
) -> List[str]:
    if answer_type in _TYPED_BANKS:
        bank = list(_TYPED_BANKS[answer_type])
    else:
        bank = _best_concept_bank(context)

    random.shuffle(bank)
    out: List[str] = []
    for item in bank:
        if (_is_fresh(item, answer, existing + out)
                and _same_shape(item, answer)):
            out.append(item)
            if len(out) >= num:
                break

    # If shape matching is too restrictive, relax it
    if len(out) < num:
        for item in bank:
            if _is_fresh(item, answer, existing + out) and not _has_verb(item):
                out.append(item)
                if len(out) >= num:
                    break

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Morphological variants (last resort)
# ─────────────────────────────────────────────────────────────────────────────

def _morph_variants(answer: str) -> List[str]:
    # Year
    m = re.match(r'^(\d{4})$', answer.strip())
    if m:
        y = int(m.group(1))
        return [str(y + d) for d in (5, 10, -5, 15)]

    # Percentage
    m = re.match(r'^(\d+(?:\.\d+)?)\s*%$', answer.strip())
    if m:
        v = float(m.group(1))
        return [f"{int(max(0, v + d))}%" for d in (10, -10, 25, -25)]

    # Word swap
    words = answer.split()
    _swaps = {
        'artificial':   ['general', 'narrow', 'human', 'digital'],
        'general':      ['narrow', 'artificial', 'broad', 'specific'],
        'machine':      ['deep', 'statistical', 'classical', 'online'],
        'learning':     ['reasoning', 'planning', 'inference', 'optimisation'],
        'intelligence': ['computing', 'automation', 'cognition', 'processing'],
        'network':      ['model', 'system', 'architecture', 'framework'],
        'deep':         ['shallow', 'recurrent', 'convolutional', 'dense'],
        'neural':       ['symbolic', 'probabilistic', 'graph-based', 'hybrid'],
        'natural':      ['computational', 'formal', 'semantic', 'pragmatic'],
    }
    variants: List[str] = []
    for i, w in enumerate(words):
        if w.lower() in _swaps:
            for rep in _swaps[w.lower()]:
                cand = ' '.join(words[:i] + [rep] + words[i+1:])
                if _is_fresh(cand, answer, variants):
                    variants.append(cand)
                if len(variants) >= 4:
                    return variants

    last = words[-1].lower() if words else 'approach'
    return [f"traditional {last}", f"alternative {last}", f"classical {last}"]


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def generate_distractors(
    answer: str,
    answer_type: str,
    all_answer_dicts: List[dict],
    num: int = 3,
    context: str = '',
) -> List[str]:
    collected: List[str] = []

    # 1. Cluster-based: same type, semantically close corpus spans
    for c in _cluster_distractors(answer, answer_type, all_answer_dicts, collected, num):
        if _is_fresh(c, answer, collected):
            collected.append(c)
        if len(collected) >= num:
            return collected[:num]

    # 2. Curated bank (type + domain matched)
    for c in _bank_distractors(answer, answer_type, context, collected, num - len(collected)):
        if _is_fresh(c, answer, collected):
            collected.append(c)
        if len(collected) >= num:
            return collected[:num]

    # 3. Morphological variants
    for c in _morph_variants(answer):
        if _is_fresh(c, answer, collected):
            collected.append(c)
        if len(collected) >= num:
            return collected[:num]

    return collected[:num]
