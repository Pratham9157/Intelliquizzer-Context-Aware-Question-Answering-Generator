"""
MODULE 7 — MCQ Builder + Quality Filter  (v4)
==============================================
Changes from v3:
  - Post-assembly quality filter:
      • Reject questions where the stem semantically gives away the answer
        (e.g. "What is goals of AI research?" → answer "goals of AI research")
      • Reject questions where all 4 options look the same (zero distractor variety)
      • Reject trivially short questions (< 5 words)
  - Improved explanation: uses the full source sentence, not just a clause
  - Dedup now normalises punctuation before comparing
"""

from __future__ import annotations

import re
import math
import random
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Difficulty / type estimation
# ─────────────────────────────────────────────────────────────────────────────

_EASY_Q   = {'what', 'who', 'when', 'where', 'which'}
_HARD_Q   = {'why', 'in what way', 'to what extent', 'explain'}
_EASY_ANS = {'DATE', 'NUMBER', 'PERSON'}
_HARD_ANS = {'CONCEPT', 'OTHER'}


def _syllables(word: str) -> int:
    return max(1, len(re.findall(r'[aeiouAEIOU]+', word)))


def _fk_grade(text: str) -> float:
    words = text.split()
    if not words:
        return 0.0
    sentences = max(1, len(re.findall(r'[.!?]+', text)))
    syl = sum(_syllables(w) for w in words)
    return max(0.0, 0.39 * (len(words) / sentences) + 11.8 * (syl / len(words)) - 15.59)


def _estimate_difficulty(question: str, answer_type: str, q_score: float) -> str:
    grade = _fk_grade(question)
    q_low = question.lower()
    hard  = any(q_low.startswith(w) for w in _HARD_Q)
    easy  = answer_type in _EASY_ANS and any(q_low.startswith(w) for w in _EASY_Q)
    if hard or (answer_type in _HARD_ANS and grade > 12):
        return 'hard'
    if easy:
        return 'easy'
    return 'medium'


def _determine_type(question: str, answer_type: str) -> str:
    q_low = question.lower()
    if any(q_low.startswith(w) for w in ('who', 'when', 'where', 'which')):
        return 'factual'
    if any(q_low.startswith(w) for w in ('why', 'how', 'explain', 'in what')):
        return 'analytical'
    return 'conceptual'


# ─────────────────────────────────────────────────────────────────────────────
# Explanation builder
# ─────────────────────────────────────────────────────────────────────────────

def _make_explanation(answer: str, context: str) -> str:
    """Return the tightest sentence fragment that contains the answer."""
    clauses = re.split(r'(?<=[.!?])\s+|[;]', context)
    for clause in clauses:
        if answer.lower() in clause.lower() and len(clause.strip()) > 15:
            c = clause.strip().rstrip('.,;')
            return f"According to the text: {c}."
    return f"According to the text: {context.strip()}"


# ─────────────────────────────────────────────────────────────────────────────
# Quality filter helpers
# ─────────────────────────────────────────────────────────────────────────────

def _normalise(text: str) -> str:
    return re.sub(r'\W+', ' ', text.lower()).strip()


def _word_overlap(a: str, b: str) -> float:
    wa = set(_normalise(a).split())
    wb = set(_normalise(b).split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def _question_gives_away_answer(question: str, answer: str) -> bool:
    """True when the question stem already contains most of the answer."""
    # Remove the question word itself before comparing
    q_body = re.sub(r'^(what|who|when|where|which|how|why)\s+(is|are|was|were|does|do|did|has|have)?\s*',
                    '', question.lower()).rstrip('?').strip()
    return _word_overlap(q_body, answer.lower()) > 0.65


def _options_too_similar(options: Dict[str, str]) -> bool:
    """True if all four options share > 70% word overlap (no real choice)."""
    vals = list(options.values())
    for i in range(len(vals)):
        for j in range(i + 1, len(vals)):
            if _word_overlap(vals[i], vals[j]) < 0.40:
                return False   # At least one genuinely different pair exists
    return True


def _is_good_mcq(mcq: dict) -> bool:
    q = mcq.get('question', '')
    a = mcq.get('answer_text', '')
    opts = mcq.get('options', {})

    if len(q.split()) < 5:
        return False
    if not q.strip().endswith('?'):
        return False
    if _question_gives_away_answer(q, a):
        logger.debug(f"Filtered (gives away answer): {q!r}")
        return False
    if _options_too_similar(opts):
        logger.debug(f"Filtered (options too similar): {q!r}")
        return False
    # Reject if answer literally appears in the question body
    q_body = re.sub(r'^(what|who|when|where|which|how|why)\b', '', q.lower()).strip(' ?')
    if a.lower() in q_body and len(a.split()) >= 3:
        logger.debug(f"Filtered (answer in question): {q!r}")
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# MCQ assembly
# ─────────────────────────────────────────────────────────────────────────────

LETTERS = ['A', 'B', 'C', 'D']


def build_mcq(
    question: str,
    answer: str,
    distractors: List[str],
    context: str,
    answer_type: str,
    q_score: float,
    source_sentence: str = '',
    seed: Optional[int] = None,
) -> dict:
    distractors = list(distractors)[:3]
    while len(distractors) < 3:
        distractors.append(f'None of the above ({len(distractors) + 1})')

    all_options = [answer] + distractors
    rng = random.Random(seed)
    rng.shuffle(all_options)

    options        = {letter: opt for letter, opt in zip(LETTERS, all_options)}
    correct_letter = next(k for k, v in options.items() if v == answer)

    return {
        'question':        question,
        'options':         options,
        'answer':          correct_letter,
        'answer_text':     answer,
        'explanation':     _make_explanation(answer, context),
        'type':            _determine_type(question, answer_type),
        'difficulty':      _estimate_difficulty(question, answer_type, q_score),
        'score':           q_score,
        'answer_type':     answer_type,
        'source_sentence': source_sentence,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def assemble_mcqs(
    question_dict: Dict[str, List[dict]],
    all_answer_dicts: List[dict],
    max_questions: int = 30,
) -> List[dict]:
    from .distractor_generator import generate_distractors

    mcqs: List[dict]  = []
    seen_qs: set[str] = set()

    for sentence, q_objects in question_dict.items():
        for qobj in q_objects:
            if len(mcqs) >= max_questions * 2:   # Build 2× buffer for filtering
                break

            question = qobj.get('question', '').strip()
            answer   = qobj.get('answer', '').strip()
            ans_type = qobj.get('type', 'CONCEPT')
            q_score  = qobj.get('question_score', 0.5)

            if not question or not answer:
                continue

            # Normalised dedup
            q_key = _normalise(question)
            if q_key in seen_qs:
                continue
            seen_qs.add(q_key)

            distractors = generate_distractors(
                answer=answer,
                answer_type=ans_type,
                all_answer_dicts=all_answer_dicts,
                num=3,
                context=sentence,
            )

            mcq = build_mcq(
                question=question, answer=answer,
                distractors=distractors, context=sentence,
                answer_type=ans_type, q_score=q_score,
                source_sentence=sentence,
            )

            if _is_good_mcq(mcq):
                mcqs.append(mcq)

    mcqs.sort(key=lambda x: -x['score'])
    logger.info(f"Assembled {min(len(mcqs), max_questions)} MCQs "
                f"({len(mcqs)} before cap, after quality filter)")
    return mcqs[:max_questions]
