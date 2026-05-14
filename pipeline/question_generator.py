"""
MODULE 5 — Question Generator  (v3 — valhalla/t5-base-qg-hl)
==============================================================
Upgraded from mrm8488/t5-base-finetuned-question-generation-ap to
valhalla/t5-base-qg-hl which uses a highlight-format input that grounds
the question more precisely to the answer span:

    Input:  "generate question: context with <hl> answer_span <hl> highlighted"
    Output: a natural question whose answer is the highlighted span

This format is trained on SQuAD, RACE, and CoQA — far more diverse than
the SQuAD-only AP model — producing much more natural, varied questions.

Falls back to rule-based questions if model is unavailable.
"""

from __future__ import annotations

import re
import logging
import threading
from typing import Dict, List

logger = logging.getLogger(__name__)

_MODEL_NAME     = "valhalla/t5-base-qg-hl"
_MAX_NEW_TOKENS = 64
_NUM_BEAMS      = 4
_QUESTION_WORDS = ('who', 'what', 'when', 'where', 'why', 'how', 'which', 'whose')

_TYPE_HINT: Dict[str, str] = {
    'PERSON':       'Who',
    'LOCATION':     'Where',
    'DATE':         'When',
    'NUMBER':       'How many',
    'ORGANIZATION': 'Which',
    'CONCEPT':      'What',
    'PROPER_NOUN':  'What',
    'OTHER':        'What',
}

# ── Lazy model loader ─────────────────────────────────────────────────────────

_model      = None
_tokenizer  = None
_model_lock = threading.Lock()


def _load_model():
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer
    with _model_lock:
        if _model is not None:
            return _model, _tokenizer
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            logger.info(f"Loading QG model '{_MODEL_NAME}' ...")
            _tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
            _model     = AutoModelForSeq2SeqLM.from_pretrained(_MODEL_NAME)
            _model.eval()
            logger.info("QG model ready.")
        except Exception as e:
            logger.error(f"Failed to load QG model: {e}")
            _model, _tokenizer = None, None
    return _model, _tokenizer


def _build_hl_input(answer: str, context: str) -> str:
    """
    Highlight the answer span within the context using <hl> tags.
    If the answer appears in the context, wrap it. Otherwise append.
    """
    # Try case-insensitive match to find the span in context
    pattern = re.compile(re.escape(answer), re.IGNORECASE)
    match   = pattern.search(context)
    if match:
        start, end = match.start(), match.end()
        highlighted = f"{context[:start]}<hl> {context[start:end]} <hl>{context[end:]}"
    else:
        # Answer not literally in context — append it
        highlighted = f"{context} <hl> {answer} <hl>"
    return f"generate question: {highlighted}"


def _generate_raw(answer: str, context: str) -> str:
    model, tokenizer = _load_model()
    if model is None or tokenizer is None:
        return ''
    import torch
    model_input = _build_hl_input(answer, context)
    inputs      = tokenizer(
        model_input,
        return_tensors="pt",
        max_length=512,
        truncation=True,
    )
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=_MAX_NEW_TOKENS,
            num_beams=_NUM_BEAMS,
            early_stopping=True,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def _clean_question(raw: str) -> str:
    q = raw.strip()
    q = re.sub(r'^question:\s*', '', q, flags=re.I)
    q = q.strip().strip('"\'')
    if q:
        q = q[0].upper() + q[1:]
    q = q.rstrip('.!?') + '?'
    q = re.sub(r'\?{2,}', '?', q)
    return q


def _is_valid(question: str, answer: str, context: str) -> bool:
    if len(question.split()) < 4:
        return False
    if not question.endswith('?'):
        return False
    if not any(question.lower().startswith(w) for w in _QUESTION_WORDS):
        return False
    # Reject if question IS basically the answer verbatim
    if answer.lower() in question.lower() and len(answer.split()) > 3:
        return False
    return True


def _fallback_question(answer: str, answer_type: str) -> str:
    t = answer_type
    if t == 'PERSON':       return f"Who is {answer}?"
    if t == 'LOCATION':     return f"Where is {answer} located?"
    if t == 'DATE':         return f"When did this occur — specifically in {answer}?"
    if t == 'NUMBER':       return f"What quantity is associated with {answer}?"
    if t == 'ORGANIZATION': return f"Which organisation is known as {answer}?"
    return f"What is {answer}?"


# ── Main class ────────────────────────────────────────────────────────────────

class QuestionGenerator:
    def __init__(self) -> None:
        _load_model()
        logger.info("QuestionGenerator initialised (valhalla/t5-base-qg-hl)")

    def generate_question(self, answer: str, context: str, answer_type: str = 'CONCEPT') -> str:
        if not answer.strip() or not context.strip():
            return ''
        try:
            raw      = _generate_raw(answer, context)
            question = _clean_question(raw)
            if not _is_valid(question, answer, context):
                logger.debug(f"Rejected: '{question}' for answer='{answer}'")
                return _fallback_question(answer, answer_type)
            return question
        except Exception as e:
            logger.error(f"QG error: {e}")
            return _fallback_question(answer, answer_type)

    def generate_from_answers(
        self,
        answer_dict: Dict[str, List[dict]],
        progress_callback=None,
    ) -> Dict[str, List[dict]]:
        results: Dict[str, List[dict]] = {}
        total = sum(len(v) for v in answer_dict.values())
        done  = 0

        for sentence, answers in answer_dict.items():
            q_objects: List[dict] = []
            seen_qs: set          = set()

            for ans_obj in answers:
                answer    = ans_obj['answer']
                ans_type  = ans_obj.get('type', 'CONCEPT')
                ans_score = ans_obj.get('score', 0.5)

                question = self.generate_question(answer, sentence, ans_type)
                done += 1
                if progress_callback:
                    progress_callback(done, total)

                if not question or question.lower() in seen_qs:
                    continue
                seen_qs.add(question.lower())

                q_score = _question_score(question, answer, sentence, ans_score, ans_type)
                q_objects.append({
                    **ans_obj,
                    'question':       question,
                    'question_score': round(q_score, 4),
                })

            results[sentence] = q_objects

        logger.info(f"Generated {sum(len(v) for v in results.values())} questions")
        return results


def _question_score(question, answer, context, answer_score, answer_type):
    score = 0.50
    q_low = question.lower()
    if any(q_low.startswith(w) for w in _QUESTION_WORDS): score += 0.15
    if question.endswith('?'):                             score += 0.05
    if len(question.split()) < 5:                         score -= 0.15
    if len(question) > 120:                               score -= 0.10
    expected_wh = _TYPE_HINT.get(answer_type, 'What').split('/')[0].strip().lower()
    if q_low.startswith(expected_wh):                     score += 0.10
    if answer.lower() not in q_low:                       score += 0.05
    score += answer_score * 0.10
    return max(0.0, min(1.0, score))
