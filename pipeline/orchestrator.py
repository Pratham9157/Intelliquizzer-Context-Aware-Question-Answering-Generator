"""
Pipeline Orchestrator
======================
Connects all 7 modules into a single callable pipeline:

  Module 1  TextExtractor  → raw text
  Module 2  Preprocessor   → clean sentences
  Module 3  SentenceRanker → important sentences
  Module 4  AnswerExtractor→ answer candidates
  Module 5  QuestionGen    → (question, answer) pairs
  Module 6  DistractorGen  → 3 distractors per pair   (called inside Module 7)
  Module 7  MCQBuilder     → final MCQ objects

Also exposes `get_all_answers_flat()` for distractor corpus building.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from . import text_extractor as M1
from . import preprocessor   as M2
from . import sentence_ranker as M3
from . import answer_extractor as M4
from .question_generator import QuestionGenerator
from .mcq_builder import assemble_mcqs

logger = logging.getLogger(__name__)


class IntelliQuizzerPipeline:
    """
    End-to-end MCQ generation pipeline.

    Usage:
        pipe = IntelliQuizzerPipeline()
        mcqs = pipe.run(text_or_file, max_questions=20)
    """

    def __init__(self) -> None:
        self._qg = QuestionGenerator()

    # ── Run ───────────────────────────────────────────────────────────────────

    def run(
        self,
        source,
        max_questions:   int  = 20,
        max_sentences:   int  = 30,
        answers_per_sent: int = 3,
        use_sentence_ranking: bool = True,
        progress_callback = None,
    ) -> Tuple[List[dict], dict]:
        """
        Full pipeline run.

        Args:
            source:             Raw text str, bytes, or Streamlit UploadedFile.
            max_questions:      Maximum MCQs to return.
            max_sentences:      Maximum sentences to process (after ranking).
            answers_per_sent:   Max answer candidates per sentence.
            use_sentence_ranking: Whether to apply TF-IDF sentence ranking.
            progress_callback:  Optional callable(stage: str, pct: int).

        Returns:
            (mcqs, metadata)  where metadata has pipeline stats.
        """
        def _cb(stage: str, pct: int) -> None:
            if progress_callback:
                progress_callback(stage, pct)

        # ── M1: Extract ───────────────────────────────────────────────────────
        _cb('Extracting text…', 5)
        raw_text = M1.extract(source)
        if not raw_text.strip():
            logger.error("Text extraction returned empty string.")
            return [], {'error': 'Empty text'}

        # ── M2: Preprocess ────────────────────────────────────────────────────
        _cb('Preprocessing sentences…', 15)
        sentences = M2.preprocess(raw_text)
        if not sentences:
            logger.error("Preprocessing produced no sentences.")
            return [], {'error': 'No sentences after preprocessing'}

        # ── M3: Rank sentences ────────────────────────────────────────────────
        _cb('Ranking sentences by importance…', 25)
        if use_sentence_ranking and len(sentences) > max_sentences:
            sentences = M3.select_sentences(
                sentences, max_sentences=max_sentences, score_threshold=0.05
            )
        else:
            sentences = sentences[:max_sentences]

        # ── M4: Extract answers ───────────────────────────────────────────────
        _cb('Extracting answer candidates…', 35)
        answer_dict = M4.extract_answers(sentences, max_per_sentence=answers_per_sent)

        # Build flat pool for distractor corpus (Module 6 Layer 1)
        all_answer_dicts: List[dict] = [
            a for answers in answer_dict.values() for a in answers
        ]

        # ── M5: Generate questions ────────────────────────────────────────────
        _cb('Generating questions (seq2seq)…', 50)

        total_answers = len(all_answer_dicts)
        generated_count = 0

        def _qg_progress(done: int, total: int) -> None:
            nonlocal generated_count
            generated_count = done
            pct = 50 + int(done / max(total, 1) * 30)
            _cb(f'Generating questions… ({done}/{total})', pct)

        question_dict = self._qg.generate_from_answers(
            answer_dict,
            progress_callback=_qg_progress,
        )

        # ── M6 + M7: Build MCQs ───────────────────────────────────────────────
        _cb('Building MCQs with distractors…', 82)
        mcqs = assemble_mcqs(
            question_dict=question_dict,
            all_answer_dicts=all_answer_dicts,
            max_questions=max_questions,
        )

        _cb('Done', 100)

        # ── Metadata ──────────────────────────────────────────────────────────
        meta = {
            'raw_chars':        len(raw_text),
            'total_sentences':  len(sentences),
            'total_answers':    total_answers,
            'total_questions':  len(mcqs),
            'type_dist':        _count_field(mcqs, 'type'),
            'diff_dist':        _count_field(mcqs, 'difficulty'),
            'avg_score':        sum(q['score'] for q in mcqs) / max(len(mcqs), 1),
        }

        return mcqs, meta


def _count_field(mcqs: List[dict], field: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for q in mcqs:
        v = q.get(field, 'unknown')
        counts[v] = counts.get(v, 0) + 1
    return counts
