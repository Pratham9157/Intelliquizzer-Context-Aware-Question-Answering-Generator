"""
MODULE 8 — Exporter + Evaluator
=================================
Converts the final MCQ list into multiple output formats and computes
basic quality metrics (no external evaluation model needed).

Export formats:
  • Plain Text   — human-readable, printable
  • JSON         — machine-readable, complete metadata
  • CSV          — spreadsheet-compatible
  • Anki         — tab-separated import format for Anki flashcards
  • GIFT         — Moodle quiz import format

Evaluation metrics (all local, no API):
  • Answer present in context (extractive rate)
  • Question WH-word distribution
  • Difficulty distribution
  • Type distribution
  • Average option length variance (proxy for distractor quality)
  • Duplicate question rate
"""

from __future__ import annotations

import csv
import io
import json
import re
from typing import Dict, List, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Export helpers
# ─────────────────────────────────────────────────────────────────────────────

def _letters_in_order(options: dict) -> list:
    return [options.get(l, '') for l in ['A', 'B', 'C', 'D']]


# ── Plain text ────────────────────────────────────────────────────────────────

def to_plain_text(mcqs: List[dict]) -> str:
    lines = []
    for i, q in enumerate(mcqs, 1):
        diff  = q.get('difficulty', '?').capitalize()
        qtype = q.get('type', '?').capitalize()
        lines.append(f"{'─'*64}")
        lines.append(f"Q{i:02d}  [{qtype}]  Difficulty: {diff}  Score: {q.get('score', 0):.2f}")
        lines.append(f"")
        lines.append(f"{q['question']}")
        lines.append(f"")
        for letter in ['A', 'B', 'C', 'D']:
            text   = q['options'].get(letter, '')
            marker = '  ✓' if letter == q['answer'] else ''
            lines.append(f"   {letter}.  {text}{marker}")
        if q.get('explanation'):
            lines.append(f"")
            lines.append(f"   💡 {q['explanation']}")
        lines.append(f"")
    return '\n'.join(lines)


# ── JSON ──────────────────────────────────────────────────────────────────────

def to_json(mcqs: List[dict], indent: int = 2) -> str:
    # Remove internal fields not useful to end users
    export = []
    for q in mcqs:
        export.append({
            'question':    q['question'],
            'options':     q['options'],
            'answer':      q['answer'],
            'answer_text': q.get('answer_text', ''),
            'explanation': q.get('explanation', ''),
            'type':        q.get('type', ''),
            'difficulty':  q.get('difficulty', ''),
            'score':       round(q.get('score', 0), 3),
        })
    return json.dumps(export, indent=indent, ensure_ascii=False)


# ── CSV ───────────────────────────────────────────────────────────────────────

def to_csv(mcqs: List[dict]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(['#', 'Question', 'A', 'B', 'C', 'D',
                     'Answer', 'Answer Text', 'Explanation',
                     'Type', 'Difficulty', 'Score'])
    for i, q in enumerate(mcqs, 1):
        opts = q.get('options', {})
        writer.writerow([
            i,
            q['question'],
            opts.get('A', ''), opts.get('B', ''),
            opts.get('C', ''), opts.get('D', ''),
            q.get('answer', ''),
            q.get('answer_text', ''),
            q.get('explanation', ''),
            q.get('type', ''),
            q.get('difficulty', ''),
            round(q.get('score', 0), 3),
        ])
    return buf.getvalue()


# ── Anki  (Basic note type: Front \t Back) ────────────────────────────────────

def to_anki(mcqs: List[dict]) -> str:
    lines = ['#separator:Tab', '#html:true', '#notetype:Basic']
    for q in mcqs:
        opts = q.get('options', {})
        front_parts = [f"<b>{q['question']}</b>"]
        for letter in ['A', 'B', 'C', 'D']:
            front_parts.append(f"{letter}. {opts.get(letter, '')}")
        front = '<br>'.join(front_parts)

        ans_letter = q.get('answer', '')
        ans_text   = opts.get(ans_letter, '')
        expl       = q.get('explanation', '')
        back = (
            f"<b>Answer: {ans_letter}. {ans_text}</b>"
            + (f"<br><br><i>{expl}</i>" if expl else '')
        )
        lines.append(f"{front}\t{back}")
    return '\n'.join(lines)


# ── GIFT (Moodle) ─────────────────────────────────────────────────────────────

def to_gift(mcqs: List[dict]) -> str:
    """GIFT format for Moodle quiz import."""
    lines = []
    for i, q in enumerate(mcqs, 1):
        opts = q.get('options', {})
        correct = q.get('answer', 'A')
        lines.append(f"// Question {i}")
        lines.append(f"::{q.get('type','Q').capitalize()} Q{i}::[html]{q['question']}{{")
        for letter in ['A', 'B', 'C', 'D']:
            text = opts.get(letter, '')
            if letter == correct:
                lines.append(f"  ={text}")
            else:
                lines.append(f"  ~{text}")
        lines.append("}")
        lines.append("")
    return '\n'.join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation / Stats
# ─────────────────────────────────────────────────────────────────────────────

def compute_stats(mcqs: List[dict]) -> Dict:
    """
    Compute quality metrics for a set of MCQs.
    All metrics are local — no external model or API needed.
    """
    if not mcqs:
        return {}

    n = len(mcqs)

    # Type distribution
    type_dist: Dict[str, int] = {}
    diff_dist: Dict[str, int] = {}
    wh_dist:   Dict[str, int] = {}
    scores: List[float] = []
    opt_len_variances: List[float] = []
    extractive_count = 0
    duplicate_qs: set = set()
    dup_count = 0

    wh_words = ['what', 'who', 'when', 'where', 'why', 'how', 'which']

    for q in mcqs:
        # Type / difficulty
        t = q.get('type', 'unknown')
        d = q.get('difficulty', 'unknown')
        type_dist[t] = type_dist.get(t, 0) + 1
        diff_dist[d] = diff_dist.get(d, 0) + 1

        # WH-word distribution
        q_low = q['question'].lower()
        matched = next((w for w in wh_words if q_low.startswith(w)), 'other')
        wh_dist[matched] = wh_dist.get(matched, 0) + 1

        # Score
        scores.append(q.get('score', 0.5))

        # Extractive rate (answer appears literally in source sentence)
        answer  = q.get('answer_text', '').lower()
        context = q.get('source_sentence', '').lower()
        if answer and context and answer in context:
            extractive_count += 1

        # Option length variance (proxy for distractor quality)
        opts = [q['options'].get(l, '') for l in ['A', 'B', 'C', 'D']]
        lengths = [len(o.split()) for o in opts if o]
        if len(lengths) >= 2:
            mean_len = sum(lengths) / len(lengths)
            var = sum((l - mean_len) ** 2 for l in lengths) / len(lengths)
            opt_len_variances.append(var)

        # Duplicate detection
        qkey = re.sub(r'\W+', ' ', q['question'].lower()).strip()
        if qkey in duplicate_qs:
            dup_count += 1
        else:
            duplicate_qs.add(qkey)

    avg_score      = sum(scores) / n
    extractive_rate = extractive_count / n
    avg_opt_var    = sum(opt_len_variances) / max(len(opt_len_variances), 1)
    duplicate_rate = dup_count / n

    # Overall quality score (0–100)
    quality = (
        avg_score * 40
        + (1 - duplicate_rate) * 20
        + min(1.0, avg_opt_var / 5) * 20   # some variance = good distractors
        + extractive_rate * 20
    )

    return {
        'total':            n,
        'avg_score':        round(avg_score, 3),
        'quality_score':    round(quality, 1),
        'type_dist':        type_dist,
        'difficulty_dist':  diff_dist,
        'wh_word_dist':     wh_dist,
        'extractive_rate':  round(extractive_rate, 3),
        'duplicate_rate':   round(duplicate_rate, 3),
        'avg_option_length_variance': round(avg_opt_var, 2),
    }
