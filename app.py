"""
IntelliQuizzer — Streamlit UI
==============================
100% local — no API keys required.

Tabs:
  1. Generate MCQs  — full pipeline from text / PDF / PPTX
  2. Quiz Mode      — interactive self-test on generated questions
  3. Ask Anything   — RAG Q&A using document + web (local RoBERTa)
  4. Export         — download in TXT / CSV / JSON / Anki / GIFT
  5. Evaluate       — quality metrics dashboard
"""

import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Page config  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IntelliQuizzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"]{background:#0d1117}
[data-testid="stSidebar"]{background:#161b22;border-right:1px solid #30363d}
h1,h2,h3{color:#e6edf3!important}
p,label,div,span{color:#8b949e}

.iq-card{background:#161b22;border:1px solid #30363d;border-radius:10px;
          padding:20px 24px;margin-bottom:14px;transition:border-color .2s}
.iq-card:hover{border-color:#58a6ff}

.q-num{background:#1f6feb;color:#e6edf3;border-radius:50%;
        width:32px;height:32px;display:inline-flex;align-items:center;
        justify-content:center;font-weight:700;font-size:13px;margin-right:10px;
        flex-shrink:0}
.q-text{color:#e6edf3;font-size:15px;font-weight:600;line-height:1.5}

.badge{font-size:10px;font-weight:700;padding:2px 8px;border-radius:12px;
       text-transform:uppercase;letter-spacing:.5px;margin-left:5px}
.b-easy{background:#0d4429;color:#3fb950}
.b-medium{background:#3d2c00;color:#d29922}
.b-hard{background:#3d0c0c;color:#f85149}
.b-factual{background:#0c2d6b;color:#58a6ff}
.b-conceptual{background:#2d1a4d;color:#bc8cff}
.b-analytical{background:#1a2d4d;color:#79c0ff}
.b-local{background:#0d3322;color:#3fb950}

.opt{padding:8px 14px;margin:4px 0;border-radius:7px;border:1px solid #30363d;
     color:#8b949e;font-size:13.5px;display:flex;align-items:center;gap:10px}
.opt-correct{background:#0d3322;border-color:#3fb950;color:#3fb950;font-weight:600}
.opt-letter{font-weight:700;min-width:20px;color:#484f58;font-size:13px}
.opt-correct .opt-letter{color:#3fb950}

.expl{background:#161b22;border-left:3px solid #1f6feb;padding:9px 13px;
      border-radius:0 7px 7px 0;margin-top:10px;font-size:12.5px;color:#8b949e}

.stat-grid{display:flex;gap:10px;flex-wrap:wrap;margin:10px 0}
.stat-box{background:#161b22;border:1px solid #30363d;border-radius:9px;
           padding:14px 18px;flex:1;min-width:90px;text-align:center}
.stat-n{font-size:26px;font-weight:700;color:#1f6feb}
.stat-l{font-size:11px;color:#484f58;margin-top:2px}

.pipeline-stage{background:#161b22;border:1px solid #30363d;border-radius:8px;
                 padding:10px 16px;margin:4px 0;font-size:13px;color:#8b949e}

.quiz-opt{padding:11px 16px;margin:6px 0;border-radius:8px;cursor:pointer;
           border:1px solid #30363d;color:#8b949e;font-size:14px;
           transition:border-color .15s,background .15s}
.quiz-opt:hover{border-color:#58a6ff;color:#e6edf3}
.quiz-correct{background:#0d3322;border-color:#3fb950!important;color:#3fb950!important}
.quiz-wrong{background:#3d0c0c;border-color:#f85149!important;color:#f85149!important}
.quiz-score{font-size:48px;font-weight:700;color:#3fb950;text-align:center}

.ans-box{background:#161b22;border:1px solid #30363d;border-radius:10px;
          padding:20px;margin-top:12px}
.web-badge{display:inline-block;background:#0c2d6b;color:#58a6ff;
            border:1px solid #1f6feb;border-radius:20px;padding:2px 9px;
            font-size:10px;font-weight:700;text-transform:uppercase;margin-left:8px}
.local-badge{display:inline-block;background:#0d3322;color:#3fb950;
              border:1px solid #3fb950;border-radius:20px;padding:2px 9px;
              font-size:10px;font-weight:700;text-transform:uppercase;margin-left:8px}

.upload-hint{text-align:center;padding:30px;border:2px dashed #30363d;
              border-radius:10px;color:#484f58;font-size:14px;margin-top:20px}

.stButton>button{background:#1f6feb;color:#e6edf3;border:none;
                   border-radius:7px;font-weight:600;transition:all .2s}
.stButton>button:hover{background:#388bfd;transform:translateY(-1px)}

.model-info{background:#0c2d0c;border:1px solid #3fb950;border-radius:8px;
             padding:10px 14px;font-size:12px;color:#3fb950;margin-bottom:12px}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
_DEFAULTS = dict(
    mcqs=[], doc_text='', meta={},
    pipeline=None,
    quiz_idx=0, quiz_answers={}, quiz_done=False,
    qa_result=None, qa_question='',
)
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────────────
# Lazy pipeline / agent factories  (no API key needed)
# ─────────────────────────────────────────────────────────────────────────────
def get_pipeline():
    from pipeline import IntelliQuizzerPipeline
    if st.session_state.pipeline is None:
        with st.spinner("Loading local T5 question-generation model (first run only)…"):
            st.session_state.pipeline = IntelliQuizzerPipeline()
    return st.session_state.pipeline


def get_qa_agent():
    from rag import QAAgent
    return QAAgent()


# ─────────────────────────────────────────────────────────────────────────────
# Rendering helpers
# ─────────────────────────────────────────────────────────────────────────────
def _badge(text, css_class):
    return f'<span class="badge {css_class}">{text}</span>'


def render_mcq_card(q: dict, idx: int, show_answer: bool = True):
    diff  = q.get('difficulty', 'medium').lower()
    qtype = q.get('type', 'factual').lower()
    diff_badge  = _badge(diff,  f'b-{diff}')
    type_badge  = _badge(qtype, f'b-{qtype}')
    score_badge = _badge(f"★ {q.get('score',0):.2f}", 'b-factual')

    opts_html = ''
    for letter in ['A', 'B', 'C', 'D']:
        text    = q['options'].get(letter, '')
        correct = show_answer and (letter == q['answer'])
        cls     = 'opt opt-correct' if correct else 'opt'
        tick    = ' ✓' if correct else ''
        opts_html += (
            f'<div class="{cls}">'
            f'<span class="opt-letter">{letter}{tick}</span>'
            f'<span>{text}</span></div>'
        )

    expl_html = ''
    if show_answer and q.get('explanation'):
        expl_html = f'<div class="expl">💡 {q["explanation"]}</div>'

    st.markdown(f"""
    <div class="iq-card">
      <div style="display:flex;align-items:flex-start;gap:10px;margin-bottom:10px">
        <div class="q-num">{idx}</div>
        <div style="flex:1">
          <div class="q-text">{q['question']}</div>
          <div style="margin-top:5px">{diff_badge}{type_badge}{score_badge}</div>
        </div>
      </div>
      {opts_html}{expl_html}
    </div>""", unsafe_allow_html=True)


def render_stats(mcqs):
    from utils import compute_stats
    s = compute_stats(mcqs)
    if not s:
        return
    dd = s.get('difficulty_dist', {})
    items = [
        (s['total'],          'Total'),
        (dd.get('easy',0),    'Easy'),
        (dd.get('medium',0),  'Medium'),
        (dd.get('hard',0),    'Hard'),
        (f"{s.get('quality_score',0):.0f}", 'Quality'),
    ]
    cards = ''.join(
        f'<div class="stat-box"><div class="stat-n">{n}</div>'
        f'<div class="stat-l">{l}</div></div>'
        for n, l in items
    )
    st.markdown(f'<div class="stat-grid">{cards}</div>', unsafe_allow_html=True)
    td = s.get('type_dist', {})
    if td:
        type_str = '  ·  '.join(f'{k.capitalize()}: {v}' for k, v in td.items())
        st.caption(f'Types — {type_str}')


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 IntelliQuizzer")
    st.markdown("*Context-Aware AQG · 100% Local*")
    st.markdown("---")

    # Model status banner
    st.markdown("""
    <div class="model-info">
      ✅ <strong>Zero API keys required</strong><br>
      🤖 QG: T5-base (SQuAD fine-tuned)<br>
      🤖 QA: RoBERTa-base (SQuAD2)<br>
      📦 Models auto-download on first run
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ⚙️ Pipeline Settings")
    max_q   = st.slider("Max questions",       5, 50, 20)
    max_s   = st.slider("Max sentences",       10, 60, 30)
    ans_ps  = st.slider("Answers per sentence", 1,  5,  3)
    rank_s  = st.toggle("TF-IDF sentence ranking", value=True,
                         help="Rank sentences by importance before processing")
    show_ans = st.toggle("Show answers in cards", value=True)

    st.markdown("---")
    st.markdown("### 📐 Pipeline Architecture")
    st.markdown("""
```
Input (PDF / PPT / Text)
    ↓ M1 Text Extractor
    ↓ M2 Preprocessor
    ↓ M3 Sentence Ranker (TF-IDF)
    ↓ M4 Answer Extractor
       (NER + TF-IDF keywords)
    ↓ M5 Question Generator
       (T5-base, local, FREE)
    ↓ M6 Distractor Generator
       (corpus + semantic bank)
    ↓ M7 MCQ Builder
       (shuffle + difficulty)
    ↓ M8 Exporter
Output (TXT/CSV/JSON/Anki/GIFT)

QA Tab:
    ↓ DuckDuckGo Web Retriever
    ↓ RoBERTa QA (local, FREE)
```
    """)


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    '<h1 style="text-align:center;margin-bottom:4px">🧠 IntelliQuizzer</h1>'
    '<p style="text-align:center;color:#484f58;margin-bottom:24px">'
    'Context-Aware Automatic Question Generation · 100% Local · Zero API Keys</p>',
    unsafe_allow_html=True
)

tab_gen, tab_quiz, tab_qa, tab_export, tab_eval = st.tabs([
    "📝 Generate", "🎯 Quiz Mode", "💬 Ask Anything", "💾 Export", "📊 Evaluate"
])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — GENERATE
# ═════════════════════════════════════════════════════════════════════════════
with tab_gen:
    col_in, col_out = st.columns([1, 1], gap="large")

    with col_in:
        st.markdown("### 📥 Input")
        mode = st.radio("Source type", ["Paste Text", "Upload File"],
                        horizontal=True, label_visibility="collapsed")

        raw_text = ''
        if mode == "Paste Text":
            raw_text = st.text_area("", height=280,
                placeholder="Paste text from any source — articles, lecture notes, research papers…",
                label_visibility="collapsed")
        else:
            upl = st.file_uploader("", type=["pdf", "pptx", "ppt", "txt"],
                                    label_visibility="collapsed")
            if upl:
                from pipeline.text_extractor import extract as _extract
                with st.spinner(f"Extracting from {upl.name}…"):
                    try:
                        raw_text = _extract(upl)
                        st.success(f"✅ Extracted {len(raw_text):,} characters from **{upl.name}**")
                    except Exception as e:
                        st.error(f"Extraction failed: {e}")

        if raw_text.strip():
            st.session_state.doc_text = raw_text

        if st.session_state.doc_text:
            st.info(f"📄 Document loaded · {len(st.session_state.doc_text):,} chars")

        st.markdown("")
        run_btn = st.button("⚡ Generate Questions", use_container_width=True,
                             type="primary")

        if run_btn:
            if not st.session_state.doc_text.strip():
                st.warning("Please enter text or upload a file first.")
            else:
                pipe = get_pipeline()

                prog_bar  = st.progress(0, "Starting pipeline…")
                stage_txt = st.empty()

                def _progress(stage: str, pct: int):
                    prog_bar.progress(pct, stage)
                    stage_txt.markdown(
                        f'<div class="pipeline-stage">🔄 {stage}</div>',
                        unsafe_allow_html=True
                    )

                t0 = time.time()
                mcqs, meta = pipe.run(
                    source=st.session_state.doc_text,
                    max_questions=max_q,
                    max_sentences=max_s,
                    answers_per_sent=ans_ps,
                    use_sentence_ranking=rank_s,
                    progress_callback=_progress,
                )
                elapsed = time.time() - t0
                prog_bar.empty()
                stage_txt.empty()

                if mcqs:
                    st.session_state.mcqs  = mcqs
                    st.session_state.meta  = meta
                    st.session_state.quiz_idx     = 0
                    st.session_state.quiz_answers = {}
                    st.session_state.quiz_done    = False
                    st.success(
                        f"✅ Generated **{len(mcqs)} questions** "
                        f"from {meta.get('total_sentences',0)} sentences "
                        f"in {elapsed:.1f}s"
                    )
                else:
                    st.error("No questions generated. Try more text or lower the sentence threshold.")

    with col_out:
        st.markdown("### 📋 Generated Questions")

        if st.session_state.mcqs:
            render_stats(st.session_state.mcqs)

            all_types = sorted({q.get('type','factual') for q in st.session_state.mcqs})
            all_diffs = sorted({q.get('difficulty','medium') for q in st.session_state.mcqs})
            fc1, fc2  = st.columns(2)
            f_type = fc1.multiselect("Type",       all_types, default=all_types, key='f_type')
            f_diff = fc2.multiselect("Difficulty", all_diffs, default=all_diffs, key='f_diff')

            filtered = [q for q in st.session_state.mcqs
                        if q.get('type','factual') in f_type
                        and q.get('difficulty','medium') in f_diff]
            st.caption(f"Showing {len(filtered)} / {len(st.session_state.mcqs)}")

            for i, q in enumerate(filtered, 1):
                render_mcq_card(q, i, show_answer=show_ans)
        else:
            st.markdown(
                '<div class="upload-hint">📄 Add content and click '
                '<b>Generate Questions</b> to begin</div>',
                unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — QUIZ MODE
# ═════════════════════════════════════════════════════════════════════════════
with tab_quiz:
    st.markdown("### 🎯 Interactive Quiz Mode")

    if not st.session_state.mcqs:
        st.markdown('<div class="upload-hint">Generate questions first.</div>',
                    unsafe_allow_html=True)
    else:
        mcqs      = st.session_state.mcqs
        q_idx     = st.session_state.quiz_idx
        answers   = st.session_state.quiz_answers
        quiz_done = st.session_state.quiz_done

        if not quiz_done:
            prog = q_idx / len(mcqs)
            st.progress(prog, f"Question {q_idx + 1} of {len(mcqs)}")
            st.markdown("")

            q = mcqs[q_idx]
            st.markdown(
                f'<div class="iq-card">'
                f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:14px">'
                f'<div class="q-num">{q_idx+1}</div>'
                f'<div class="q-text">{q["question"]}</div>'
                f'</div></div>',
                unsafe_allow_html=True
            )

            already_answered = q_idx in answers
            cols = st.columns(2)
            for i, letter in enumerate(['A', 'B', 'C', 'D']):
                text = q['options'].get(letter, '')
                col  = cols[i % 2]
                with col:
                    if not already_answered:
                        if st.button(f"{letter}.  {text}", key=f"quiz_{q_idx}_{letter}",
                                     use_container_width=True):
                            answers[q_idx] = letter
                            st.session_state.quiz_answers = answers
                            st.rerun()
                    else:
                        chosen  = answers[q_idx]
                        correct = q['answer']
                        if letter == correct:
                            st.markdown(
                                f'<div class="quiz-opt quiz-correct">✓ {letter}. {text}</div>',
                                unsafe_allow_html=True)
                        elif letter == chosen:
                            st.markdown(
                                f'<div class="quiz-opt quiz-wrong">✗ {letter}. {text}</div>',
                                unsafe_allow_html=True)
                        else:
                            st.markdown(
                                f'<div class="quiz-opt">{letter}. {text}</div>',
                                unsafe_allow_html=True)

            if already_answered:
                chosen  = answers[q_idx]
                correct = q['answer']
                if chosen == correct:
                    st.success("✅ Correct!")
                else:
                    st.error(f"❌ Wrong — correct answer: **{correct}. {q['options'][correct]}**")
                if q.get('explanation'):
                    st.info(f"💡 {q['explanation']}")

                nav1, nav2 = st.columns(2)
                with nav1:
                    if q_idx > 0:
                        if st.button("← Previous", use_container_width=True):
                            st.session_state.quiz_idx -= 1
                            st.rerun()
                with nav2:
                    if q_idx < len(mcqs) - 1:
                        if st.button("Next →", use_container_width=True, type="primary"):
                            st.session_state.quiz_idx += 1
                            st.rerun()
                    else:
                        if st.button("🏁 See Results", use_container_width=True, type="primary"):
                            st.session_state.quiz_done = True
                            st.rerun()
        else:
            correct_count  = sum(1 for i, q in enumerate(mcqs) if answers.get(i) == q['answer'])
            total_answered = len([i for i in range(len(mcqs)) if i in answers])
            pct = int(correct_count / max(total_answered, 1) * 100)

            st.markdown(
                f'<div class="quiz-score">{pct}%</div>'
                f'<p style="text-align:center;color:#484f58">'
                f'{correct_count} / {total_answered} correct</p>',
                unsafe_allow_html=True
            )
            grade = ("🏆 Excellent!" if pct >= 80 else
                     "👍 Good job!"  if pct >= 60 else
                     "📚 Keep studying!")
            st.markdown(f'<p style="text-align:center;font-size:20px">{grade}</p>',
                         unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("#### Review answers")
            for i, q in enumerate(mcqs):
                if i not in answers:
                    continue
                chosen  = answers[i]
                correct = q['answer']
                icon    = "✅" if chosen == correct else "❌"
                with st.expander(f"{icon} Q{i+1}: {q['question'][:80]}…"):
                    st.markdown(f"**Your answer:** {chosen}. {q['options'].get(chosen,'')}")
                    st.markdown(f"**Correct:**     {correct}. {q['options'].get(correct,'')}")
                    if q.get('explanation'):
                        st.info(q['explanation'])

            if st.button("🔄 Restart Quiz", type="primary"):
                st.session_state.quiz_idx     = 0
                st.session_state.quiz_answers = {}
                st.session_state.quiz_done    = False
                st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — ASK ANYTHING (Local RAG Q&A)
# ═════════════════════════════════════════════════════════════════════════════
with tab_qa:
    st.markdown("### 💬 Ask Anything")
    st.markdown(
        "Answers are extracted from your document and/or live web search using a "
        "<span class='local-badge'>LOCAL</span> RoBERTa model — no API key needed.",
        unsafe_allow_html=True
    )

    user_q  = st.text_input("Your question", label_visibility="collapsed",
                              placeholder="e.g. What is the difference between AGI and ANI?")
    use_web = st.toggle("🌐 Augment with web search (DuckDuckGo RAG)", value=True)

    if st.button("🔍 Get Answer", type="primary", use_container_width=True):
        if not user_q.strip():
            st.warning("Please enter a question.")
        else:
            agent = get_qa_agent()
            with st.spinner("Thinking" + (" + searching web…" if use_web else "…")):
                result = agent.answer(
                    question=user_q,
                    document_text=st.session_state.doc_text,
                    use_web=use_web,
                )
            st.session_state.qa_result   = result
            st.session_state.qa_question = user_q

    if st.session_state.qa_result:
        r = st.session_state.qa_result
        web_badge   = '<span class="web-badge">🌐 Web-enriched</span>' if r.get('used_web') else ''
        local_badge = '<span class="local-badge">🤖 Local RoBERTa</span>'
        conf_str    = f"Confidence: {r.get('score', 0):.0%}" if r.get('score', 0) > 0 else ''

        st.markdown(f"""
        <div class="ans-box">
          <div style="display:flex;justify-content:space-between;margin-bottom:10px">
            <span style="color:#e6edf3;font-weight:600">
              Q: {st.session_state.qa_question}
            </span>
            <span>{local_badge}{web_badge}</span>
          </div>
          <div style="color:#c9d1d9;line-height:1.7;white-space:pre-wrap">{r['answer']}</div>
          {'<div style="margin-top:8px;font-size:11px;color:#484f58">' + conf_str + '</div>' if conf_str else ''}
        </div>""", unsafe_allow_html=True)

        if r.get('sources'):
            with st.expander("📎 Sources"):
                for url in r['sources']:
                    st.markdown(f"- [{url}]({url})")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — EXPORT
# ═════════════════════════════════════════════════════════════════════════════
with tab_export:
    st.markdown("### 💾 Export Questions")
    from utils import to_plain_text, to_json, to_csv, to_anki, to_gift

    if not st.session_state.mcqs:
        st.markdown('<div class="upload-hint">Generate questions first to enable export.</div>',
                    unsafe_allow_html=True)
    else:
        mcqs = st.session_state.mcqs
        render_stats(mcqs)
        st.markdown("")

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.download_button("📄 Plain Text", data=to_plain_text(mcqs),
                                file_name="intelliquizzer_questions.txt",
                                mime="text/plain", use_container_width=True)
        with c2:
            st.download_button("📊 CSV", data=to_csv(mcqs),
                                file_name="intelliquizzer_questions.csv",
                                mime="text/csv", use_container_width=True)
        with c3:
            st.download_button("🔧 JSON", data=to_json(mcqs),
                                file_name="intelliquizzer_questions.json",
                                mime="application/json", use_container_width=True)
        with c4:
            st.download_button("🃏 Anki", data=to_anki(mcqs),
                                file_name="intelliquizzer_anki.txt",
                                mime="text/plain", use_container_width=True)
        with c5:
            st.download_button("🎓 GIFT (Moodle)", data=to_gift(mcqs),
                                file_name="intelliquizzer_moodle.gift",
                                mime="text/plain", use_container_width=True)

        st.markdown("---")
        st.markdown("#### Preview (first 3 questions)")
        st.code(to_plain_text(mcqs[:3]), language=None)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — EVALUATE
# ═════════════════════════════════════════════════════════════════════════════
with tab_eval:
    st.markdown("### 📊 Quality Evaluation")
    from utils import compute_stats

    if not st.session_state.mcqs:
        st.markdown('<div class="upload-hint">Generate questions first.</div>',
                    unsafe_allow_html=True)
    else:
        mcqs  = st.session_state.mcqs
        stats = compute_stats(mcqs)
        meta  = st.session_state.meta

        qs    = stats.get('quality_score', 0)
        color = '#3fb950' if qs >= 70 else '#d29922' if qs >= 50 else '#f85149'
        st.markdown(f"""
        <div style="text-align:center;padding:20px">
          <div style="font-size:56px;font-weight:700;color:{color}">{qs:.0f}</div>
          <div style="color:#484f58">Overall Quality Score (0–100)</div>
        </div>""", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Distribution")
            diff      = stats.get('difficulty_dist', {})
            diff_data = {'Easy': diff.get('easy',0), 'Medium': diff.get('medium',0),
                         'Hard': diff.get('hard',0)}
            st.bar_chart(diff_data, height=200)
            st.caption("Difficulty distribution")

            wh = stats.get('wh_word_dist', {})
            if wh:
                st.bar_chart(wh, height=200)
                st.caption("Question-word (WH) distribution")

        with col2:
            st.markdown("#### Metrics")
            metrics = {
                'Total Questions':         stats.get('total', 0),
                'Avg Quality Score':       f"{stats.get('avg_score', 0):.3f}",
                'Extractive Rate':         f"{stats.get('extractive_rate',0)*100:.1f}%",
                'Duplicate Rate':          f"{stats.get('duplicate_rate',0)*100:.1f}%",
                'Avg Option Len Variance': f"{stats.get('avg_option_length_variance',0):.2f}",
            }
            for k, v in metrics.items():
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;'
                    f'padding:8px 0;border-bottom:1px solid #21262d">'
                    f'<span style="color:#8b949e">{k}</span>'
                    f'<span style="color:#e6edf3;font-weight:600">{v}</span></div>',
                    unsafe_allow_html=True
                )

            if meta:
                st.markdown("#### Pipeline Stats")
                pipeline_info = {
                    'Raw characters':    f"{meta.get('raw_chars',0):,}",
                    'Sentences used':    meta.get('total_sentences', 0),
                    'Answer candidates': meta.get('total_answers', 0),
                    'Questions built':   meta.get('total_questions', 0),
                }
                for k, v in pipeline_info.items():
                    st.markdown(
                        f'<div style="display:flex;justify-content:space-between;'
                        f'padding:8px 0;border-bottom:1px solid #21262d">'
                        f'<span style="color:#8b949e">{k}</span>'
                        f'<span style="color:#e6edf3;font-weight:600">{v}</span></div>',
                        unsafe_allow_html=True
                    )
