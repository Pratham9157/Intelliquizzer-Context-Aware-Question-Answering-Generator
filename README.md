# 🧠 IntelliQuizzer v4

**Context-Aware Automatic Question Generation — 100% Local, Zero API Keys**

---

## 📚 Project Overview

IntelliQuizzer is an advanced **Automatic Multiple Choice Question (MCQ) Generation System** that transforms documents into high-quality quiz content. Unlike cloud-based solutions, it runs entirely on your machine with zero API dependencies.

### Key Features
- ✅ **100% Local** — No internet required, no API keys, privacy guaranteed
- 🎯 **Context-Aware** — Extracts meaningful answers and generates relevant questions
- 🔧 **Multi-Format Support** — PDF, PPTX, TXT files
- 💾 **Multiple Export Formats** — TXT, CSV, JSON, Anki, GIFT
- 📊 **Quality Evaluation** — Built-in metrics dashboard
- 🎮 **Interactive Quiz Mode** — Self-test on generated questions
- 🤖 **RAG Q&A** — "Ask Anything" tab for document Q&A with web fallback
- ⚡ **Smart Filtering** — Rejects low-quality questions automatically

---

## ⚙️ Installation & Setup

### Prerequisites
- **Python 3.11** (required; NOT 3.12, 3.13, or 3.14 due to `torch` compatibility)
- Windows, macOS, or Linux
- ~2 GB disk space for models

### Setup Steps

```bash
# 1. Navigate to project directory
cd intelliquizzer

# 2. Create virtual environment with Python 3.11
py -3.11 -m venv venv              # Windows
python3.11 -m venv venv            # macOS/Linux

# 3. Activate virtual environment
venv\Scripts\activate              # Windows
source venv/bin/activate           # macOS/Linux

# 4. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 5. Download pre-trained models (one-time, ~1.8 GB)
python download_models.py

# 6. Launch the app
streamlit run app.py
```

The app opens at `http://localhost:8501`

---

## 🎨 Application Tabs

The Streamlit interface provides 5 main features:

| Tab | Purpose |
|-----|---------|
| **Generate MCQs** | Upload text/PDF/PPTX and generate questions with automatic quality filtering |
| **Quiz Mode** | Take interactive self-tests on generated questions |
| **Ask Anything** | RAG-powered Q&A — query documents with web search fallback |
| **Export** | Download questions in TXT, CSV, JSON, Anki, or GIFT format |
| **Evaluate** | View quality metrics dashboard for generated questions |

---

## 🤖 Models Used (all local, all free)

| Module | Model | Size | Purpose |
|--------|-------|------|---------|
| M4 Answer Extraction | `dslim/bert-base-NER` | ~430 MB | Named entity recognition |
| M5 Question Generation | `valhalla/t5-base-qg-hl` | ~850 MB | Seq2seq QG with highlight format |
| M6 Distractor Clustering | `all-MiniLM-L6-v2` | ~80 MB | Semantic similarity for distractors |
| QA Tab | `deepset/roberta-base-squad2` | ~480 MB | Extractive question answering |

---

## � Pipeline Architecture (7 Modules)

The system processes text through a carefully orchestrated 7-stage pipeline:

```
Raw Input
    ↓
M1: Text Extraction (PDF/PPTX/TXT) → Raw text
    ↓
M2: Preprocessing → Clean sentences
    ↓
M3: Sentence Ranking → Important sentences (TF-IDF + density)
    ↓
M4: Answer Extraction → Answer candidates (Pattern + NER + Subject)
    ↓
M5: Question Generation → Q&A pairs (valhalla/t5-base-qg-hl)
    ↓
M6: Distractor Generation → 3 distractors per Q (Cluster-based, semantic similarity)
    ↓
M7: MCQ Assembly → Final questions with quality filtering
    ↓
Output: High-quality MCQs
```

### Stage Details

| Module | Name | Key Algorithm | Size |
|--------|------|---------------|------|
| **M1** | Text Extractor | PDF/PPTX/TXT parsing | — |
| **M2** | Preprocessor | Sentence tokenization | — |
| **M3** | Sentence Ranker | TF-IDF + info density scoring | — |
| **M4** | Answer Extractor | Pattern matching + BERT NER | 430 MB |
| **M5** | Question Generator | T5 seq2seq with highlight format | 850 MB |
| **M6** | Distractor Generator | Semantic clustering + parallelism | 80 MB |
| **M7** | MCQ Builder | Stem-overlap filtering + quality checks | — |

### Quality Improvements in v4

#### M4 — Answer Extractor (Pattern-First Approach)
- **Before:** TF-IDF n-grams → fragmented, verb-heavy answers
- **After:** 
  - Detects explicit patterns: "such as X, Y", "(AGI)", "known as X"
  - BERT NER for entity recognition
  - Subject-of-sentence extraction
  - Hard verb gate: **no finite verbs allowed in answers**

#### M5 — Question Generation (Multi-Dataset T5)
- **Before:** `mrm8488/t5-base-finetuned-question-generation-ap` (SQuAD only)
- **After:** `valhalla/t5-base-qg-hl` trained on **SQuAD + RACE + CoQA**
  - Uses `<hl>highlight</hl>` tags around answer span
  - Better generalization across domains

#### M6 — Distractor Generation (Semantic Clustering)
- **Before:** Fixed semantic bank → same 4 distractors everywhere
- **After:** 
  - Cluster-based selection from corpus
  - Structural parallelism enforcement
  - Same length bucket, capitalization, digit presence matching
  - Domain-aware banks: ML, DL, NLP, CV, AI

#### M7 — MCQ Builder (Quality Filter)
- **Rejects stem-biased questions** (>65% word overlap with answer)
- **Rejects low-diversity options** (all options too similar)
- **Rejects trivial questions** (<5 words)
- **Builds 2× buffer** then filters to target count

---

## 📁 Project Structure

```
intelliquizzer/
│
├── app.py                          # Main Streamlit UI (5 tabs)
├── download_models.py              # Pre-download all models script
├── requirements.txt                # Python dependencies
├── __init__.py                     # Package initialization
│
├── pipeline/                       # MCQ Generation Pipeline
│   ├── __init__.py
│   ├── orchestrator.py             # Pipeline coordinator (M1–M7)
│   ├── text_extractor.py           # M1: Text extraction
│   ├── preprocessor.py             # M2: Sentence splitting
│   ├── sentence_ranker.py          # M3: TF-IDF + density ranking
│   ├── answer_extractor.py         # M4: Pattern + NER
│   ├── question_generator.py       # M5: T5 seq2seq
│   ├── distractor_generator.py     # M6: Semantic clustering
│   └── mcq_builder.py              # M7: Assembly + filtering
│
├── rag/                            # Retrieval-Augmented Generation
│   └── __init__.py                 # DuckDuckGo + RoBERTa Q&A
│
└── utils/                          # Utilities
    ├── __init__.py
    └── exporter.py                 # Export to TXT/CSV/JSON/Anki/GIFT
```

---

## 💡 Usage Examples

### Example 1: Generate MCQs from Text

```python
from pipeline.orchestrator import IntelliQuizzerPipeline

# Initialize
pipe = IntelliQuizzerPipeline()

# Generate questions
text = "Artificial Intelligence is transforming industries..."
mcqs, metadata = pipe.run(
    source=text,
    max_questions=10,
    max_sentences=20,
    use_sentence_ranking=True
)

# Access results
for mcq in mcqs:
    print(f"Q: {mcq['question']}")
    print(f"Answer: {mcq['answer']}")
    print(f"Options: {mcq['options']}\n")
```

### Example 2: Generate from PDF (via UI)
1. Open `http://localhost:8501`
2. Go to **Generate MCQs** tab
3. Upload PDF file
4. Set number of questions (default: 20)
5. Click "Generate"

### Example 3: Export to Multiple Formats
1. After generation, go to **Export** tab
2. Select format (TXT, CSV, JSON, Anki, GIFT)
3. Download file

---

## 🐛 Troubleshooting

### Issue: ModuleNotFoundError when running app
**Solution:**
```bash
# Ensure virtual environment is activated
venv\Scripts\activate              # Windows
source venv/bin/activate           # macOS/Linux

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue: "No module named 'torch'" or GPU errors
**Solution:**
- Project auto-detects GPU; CPU mode is default
- For GPU support, install: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

### Issue: Models not downloading
**Solution:**
```bash
python download_models.py
# This may take 5-10 minutes depending on internet speed
```

### Issue: "Python version not supported" error
**Solution:**
- Ensure Python 3.11 is installed: `python --version`
- Delete venv and recreate: `rmdir /s venv` (Windows) or `rm -rf venv` (macOS/Linux)
- Recreate: `py -3.11 -m venv venv`

### Issue: Memory error or crash on large PDFs
**Solution:**
- Reduce `max_sentences` in the UI (default: 30)
- Split large PDFs into smaller chunks before uploading

---

## � Supported Formats

### Input Files
- ✅ **PDF** (`.pdf`) — via `pdfplumber`
- ✅ **PowerPoint** (`.pptx`) — via `python-pptx`
- ✅ **Plain Text** (`.txt`) — raw string input

### Output Formats
- 📝 **TXT** — Human-readable plain text
- 📊 **CSV** — Spreadsheet-compatible format
- 📦 **JSON** — Machine-readable structured data
- 🧠 **Anki** — Compatible with Anki flashcard app
- 🎓 **GIFT** — Moodle/Blackboard compatible format

---

## 📊 Configuration Options

### In-App Settings (Generate MCQs Tab)
- **Number of Questions** (1–50) — target MCQ count
- **Max Sentences** (1–100) — limit sentences to process
- **Use Sentence Ranking** (toggle) — enable TF-IDF importance scoring

### Pipeline Parameters (`orchestrator.py`)
```python
pipe.run(
    source,                          # Text or file
    max_questions=20,                # Target MCQs
    max_sentences=30,                # Max sentences to process
    answers_per_sent=3,              # Answer candidates per sentence
    use_sentence_ranking=True,       # Enable importance scoring
    progress_callback=None           # Optional progress callback
)
```

---

## 🤝 Contributing & Development

### Key Files to Modify

| Goal | File |
|------|------|
| Add new export format | [utils/exporter.py](utils/exporter.py) |
| Improve question filtering | [pipeline/mcq_builder.py](pipeline/mcq_builder.py) |
| Change models | [pipeline/question_generator.py](pipeline/question_generator.py) / [pipeline/distractor_generator.py](pipeline/distractor_generator.py) |
| Add UI tab | [app.py](app.py) |
| Modify distractor logic | [pipeline/distractor_generator.py](pipeline/distractor_generator.py) |

### Local Testing
```bash
# Run a single module
python -c "from pipeline import preprocessor; print(preprocessor.preprocess('Your text here.'))"

# Run full pipeline
python -c "from pipeline.orchestrator import IntelliQuizzerPipeline; pipe = IntelliQuizzerPipeline(); mcqs, meta = pipe.run('Sample text'); print(len(mcqs), 'questions generated')"
```

---

## ⚖️ License & Attribution

- **Models:** All models are open-source (Hugging Face)
- **Libraries:** See [requirements.txt](requirements.txt)
- **RAG Backend:** DuckDuckGo API (free, no key required)

---

## 📝 Notes

- **First Run:** Model download (~1.8 GB) takes 5–10 minutes
- **Subsequent Runs:** Models cached locally; ~30–60 seconds per 20 questions
- **Best Performance:** Python 3.11, Windows 10+, macOS 10.15+, Linux (Ubuntu 20.04+)
- **Privacy:** All data stays local; no telemetry or tracking

---

## 🚀 Roadmap (Potential Future Enhancements)

- [ ] GPU acceleration (CUDA support guide)
- [ ] Fine-tuning on domain-specific corpora
- [ ] Multi-language support
- [ ] Question difficulty classification
- [ ] Bloom's taxonomy alignment
- [ ] REST API interface
- [ ] Web-based frontend alternative

---

**Questions or Issues?** Create an issue or check the [LLM_prompts/](LLM_prompts/) folder for detailed technical documentation.

Happy Learning! 🎓
