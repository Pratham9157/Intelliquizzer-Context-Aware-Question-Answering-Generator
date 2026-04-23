# 🧠 Context-Aware Automatic Question Generation
### Using Transformer-based Sequence Modeling
**Semester Project — Complete Execution Plan**

---

## TABLE OF CONTENTS
1. [System Architecture](#1-system-architecture)
2. [Module Design](#2-module-design)
3. [Model Design](#3-model-design)
4. [3-Week Execution Plan](#4-3-week-execution-plan)
5. [Project Folder Structure](#5-project-folder-structure)
6. [Dataset Strategy](#6-dataset-strategy)
7. [Evaluation Strategy](#7-evaluation-strategy)
8. [Enhancements (If Time Permits)](#8-enhancements-if-time-permits)
9. [Interview Preparation](#9-interview-preparation)

---

## 1. 📐 SYSTEM ARCHITECTURE

### High-Level Pipeline

```
INPUT (PDF / PPT / TXT / Paragraph)
        │
        ▼
┌─────────────────────────┐
│  1. TEXT EXTRACTION     │  ← PyMuPDF, python-pptx, plain text
│     LAYER               │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  2. PREPROCESSING       │  ← Tokenization, cleaning, sentence split
│     MODULE              │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  3. SENTENCE RANKING /  │  ← TF-IDF + cosine similarity (custom)
│     IMPORTANCE MODULE   │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  4. ANSWER EXTRACTION   │  ← Named Entity Recognition + KeyBERT
│     MODULE              │     (answer candidates = key phrases)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  5. QUESTION GENERATION │  ← Fine-tuned T5-small (SQuAD)
│     MODULE              │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  6. DIFFICULTY          │  ← Custom rule-based + readability scores
│     ESTIMATOR           │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  7. OUTPUT FORMATTER    │  ← MCQ distractors, FAQs, short answers
│                         │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  8. EVALUATION MODULE   │  ← BLEU, ROUGE, BERTScore
│                         │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  9. UI / API LAYER      │  ← Streamlit or FastAPI
│     (Optional)          │
└─────────────────────────┘
```

### Module Interaction Map

```
TextExtractor ──► Preprocessor ──► SentenceRanker
                                         │
                              ┌──────────┴──────────┐
                              ▼                     ▼
                       AnswerExtractor       (Context window)
                              │                     │
                              └──────────┬──────────┘
                                         ▼
                                QuestionGenerator (T5)
                                         │
                              ┌──────────┴──────────┐
                              ▼                     ▼
                       DistractorGen         DifficultyEstimator
                              │                     │
                              └──────────┬──────────┘
                                         ▼
                                  OutputFormatter
                                         │
                                         ▼
                                  EvaluationModule
```

---

## 2. 🧩 MODULE DESIGN

---

### MODULE 1 — Text Extraction Layer

**Purpose:** Accept any input format and return clean plain text.

| Item | Detail |
|------|--------|
| **Implement Yourself** | File-type router, text cleaning post-extraction |
| **Libraries** | `PyMuPDF` (fitz) for PDF, `python-pptx` for PPT, `chardet` for encoding |
| **Output** | Raw text string |

**Implementation:**
```python
# file_extractor.py
import fitz  # PyMuPDF
from pptx import Presentation

def extract_from_pdf(path):
    doc = fitz.open(path)
    return "\n".join([page.get_text() for page in doc])

def extract_from_pptx(path):
    prs = Presentation(path)
    text = []
    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                text.append(shape.text)
    return "\n".join(text)

def extract_text(path):
    ext = path.split(".")[-1].lower()
    if ext == "pdf": return extract_from_pdf(path)
    elif ext in ["pptx", "ppt"]: return extract_from_pptx(path)
    elif ext == "txt": return open(path, encoding="utf-8").read()
    else: raise ValueError("Unsupported file type")
```

---

### MODULE 2 — Text Preprocessing Module

**Purpose:** Clean and structure raw text for downstream NLP tasks.

| Item | Detail |
|------|--------|
| **Implement Yourself** | Custom tokenizer, stopword filter, sentence boundary detector |
| **Libraries** | `NLTK` (sent_tokenize), `re`, `unicodedata` |
| **Output** | List of clean sentences |

**Key Steps (implement each):**
1. Unicode normalization (`unicodedata.normalize`)
2. Remove headers, page numbers, special chars (`re.sub`)
3. Sentence tokenization using NLTK punkt tokenizer
4. Filter sentences < 6 words (noise removal)
5. Lowercase (for matching), preserve case for generation

```python
# preprocessor.py
import re, nltk, unicodedata
nltk.download('punkt')

def clean_text(text):
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r'\s+', ' ', text)           # multiple spaces
    text = re.sub(r'\[.*?\]|\(.*?\)', '', text) # brackets
    text = re.sub(r'[^\w\s\.\,\!\?\;\:]', '', text)
    return text.strip()

def tokenize_sentences(text):
    from nltk.tokenize import sent_tokenize
    sentences = sent_tokenize(text)
    return [s for s in sentences if len(s.split()) >= 6]

def preprocess(raw_text):
    cleaned = clean_text(raw_text)
    sentences = tokenize_sentences(cleaned)
    return sentences
```

---

### MODULE 3 — Sentence Importance Ranking

**Purpose:** Select the most information-dense sentences to generate questions from. You don't want to generate questions from trivial sentences.

| Item | Detail |
|------|--------|
| **Implement Yourself** | TF-IDF vectorizer from scratch OR sklearn, cosine similarity scoring |
| **Libraries** | `sklearn.feature_extraction.text`, `numpy` |
| **Key Algorithm** | TF-IDF + Cosine Similarity vs. document centroid |

**Algorithm (implement this):**
1. Build TF-IDF matrix for all sentences
2. Compute document centroid (mean of all sentence vectors)
3. Rank each sentence by cosine similarity to centroid
4. Also boost sentences containing Named Entities (NER signal)
5. Return top-K sentences

```python
# sentence_ranker.py
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def rank_sentences(sentences, top_k=10):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(sentences)  # (N, vocab)
    
    # Document centroid = mean of all sentence vectors
    centroid = np.mean(tfidf_matrix.toarray(), axis=0).reshape(1, -1)
    
    # Score each sentence
    scores = cosine_similarity(tfidf_matrix, centroid).flatten()
    
    # Get top-k indices
    top_indices = scores.argsort()[::-1][:top_k]
    top_indices_sorted = sorted(top_indices)  # preserve order
    
    return [(sentences[i], scores[i]) for i in top_indices_sorted]
```

**Enhancement (add NER boost):**
```python
import spacy
nlp = spacy.load("en_core_web_sm")

def ner_boost(sentence, base_score):
    doc = nlp(sentence)
    ner_count = len(doc.ents)
    return base_score + 0.05 * ner_count  # small boost per entity
```

---

### MODULE 4 — Answer Extraction Module

**Purpose:** Identify answer candidates (key phrases/entities) within each sentence. These become the "answers" your questions will target.

| Item | Detail |
|------|--------|
| **Implement Yourself** | NER-based extraction, POS tagging filter, noun phrase chunking |
| **Libraries** | `spaCy` (en_core_web_sm), no large models needed |
| **What NOT to do** | Don't use GPT APIs for this — use rule-based NLP |

**Strategy:**
- **Named Entities** → proper answers (people, places, dates, orgs)
- **Noun Phrases** → conceptual answers
- **Numbers/Dates** → factual answers (good for hard questions)

```python
# answer_extractor.py
import spacy
nlp = spacy.load("en_core_web_sm")

def extract_answers(sentence):
    doc = nlp(sentence)
    candidates = []
    
    # Priority 1: Named Entities
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "ORG", "GPE", "DATE", 
                           "EVENT", "PRODUCT", "LAW", "NORP"]:
            candidates.append({
                "text": ent.text,
                "type": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            })
    
    # Priority 2: Noun Chunks (if no NER found)
    if not candidates:
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) <= 5:  # avoid long phrases
                candidates.append({
                    "text": chunk.text,
                    "type": "NOUN_PHRASE",
                    "start": chunk.start_char,
                    "end": chunk.end_char
                })
    
    return candidates
```

---

### MODULE 5 — Question Generation Model (Core ML)

**Purpose:** Given a `(context, answer)` pair, generate a natural language question.

| Item | Detail |
|------|--------|
| **Model** | Fine-tuned `t5-small` on SQuAD |
| **Why T5** | Seq2seq, text-to-text, small enough for local training |
| **What You Implement** | Fine-tuning pipeline, custom data collator, training loop |
| **Libraries** | `transformers`, `datasets`, `torch` |

**Input format for T5:**
```
"answer: {answer} context: {context}"
```
**Output:**
```
"What is the capital of France?"
```

**Fine-tuning script (implement yourself):**
```python
# train_qg.py
from transformers import T5ForConditionalGeneration, T5Tokenizer
from torch.utils.data import DataLoader, Dataset
import torch

MODEL_NAME = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

class QGDataset(Dataset):
    def __init__(self, data, tokenizer, max_src=512, max_tgt=64):
        self.data = data
        self.tokenizer = tokenizer
        self.max_src = max_src
        self.max_tgt = max_tgt

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        src = f"answer: {item['answer']} context: {item['context']}"
        tgt = item['question']
        
        src_enc = self.tokenizer(src, max_length=self.max_src,
                                  padding='max_length', truncation=True,
                                  return_tensors='pt')
        tgt_enc = self.tokenizer(tgt, max_length=self.max_tgt,
                                  padding='max_length', truncation=True,
                                  return_tensors='pt')
        
        labels = tgt_enc['input_ids'].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100  # ignore padding in loss
        
        return {
            'input_ids': src_enc['input_ids'].squeeze(),
            'attention_mask': src_enc['attention_mask'].squeeze(),
            'labels': labels
        }

def train(model, dataloader, optimizer, device, epochs=3):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids,
                           attention_mask=attention_mask,
                           labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(dataloader):.4f}")
```

**Inference:**
```python
def generate_question(context, answer, model, tokenizer, device):
    input_text = f"answer: {answer} context: {context}"
    inputs = tokenizer(input_text, return_tensors="pt", 
                       max_length=512, truncation=True).to(device)
    
    outputs = model.generate(
        inputs['input_ids'],
        max_length=64,
        num_beams=4,           # beam search
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

---

### MODULE 6 — Difficulty Estimation Module

**Purpose:** Tag each generated question as Easy / Medium / Hard.

| Item | Detail |
|------|--------|
| **Implement Yourself** | 100% — this is your custom logic, great for interviews |
| **Libraries** | `textstat`, `spaCy`, basic Python |
| **Algorithm** | Multi-factor scoring |

**Scoring Factors (implement all):**

| Factor | Easy | Medium | Hard |
|--------|------|--------|------|
| Flesch Reading Ease | > 70 | 50–70 | < 50 |
| Answer type | NOUN/simple | ORG/GPE | DATE/NUMBER/concept |
| Question word | What/Who | When/Where | Why/How/Which |
| Sentence length | < 15 words | 15–25 | > 25 |
| NE count in context | 0–1 | 2–3 | 4+ |

```python
# difficulty_estimator.py
import textstat
import spacy
nlp = spacy.load("en_core_web_sm")

QUESTION_WORDS = {
    "easy":   ["what", "who", "is", "are", "was", "were"],
    "medium": ["when", "where", "which", "how many", "how much"],
    "hard":   ["why", "how", "explain", "describe", "analyze"]
}

def estimate_difficulty(question, context, answer_type):
    score = 0
    
    # Factor 1: Readability
    fre = textstat.flesch_reading_ease(context)
    if fre > 70: score += 1
    elif fre < 50: score += 3
    else: score += 2
    
    # Factor 2: Question word
    q_lower = question.lower()
    if any(q_lower.startswith(w) for w in QUESTION_WORDS["hard"]):
        score += 3
    elif any(q_lower.startswith(w) for w in QUESTION_WORDS["medium"]):
        score += 2
    else:
        score += 1
    
    # Factor 3: Answer type
    hard_types = ["DATE", "CARDINAL", "PERCENT", "MONEY", "LAW"]
    easy_types = ["PERSON", "ORG", "NOUN_PHRASE"]
    if answer_type in hard_types: score += 3
    elif answer_type in easy_types: score += 1
    else: score += 2
    
    # Factor 4: Sentence complexity
    doc = nlp(context)
    nes = len(doc.ents)
    score += min(nes, 3)  # cap at 3
    
    # Map score to label
    if score <= 4: return "easy"
    elif score <= 7: return "medium"
    else: return "hard"
```

---

### MODULE 7 — MCQ Distractor Generator

**Purpose:** Generate 3 wrong-but-plausible answer choices for MCQs.

| Item | Detail |
|------|--------|
| **Implement Yourself** | WordNet-based semantic distractor logic |
| **Libraries** | `NLTK WordNet`, `sense2vec` (optional), `random` |
| **Strategy** | Semantic neighbors + same POS + different meaning |

```python
# distractor_generator.py
from nltk.corpus import wordnet
import random

def get_wordnet_distractors(answer, n=3):
    synsets = wordnet.synsets(answer.lower())
    distractors = set()
    
    for syn in synsets:
        # Hypernyms (broader category)
        for hypernym in syn.hypernyms():
            for lemma in hypernym.hyponyms():
                for name in lemma.lemma_names():
                    candidate = name.replace('_', ' ')
                    if candidate.lower() != answer.lower():
                        distractors.add(candidate)
    
    distractors = list(distractors)
    random.shuffle(distractors)
    return distractors[:n]

def build_mcq(question, correct_answer, distractors):
    choices = distractors[:3] + [correct_answer]
    random.shuffle(choices)
    return {
        "question": question,
        "options": {chr(65+i): c for i, c in enumerate(choices)},
        "answer": next(k for k, v in 
                       {chr(65+i): c for i, c in enumerate(choices)}.items()
                       if v == correct_answer)
    }
```

---

### MODULE 8 — Evaluation Module

**Purpose:** Quantitatively measure quality of generated questions.

| Item | Detail |
|------|--------|
| **Implement Yourself** | BLEU computation logic, ROUGE wrapper |
| **Libraries** | `nltk.translate.bleu_score`, `rouge-score`, `bert_score` |

```python
# evaluator.py
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn

def compute_bleu(reference, hypothesis):
    ref_tokens = [reference.lower().split()]
    hyp_tokens = hypothesis.lower().split()
    smoothie = SmoothingFunction().method4
    return sentence_bleu(ref_tokens, hyp_tokens, smoothing_function=smoothie)

def compute_rouge(reference, hypothesis):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], 
                                       use_stemmer=True)
    return scorer.score(reference, hypothesis)

def compute_bertscore(references, hypotheses):
    P, R, F1 = bert_score_fn(hypotheses, references, lang="en", verbose=False)
    return {"precision": P.mean().item(), 
            "recall": R.mean().item(), 
            "f1": F1.mean().item()}

def evaluate_batch(reference_questions, generated_questions):
    results = []
    for ref, gen in zip(reference_questions, generated_questions):
        results.append({
            "bleu": compute_bleu(ref, gen),
            "rouge": compute_rouge(ref, gen),
            "reference": ref,
            "generated": gen
        })
    
    avg_bleu = sum(r["bleu"] for r in results) / len(results)
    print(f"Average BLEU: {avg_bleu:.4f}")
    return results
```

---

### MODULE 9 — Output Formatter

**Purpose:** Assemble all outputs into a structured, human-readable format.

```python
# output_formatter.py
import json

def format_output(generated_items):
    """
    generated_items: list of dicts with keys:
    question, answer, question_type, difficulty, options (for MCQ)
    """
    output = {
        "MCQ": [],
        "ShortAnswer": [],
        "FAQ": []
    }
    
    for item in generated_items:
        base = {
            "question": item["question"],
            "answer": item["answer"],
            "difficulty": item["difficulty"]
        }
        
        if item["type"] == "mcq":
            base["options"] = item["options"]
            output["MCQ"].append(base)
        elif item["type"] == "short":
            output["ShortAnswer"].append(base)
        elif item["type"] == "faq":
            output["FAQ"].append(base)
    
    return output

def save_to_json(output, path="output/questions.json"):
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved to {path}")
```

---

## 3. 🤖 MODEL DESIGN

### Decision: Fine-tune T5-small (Not build from scratch)

**Rationale:** Building a transformer encoder-decoder from scratch in 3 weeks is possible but gives inferior results. The real skill being demonstrated here is:
- Understanding the architecture deeply (for interviews)
- Writing the fine-tuning loop manually (not using Trainer API)
- Custom data preprocessing for QG task

### T5 Architecture — What You Must Understand

```
T5 = Text-To-Text Transfer Transformer

ENCODER:                          DECODER:
[Input tokens]                    [Output tokens (teacher-forced)]
     │                                    │
[Embedding + PosEnc]              [Embedding + PosEnc]
     │                                    │
[Self-Attention]×N                [Masked Self-Attention]×N
     │                                    │
[Feed Forward]                    [Cross-Attention ← Encoder]
     │                                    │
[Layer Norm]                      [Feed Forward]
     │                                    │
[Encoder output]              [Softmax → next token]

T5-small specs:
- 6 encoder layers, 6 decoder layers
- 8 attention heads
- d_model = 512
- ~60M parameters
```

### Fine-tuning Strategy

```
SQuAD Dataset (train split: ~87K examples)
    ↓ Use 20K subset (time constraint)
    ↓ Format: "answer: {a} context: {c}" → "{question}"
    ↓ Train for 3 epochs, batch_size=8, lr=3e-4
    ↓ Save checkpoint every 500 steps
    ↓ Validate on SQuAD dev set (2K samples)
```

**Training Configuration:**
```python
# config.py
CONFIG = {
    "model_name": "t5-small",
    "max_source_length": 512,
    "max_target_length": 64,
    "batch_size": 8,
    "learning_rate": 3e-4,
    "epochs": 3,
    "warmup_steps": 200,
    "save_steps": 500,
    "eval_steps": 500,
    "weight_decay": 0.01,
    "gradient_clip": 1.0,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}
```

**Why NOT build from scratch:**
- T5 took ~1 TPU year to pre-train
- Your value-add is the full pipeline, not reinventing the wheel
- Fine-tuning = you understand what you're doing + better results
- You still write the training loop manually = shows deep understanding

**What you DO implement yourself (to impress):**
- Manual training loop (no `Trainer` API)
- Custom learning rate scheduler (linear warmup + decay)
- Gradient accumulation (simulate larger batch)
- Checkpoint saving + loading
- Custom data collator with dynamic padding

---

## 4. 🗓️ 3-WEEK EXECUTION PLAN

---

### 📅 WEEK 1 — Foundation & Infrastructure (Days 1–7)

**Goal:** Have a complete non-ML pipeline working by end of week.

| Day | Task | Deliverable |
|-----|------|-------------|
| **Day 1** | Set up project structure, virtual env, dependencies | `requirements.txt`, folder skeleton |
| **Day 2** | Build Text Extraction module (PDF, PPTX, TXT) | `file_extractor.py` tested on 3 file types |
| **Day 3** | Build Preprocessing module (clean, tokenize, filter) | `preprocessor.py` with unit tests |
| **Day 4** | Build Sentence Ranking module (TF-IDF + cosine) | `sentence_ranker.py` + visual score output |
| **Day 5** | Build Answer Extraction module (NER + noun phrases) | `answer_extractor.py` tested on 10 sentences |
| **Day 6** | Build Difficulty Estimator | `difficulty_estimator.py` with test cases |
| **Day 7** | Integration test of pipeline (Week 1 modules only) | End-to-end test: PDF → sentences → answers → difficulty |

**Week 1 Checkpoint:**
- Feed a PDF → get ranked sentences + answer candidates + difficulty tags
- All modules tested independently with sample inputs

---

### 📅 WEEK 2 — Core ML: Dataset, Training, Generation (Days 8–14)

**Goal:** Fine-tuned T5 model generating questions.

| Day | Task | Deliverable |
|-----|------|-------------|
| **Day 8** | Download + explore SQuAD v1.1, understand schema | EDA notebook, data stats |
| **Day 9** | Write SQuAD preprocessing script → T5 input format | `data_processor.py`, processed 20K pairs |
| **Day 10** | Implement `QGDataset`, `DataLoader`, `T5Tokenizer` setup | `dataset.py` tested |
| **Day 11** | Write training loop from scratch (loss, optimizer, scheduler) | `train_qg.py` first run |
| **Day 12** | Full training run (3 epochs on 20K samples) | Saved model checkpoint |
| **Day 13** | Inference module + beam search generation | `question_generator.py`, 10 test outputs |
| **Day 14** | Build MCQ Distractor generator + Output Formatter | `distractor_gen.py`, `output_formatter.py` |

**Week 2 Checkpoint:**
- Fine-tuned T5 generating questions from (context, answer) pairs
- MCQs being formatted correctly
- Training loss curve plotted

---

### 📅 WEEK 3 — Integration, Evaluation, UI & Polish (Days 15–21)

**Goal:** Complete, demo-ready system.

| Day | Task | Deliverable |
|-----|------|-------------|
| **Day 15** | Integrate all modules into `pipeline.py` main controller | `pipeline.py` — end-to-end working |
| **Day 16** | Build Evaluation module (BLEU, ROUGE, BERTScore) | `evaluator.py` + results on 100 SQuAD samples |
| **Day 17** | Build Streamlit UI — file upload + question display | `app/streamlit_app.py` |
| **Day 18** | Add FAQ generation (question type classifier/heuristic) | FAQ output working |
| **Day 19** | Testing with real documents (lecture notes, Wikipedia articles) | Demo video material |
| **Day 20** | Generate evaluation report + plots (loss curve, BLEU scores) | `evaluation_report.md` |
| **Day 21** | Clean code, write README, prepare demo + slides | Submission-ready project |

**Week 3 Checkpoint:**
- Demo-ready Streamlit app
- Evaluation metrics computed and visualized
- README with setup instructions

---

## 5. 📁 PROJECT FOLDER STRUCTURE

```
aqg-project/
│
├── 📂 data/
│   ├── raw/
│   │   ├── squad_train.json          # SQuAD v1.1 train
│   │   └── squad_dev.json            # SQuAD v1.1 dev
│   ├── processed/
│   │   ├── train_pairs.json          # (context, answer, question) triplets
│   │   └── dev_pairs.json
│   └── sample_inputs/
│       ├── sample.pdf
│       ├── sample.pptx
│       └── sample.txt
│
├── 📂 preprocessing/
│   ├── __init__.py
│   ├── file_extractor.py             # PDF/PPTX/TXT extraction
│   ├── preprocessor.py               # Text cleaning + sentence tokenization
│   ├── sentence_ranker.py            # TF-IDF importance ranking
│   └── answer_extractor.py           # NER + noun phrase extraction
│
├── 📂 models/
│   ├── __init__.py
│   ├── dataset.py                    # QGDataset class
│   ├── train_qg.py                   # Training loop
│   ├── question_generator.py         # Inference module
│   └── checkpoints/                  # Saved model weights
│       └── t5_qg_epoch3.pt
│
├── 📂 generation/
│   ├── __init__.py
│   ├── distractor_generator.py       # MCQ distractors (WordNet)
│   ├── difficulty_estimator.py       # Rule-based difficulty
│   └── output_formatter.py           # Format MCQ/Short/FAQ output
│
├── 📂 evaluation/
│   ├── __init__.py
│   ├── evaluator.py                  # BLEU, ROUGE, BERTScore
│   ├── run_eval.py                   # Evaluation script
│   └── results/
│       ├── eval_results.json
│       └── plots/
│           ├── loss_curve.png
│           └── bleu_rouge_scores.png
│
├── 📂 app/
│   ├── streamlit_app.py              # Main UI
│   ├── api.py                        # FastAPI (optional)
│   └── static/
│       └── style.css
│
├── 📂 notebooks/
│   ├── 01_data_exploration.ipynb     # SQuAD EDA
│   ├── 02_preprocessing_demo.ipynb   # Module testing
│   ├── 03_training_analysis.ipynb    # Loss curves
│   └── 04_evaluation_analysis.ipynb  # Metric results
│
├── 📂 config/
│   └── config.py                     # All hyperparameters
│
├── pipeline.py                       # Master orchestrator
├── requirements.txt
└── README.md
```

---

## 6. 📊 DATASET STRATEGY

### SQuAD v1.1 — Primary Dataset

**Why SQuAD:**
- 87,599 question-answer pairs from 536 Wikipedia articles
- Each entry has: `context`, `question`, `answer` (with char position)
- Perfect fit for (context, answer) → question generation

**Download:**
```bash
# SQuAD v1.1
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
```

**Schema:**
```json
{
  "data": [{
    "title": "Article Title",
    "paragraphs": [{
      "context": "Full paragraph text...",
      "qas": [{
        "question": "What is X?",
        "answers": [{"text": "X", "answer_start": 42}]
      }]
    }]
  }]
}
```

**Preprocessing Script:**
```python
# data/preprocess_squad.py
import json

def load_squad(path, max_samples=20000):
    with open(path) as f:
        data = json.load(f)
    
    pairs = []
    for article in data['data']:
        for para in article['paragraphs']:
            context = para['context']
            for qa in para['qas']:
                question = qa['question']
                if qa['answers']:
                    answer = qa['answers'][0]['text']
                    pairs.append({
                        "context": context,
                        "answer": answer,
                        "question": question
                    })
                if len(pairs) >= max_samples:
                    return pairs
    return pairs
```

**Subset Strategy (20K for training, 2K for eval):**
- Filter pairs where answer length > 1 word (avoid trivial)
- Filter pairs where context length < 450 tokens
- Shuffle before splitting

**Input → Output Format:**
```
INPUT:  "answer: {answer_text}  context: {paragraph_text}"
OUTPUT: "{question_text}"

Example:
INPUT:  "answer: 1969  context: The Apollo 11 mission landed on the Moon in 1969..."
OUTPUT: "In what year did the Apollo 11 mission land on the Moon?"
```

---

## 7. 🧪 EVALUATION STRATEGY

### Metrics Used

| Metric | What It Measures | Good Score |
|--------|-----------------|------------|
| **BLEU-1** | 1-gram overlap | > 0.40 |
| **BLEU-4** | 4-gram precision | > 0.15 |
| **ROUGE-1** | Recall of unigrams | > 0.45 |
| **ROUGE-L** | Longest common subsequence | > 0.38 |
| **BERTScore F1** | Semantic similarity | > 0.85 |

### Evaluation Dimensions

**1. Linguistic Quality**
- Are questions grammatically correct?
- Do they make sense in isolation?

**2. Relevance**
- Is the question answerable from the context?
- Is the generated answer correct?

**3. Diversity**
- Are questions varied (not all "What is...")?
- Do they cover different aspects of the text?

**4. Difficulty Accuracy (Manual)**
- Human spot-check: do "hard" questions actually feel harder?

### Running Evaluation

```python
# evaluation/run_eval.py
from evaluator import evaluate_batch
import json

# Load SQuAD dev set reference questions
with open("data/processed/dev_pairs.json") as f:
    dev_data = json.load(f)

references = [item["question"] for item in dev_data[:500]]
generated = []  # Your model's output

for item in dev_data[:500]:
    gen_q = generate_question(item["context"], item["answer"], 
                               model, tokenizer, device)
    generated.append(gen_q)

results = evaluate_batch(references, generated)

# BERTScore
bs = compute_bertscore(references, generated)
print(f"BERTScore F1: {bs['f1']:.4f}")
```

### Plotting Results

```python
import matplotlib.pyplot as plt

def plot_training_loss(loss_history):
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, color='steelblue', linewidth=2)
    plt.title("Training Loss Curve — T5-small QG Fine-tuning")
    plt.xlabel("Steps"); plt.ylabel("Cross-Entropy Loss")
    plt.grid(True, alpha=0.3)
    plt.savefig("evaluation/results/plots/loss_curve.png", dpi=150)
```

---

## 8. 🚀 ENHANCEMENTS (IF TIME PERMITS)

### 1. Multi-Question Type Control (High Impact)

Add a prefix to T5 input to control question type:
```
"generate mcq: answer: {a} context: {c}"
"generate short: answer: {a} context: {c}"
"generate why: answer: {a} context: {c}"
```

### 2. Answer-Agnostic Question Generation

Instead of providing the answer, let the model generate both:
```
Input: "generate questions: {context}"
Output: "Q1: ...\nA1: ...\nQ2: ...\nA2: ..."
```

### 3. Better Distractor Generation with Sense2Vec

```python
# Instead of WordNet (too generic):
import sense2vec
s2v = sense2vec.load()
query = f"{answer}|NOUN"
similar = s2v.most_similar(query, n=10)
distractors = [w.split("|")[0] for w, _ in similar if w != answer]
```

### 4. Reading Comprehension Score Check

Use a QA model to verify generated questions are answerable:
```python
from transformers import pipeline
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased")

def verify_question(context, question, expected_answer):
    result = qa_pipeline(question=question, context=context)
    # Check if predicted answer matches expected
    return expected_answer.lower() in result['answer'].lower()
```

### 5. Question Diversity Filter

Cluster questions and pick the most diverse set:
```python
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

def diversify_questions(questions, n_clusters=5):
    tfidf = TfidfVectorizer().fit_transform(questions)
    kmeans = KMeans(n_clusters=min(n_clusters, len(questions)))
    kmeans.fit(tfidf)
    
    # Pick question closest to each cluster center
    selected = []
    for c in range(kmeans.n_clusters):
        cluster_idxs = [i for i, l in enumerate(kmeans.labels_) if l == c]
        selected.append(questions[cluster_idxs[0]])
    return selected
```

---

## 9. 💬 INTERVIEW PREPARATION

---

### TIER 1 — Concepts You Must Know Cold

**Transformers:**
- Q: Explain the attention mechanism mathematically.
  - A: `Attention(Q,K,V) = softmax(QKᵀ/√d_k)V`
  - Know: why √d_k scaling, what Q/K/V represent, multi-head purpose

- Q: Why is self-attention O(n²) in complexity?
  - A: Every token attends to every other token → n×n attention matrix

- Q: What is positional encoding and why is it needed?
  - A: Transformers have no recurrence, so position info must be injected

- Q: What is the difference between encoder-only, decoder-only, encoder-decoder?
  - Encoder-only: BERT (classification/understanding)
  - Decoder-only: GPT (generation)
  - Encoder-Decoder: T5/BART (seq2seq translation/summarization/QG)

**T5 Specifically:**
- Q: Why is T5's "text-to-text" framing powerful?
  - A: Unifies all NLP tasks under one format → same model for many tasks

- Q: How does teacher forcing work in seq2seq training?
  - A: Feed ground truth tokens as decoder input at each step, not model predictions

- Q: What is beam search and why use it over greedy?
  - A: Maintains k best hypotheses at each step → higher quality output

**NLP Pipeline:**
- Q: What is TF-IDF and what are its limitations?
  - A: Term frequency × inverse document frequency; doesn't capture semantics

- Q: Why use cosine similarity for sentence ranking?
  - A: Direction of vector matters more than magnitude; normalized comparison

- Q: What is NER and what model powers spaCy's en_core_web_sm?
  - A: Named Entity Recognition; uses CNN + transition-based parser

**Training:**
- Q: Why do we set labels = -100 for padding tokens?
  - A: PyTorch's CrossEntropyLoss ignores index -100 → no gradient from padding

- Q: What is gradient accumulation and when do you use it?
  - A: Sum gradients over N mini-batches before stepping → simulate larger batch

- Q: What is learning rate warmup?
  - A: Gradually increase LR from 0 → target over first K steps → stable training

---

### TIER 2 — System Design Questions

- Q: How would you scale this system to handle 1000-page documents?
  - A: Chunk document into overlapping windows (512 tokens), process in parallel, deduplicate output questions

- Q: How would you improve MCQ quality beyond WordNet distractors?
  - A: Sense2vec, fine-tune T5 for distractor generation, use TF-IDF to find in-corpus distractors

- Q: How does your difficulty estimation compare to supervised approaches?
  - A: Rule-based is interpretable and zero-shot; supervised requires labeled difficulty data but learns nuanced patterns

- Q: What are the limitations of BLEU for evaluating question generation?
  - A: BLEU measures n-gram overlap but not semantic correctness, question type, or answerability. BERTScore better captures meaning.

- Q: How would you evaluate whether a generated question is unanswerable?
  - A: Run a QA model on (context, question) → if confidence < threshold → unanswerable

---

### TIER 3 — Likely Technical Deep-Dives

1. Walk me through your fine-tuning loop in `train_qg.py`
2. Why did you choose T5-small over BART?
3. How does your sentence ranking handle repeated sentences?
4. What tokenizer does T5 use? (SentencePiece / Unigram LM)
5. What is cross-attention and how does the decoder use encoder output?
6. How does your NER-based answer extraction avoid extracting stop words?
7. What would you change if you had 3 months instead of 3 weeks?
8. How would you add multilingual support to this system?

---

### Key Formulas to Know

| Formula | Name |
|---------|------|
| `Attention(Q,K,V) = softmax(QKᵀ/√d_k)V` | Scaled Dot-Product Attention |
| `BLEU = BP × exp(Σ wₙ log pₙ)` | BLEU Score |
| `TF-IDF(t,d) = TF(t,d) × log(N/df(t))` | TF-IDF |
| `cosine_sim(A,B) = (A·B)/(‖A‖‖B‖)` | Cosine Similarity |
| `Flesch = 206.835 - 1.015(words/sent) - 84.6(syllables/word)` | Readability |

---

## REQUIREMENTS.TXT

```txt
torch>=2.0.0
transformers>=4.35.0
datasets>=2.14.0
sentencepiece>=0.1.99
tokenizers>=0.14.0
spacy>=3.6.0
nltk>=3.8.1
scikit-learn>=1.3.0
numpy>=1.24.0
pandas>=2.0.0
textstat>=0.7.3
rouge-score>=0.1.2
bert-score>=0.3.13
PyMuPDF>=1.23.0
python-pptx>=0.6.21
streamlit>=1.28.0
fastapi>=0.104.0
uvicorn>=0.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
chardet>=5.2.0
```

---

## README.md OUTLINE

```markdown
# Context-Aware Automatic Question Generation

## Overview
End-to-end system that generates MCQs, short-answer, and FAQ questions
from any text document using fine-tuned T5-small + custom NLP pipeline.

## Setup
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt wordnet

## Train
python models/train_qg.py --data_path data/processed/train_pairs.json

## Run
streamlit run app/streamlit_app.py

## Evaluate
python evaluation/run_eval.py

## Architecture
[Diagram here]
```

---

*End of Project Plan — Good luck! Build it step by step.*
