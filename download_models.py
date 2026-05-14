"""
Run this once after pip install to pre-download all models.
They cache in ~/.cache/huggingface/ and never re-download.

Usage:
    python download_models.py
"""

print("=" * 60)
print("IntelliQuizzer — Model Downloader")
print("=" * 60)
print()

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# M5 — Question Generator (~850 MB)
print("[1/4] Downloading Question Generation model (valhalla/t5-base-qg-hl)...")
AutoTokenizer.from_pretrained("valhalla/t5-base-qg-hl")
AutoModelForSeq2SeqLM.from_pretrained("valhalla/t5-base-qg-hl")
print("      ✅ Done\n")

# M4 — BERT NER (~430 MB)
print("[2/4] Downloading NER model (dslim/bert-base-NER)...")
pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
print("      ✅ Done\n")

# M4+M6 — Sentence embeddings (~80 MB)
print("[3/4] Downloading embedding model (all-MiniLM-L6-v2)...")
from sentence_transformers import SentenceTransformer
SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("      ✅ Done\n")

# QA Tab — RoBERTa (~480 MB)
print("[4/4] Downloading QA model (deepset/roberta-base-squad2)...")
pipeline("question-answering", model="deepset/roberta-base-squad2")
print("      ✅ Done\n")

print("=" * 60)
print("All models downloaded. Run:  streamlit run app.py")
print("=" * 60)
