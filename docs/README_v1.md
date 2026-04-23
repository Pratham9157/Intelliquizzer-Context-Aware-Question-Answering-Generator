# 🧠 Context-Aware Automatic Question Generation (AQG)

## Overview

AQG is a fully self-contained system for generating multiple choice questions, short-answer questions, and FAQs from any text document using fine-tuned T5-small transformer model combined with custom NLP pipeline.

## ✨ Features

- **🔄 Fully Self-Contained**: All dependencies automatically installed on first use
- **📄 Multi-Format Support**: PDF, PowerPoint, TXT files
- **🤖 Intelligent Preprocessing**: Unicode normalization, noise filtering, sentence ranking
- **🎯 Question Generation**: Fine-tuned T5-small for high-quality questions
- **📊 Multiple Question Types**: MCQ, Short Answer, FAQ
- **⚡ Production Ready**: Clean, modular, well-tested code

## 🚀 Quick Start

### 1. Clone & Setup (One-Time)

```bash
# Clone the repository
git clone <repo-url>
cd aqg-project

# Option A: Automatic setup (recommended)
python -m setup --verbose

# Option B: Manual pip install
pip install -r requirements.txt
python -m nltk.downloader punkt_tab wordnet
```

### 2. Basic Usage

```python
from preprocessing import TextExtractor, preprocess

# Extract text from PDF
extractor = TextExtractor()
text = extractor.extract("document.pdf")

# Preprocess into sentences
sentences = preprocess(text)

print(f"Extracted {len(sentences)} quality sentences")
```

That's it! All dependencies are automatically installed on first import.

## 📁 Project Structure

```
aqg-project/
├── preprocessing/           # Text extraction & preprocessing
│   ├── __init__.py
│   ├── file_extractor.py   # PDF/PPTX/TXT extraction
│   ├── preprocessor.py     # Text cleaning & tokenization
│   ├── sentence_ranker.py  # (Coming soon)
│   ├── answer_extractor.py # (Coming soon)
│   └── INFO.md             # Module documentation
│
├── models/                  # ML models (coming soon)
│   ├── dataset.py
│   ├── train_qg.py
│   └── question_generator.py
│
├── generation/              # Question generation (coming soon)
│   ├── difficulty_estimator.py
│   ├── distractor_generator.py
│   └── output_formatter.py
│
├── evaluation/              # Evaluation metrics (coming soon)
│   └── evaluator.py
│
├── setup.py                # Automatic dependency setup
├── requirements.txt        # Python dependencies
├── __init__.py            # Package initialization
└── README.md              # This file
```

## 🔧 Auto-Setup System

The project includes a fully automated setup system that:

1. **Checks Python version** (requires 3.7+)
2. **Installs missing packages** from requirements.txt
3. **Downloads NLTK resources** (punkt_tab, wordnet, etc.)
4. **Validates environment** before code runs

### Automatic Setup (No User Action Required)

```python
# First import - automatically sets up dependencies
from preprocessing import preprocess

# Everything is ready to use
sentences = preprocess(raw_text)
```

### Manual Setup (If Needed)

```bash
# Run setup interactively
python -m setup --verbose

# Check status
python -m setup
```

## 📚 Module Documentation

Each module has comprehensive documentation in `INFO.md`:

- **Module 1**: [Text Extraction](preprocessing/INFO.md#module-1-text-extraction-layer)
- **Module 2**: [Preprocessing](preprocessing/INFO.md#module-2-text-preprocessing-module)
- **Module 3**: Sentence Ranking (coming soon)
- **Module 4**: Answer Extraction (coming soon)
- **Module 5**: Question Generation (coming soon)

## 🧪 Testing

### Run All Tests

```bash
# Test preprocessing module
python preprocessing/preprocessor.py

# Test file extraction
python preprocessing/file_extractor.py
```

### Example Test Output

```
================================================================================
PREPROCESSING MODULE - TEST SUITE
================================================================================

[TEST 1] Messy text with extra spaces and special characters...
[OK] Successfully processed 4 sentences
  1. Natural Language Processing is a field of AI.
  2. It involves working with text data.
  ...

[TEST 7] Full pipeline with statistics...
[OK] Pipeline complete!
  Results:
    - Sentences: 7
    - Total words: 53
    - Avg sentence length: 7.6 words
    ...
```

## 💻 Requirements

### System
- Python 3.7+
- 2GB RAM (minimum)
- 500MB disk space for dependencies

### Automatic Installation
- All Python packages from `requirements.txt`
- NLTK data resources (punkt_tab, wordnet, etc.)

**No manual installation required!** Everything is installed automatically.

## 📖 Usage Examples

### Example 1: Extract and Preprocess PDF

```python
from preprocessing import TextExtractor, preprocess
import logging

# Setup extraction (auto-downloads PyMuPDF if needed)
extractor = TextExtractor(log_level=logging.DEBUG)

# Extract text from PDF
text = extractor.extract("lecture.pdf")

# Preprocess into sentences
sentences = preprocess(text)

print(f"Got {len(sentences)} sentences from PDF")
```

### Example 2: Batch Process Multiple Files

```python
from pathlib import Path
from preprocessing import TextExtractor, preprocess, get_preprocessing_stats

extractor = TextExtractor()
results = {}

for pdf_file in Path("documents").glob("*.pdf"):
    try:
        text = extractor.extract(str(pdf_file))
        sentences = preprocess(text)
        stats = get_preprocessing_stats(sentences)
        
        results[pdf_file.name] = {
            'sentences': len(sentences),
            'avg_length': stats['avg_sentence_length'],
        }
        print(f"✓ {pdf_file.name}: {len(sentences)} sentences")
    except Exception as e:
        print(f"✗ {pdf_file.name}: {e}")

# Print summary
for file, data in results.items():
    print(f"{file}: {data['sentences']} sentences")
```

### Example 3: Check Preprocessing Status

```python
from setup import get_setup_status

status = get_setup_status()
print(f"Python: {status['python_version']}")
print(f"Missing packages: {len(status['missing_packages'])}")
print(f"NLTK ready: {status['nltk_available']}")
print(f"Overall: {'Ready ✓' if status['all_ready'] else 'Setup needed'}")
```

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'nltk'"

**Solution**: The auto-setup failed. Run manually:
```bash
python -m setup --verbose
```

### Issue: "NLTK resource 'punkt_tab' not found"

**Solution**: NLTK data download failed. Run:
```bash
python -c "import nltk; nltk.download('punkt_tab')"
```

### Issue: "PyMuPDF not installed"

**Solution**: Run auto-setup:
```bash
python -m setup --verbose
```

### Issue: "Permission denied" during installation

**Solution**: Use user-level installation:
```bash
pip install --user -r requirements.txt
```

## 📈 Roadmap

- [x] Module 1: Text Extraction (Complete)
- [x] Module 2: Preprocessing (Complete)
- [x] Auto-Setup System (Complete)
- [ ] Module 3: Sentence Ranking
- [ ] Module 4: Answer Extraction
- [ ] Module 5: Question Generation Model
- [ ] Module 6: Difficulty Estimation
- [ ] Module 7: Distractor Generation
- [ ] Module 8: Evaluation
- [ ] Module 9: Streamlit UI

## 🤝 Contributing

This project is built module-by-module with:
- Clean, testable code
- Comprehensive documentation
- Built-in test suites
- Production-ready quality

Each module is independent and can be used separately.

## 📄 License

This is a semester project for educational purposes.

## 📞 Support

For issues or questions:
1. Check [Module Documentation](preprocessing/INFO.md)
2. Run tests to validate your setup
3. Check troubleshooting section above

## 🎓 Learning Goals

This project demonstrates:
- Modular code design
- NLP pipeline architecture
- Transformer fine-tuning
- Data preprocessing best practices
- Production code quality
- Automated dependency management

---

**Status**: 🟢 Ready to use (Modules 1-2 complete)

**Last Updated**: 2026-04-22
