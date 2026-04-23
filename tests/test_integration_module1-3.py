"""
INTEGRATION TEST: Modules 1 + 2 + 3
===================================
End-to-end pipeline: Text Extraction → Preprocessing → Sentence Ranking
"""

import logging
from preprocessing import (
    TextExtractor,
    preprocess,
    rank_sentences,
    get_ranking_stats,
)

def test_integration_pipeline():
    """
    Test the complete pipeline from raw text to ranked sentences.
    """
    print("="*80)
    print("INTEGRATION TEST: Modules 1 + 2 + 3")
    print("="*80 + "\n")
    
    # Simulate Module 1 output (text extraction)
    # In real scenario: text = TextExtractor().extract("document.pdf")
    raw_text = """
    Artificial Intelligence and Machine Learning are transforming modern industries.
    
    Machine learning is a subset of Artificial Intelligence where machines learn from data
    without being explicitly programmed. It has applications in healthcare, finance, and
    education. Deep learning, a subset of machine learning, uses neural networks with
    multiple layers to process complex data.
    
    The transformer architecture, introduced by Vaswani et al., revolutionized natural
    language processing. BERT and GPT models achieved state-of-the-art results on various
    NLP benchmarks. These models can perform tasks like translation, question answering,
    and sentiment analysis.
    
    Data preprocessing is crucial for model performance. Quality data leads to better models.
    Python is the most popular programming language for AI and machine learning development.
    TensorFlow and PyTorch are widely used frameworks.
    
    In conclusion, AI and ML will continue to advance rapidly. Organizations should invest
    in AI talent and infrastructure to stay competitive.
    """
    
    print("[STEP 1] TEXT EXTRACTION (Module 1)")
    print("-" * 80)
    print(f"Raw text length: {len(raw_text)} characters")
    print(f"Preview: {raw_text[:100]}...\n")
    
    # Module 2: Preprocessing
    print("[STEP 2] TEXT PREPROCESSING (Module 2)")
    print("-" * 80)
    sentences = preprocess(raw_text, log_level=logging.WARNING)
    print(f"Extracted {len(sentences)} sentences:\n")
    for i, sent in enumerate(sentences[:5], 1):
        print(f"  {i}. {sent[:70]}...")
    if len(sentences) > 5:
        print(f"  ... and {len(sentences) - 5} more sentences\n")
    else:
        print()
    
    # Module 3: Sentence Ranking
    print("[STEP 3] SENTENCE RANKING (Module 3)")
    print("-" * 80)
    ranked = rank_sentences(sentences, top_k=5, log_level=logging.WARNING)
    stats = get_ranking_stats(ranked)
    
    print(f"Ranked {stats['count']} top sentences:\n")
    for i, (sent, score) in enumerate(ranked, 1):
        word_count = len(sent.split())
        print(f"  {i}. [{score:.4f}] ({word_count} words)")
        print(f"     {sent}\n")
    
    print(f"Ranking Statistics:")
    print(f"  - Average importance score: {stats['avg_score']:.4f}")
    print(f"  - Score range: {stats['score_range'][0]:.4f} - {stats['score_range'][1]:.4f}")
    print(f"  - Average sentence length: {stats['avg_length']:.1f} words\n")
    
    print("="*80)
    print("INTEGRATION TEST COMPLETED SUCCESSFULLY")
    print("="*80 + "\n")
    
    print("Pipeline Summary:")
    print(f"  Raw text: {len(raw_text)} chars")
    print(f"  → Preprocessed: {len(sentences)} sentences")
    print(f"  → Top-ranked: {stats['count']} sentences (importance-ordered)")
    print()


if __name__ == "__main__":
    test_integration_pipeline()
