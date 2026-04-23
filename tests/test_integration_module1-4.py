"""
INTEGRATION TEST: Modules 1 + 2 + 3 + 4
========================================
End-to-end pipeline: Text Extraction → Preprocessing → Sentence Ranking → Answer Extraction
"""

import logging
from preprocessing import (
    preprocess,
    rank_sentences,
)
from generation import extract_answers, get_answer_stats

def test_integration_pipeline_full():
    """
    Test the complete pipeline from raw text to extracted answers.
    """
    print("="*80)
    print("INTEGRATION TEST: Modules 1 + 2 + 3 + 4")
    print("="*80 + "\n")
    
    # Simulate Module 1 output (text extraction)
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
    print(f"Extracted {len(sentences)} sentences\n")
    
    # Module 3: Sentence Ranking
    print("[STEP 3] SENTENCE RANKING (Module 3)")
    print("-" * 80)
    ranked = rank_sentences(sentences, top_k=5, log_level=logging.WARNING)
    print(f"Ranked {len(ranked)} top sentences\n")
    
    # Module 4: Answer Extraction
    print("[STEP 4] ANSWER EXTRACTION (Module 4)")
    print("-" * 80)
    ranked_sents = [sent for sent, _ in ranked]
    answers_results = extract_answers(ranked_sents, max_answers_per_sentence=3, log_level=logging.WARNING)
    stats = get_answer_stats(answers_results)
    
    print(f"Extracted answers from {len(ranked_sents)} ranked sentences:\n")
    
    for idx, (sent, rank_score) in enumerate(ranked, 1):
        print(f"{idx}. [Rank Score: {rank_score:.4f}] {sent}")
        
        if sent in answers_results:
            answers = answers_results[sent]
            if answers:
                print(f"   Extracted Answers ({len(answers)}):")
                for i, ans in enumerate(answers, 1):
                    print(f"     {i}. '{ans['answer']}' "
                          f"(Type: {ans['type']}, Score: {ans['score']:.3f}, Source: {ans['source']})")
            else:
                print(f"   No answers extracted")
        print()
    
    print(f"Overall Statistics:")
    print(f"  - Total sentences ranked: {stats['total_sentences']}")
    print(f"  - Total answers extracted: {stats['total_answers']}")
    print(f"  - Avg answers per sentence: {stats['avg_answers_per_sentence']:.1f}")
    print(f"  - Answer types distribution: {stats['answer_types_distribution']}")
    print(f"  - Extraction sources: {stats['source_distribution']}")
    print(f"  - Avg answer length: {stats['avg_answer_length']:.1f} characters\n")
    
    print("="*80)
    print("INTEGRATION TEST COMPLETED SUCCESSFULLY")
    print("="*80 + "\n")
    
    print("Pipeline Summary:")
    print(f"  Raw text: {len(raw_text)} chars")
    print(f"  → Preprocessed: {len(sentences)} sentences")
    print(f"  → Ranked top: {len(ranked)} sentences")
    print(f"  → Answers extracted: {stats['total_answers']} total ({stats['avg_answers_per_sentence']:.1f} per sentence)")
    print()


if __name__ == "__main__":
    test_integration_pipeline_full()
