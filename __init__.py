"""
IntelliQuizzer v4 — Context-Aware Automatic Question Generation
================================================================

Quick start:
    from intelliquizzer.pipeline import IntelliQuizzerPipeline
    pipe = IntelliQuizzerPipeline()
    mcqs, meta = pipe.run("Your text here...", max_questions=15)
"""
__version__ = "4.0.0"
__author__  = "IntelliQuizzer"

from .pipeline import IntelliQuizzerPipeline
