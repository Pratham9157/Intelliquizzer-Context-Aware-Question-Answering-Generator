"""
AQG (Automatic Question Generation) Project
============================================

Context-Aware Automatic Question Generation using Transformer-based Sequence Modeling

This package is fully self-contained. All dependencies are automatically installed
and configured on first use.

Quick Start:
    1. Clone the repository
    2. Run: python -m setup
    3. Import and use: from preprocessing import preprocess
    
All dependencies will be automatically downloaded and installed.
"""

__version__ = "1.0.0"
__author__ = "AI Development Team"

# Auto-setup on first import
try:
    from setup import ensure_dependencies, get_setup_status
    
    # Ensure all dependencies are available
    if not ensure_dependencies():
        import warnings
        warnings.warn(
            "Some dependencies failed to install. "
            "Run: python -m setup --verbose"
        )
    
    # Import submodules
    from preprocessing import TextExtractor, preprocess
    
    __all__ = [
        'TextExtractor',
        'preprocess',
        'get_setup_status',
    ]

except Exception as e:
    import warnings
    warnings.warn(f"Failed to initialize AQG package: {e}")
    __all__ = []
