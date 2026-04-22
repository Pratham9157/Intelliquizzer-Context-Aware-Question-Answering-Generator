"""
AUTOMATIC SETUP AND DEPENDENCY MANAGEMENT
==========================================

This module ensures all dependencies are automatically installed and configured
on first use. Users don't need to manually install packages or download NLTK data.

Features:
    - Auto-installs missing packages from requirements.txt
    - Auto-downloads NLTK resources (punkt_tab, wordnet, etc.)
    - Checks Python version
    - Provides setup status and logging
    - Can be imported safely (won't break if used in production)
"""

import sys
import subprocess
import importlib.util
import os
import logging
from pathlib import Path
from typing import List, Tuple

# Configure logging
logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS
# ============================================================================

PYTHON_MIN_VERSION = (3, 7)
PROJECT_ROOT = Path(__file__).parent
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"

# NLTK resources needed by the project
NLTK_RESOURCES = [
    'punkt_tab',      # Sentence tokenizer (new version)
    'punkt',          # Sentence tokenizer (fallback)
    'wordnet',        # Word synonyms for distractor generation
    'averaged_perceptron_tagger',  # POS tagging
]

# Packages that are optional (nice to have, but won't break core functionality)
OPTIONAL_PACKAGES = {
    'torch': 'CPU version sufficient for development',
    'spacy': 'en_core_web_sm model needed for NER',
    'streamlit': 'Only needed if running UI',
    'fastapi': 'Only needed if running API server',
}

# Core packages (required for basic functionality)
CORE_PACKAGES = {
    'nltk': 'Sentence tokenization',
    'scikit-learn': 'TF-IDF and similarity calculations',
    'numpy': 'Numerical operations',
}

# ============================================================================
# SETUP FUNCTIONS
# ============================================================================


def check_python_version() -> Tuple[bool, str]:
    """
    Check if Python version meets minimum requirements.
    
    Returns:
        (is_compatible, message)
    """
    current = sys.version_info[:2]
    if current >= PYTHON_MIN_VERSION:
        return True, f"Python {current[0]}.{current[1]} (OK)"
    else:
        return False, f"Python {current[0]}.{current[1]} (minimum: {PYTHON_MIN_VERSION[0]}.{PYTHON_MIN_VERSION[1]})"


def package_installed(package_name: str) -> bool:
    """
    Check if a package is installed.
    
    Args:
        package_name: Name of the package to check
        
    Returns:
        True if installed, False otherwise
    """
    spec = importlib.util.find_spec(package_name)
    return spec is not None


def get_missing_packages(requirements_file: Path) -> List[str]:
    """
    Parse requirements.txt and return list of missing packages.
    
    Args:
        requirements_file: Path to requirements.txt
        
    Returns:
        List of package names that are not installed
    """
    if not requirements_file.exists():
        logger.warning(f"requirements.txt not found at {requirements_file}")
        return []
    
    missing = []
    with open(requirements_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Parse "package>=version" -> "package"
            package_name = line.split('>')[0].split('<')[0].split('=')[0].split('!')[0].strip()
            
            if not package_installed(package_name):
                missing.append(package_name)
    
    return missing


def install_packages(package_names: List[str]) -> bool:
    """
    Install packages using pip.
    
    Args:
        package_names: List of package names to install
        
    Returns:
        True if successful, False otherwise
    """
    if not package_names:
        return True
    
    logger.info(f"Installing {len(package_names)} package(s): {', '.join(package_names)}")
    
    try:
        # Use pip to install packages
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q"] + package_names,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        logger.info(f"Successfully installed {len(package_names)} package(s)")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install packages: {e}")
        return False


def install_nltk_resources(resources: List[str]) -> bool:
    """
    Download NLTK data resources.
    
    Args:
        resources: List of resource names to download
        
    Returns:
        True if successful, False otherwise
    """
    try:
        import nltk
    except ImportError:
        logger.warning("NLTK not installed, skipping resource download")
        return False
    
    logger.info(f"Downloading NLTK resources: {', '.join(resources)}")
    
    all_success = True
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
            logger.debug(f"Downloaded NLTK resource: {resource}")
        except Exception as e:
            logger.debug(f"Failed to download NLTK resource '{resource}': {e}")
            all_success = False
    
    return all_success


def get_setup_status() -> dict:
    """
    Get comprehensive setup status.
    
    Returns:
        Dictionary with setup status information
    """
    # Check Python version
    python_ok, python_msg = check_python_version()
    
    # Check missing packages
    missing_packages = get_missing_packages(REQUIREMENTS_FILE)
    
    # Check NLTK resources
    nltk_ok = True
    try:
        import nltk
        for resource in NLTK_RESOURCES[:1]:  # Just check ersten resource
            try:
                nltk.data.find(f'tokenizers/{resource}')
            except LookupError:
                nltk_ok = False
                break
    except ImportError:
        nltk_ok = False
    
    return {
        'python_compatible': python_ok,
        'python_version': python_msg,
        'missing_packages': missing_packages,
        'nltk_available': nltk_ok,
        'all_ready': python_ok and not missing_packages and nltk_ok,
    }


def auto_setup(verbose: bool = False) -> bool:
    """
    Automatically set up the development environment.
    
    This function:
    1. Checks Python version
    2. Installs missing packages from requirements.txt
    3. Downloads NLTK resources
    
    Args:
        verbose: If True, print detailed status messages
        
    Returns:
        True if setup successful, False otherwise
    """
    if verbose:
        print("\n" + "="*70)
        print("AUTOMATIC SETUP - AQG PROJECT")
        print("="*70 + "\n")
    
    # Step 1: Check Python version
    python_ok, python_msg = check_python_version()
    if verbose:
        print(f"[CHECK] Python version: {python_msg}")
    
    if not python_ok:
        logger.error(f"Python version requirement not met: {python_msg}")
        return False
    
    # Step 2: Check and install missing packages
    missing = get_missing_packages(REQUIREMENTS_FILE)
    if missing:
        if verbose:
            print(f"[INSTALL] Missing {len(missing)} package(s): {', '.join(missing[:5])}{'...' if len(missing) > 5 else ''}")
        
        if not install_packages(missing):
            logger.error("Failed to install required packages")
            return False
        
        if verbose:
            print(f"[OK] All packages installed")
    else:
        if verbose:
            print(f"[OK] All packages already installed")
    
    # Step 3: Download NLTK resources
    if verbose:
        print(f"[DOWNLOAD] NLTK resources...")
    
    if not install_nltk_resources(NLTK_RESOURCES):
        logger.warning("Some NLTK resources failed to download (will use fallback)")
    else:
        if verbose:
            print(f"[OK] NLTK resources ready")
    
    if verbose:
        print("\n" + "="*70)
        print("SETUP COMPLETE - Ready to use!")
        print("="*70 + "\n")
    
    return True


def ensure_dependencies() -> bool:
    """
    Ensure all dependencies are available.
    
    This is a silent version of auto_setup() suitable for being called
    at module import time. It won't print anything unless there's an error.
    
    Returns:
        True if all dependencies available, False otherwise
    """
    status = get_setup_status()
    
    if status['all_ready']:
        return True
    
    logger.debug("Dependencies not fully set up, running auto_setup...")
    return auto_setup(verbose=False)


# ============================================================================
# CLI INTERFACE
# ============================================================================

if __name__ == "__main__":
    # Configure logging for CLI
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )
    
    # Run setup
    success = auto_setup(verbose=True)
    
    # Print status
    print("\nSetup Status:")
    status = get_setup_status()
    print(f"  Python version: {status['python_version']}")
    print(f"  Missing packages: {len(status['missing_packages'])}")
    print(f"  NLTK ready: {status['nltk_available']}")
    print(f"  Overall: {'Ready (OK)' if status['all_ready'] else 'Not ready'}")
    
    sys.exit(0 if status['all_ready'] else 1)
