import os
import re
from setuptools import find_packages, setup
from typing import Dict, List, Optional

# Package meta-data
NAME = "visionboard"
DESCRIPTION = "YOLOv8-based object detection system for signboard detection and analysis"
AUTHOR = "Harshavardhan Kurtkoti"
AUTHOR_EMAIL = "harshavardhan.kurtkoti@gmail.com"
URL = "https://github.com/HarshavardhanKurtkoti/VisionBoard"
REQUIRES_PYTHON = ">=3.8.0"
VERSION = "0.1.0"

# Required packages for different environments
REQUIRED = [
    "numpy>=1.21.0",
    "opencv-python>=4.5.0",
    "torch>=2.0.0",
    "ultralytics>=8.0.0",
    "boto3>=1.26.0",
    "PyYAML>=6.0",
    "tqdm>=4.65.0",
    "pandas>=1.5.0",
    "python-dotenv>=1.0.0",
    "pillow>=9.0.0",
    "requests>=2.28.0",
    "matplotlib>=3.5.0"
]

# Optional packages
EXTRAS = {
    "dev": [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "black>=22.0.0",
        "isort>=5.10.0",
        "flake8>=4.0.0",
        "mypy>=0.950",
        "pre-commit>=2.20.0",
    ],
    "docs": [
        "sphinx>=4.5.0",
        "sphinx-rtd-theme>=1.0.0",
        "sphinx-autodoc-typehints>=1.18.0",
    ],
    "training": [
        "wandb>=0.13.0",  # For experiment tracking
        "albumentations>=1.3.0",  # For advanced augmentations
        "scikit-learn>=1.0.0",  # For metrics and evaluation
    ]
}

def get_version() -> str:
    """Get package version from __init__.py"""
    init_py = open(os.path.join(NAME, "__init__.py")).read()
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", init_py, re.M)
    if version_match:
        return version_match.group(1)
    return VERSION

def get_long_description() -> str:
    """Get long description from README.md"""
    try:
        with open("README.md", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return DESCRIPTION

def get_requirements(requirements_path: str = "requirements.txt") -> List[str]:
    """
    Get list of requirements from requirements.txt
    Args:
        requirements_path: Path to requirements file
    Returns:
        List[str]: List of required packages
    """
    try:
        with open(requirements_path, "r", encoding="utf-8") as f:
            requirements = [
                line.strip()
                for line in f
                if line.strip() and not line.startswith(("#", "-e", "git+"))
            ]
        return requirements
    except FileNotFoundError:
        return REQUIRED

def get_package_data() -> Dict[str, List[str]]:
    """Get package data files"""
    return {
        NAME: [
            "config/*.yaml",  # Configuration files
            "data/*.json",    # Data files
            "models/*.pt",    # Model files
            "utils/*.py",     # Utility modules
        ]
    }

setup(
    name=NAME,
    version=get_version(),
    description=DESCRIPTION,
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    package_data=get_package_data(),
    install_requires=get_requirements(),
    extras_require=EXTRAS,
    include_package_data=True,
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords=[
        "computer-vision",
        "object-detection",
        "yolov8",
        "deep-learning",
        "machine-learning",
        "signboard-detection",
        "pytorch",
    ],
    project_urls={
        "Documentation": f"{URL}/docs",
        "Source": URL,
        "Tracker": f"{URL}/issues",
        "Changelog": f"{URL}/blob/main/CHANGELOG.md",
    },
    entry_points={
        "console_scripts": [
            f"{NAME}={NAME}.main:main",
        ],
    },
    zip_safe=False,
    platforms="any"
)