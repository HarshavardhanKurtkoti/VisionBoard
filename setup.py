import os
import re
from setuptools import find_packages, setup
from typing import Dict, List

NAME = "visionboard"
DESCRIPTION = "Production-ready YOLOv8 and OCR object detection system for signboard analysis"
AUTHOR = "Harshavardhan Kurtkoti"
AUTHOR_EMAIL = "harshavardhan.kurtkoti@gmail.com"
URL = "https://github.com/HarshavardhanKurtkoti/VisionBoard"
REQUIRES_PYTHON = ">=3.8.0"
VERSION = "0.1.0"

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
    "matplotlib>=3.5.0",
    "pytesseract>=0.3.10",
    "scikit-learn>=1.0.0"
]

EXTRAS = {
    "dev": [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "black>=22.0.0",
        "flake8>=4.0.0"
    ]
}

def get_long_description() -> str:
    """Get long description from README.md"""
    try:
        with open("README.md", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return DESCRIPTION

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license="MIT",
    entry_points={
        "console_scripts": [
            f"{NAME}=main:main",
        ],
    },
    zip_safe=False
)