#!/usr/bin/env python3
"""
Setup script for Advanced Multi-Modal AI Engineering System
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ai-engineering-system",
    version="1.0.0",
    author="Joshua Terranova",
    author_email="joshua@example.com",
    description="Advanced Multi-Modal AI System for Engineering Applications",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/joshuaterranova/ai-engineering-system",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.9.0",
            "flake8>=6.1.0",
            "mypy>=1.6.0",
            "pre-commit>=3.5.0",
        ],
        "gpu": [
            "cupy-cuda12x>=12.2.0",
            "tensorflow-gpu>=2.15.0",
        ],
        "docs": [
            "sphinx>=7.1.0",
            "sphinx-rtd-theme>=1.3.0",
            "myst-parser>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ai-engineering=ai_engineering_system.cli:main",
            "ai-setup=ai_engineering_system.setup:download_models",
        ],
    },
    include_package_data=True,
    package_data={
        "ai_engineering_system": [
            "data/*.json",
            "models/*.pkl",
            "configs/*.yaml",
            "templates/*.html",
        ],
    },
    zip_safe=False,
)
