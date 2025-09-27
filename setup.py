#!/usr/bin/env python3
"""
Setup script for Genuity OS - A comprehensive synthetic tabular data generation framework.
"""

from setuptools import setup, find_packages
import os


# Read the README file for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return "A comprehensive synthetic tabular data generation framework with multiple state-of-the-art algorithms."


# Read requirements from requirements.txt
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    if os.path.exists(requirements_path):
        with open(requirements_path, "r", encoding="utf-8") as f:
            return [
                line.strip() for line in f if line.strip() and not line.startswith("#")
            ]
    return []


setup(
    name="genuity-os",
    version="1.0.0",
    author="Genuity IO",
    author_email="shivansh231223@gmail.com",
    description="A comprehensive synthetic tabular data generation framework with multiple state-of-the-art algorithms",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/genuity_os",
    project_urls={},
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
            "pre-commit>=2.0",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "myst-parser>=0.15",
        ],
        "gpu": [
            "torch>=1.9.0",
            "torchvision>=0.10.0",
        ],
    },
    # Removed console_scripts for security - library is API-only
    # entry_points={
    #     "console_scripts": [
    #         "genuity-preprocess=genuity_os.data_processor.data_preprocess:main",
    #         "genuity-generate=genuity_os.core_generator.ctgan.ctgan.utils.api:main",
    #     ],
    # },
    include_package_data=True,
    package_data={
        "genuity_os": [
            "*.yaml",
            "*.yml",
            "*.json",
        ],
    },
    zip_safe=False,
    keywords=[
        "synthetic-data",
        "tabular-data",
        "generative-models",
        "gan",
        "diffusion",
        "vae",
        "differential-privacy",
        "data-generation",
        "machine-learning",
        "privacy-preserving",
    ],
)
