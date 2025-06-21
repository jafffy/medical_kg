#!/usr/bin/env python3
"""
Setup script for SOAP Knowledge Graph Generator
"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
try:
    with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = """
    SOAP Knowledge Graph Generator
    
    A Python package for constructing knowledge graphs from clinical data
    organized by SOAP (Subjective, Objective, Assessment, Plan) categories.
    """

# Read requirements
def read_requirements(filename):
    try:
        with open(filename, 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    except FileNotFoundError:
        return []

requirements = read_requirements('requirements.txt')

setup(
    name="soap-knowledge-graph",
    version="0.1.0",
    author="SOAP-KG Team",
    author_email="your-email@example.com",
    description="A Python package for constructing SOAP-organized knowledge graphs from clinical data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/soap-knowledge-graph",
    
    # Package configuration
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    
    # Dependencies
    install_requires=requirements,
    
    # Optional dependencies
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.0',
            'black>=21.0',
            'flake8>=3.8',
            'mypy>=0.800',
        ],
        'visualization': [
            'plotly>=5.0',
            'dash>=2.0',
            'kaleido>=0.2',
        ],
    },
    
    # Entry points
    entry_points={
        'console_scripts': [
            'soap-kg=soap_kg.cli:main',
        ],
    },
    
    # Scripts
    scripts=[
        'scripts/main.py',
        'scripts/setup_environment.sh',
    ],
    
    # Metadata
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    
    # Requirements
    python_requires=">=3.8",
    
    # Include additional files
    include_package_data=True,
    package_data={
        'soap_kg': ['*.yaml', '*.yml', '*.json'],
    },
    
    # Keywords
    keywords="medical nlp knowledge-graph soap clinical-data mimic healthcare",
    
    # Project URLs
    project_urls={
        "Bug Reports": "https://github.com/your-username/soap-knowledge-graph/issues",
        "Source": "https://github.com/your-username/soap-knowledge-graph",
        "Documentation": "https://soap-knowledge-graph.readthedocs.io/",
    },
)