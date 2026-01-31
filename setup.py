"""Setup script for PERSONA: Contextual Behavioral Intelligence Engine"""

from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="persona-behavioral-intelligence",
    version="0.1.0",
    author="Chidghana Hemantharaju",
    author_email="your-email@domain.com",
    description="A cutting-edge ML system for organizational behavioral analysis with permission-controlled access",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ChidghanaH/PERSONA-behavioral-intelligence",
    project_urls={
        "Bug Tracker": "https://github.com/ChidghanaH/PERSONA-behavioral-intelligence/issues",
        "Documentation": "https://github.com/ChidghanaH/PERSONA-behavioral-intelligence/blob/main/README.md",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    package_dir={"persona": "src/persona"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "persona=persona.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "machine-learning",
        "behavioral-analysis",
        "graph-neural-networks",
        "temporal-networks",
        "explainable-ai",
        "federated-learning",
        "privacy",
        "hr-analytics",
        "organizational-intelligence",
        "permission-control",
    ],
)
