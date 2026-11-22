# -*- coding: utf-8 -*-
import os

from setuptools import find_packages, setup

try:
    with open("README.md") as f:
        readme = f.read()
except FileNotFoundError:
    readme = ""

LICENSE = "MIT"

setup(
    name="timecorr",
    version="0.3.1",
    description="Compute dynamic correlations, dynamic higher-order correlations, and dynamic graph theoretic measures in timeseries data",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Contextual Dynamics Laboratory",
    author_email="contextualdynamics@gmail.com",
    url="https://github.com/ContextLab/timecorr",
    license=LICENSE,
    python_requires=">=3.10",
    install_requires=[
        "pydata-wrangler",
        "hypertools>=0.8",
        "numpy",
        "scipy",
        "scikit-learn",
        "pandas",
        "matplotlib",
    ],
    extras_require={
        "segment": ["hmmlearn"],
        "test": [
            "pytest",
            "polars",
            "hmmlearn",
        ],
        # building the documentation (the notebooks are not executed)
        "docs": [
            "sphinx>=4.0.0",
            "furo",
            "numpydoc",
            "nbsphinx",
            "ipython",
        ],
        # what the real-data tutorial notebooks import (htfa_network_animations,
        # stocks, trajectories); the other notebooks need only timecorr
        "tutorials": [
            "htfa[plot,brain3d]>=0.1.0",
            "hypertools>=1.1",
            "pyvista>=0.43",
            "imageio-ffmpeg",
            "pillow",
            "joblib",
            "ipython",
        ],
    },
    packages=find_packages(exclude=("tests", "docs")),
)

