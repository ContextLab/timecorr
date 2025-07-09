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
    version="0.2.0",
    description="Compute dynamic correlations, dynamic higher-order correlations, and dynamic graph theoretic measures in timeseries data",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Contextual Dynamics Laboratory",
    author_email="contextualdynamics@gmail.com",
    url="https://github.com/ContextLab/timecorr",
    license=LICENSE,
    install_requires=[
        "nose>=1.3.7",
        "sphinx>=2.0.0",
        "duecredit>=0.7.0",
        "hypertools>=0.7.0",
        "numpy>=1.14.2",
        "scipy>=1.2.1",
        "scikit-learn>=0.19.2",
    ],
    extras_require={
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-book-theme",
            "sphinx-gallery",
            "numpydoc",
            "nbsphinx",
            "matplotlib>=2.1.0",
            "seaborn>=0.8.1",
        ],
    },
    packages=find_packages(exclude=("tests", "docs")),
)

#'brainconn @ git+https://github.com/FIU-Neuro/brainconn#egg=brainconn-0.1.0'
