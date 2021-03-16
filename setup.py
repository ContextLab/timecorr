# -*- coding: utf-8 -*-
import os
from setuptools import setup, find_packages

try:
    with open('README.md') as f:
        readme = f.read()
except FileNotFoundError:
    readme = ""

LICENSE = 'MIT'

setup(
    name='timecorr',
    version='0.1.5',
    description='Compute dynamic correlations, dynamic higher-order correlations, and dynamic graph theoretic measures in timeseries data',
    long_description=' ',
    author='Contextual Dynamics Laboratory',
    author_email='contextualdynamics@gmail.com',
    url='https://github.com/ContextLab/timecorr',
    license=LICENSE,
    install_requires=[
        'nose',
        'sphinx',
        'duecredit',
        'numpy>=1.14.2',
        'pandas>=0.22.0',
        'hypertools==0.5.1',
        'scipy==1.2.1',
        'matplotlib>=2.1.0',
        'seaborn>=0.8.1',
        'scikit-learn>=0.19.1',
    ],
    packages=find_packages(exclude=('tests', 'docs')),
)

#'brainconn @ git+https://github.com/FIU-Neuro/brainconn#egg=brainconn-0.1.0'