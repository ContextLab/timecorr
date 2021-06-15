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
    version='0.1.6',
    description='Compute dynamic correlations, dynamic higher-order correlations, and dynamic graph theoretic measures in timeseries data',
    long_description=' ',
    long_description_content_type='text/markdown',
    author='Contextual Dynamics Laboratory',
    author_email='contextualdynamics@gmail.com',
    url='https://github.com/ContextLab/timecorr',
    license=LICENSE,
    install_requires=[
        'nose==1.3.7',
        'sphinx==2.0.0',
        'duecredit==0.7.0',
        'numpy==1.16.2',
        'pandas==0.24.2',
        'hypertools==0.7.0',
        'scipy==1.2.1',
        'matplotlib==3.0.3',
        'seaborn==0.9.0',
        'scikit-learn==0.19.2'
    ],
    packages=find_packages(exclude=('tests', 'docs')),
)

#'brainconn @ git+https://github.com/FIU-Neuro/brainconn#egg=brainconn-0.1.0'
