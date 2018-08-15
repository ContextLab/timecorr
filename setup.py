# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='timecorr',
    version='0.1.1',
    description='Compute moment-by-moment correlations in timeseries data',
    long_description='Write this...',
    author='Contextual Dynamics Laboratory',
    author_email='contextualdynamics@gmail.com',
    url='https://github.com/ContextLab/timecorr',
    install_requires = requirements,
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)
