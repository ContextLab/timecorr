# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

def get_requirements(remove_links=True):
    """
    lists the requirements to install.
    """
    requirements = []
    with open('requirements.txt') as f:
        requirements = f.read().splitlines()

    if remove_links:
        for requirement in requirements:
            # git repository url.
            if requirement.startswith("git+"):
                requirements.remove(requirement)
            # subversion repository url.
            if requirement.startswith("svn+"):
                requirements.remove(requirement)
            # mercurial repository url.
            if requirement.startswith("hg+"):
                requirements.remove(requirement)
    return requirements


def get_links():
    """
    gets URL Dependency links.
    """
    links_list = get_requirements(remove_links=False)
    for link in links_list:
        keep_link = False
        already_removed = False
        # git repository url.
        if not link.startswith("git+"):
            if not link.startswith("svn+"):
                if not link.startswith("hg+"):
                    links_list.remove(link)
                    already_removed = True
                else:
                    keep_link = True
                if not keep_link and not already_removed:
                    links_list.remove(link)
                    already_removed = True
            else:
                keep_link = True
            if not keep_link and not already_removed:
                links_list.remove(link)
    return links_list


try:
    with open('README.md') as f:
        readme = f.read()
except FileNotFoundError:
    readme = ""

with open('LICENSE') as f:
    license = f.read()

setup(
    name='timecorr',
    version='0.1.2',
    description='Compute dynamic correlations, dynamic higher-order correlations, and dynamic graph theoretic measures in timeseries data',
    long_description=readme,
    author='Contextual Dynamics Laboratory',
    author_email='contextualdynamics@gmail.com',
    url='https://github.com/ContextLab/timecorr',
    install_requires=get_requirements(),
    dependency_links=get_links(),
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)
