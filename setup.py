# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='fdfdpy',
    version='0.1.0',
    description='Electromagnetic Finite Difference Frequency Domain Solver',
    long_description=readme,
    author='Tyler Hughes, Momchil Minkov, Ian Williamson',
    author_email='twhughes@stanford.edu',
    url='https://github.com/fancompute/fdfdpy',
    license=license,
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=[
          'pyMKL',
          'numpy',
          'scipy',
          'matplotlib',
          'progressbar'
      ],
)
