"""Setup script for particle filtering framework."""

from setuptools import setup, find_packages

setup(
    name='particle-filtering-framework',
    version='0.1.0',
    description='Modular particle filtering framework for Part 1 experiments',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.20.0',
        'scipy>=1.7.0',
        'matplotlib>=3.3.0',
        'pytest>=7.0.0',
        'tensorflow>=2.13.0',
        'tensorflow-probability>=0.21.0',
        'hydra-core>=1.3.0',
        'omegaconf>=2.3.0',
    ],
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
        ]
    },
)
