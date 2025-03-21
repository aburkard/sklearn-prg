from setuptools import setup, find_packages

setup(
    name='sklearn_prg',
    version='0.1.1',
    description='Precision-Recall-Gain curves and metrics for scikit-learn',
    readme='README.md',
    author='Andrew Burkard',
    author_email='andrewburkard@gmail.com',
    url='https://github.com/aburkard/sklearn_prg',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.20',
        'scikit-learn>=1.0',
        'matplotlib>=3.0',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.8',
)
