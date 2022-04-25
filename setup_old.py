import pathlib
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()


setup(
    name='fenicsconcrete',
    version='0.1.0',
    description='A Python package for a FEM concrete model',
    long_description=README,
    long_description_content_type="text/markdown",
    url='https://github.com/BAMresearch/FenicsConcrete',
    author='Erik Tamsen',
    author_email='erik.tamsen@bam.de',
    license="MIT",
    packages=['concrete_model'],
    install_requires=['mpi4py>=2.0',
                      'numpy',
                      ],

    classifiers=[
        'Development Status :: 1 - Implementation',
        'Topic :: Scientific/Engineering',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.5',
        "Programming Language :: Python :: 3.7",
    ],
)
