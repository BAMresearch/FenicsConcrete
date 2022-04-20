from setuptools import setup

setup(
    name='fenicsconcrete',
    version='0.1.0',
    description='A Python package for a FEM concrete model',
    url='https://github.com/BAMresearch/FenicsConcrete',
    author='Erik Tamsen',
    author_email='erik.tamsen@bam.de',
    license='BSD 2-clause',
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
    ],
)
