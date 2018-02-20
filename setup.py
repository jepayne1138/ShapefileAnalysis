#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import io
from glob import glob
from os.path import basename
from os.path import dirname
from os.path import join
from os.path import splitext

from setuptools import find_packages
from setuptools import setup


def read(*names, **kwargs):
    return io.open(
        join(dirname(__file__), *names),
        encoding=kwargs.get('encoding', 'utf8')
    ).read()


requires = [
    'pyshp',
    'numpy',
    'scipy',
]

tests_require = [
    'pytest',
]

setup(
    name='shapeanalysis',
    author='James Payne',
    license='MIT',
    description='Shapefile analysis',
    long_description=read('README.rst'),
    author_email='jepayne1138@gmail.com',
    url='https://github.com/jepayne1138/ShapefileAnalysis',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    include_package_data=True,
    zip_safe=False,
    extras_require={
        'test': tests_require,
    },
    install_requires=requires,
    entry_points={
        'console_scripts': [
            'analyze = shapeanalysis:main',
        ],
    },
)
