from setuptools import setup

requires = [
    'pyshp',
    'numpy',
]

tests_require = [
    'pytest',
]

setup(
    name='shapefile',
    author='James Payne',
    author_email='jepayne1138@gmail.com',
    extras_require={
        'testing': tests_require,
    },
    install_requires=requires,
    entry_points={
        'console_scripts': [
            'analyize = analyize:main',
        ],
    },
)
