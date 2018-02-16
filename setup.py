from setuptools import setup

requires = [
    'pyshp',
    'numpy',
    'scipy',
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
            'analyzeshapefile = analyze:main',
        ],
    },
)
