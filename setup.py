#!/usr/bin/env python
from setuptools import setup, find_packages

# requirements
with open('requirements.txt') as fh:
    requirements = fh.read().splitlines()

# install main application
setup(
    name='rnatargeting',
    version='0.1.0',
    description='RNA Targeting Guide Design',
    long_description='See README.md',
    author='Jingyi Wei',
    author_email='jingyiw@arcinstitute.org',
    # entry_points={
    #     'console_scripts': [
    #         'genomics-guide = genomics_guide.__main__:main'
    #     ]
    # },
    package_data={'rnatargeting': ['saved_model/*']},
    install_requires=requirements,
    license="MIT license",
    packages=find_packages(),
    package_dir={'rnatargeting': 'rnatargeting'},
    python_requires='>=3.9',
    url='https://github.com/arcinstitute/RNAtargeting_web_custom'
)