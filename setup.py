# -*- coding: utf-8 -*-
from setuptools import setup, find_packages
try:
    import kdiagram 
    VERSION = kdiagram.__version__
except:
    VERSION= "1.0.2" 
    
# Package metadata
DISTNAME = "k-diagram"
DESCRIPTION = "Rethinking Forecasting Uncertainty via Polar-Based Visualization"
LONG_DESCRIPTION = open('README.md', 'r', encoding='utf8').read()
MAINTAINER = "Laurent Kouadio"
MAINTAINER_EMAIL = 'etanoyau@gmail.com'
URL = "https://github.com/earthai-tech/k-diagram"
LICENSE = "Apache-2.0"
PROJECT_URLS = {
    "API Documentation": "https://k-diagram.readthedocs.io/en/latest/api_references.html",
    "Home page": "https://k-diagram.readthedocs.io",
    "Bugs tracker": "https://github.com/earthai-tech/k-diagram/issues",
    "Installation guide": "https://k-diagram.readthedocs.io/en/latest/installation.html",
    "User guide": "https://k-diagram.readthedocs.io/en/latest/user_guide.html",
}
KEYWORDS = "forecasting, uncertainty visualization, model evaluation"

# Ensure dependencies are installed
_required_dependencies = [
    "numpy<2",
    "pandas",
    "scipy",
    "matplotlib",
    "seaborn",
    "scikit-learn"
]

# Package data specification
PACKAGE_DATA = {
    'kdiagram': [
        # 'data/*.json', 
        # 'assets/*.txt'
    ],
}
setup_kwargs = {
    'entry_points': {
        'console_scripts': [
            'k-diagram=kdiagram.cli:main',
        ]
    },
    'packages': find_packages(),
    'install_requires': _required_dependencies,
    'extras_require': {
        "dev": [
            "pytest",
            "sphinx",
        ]
    },
    'python_requires': '>=3.9',
}

setup(
    name=DISTNAME,
    version=VERSION, 
    author=MAINTAINER,
    author_email=MAINTAINER_EMAIL,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url=URL,
    license=LICENSE,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
    ],
    keywords=KEYWORDS,
    package_data=PACKAGE_DATA,
    **setup_kwargs
)
