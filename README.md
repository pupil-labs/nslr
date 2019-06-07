# Naive Segmented Linear Regression

Implements the NSLR algorithm described in the article [upcoming].

## Installation

Currently the preferred way is to install directly from gitlab:

    pip install git+https://gitlab.com/nslr/nslr

This will try to build the C++ version, but will (silently, sadly) use
the very much slower Python version if the building fails. There's also
a version in PyPI, but it installs only the pure-python version:

    pip install nslr

**Note**: The cpp code is not compatible with gcc 5 (default for Ubuntu 14.04). We recommend using gcc 7 or higher.

## Usage

Currently the only "supported" function is `nslr.fit_gaze`. For
usage, see `demo.py`.
