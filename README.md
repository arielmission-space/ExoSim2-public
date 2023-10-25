# ExoSim 2.0

[![release-build](https://github.com/arielmission-space/ExoSim2-public/workflows/release-build/badge.svg)](https://github.com/arielmission-space/ExoSim2.0/actions/workflows/build.yml)
[![develop-build](https://github.com/arielmission-space/ExoSim2-public/workflows/develop-build/badge.svg)](https://github.com/arielmission-space/ExoSim2.0/actions/workflows/ci_linux.yml)
[![codecov](https://github.com/arielmission-space/ExoSim2-public/branch/develop/graph/badge.svg?token=1Z0KNUKKI7)](https://codecov.io/gh/arielmission-space/ExoSim2-public)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

## Introduction <a name="introduction"></a>

ExoSim 2.0 is a Python package for simulating an astrophysics observation. It is based on the ExoSim 1.0 package, which was developed for the ARIEL mission concept study. ExoSim 2.0 is a complete rewrite of the original code, with a focus on modularity and extensibility. It is designed to be used as a library and can be easily integrated into other Python projects.

## Table of contents

- [ExoSim 2.0](#exosim-20)
  - [Introduction ](#introduction-)
  - [Table of contents](#table-of-contents)
  - [How to install ](#how-to-install-)
    - [Install from PyPI ](#install-from-pypi-)
    - [Install from source code ](#install-from-source-code-)
    - [Test your installation ](#test-your-installation-)
  - [Documentation ](#documentation-)
    - [Build the html documentation ](#build-the-html-documentation-)
    - [Build the pdf documentation  ](#build-the-pdf-documentation--)
  - [How to contribute ](#how-to-contribute-)

## How to install <a name="how-to-install"></a>

### Install from PyPI <a name="install-from-source-code"></a>

ExoSim 2.0 is available on PyPI and can be installed via pip as 

    pip install exosim


### Install from source code <a name="install-from-source-code"></a>

ExoSim 2.0 is compatible (tested) with Python 3.8, 3.9 and 3.10

To install from source, clone the repository and move inside the directory.

Then use `pip` as

    pip install .

### Test your installation <a name="test-your-installation"></a>


If you have installed ExoSim from source-code, to test your ExoSim2 installation simply run

    python -m unittest discover -s tests

## Documentation <a name="documentation"></a>

ExoSim2 comes with an extensive documentation, which can be built using Sphinx.
The documentation includes a tutorial, a user guide and a reference guide.

To build the documentation, install the following packages first:

    pip install `sphinx==4.5` sphinxcontrib-napoleon sphinxcontrib-jsmath nbsphinx pydata-sphinx-theme phinx-panels


### Build the html documentation <a name="build-the-html-documentation"></a>

To build the html documentation, move into the `docs` directory and run

    make html

The documentation will be produced into the `build/html` directory inside `docs`.
Open `index.html` to read the documentation.

### Build the pdf documentation  <a name="build-the-pdf-documentation"></a>

To build the pdf, move into the `docs` directory and run

    make latexpdf

The documentation will be produced into the `build/latex` directory inside `docs`.
Open `exosim2.pdf` to read the documentation.

Here is reported the use of `pdflatex`, if have another compiler for LaTex, please refer to [sphinx documentation](https://www.sphinx-doc.org/en/master/usage/configuration.html#latex-options).

## How to contribute <a name="how-to-contribute"></a>

You can contribute to ExoSim 2.0 by reporting bugs, suggesting new features, or contributing to the code itself. If you want to contribute to the code, please follow the steps described in the documentation under `Developer guide/Contributing guidelines`.
