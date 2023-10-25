# ExoSim 2.0
[![release-build](https://github.com/arielmission-space/ExoSim2-public/workflows/release-build/badge.svg)](https://github.com/arielmission-space/ExoSim2.0/actions/workflows/build.yml)
[![develop-build](https://github.com/arielmission-space/ExoSim2-public/workflows/develop-build/badge.svg)](https://github.com/arielmission-space/ExoSim2.0/actions/workflows/ci_linux.yml)
[![codecov](https://codecov.io/gh/arielmission-space/ExoSim2-public/graph/badge.svg?token=8LDBCU43CK)](https://codecov.io/gh/arielmission-space/ExoSim2-public)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![PyPI version](https://badge.fury.io/py/exosim.svg)](https://badge.fury.io/py/exosim)
![GitHub tag (with filter)](https://img.shields.io/github/v/tag/arielmission-space/ExoSim2-public?label=GitHub%20version&color=green)
[![Documentation Status](https://readthedocs.org/projects/exosim2-public/badge/?version=latest)](https://exosim2-public.readthedocs.io/en/latest/?badge=latest)
<!-- [![Downloads](https://pepy.tech/badge/exosim)](https://pepy.tech/project/exosim) -->

## Introduction <a name="introduction"></a>

ExoSim 2 is the next generation of the Exoplanet Observation Simulator [ExoSim](https://github.com/ExoSim/ExoSimPublic) tailored for spectro-photometric observations of transiting exoplanets from space, ground, and sub-orbital platforms. This software is a complete rewrite implemented in Python 3, embracing object-oriented design principles, which allow users to replace each component with their functions when required.


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
  - [How to cite](#how-to-cite)

## How to install <a name="how-to-install"></a>

### Install from PyPI <a name="install-from-source-code"></a>

ExoSim 2.0 is available on PyPI and can be installed via pip as 

    pip install exosim


### Install from source code <a name="install-from-source-code"></a>

ExoSim 2.0 is compatible (tested) with Python 3.8, 3.9 and 3.10

To install from source, clone the [repository](https://github.com/arielmission-space/ExoSim2-public/) and move inside the directory.

Then use `pip` as

    pip install .

#### Test your installation <a name="test-your-installation"></a>


If you have installed ExoSim from source-code, to test your ExoSim2 installation simply run from the main ExoSim2 folder

    pip install -e ".[test]"
    python -m unittest discover -s tests

## Documentation <a name="documentation"></a>

ExoSim2 comes with an extensive documentation, which can be built using Sphinx.
The documentation includes a tutorial, a user guide and a reference guide.

To build the documentation, install the needed packages first via:

    pip install -e ".[docs]"


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

## How to cite
A dedicated publication has been submitted and the relative information will be published soon. In the meanwhile, please, send an email to the developers.  
