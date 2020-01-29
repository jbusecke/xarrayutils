[![Build Status](https://travis-ci.org/jbusecke/xarrayutils.svg?branch=master)](https://travis-ci.org/jbusecke/xarrayutils)
[![codecov](https://codecov.io/gh/jbusecke/xarrayutils/branch/master/graph/badge.svg)](https://codecov.io/gh/jbusecke/xarrayutils)
[![License:MIT](https://img.shields.io/badge/License-MIT-lightgray.svg?style=flt-square)](https://opensource.org/licenses/MIT)
[![pypi](https://img.shields.io/pypi/v/xarrayutils.svg)](https://pypi.org/project/xarrayutils)


# A collection of various tools for data analysis built on top of xarray and xgcm

This package contains a variety of utility functions I have used in the past few years for data analysis.

Please be aware that I am currently refactoring a lot of the code, which might cause breaking changes.

Hopefully soon it will be better tested and documented.

## Installation

To install from source use pip:

`pip install git+https://github.com/jbusecke/xarrayutils.git`


## Contents (not complete yet)

### Converting between vertical coordinate systems

`xarrayutils.vertical_coordinates` provides several tool to move between various vertical ocean grids, e.g. z, density, sigma or hybrid coordinates. See [this notebook](https://github.com/jbusecke/xarrayutils/blob/master/doc/vertical_coords.ipynb) for an early example.


