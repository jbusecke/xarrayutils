.. xarrayutils documentation master file, created by
   sphinx-quickstart on Wed Jan 13 12:39:16 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to xarrayutils's documentation!
=======================================

This package contains a variety of utility functions I have used in the past few years for data analysis.

Installation
------------

Installation from Conda Forge
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The easiest way to install xgcm along with its dependencies is via conda
forge::

    conda install -c conda-forge xarrayutils

Installation from Pip
^^^^^^^^^^^^^^^^^^^^^

An alternative is to use pip::

    pip install xarrayutils

Installation from GitHub
^^^^^^^^^^^^^^^^^^^^^^^^

You can get the newest version by installing directly from GitHub::

    pip install git+https://github.com/jbusecke/xarrayutils.git



Contents
--------

Utilities for large scale climate data analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`xarrayutils.utils` provides some helpful tools to simplify common tasks in climate data
analysis workflows.

Plotting utilities
^^^^^^^^^^^^^^^^^^

`xarrayutils.plotting` provides several small utility functions to make common tasks I find in my workflow in matplotlib easier.

Convenience functions for file handling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`xarrayutils.file_handling` contains a mix of functions that I found useful across a variety to save/load files. In particular there are functions in there to efficiently save large dask arrays out as temporary files to avoid large task graphs from causing memory problems. Hopefully these will become obsolete as dask continues to improve.

The utilities in `xarrayutils.vertical_coordinates` are superseeded by the new `xgcm transform module <https://xgcm.readthedocs.io/en/latest/transform.html>`_.

Converting between vertical coordinate systems
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The utilities in `xarrayutils.vertical_coordinates` are superseeded by the new `xgcm transform module <https://xgcm.readthedocs.io/en/latest/transform.html>`_.



.. toctree::
   :maxdepth: 2
   :caption: Contents:

   utils
   plotting
   api
   whats-new



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
