.. currentmodule:: xarrayutils

What's New
===========
.. _whats-new.2.0.0:

v2.0.0 (unreleased)

Breaking changes
~~~~~~~~~~~~~~~~
- Dropped support for python 3.8 (:pull:`141`)
  By `Julius Busecke <https://github.com/jbusecke>`_

- Deprecated multiple modules:
  `visualization.py`
  `vertical_coordinates.py/vertical_remapping.py`
  `weighted_operations.py`
  `numpy_utils.py`
  (:pull:`146`, :issue:`84`)
  By `Julius Busecke <https://github.com/jbusecke>`_

Documentation
~~~~~~~~~~~~~

- Added CITATION.cff to repo (:pull:`129`).
  By `Julius Busecke <https://github.com/jbusecke>`_

- Switch RTD env build to mamba (:pull:`132`).
  By `Julius Busecke <https://github.com/jbusecke>`_
  
Internal Changes
~~~~~~~~~~~~~~~~

- Switch linting CI to pre-commit.ci github app (:pull:`147`).
  By `Julius Busecke <https://github.com/jbusecke>`_


.. _whats-new.1.1.0:

v1.1.0 (2022/2/15)

New Features
~~~~~~~~~~~~
- `sign_agreement` now supports the option to not count nans along the given dimension (:pull:`118`). By `Julius Busecke <https://github.com/jbusecke>`_

Bugfixes
~~~~~~~~
- Fixed bug in `shaded_line_plot`, where std bounds were displayed incorrectly (:issue:`74`, :pull:`123`). By `Julius Busecke <https://github.com/jbusecke>`_

.. _whats-new.1.0.0:

v1.0.0 (2021/6/22)

Breaking changes
~~~~~~~~~~~~~~~~
- Dropped support for python 3.7 (:pull:`97`)
  By `Julius Busecke <https://github.com/jbusecke>`_

- Refactored implementation of  :py:meth:`utils.xr_linregress`: Scales better and
  does not require rechunking the data anymore. No more `nanmask` option available
  and parameters are always variables no more optioin to put out a dataarray (:pull:`62`). 
  By `Julius Busecke <https://github.com/jbusecke>`_

New Features
~~~~~~~~~~~~
- Added `file_handling` module (:pull:`93`, :pull:`95`). By `Julius Busecke <https://github.com/jbusecke>`_


Documentation
~~~~~~~~~~~~~
- Added example notebook for `xarrayutils.utils`


Internal Changes
~~~~~~~~~~~~~~~~
- Upgraded internal CI/Publishing workflows (:pull:`97`, :pull:`101`)

.. _whats-new.0.1.3:

v0.1.3
----------------------
Changes not documented for this release and earlier
