.. currentmodule:: xarrayutils

What's New
===========

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
