.. currentmodule:: xarrayutils

What's New
===========

.. _whats-new.0.3.0:

v0.2.0 (unreleased)

Breaking changes
~~~~~~~~~~~~~~~~
- Dropped support for python 3.7 (:pull:`97`)
  By `Julius Busecke <https://github.com/jbusecke>`_

- Refactored implementation of  :py:meth:`utils.xr_linregress`: Scales better and
  does not require rechunking the data anymore. No more `nanmask` option available
  (:pull:`62`). By `Julius Busecke <https://github.com/jbusecke>`_

New Features
~~~~~~~~~~~~
- Added `file_handling` module (:pull:`93`, :pull:`95`). By `Julius Busecke <https://github.com/jbusecke>`_

Bug fixes
~~~~~~~~~


Documentation
~~~~~~~~~~~~~
- Added example notebook for `xarrayutils.utils`


Internal Changes
~~~~~~~~~~~~~~~~



v0.1.3
----------------------
Changes not documented for this release and earlier
