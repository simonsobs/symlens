.. sectnum::

=======
symlens
=======

.. image:: https://img.shields.io/pypi/v/symlens.svg
        :target: https://pypi.python.org/pypi/symlens

.. image:: https://img.shields.io/travis/simonsobs/symlens.svg
        :target: https://travis-ci.org/simonsobs/symlens

.. image:: https://readthedocs.org/projects/symlens/badge/?version=latest
        :target: https://symlens.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status




This library allows one to build and evaluate arbitrary separable mode-coupling
estimators. In practice, its main purpose is to provide a flat-sky lensing estimator
code. More generally, one can build estimators and noise functions for
convergence, magnification, shear, mixed estimators (for gradient cleaning),
split-based lensing, birefringence, patchy tau, etc. and cross-covariances
between these.

Instead of having to calculate by hand the separable forms of the above, one
simply provides the mode-coupling and filter expressions, and a ``sympy``-based
(Mathematica-like) backend factorizes these expressions into FFT-only form
(i.e., no explicit convolutions are required).

Curved sky support does not exist. Adding it is possibly non-trivial, but
thoughts and ideas (and PRs!) are highly appreciated. Still, this package can
serve as the backend for quick exploration of various kinds of estimators.

* Free software: BSD license
* Documentation: https://symlens.readthedocs.io.

Dependencies
============

* Python>=2.7 or Python>=3.4
* pixell_
* numpy, sympy

Installing
==========

To install, run:

.. code-block:: console
		
   $ python setup.py install --user

Usage
=====

See the Usage_ guide and the API Reference_.

Contributing
------------

If you have write access to this repository, please:

1. create a new branch
2. push your changes to that branch
3. merge or rebase to get in sync with master
4. submit a pull request on github

If you do not have write access, create a fork of this repository and proceed as described above. 

.. _pixell: https://github.com/simonsobs/pixell/
.. _Usage: https://symlens.readthedocs.io/en/latest/usage.html
.. _Reference: https://symlens.readthedocs.io/en/latest/reference.html
