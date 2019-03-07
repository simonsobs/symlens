symlens documentation
=====================

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

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   readme
   usage
   reference
   history

Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
