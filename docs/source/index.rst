.. pyimpspec documentation master file, created by
   sphinx-quickstart on Wed Jan 11 19:11:26 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. include:: ./substitutions.rst

Welcome to pyimpspec's documentation!
=====================================

.. only:: html

   .. image:: https://github.com/vyrjana/pyimpspec/actions/workflows/test-package.yml/badge.svg
      :alt: tests
      :target: https://github.com/vyrjana/pyimpspec/actions/workflows/test-package.yml
   
   .. image:: https://github.com/vyrjana/pyimpspec/actions/workflows/test-wheel.yml/badge.svg
      :alt: build
      :target: https://github.com/vyrjana/pyimpspec/actions/workflows/test-wheel.yml
   
   .. image:: https://img.shields.io/pypi/pyversions/pyimpspec
      :alt: Supported Python versions

   .. image:: https://img.shields.io/github/license/vyrjana/pyimpspec
      :alt: GitHub
      :target: https://www.gnu.org/licenses/gpl-3.0.html

   .. image:: https://img.shields.io/pypi/v/pyimpspec
      :alt: PyPI
      :target: https://pypi.org/project/pyimpspec/


Pyimpspec is a Python package for processing, analyzing, and visualizing impedance spectra.
The primary interface for using pyimpspec is the application programming interface (API).

.. doctest::

   >>> import pyimpspec

A command-line interface (CLI) is also included to provide a way to, e.g., quickly plot some experimental data via a terminal.

.. code:: bash

   # The CLI should be accessible in the terminal in the following ways
   pyimpspec --version
   python -m pyimpspec --version


The changelog can be found `here <https://github.com/vyrjana/pyimpspec/blob/main/CHANGELOG.md>`_.
If you encounter bugs or wish to request a feature, then please open an `issue on GitHub <https://github.com/vyrjana/pyimpspec/issues>`_.
If you wish to contribute to the project, then please read the `readme <https://github.com/vyrjana/pyimpspec/blob/main/README.md>`_ before submitting a `pull request via GitHub <https://github.com/vyrjana/pyimpspec/pulls>`_.

Pyimpspec is licensed under GPLv3_ or later.

.. only:: html

   PDF copies of the documentation are available in the `releases section <https://github.com/vyrjana/pyimpspec/releases>`_.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   guide
   apidocs

