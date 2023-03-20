.. include:: ./substitutions.rst

Installing
==========

Supported platforms
-------------------

- Linux
- Windows
- MacOS

The package **may** also work on other platforms depending on whether or not those platforms are supported by pyimpspec's dependencies.


Requirements
------------

- `Python <https://www.python.org>`_ (3.8, 3.9, or 3.10)
- The following Python packages

  - `lmfit <https://lmfit.github.io/lmfit-py/>`_
  - matplotlib_ 
  - `numdifftools <https://numdifftools.readthedocs.io/en/latest/>`_
  - `numpy <https://numpy.org/>`_
  - `odfpy <https://github.com/eea/odfpy>`_
  - `openpyxl <https://openpyxl.readthedocs.io/>`_
  - `pandas <https://pandas.pydata.org/>`_
  - schemdraw_
  - `scipy <https://scipy.org/>`_
  - `statsmodels <https://www.statsmodels.org/>`_
  - sympy_
  - `tabulate <https://github.com/astanin/python-tabulate>`_
  - `xdg <https://github.com/srstevenson/xdg>`_

These Python packages (and their dependencies) are installed automatically when pyimpspec is installed using `pip <https://pip.pypa.io/en/stable/>`_.

The following Python packages can be installed as optional dependencies for additional functionality:

- DRT calculations using the `TR-RBF method <https://doi.org/10.1016/j.electacta.2015.09.097>`_ (at least one of the following is required):
	- `cvxopt <https://github.com/cvxopt/cvxopt>`_
	- `kvxopt <https://github.com/sanurielf/kvxopt>`_ (this fork of cvxopt may support additional platforms)
	- `cvxpy <https://github.com/cvxpy/cvxpy>`_

.. note::

   Windows and MacOS users who wish to install CVXPY **must** follow the steps described in the `CVXPY documentation <https://www.cvxpy.org/install/index.html>`_!


Installing
----------

Make sure that Python and pip are installed first (see previous section for supported Python versions).
For example, open a terminal and run the command:

.. code:: bash

   pip --version

.. note::

   If you only intend to use pyimpspec via the CLI or are familiar with `virtual environments <https://docs.python.org/3/tutorial/venv.html>`_, then you should consider using `pipx <https://pypa.github.io/pipx/>`_ instead of pip to install pyimpspec.
   Pipx will install pyimpspec inside of a virtual environment, which can help with preventing potential version conflicts that may arise if pyimpspec requires an older or a newer version of a dependency than another package.
   Pipx also manages these virtual environments and makes it easy to run applications/packages.


If there are no errors, then run the following command to install pyimpspec and its dependencies:

.. code:: bash

   pip install pyimpspec

Pyimpspec should now be importable in, e.g., Python scripts and Jupyter notebooks.

If you wish to install the optional dependencies, then they must either be specified explicitly when installing pyimpspec or installed separately later:

.. code:: bash

   pip install pyimpspec[cvxpy]

Newer versions of pyimpspec can be installed at a later date by adding the ``--upgrade`` option to the command:

.. code:: bash
   
   pip install --upgrade pyimpspec


Using the API
-------------

Pyimpspec should now be accessible in Python:

.. doctest::

   >>> import pyimpspec


Running the CLI program
-----------------------

You should now be able to run pyimpspec in a terminal:

.. code:: bash

   pyimpspec


Alternatively, the CLI can be accessed by running pyimpspec as a module:

.. code:: bash

   python -m pyimpspec


.. raw:: latex

    \clearpage

