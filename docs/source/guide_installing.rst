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

- `Python <https://www.python.org>`_ (3.10, 3.11, or 3.12)
- The following Python packages

  - `Jinja <https://jinja.palletsprojects.com/>`_
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

These Python packages (and their dependencies) are installed automatically when pyimpspec is installed using, e.g., `pip <https://pip.pypa.io/en/stable/>`_.

The following Python packages can be installed as optional dependencies for additional functionality:

- DRT calculations using the `TR-RBF method <https://doi.org/10.1016/j.electacta.2015.09.097>`_ (at least one of the following is required):
	- `cvxopt <https://github.com/cvxopt/cvxopt>`_
	- `kvxopt <https://github.com/sanurielf/kvxopt>`_ (this fork of cvxopt may support additional platforms)


Installing
----------

Make sure that both Python and pip are installed first (see previous section for supported Python versions).
For example, open a terminal and run the following command to confirm that pip (or pipx) is indeed installed:

.. code:: bash

   pip --version


.. note::

   Using a Python `virtual environment <https://docs.python.org/3/tutorial/venv.html>`_ is highly recommended in order to avoid possible issues related to conflicting versions of dependencies installed on a system.
   Such a virtual environment needs to be activated before running a script that imports a package installed inside the virtual environment.
   The system-wide Python environment may also be `externally managed <https://peps.python.org/pep-0668/>`_ in order to prevent the user from accidentally breaking that environment since the operating system depends upon the packages in that environment.

   A third-party tool called `pipx <https://pypa.github.io/pipx/>`_ can automatically manage such virtual environments but it is primarily for installing programs that provide, e.g., a command-line interface (CLI) or a graphical user interface (GUI).
   These programs can then be run without having to manually activate the virtual environment since pipx handles that.
   The virtual environment would still need to be activated before running a script that imports DearEIS and makes use of DearEIS's application programming interface (API).

If using pipx, then run the following command to make sure that pipx is available.
If pipx is not available, then follow the `instructions to install pipx <https://pypa.github.io/pipx/installation/>`_.

.. code:: bash

   pipx --version


If there are no errors, then run one of the following commands to install pyimpspec and its dependencies:

.. code:: bash

   # If manually managing the virtual environment,
   # follow the relevant pip documentation for creating
   # and activating a virtual environment before running
   # the following command.
   pip install pyimpspec
   
   # If pipx is used to automatically manage the virtual environment.
   pipx install pyimpspec

Pyimpspec should now be importable in, e.g., Python scripts and Jupyter notebooks provided that the virtual environment has been activated.

If you wish to install the optional dependencies, then they can be specified explicitly when installing pyimpspec via pip:

.. code:: bash

   pip install pyimpspec[cvxopt]


Optional dependencies can also be install after the fact if pipx was used:

.. code:: bash

   pipx inject pyimpspec cvxopt


Newer versions of pyimpspec can be installed in the following ways:

.. code:: bash
   
   pip install pyimpspec --upgrade

   pipx upgrade pyimpspec --include-injected


Using the API
-------------

Pyimpspec should be accessible in Python provided that the virtual environment has been activated:

.. doctest::

   >>> import pyimpspec


Running the CLI program
-----------------------

You should now also be able to access pyimpspec's CLI in a terminal:

.. code:: bash

   pyimpspec


Alternatively, the CLI can be accessed by running pyimpspec as a module:

.. code:: bash

   python -m pyimpspec


.. raw:: latex

    \clearpage

