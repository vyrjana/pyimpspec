.. include:: ./substitutions.rst

API Documentation
=================

The following two lines should provide access to most of what you are likely to need.

.. doctest::

   >>> import pyimpspec          # The main functions, classes, etc.
   >>> from pyimpspec import mpl # Plotting functions based on matplotlib.


.. note::

   The API makes use of multiple processes where possible to perform tasks in parallel.
   Functions that implement this parallelization have a ``num_procs`` keyword argument that can be used to override the maximum number of processes allowed.
   Using this keyword argument should not be necessary for most users under most circumstances.
   Call the |get_default_num_procs| function to get the automatically determined value for your system.
   There is also a |set_default_num_procs| function that can be used to set a global override rather than using the ``num_procs`` keyword argument when calling various functions.

   If NumPy is linked against a multithreaded linear algebra library like OpenBLAS or MKL, then this may in some circumstances result in unusually poor performance despite heavy CPU utilization.
   It may be possible to remedy the issue by specifying a lower number of processes via the ``num_procs`` keyword argument and/or limiting the number of threads that, e.g., OpenBLAS should use by setting the appropriate environment variable (e.g., ``OPENBLAS_NUM_THREADS``).
   Again, this should not be necessary for most users and reporting this as an issue to the pyimpspec repository on GitHub would be preferred.


.. automodule:: pyimpspec
   :members: get_default_num_procs, set_default_num_procs


.. raw:: latex

    \clearpage

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   apidocs_data
   apidocs_kramers_kronig
   apidocs_zhit
   apidocs_drt
   apidocs_circuit
   apidocs_fitting
   apidocs_plot_mpl
   apidocs_typing
   apidocs_exceptions
