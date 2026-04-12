.. include:: ./substitutions.rst

Data parsing
============

A collection of functions and classes for parsing experimental data into data sets that can then be analyzed and/or visualized.

.. _pandas.read_csv: https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html#pandas-read-csv
.. _pandas.read_excel: https://pandas.pydata.org/docs/reference/api/pandas.read_excel.html#pandas-read-excel

Wrapper functions
-----------------
.. automodule:: pyimpspec
   :members: parse_data, get_parsers, dataframe_to_data_sets


Format-specific functions
-------------------------
.. automodule:: pyimpspec.data.formats
   :members:
   :imported-members:


Class
-----
.. automodule:: pyimpspec
   :members: DataSet

.. raw:: latex

    \clearpage
