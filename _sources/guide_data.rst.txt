.. include:: ./substitutions.rst

Data parsing
============

Individual impedance spectra are represented in pyimpspec as |DataSet| objects.
The |parse_data| function acts as a wrapper for the various parsing functions available for different file formats such as:

- BioLogic: ``.mpt``
- Eco Chemie: ``.dfr``
- Gamry: ``.dta``
- Ivium: ``.idf`` and ``.ids``
- PalmSens: ``.pssession``
- ZView: ``.z``
- Spreadsheets: ``.xlsx`` and ``.ods``
- Plain-text character-separated values (CSV)

The parsing functions and |parse_data| always return a list of |DataSet| objects since some files may contain multiple impedance spectra:

.. doctest::

   >>> from pyimpspec import DataSet, Frequencies, ComplexImpedances, parse_data
   >>>
   >>> data: DataSet
   >>> for data in parse_data("./tests/data.dta"):
   ...   # Do something with 'data'.
   ...   f: Frequencies = data.get_frequencies()
   ...   Z: ComplexImpedances = data.get_impedances()

.. note::

  The data points are sorted from highest to lowest frequency when a |DataSet| instance is created.
  If a file contains multiple frequency sweeps, then they are returned as separate |DataSet| instances.


Plotting data
-------------

A number of functions are included for plotting data (and various analysis results) using matplotlib_ as the backend.
Below is a Nyquist plot of some example data (test circuit 1 or TC-1 from `Boukamp (1995)`_).

.. _`Boukamp (1995)`: https://doi.org/10.1149/1.2044210


.. doctest::
   
   >>> from pyimpspec import DataSet, parse_data
   >>> from pyimpspec import mpl
   >>>
   >>> data: DataSet
   >>> for data in parse_data("./tests/data.dta"):
   ...   figure, axes = mpl.plot_nyquist(data)
   >>>
   >>> mpl.show()


.. note::

   The ``figure`` and ``axes`` values can be reused if one wishes to plot multiple immittance spectra in the same figure: ``mpl.plot_nyquist(data, figure=figure, axes=axes)``


.. plot::

   from pyimpspec import mpl
   from pyimpspec import generate_mock_data

   data = generate_mock_data("CIRCUIT_1")[0]
   figure, axes = mpl.plot_nyquist(data)


More information and examples about these functions can be found in the API documentation (:doc:`/apidocs_plot_mpl`).

.. raw:: latex

    \clearpage


Masking data points
-------------------

|DataSet| objects support non-destructive masking of data points.
Masking a data point means that the data point will not be processed when analyzing or plotting the |DataSet| object.

.. doctest::

   >>> from pyimpspec import DataSet, parse_data
   >>> from typing import Dict
   >>>
   >>> data: DataSet = parse_data("./tests/data.dta")[0]
   >>> data.get_num_points()  # Included points
   29
   >>> # Apply a low-pass filter
   >>> data.low_pass(1e3)
   >>> data.get_num_points()
   22
   >>> # Apply a high-pass filter as well on top of the low-pass filter
   >>> data.high_pass(1e1)
   >>> data.get_num_points()
   15
   >>> # Check how many points are included or excluded
   >>> data.get_num_points(masked=False)  # Included points
   15
   >>> data.get_num_points(masked=True)  # Excluded points
   14
   >>> data.get_num_points(masked=None)  # All points
   29
   >>> # Get the current mask
   >>> mask: Dict[int, bool] = data.get_mask()
   >>> # Include the highest frequency point again
   >>> mask[0] = False
   >>> data.set_mask(mask)
   >>> data.get_num_points()
   16
   >>> # Clear the mask
   >>> data.set_mask({})
   >>> data.get_num_points()
   29

|DataSet| objects have various methods for getting certain types of values while taking the applied mask into account (e.g., |DataSet.get_frequencies|  or |DataSet.get_phases|).

Below are two Bode plots of the example above before and after the low- and high-pass filters were applied.

.. plot::

   from pyimpspec import DataSet
   from pyimpspec import mpl
   from pyimpspec import generate_mock_data

   data = generate_mock_data("CIRCUIT_1")[0]
   figure, axes = mpl.plot_bode(data)

   data.low_pass(1e3)
   data.high_pass(1e1)
   figure, axes = mpl.plot_bode(data)

.. raw:: latex

    \clearpage


Custom parsers
--------------

Adding support for parsing additional file formats is quite easy.
Simply write a function that parses a file and returns a pandas.DataFrame_ object, and then pass that object to the |dataframe_to_data_sets| function.
The `pandas.read_json <https://pandas.pydata.org/docs/reference/api/pandas.read_json.html>`_ function is used in the example below, but one could also turn a dictionary into a DataFrame using the `pandas.DataFrame.from_dict <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.from_dict.html>`_ method.

.. doctest::

   >>> from pyimpspec import DataSet, dataframe_to_data_sets
   >>> from pandas import read_json
   >>> from typing import List
   >>>
   >>> def parse_json(path: str) -> List[DataSet]:
   ...   return dataframe_to_data_sets(df=read_json(path), path=path)
   >>>
   >>> data: DataSet = parse_json("./tests/data.json")[0]
   >>> data.get_num_points()
   29
   >>> print(data.to_dataframe().to_markdown(index=False))
   |      f (Hz) |   Re(Z) (ohm) |   Im(Z) (ohm) |   Mod(Z) (ohm) |   Phase(Z) (deg.) |
   |------------:|--------------:|--------------:|---------------:|------------------:|
   | 10000       |       109.009 |      -26.5557 |        112.197 |         -13.6911  |
   |  7196.86    |       112.058 |      -35.1662 |        117.446 |         -17.423   |
   |  5179.47    |       116.906 |      -46.4664 |        125.802 |         -21.6762  |
   |  3727.59    |       124.835 |      -60.8523 |        138.876 |         -25.9875  |
   |  2682.7     |       137.77  |      -78.0894 |        158.362 |         -29.5449  |
   |  1930.7     |       157.972 |      -96.4806 |        185.104 |         -31.4143  |
   |  1389.5     |       186.637 |     -112.205  |        217.769 |         -31.014   |
   |  1000       |       221.691 |     -120.399  |        252.275 |         -28.5062  |
   |   719.686   |       257.437 |     -118.651  |        283.464 |         -24.7446  |
   |   517.947   |       288.118 |     -109.31   |        308.157 |         -20.7764  |
   |   372.759   |       311.563 |      -97.2957 |        326.402 |         -17.3427  |
   |   268.27    |       328.959 |      -86.554  |        340.155 |         -14.7413  |
   |   193.07    |       342.639 |      -78.8887 |        351.603 |         -12.9657  |
   |   138.95    |       354.649 |      -74.5879 |        362.408 |         -11.877   |
   |   100       |       366.4   |      -73.2559 |        373.651 |         -11.3063  |
   |    71.9686  |       378.778 |      -74.2957 |        385.996 |         -11.0974  |
   |    51.7947  |       392.326 |      -77.1022 |        399.831 |         -11.1184  |
   |    37.2759  |       407.37  |      -81.1148 |        415.368 |         -11.2613  |
   |    26.827   |       424.086 |      -85.8173 |        432.682 |         -11.4398  |
   |    19.307   |       442.528 |      -90.7275 |        451.733 |         -11.5863  |
   |    13.895   |       462.638 |      -95.3932 |        472.371 |         -11.6508  |
   |    10       |       484.246 |      -99.3993 |        494.342 |         -11.5998  |
   |     7.19686 |       507.068 |     -102.385  |        517.302 |         -11.4155  |
   |     5.17947 |       530.727 |     -104.069  |        540.835 |         -11.0943  |
   |     3.72759 |       554.773 |     -104.271  |        564.487 |         -10.6447  |
   |     2.6827  |       578.721 |     -102.924  |        587.802 |         -10.0845  |
   |     1.9307  |       602.093 |     -100.083  |        610.354 |          -9.43768 |
   |     1.3895  |       624.462 |      -95.9026 |        631.783 |          -8.73106 |
   |     1       |       645.479 |      -90.6181 |        651.809 |          -7.99147 |


If you wish for pyimpspec to include support for additional file formats, then you can either:

- Open an issue on GitHub_ and provide an example of a data file.
- Submit a pull request on GitHub_. Note that pyimpspec is licensed under GPLv3_ or later.


.. raw:: latex

    \clearpage
