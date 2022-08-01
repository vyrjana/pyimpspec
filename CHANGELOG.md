# 2.0.1

- Added GitHub Actions workflow for testing the package on Linux (Ubuntu), MacOS, and Windows.
- Fixed issues that prevented using the package with anything but Python 3.10.


# 2.0.0

- Added support for parsing the `.i2b` data format.
- Added support for parsing the `.P00` data format.
- Updated minimum Python and dependency versions.
- Updated documentation.
- Refactored code.
- Removed support for parsing `.xls` files.


# 1.1.0

- Added support for parsing the `.dfr` data format.
- Refactored `pyimpspec.data.formats`.
- Updated the API and its documentation.


# 1.0.1

- Updated docstrings.


# 1.0.0

- Added assertions to ensure that DataSet objects have at least one unmasked data point before performing a Kramers-Kronig test or a circuit fit.
- Updated the parsing of .dta files to support files generated with the THD (total harmonic distortion) setting enabled.
- Updated the validation of circuits prior to fitting.
- Updated some of the messages of type assertions to print the variable(s) being tested.
- Updated type assertions regarding DataSet objects to be more permissive in order to better support DearEIS.
- Refactored the code related to turning dictionaries into DataSet objects in order to better support DearEIS.
- Refactored the fitting and Kramers-Kronig testing modules to reduce code duplication.
- Refactored pyimpspec.plot.mpl so that `import *` doesn't pollute the namespace.


# 0.1.3

- Updated the implementation of the `.idf/.ids` parser.
- Updated assertion messages in `.idf/.ids` and `.dta` parsers.


# 0.1.2

- Added a missing trigger for raising an exception when encountering an unsupported file format.
- Fixed a bug that prevented parsing of some `.idf/.ids` data files.

# 0.1.1

- Added a circuit validation step prior to fitting a circuit.


# 0.1.0

- Initial public beta release.
