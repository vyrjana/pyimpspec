# pyimpspec

A package for parsing, validating, analyzing, and simulating impedance spectra.


## Table of contents

- [Installing](#installing)
- [Features](#features)
	- [Circuits](#circuits)
	- [Data parsing](#data-parsing)
	- [Kramers-Kronig tests](#kramers-kronig-tests)
	- [Equivalent circuit fitting](#equivalent-circuit-fitting)
	- [Plotting](#plotting)
- [Contributors](#contributors)
- [License](#license)


## Installing

_pyimpspec_ can be installed with _pip_.

```
pip install pyimpspec
```


## Features

The sections below are merely brief descriptions of _pyimpspec_'s features.
See [the Jupyter notebook](https://github.com/vyrjana/pyimpspec/blob/main/examples/examples.ipynb) for examples of those features in action.


### Circuits

_pyimpspec_ supports the creation of `Circuit` objects, which can be used to simulate impedance spectra or to extract information from experimental data by means of complex non-linear least squares (CNLS) fitting.
The easiest way to create circuits is by letting _pyimpspec_ parse a circuit description code (CDC).
An extended CDC syntax, which makes it possible to define e.g. initial values, is also supported.


### Data parsing

Several file formats are supported by _pyimpspec_ and the data within is used to generate a `DataSet` object.
The file formats include:
- Files containing the data as character-separate values (CSV).
- Spreadsheets (`.xls`, `.xlsx`, `.ods`).
- Ivium data files (`.idf`, `.ids`).
- Gamry data files (`.dta`).

Not all CSV files and spreadsheets are necessarily supported as-is but the parsing of those types of files should be quite flexible.
The parsers expect to find at least a column with frequencies and columns for either the real and imaginary parts of the impedance, or the absolute magnitude and the phase angle/shift.
The sign of the imaginary part of the impedance and/or the phase angle/shift may be negative, but then that has to be indicated in the column header with a `-` prefix.
Additional file formats may be supported in the future.

`DataSet` objects can also be created from dictionaries.
The contents of the `DataSet` can be transformed into a `pandas.DataFrame` object, which in turn can be used to output the data in a variety of formats (CSV, Markdown, LaTeX, etc.).


### Kramers-Kronig tests

The three tests (i.e. complex, real, and imaginary) described in _"A linear Kramers-Kronig transform test for immittance data validation"_ by Bernard A. Boukamp (_Journal of the Electrochemical Society_, **1995**, 142, 6, pp. 1885-1894, DOI: 10.1149/1.2044210) are implemented in _pyimpspec_.
A variant of the complex test that uses CNLS to perform the fitting is also included.

The procedure described in _"A method for improving the robustness of linear Kramers-Kronig validity tests"_ by Michael Schönleber, Dino Klotz, and Ellen Ivers-Tiffée (_Electrochimica Acta_, **2014**, 131, pp. 20-27, DOI: 10.1016/j.electacta.2014.01.034) is also implemented in _pyimpspec_.

The relevant functions return `KramersKronigResult` objects that include:
- The fitted `Circuit` object that is generated as part of the test.
- The pseudo chi-squared and the µ values.
- The frequencies of the data points that were tested.
- The complex impedances produced by the fitted circuit at each of the tested frequencies.
- The residuals of the real and imaginary parts of the impedances.


### Equivalent circuit fitting

Fitting equivalent circuits to impedance spectra is easy with _pyimpspec_ and generates a `FittingResult` object.
The `FittingResult` object includes:
- The fitted `Circuit` object.
- Information about all of the parameters (e.g. final fitted value, estimated error, and whether or not the parameter had a fixed value during fitting).
- The frequencies that were used during the fitting.
- The complex impedances produced by the fitted circuit at each of the frequencies.
- The residuals of the real and imaginary parts of the impedances.
- The `MinimizerResult` object returned by _lmfit_.


### Plotting

_pyimpspec_ includes functions for visualizing `Circuit`, `DataSet`, `KramersKronigResult`, and `FittingResult` objects.
The only backend that is currently supported is _matplotlib_.


## Contributors

See [CONTRIBUTORS](https://github.com/vyrjana/pyimpspec/blob/main/CONTRIBUTORS) for a list of people who have contributed to the _pyimpspec_ project.


## License

Copyright 2022 pyimpspec developers

_pyimpspec_ is licensed under the [GPLv3 or later](https://www.gnu.org/licenses/gpl-3.0.html).

The licenses of _pyimpspec_'s dependencies and/or sources of portions of code are included in the LICENSES folder.

