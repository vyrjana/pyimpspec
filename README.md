# pyimpspec

A package for parsing, validating, analyzing, and simulating impedance spectra.

[![tests](https://github.com/vyrjana/pyimpspec/actions/workflows/test-package.yml/badge.svg)](https://github.com/vyrjana/pyimpspec/actions/workflows/test-package.yml)
[![build](https://github.com/vyrjana/pyimpspec/actions/workflows/test-wheel.yml/badge.svg)](https://github.com/vyrjana/pyimpspec/actions/workflows/test-wheel.yml)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pyimpspec)
[![GitHub](https://img.shields.io/github/license/vyrjana/pyimpspec)](https://www.gnu.org/licenses/gpl-3.0.html)
[![PyPI](https://img.shields.io/pypi/v/pyimpspec)](https://pypi.org/project/pyimpspec/)


## Table of contents

- [About](#about)
- [Installing](#installing)
- [Features](#features)
	- [Circuits](#circuits)
	- [Data parsing](#data-parsing)
	- [Kramers-Kronig tests](#kramers-kronig-tests)
	- [Equivalent circuit fitting](#equivalent-circuit-fitting)
	- [Distribution of relaxation times](#distribution-of-relaxation-times)
	- [Plotting](#plotting)
- [Contributing](#contributing)
- [License](#license)
- [Changelog](#changelog)


## About

pyimpspec is a Python package that provides an application programming interface (API) for working with impedance spectra.
The target audience is researchers who use electrochemical impedance spectroscopy (EIS).
Those looking for a program with a graphical user interface may wish to instead use [DearEIS](https://github.com/vyrjana/DearEIS), which is based on pyimpspec.

The API of pyimpspec implements the functionality required to:

- read certain data formats and parse the experimental data contained within
- validate impedance spectra by checking if the data is Kramers-Kronig transformable
- construct circuits by parsing a circuit description code
- extract quantitative data from an impedance spectrum through complex non-linear least squares fitting of a circuit
- simulate the impedance response of circuits
- perform basic visualization of impedance spectra and test/fit/simulation results

See the [Features](#features) section for more information.

Check out [this Jupyter notebook](examples/examples.ipynb) for examples of how to use pyimpspec.
Documentation about the API can be found [here](https://vyrjana.github.io/pyimpspec/api).

If you encounter issues, then please open an issue on [GitHub](https://github.com/vyrjana/pyimpspec/issues).


## Getting started

### Requirements

- [Python](https://www.python.org)
- The following Python packages
	- [cvxopt](https://github.com/cvxopt/cvxopt): convex optimization
	- [lmfit](https://lmfit.github.io/lmfit-py/): non-linear least squares minimization
	- [matplotlib](https://matplotlib.org/): visualization
	- [numpy](https://numpy.org/): numerical computation
	- [odfpy](https://github.com/eea/odfpy): reading and writing OpenDocument files
	- [openpyxl](https://openpyxl.readthedocs.io/en/stable/): reading and writing Excel files
	- [pandas](https://pandas.pydata.org/): data manipulation and analysis
	- [scipy](https://github.com/scipy/scipy): numerical computation
	- [sympy](https://www.sympy.org/en/index.html): symbolic computation
	- [tabulate](https://github.com/astanin/python-tabulate): formatting of Markdown tables

The Python packages (and their dependencies) are installed automatically when pyimpspec is installed using [pip](https://pip.pypa.io/en/stable/).

### Installing

The latest version of pyimpspec requires a **recent version of Python (3.8+)** and the most straightforward way to install pyimpspec is by using [pip](https://pip.pypa.io/en/stable/):
Make sure that Python and pip are installed first and then type the following command into a terminal of your choice (e.g. PowerShell in Windows).

```
pip install pyimpspec
```

pyimpspec should now be importable in, e.g., Python scripts and Jupyter notebooks.

Newer versions of pyimpspec can be installed at a later date by adding the `--upgrade` option to the command:

```
pip install --upgrade pyimpspec
```

Supported platforms:

- Linux
- Windows
- MacOS

The package **may** also work on other platforms depending on whether or not those platforms are supported by pyimpspec's [dependencies](setup.py).


## Features

### Circuits

pyimpspec supports the creation of `Circuit` objects, which can be used to simulate impedance spectra or to extract information from experimental data by means of complex non-linear least squares (CNLS) fitting.
The recommended way to create circuits is by letting pyimpspec parse a circuit description code (CDC).
An extended CDC syntax, which makes it possible to define e.g. initial values, is also supported.
`Circuit` objects also have additional features such as generation of LaTeX source for drawing circuit diagrams (requires `\usepackage{circuitikz}` in the header of the LaTeX document).


### Data parsing

Several file formats are supported by pyimpspec and the data within are used to generate a `DataSet` object.
The supported file formats include, for example:

- BioLogic: `.mpt`
- Eco Chemie: `.dfr`
- Gamry: `.dta`
- Ivium: `.idf` and `.ids`
- Spreadsheets: `.xlsx` and `.ods`
- Plain-text character-separated values (CSV)

Not all CSV files and spreadsheets are necessarily supported as-is but the parsing of those types of files should be quite flexible.
The parsers expect to find at least a column with frequencies and columns for either the real and imaginary parts of the impedance, or the absolute magnitude and the phase angle/shift.
The sign of the imaginary part of the impedance and/or the phase angle/shift may be negative, but then that has to be indicated in the column header with a `-` prefix.
Additional file formats may be supported in the future.

`DataSet` objects can also be turned into `dict` objects as well as created from them, which is convenient for serialization (e.g. using Javascript Object Notation).
The contents of the `DataSet` can also be transformed into a `pandas.DataFrame` object, which in turn can be used to output the data in a variety of formats (CSV, Markdown, LaTeX, etc.).


### Kramers-Kronig tests

Implementations of the three variants of the linear Kramers-Kronig tests (see [DOI:10.1149/1.2044210](https://doi.org/10.1149/1.2044210)) are included.
A variant of the complex test that uses CNLS fitting is also included.
An implementation of the procedure for finding a suitable number of RC elements to avoid under- and overfitting (see [DOI:10.1016/j.electacta.2014.01.034](https://doi.org/10.1016/j.electacta.2014.01.034)) is also included.
A variant on this procedure is also included to help avoid false negatives that may occasionally occur.

The relevant functions return `TestResult` objects that include:

- The fitted `Circuit` object that is generated as part of the test.
- The corresponding pseudo chi-squared and µ-values.
- The frequencies of the data points that were tested.
- The complex impedances produced by the fitted circuit at each of the tested frequencies.
- The residuals of the real and imaginary parts of the impedances.


### Equivalent circuit fitting

The `Circuit` objects can be fitted to impedance spectra to obtain fitted values for the various parameters included in circuit elements such as resistors, capacitors, constant phase elements, and Warburg elements.
The `FitResult` object produced by this process includes:

- The fitted `Circuit` object.
- Information about the parameters (e.g., final fitted value, estimated error, and whether or not the parameter had a fixed value during fitting).
- The frequencies that were used during the fitting.
- The complex impedances produced by the fitted circuit at each of the frequencies.
- The residuals of the real and imaginary parts of the impedances.
- The `MinimizerResult` object returned by lmfit.


### Distribution of relaxation times

A few implementations for calculating the distribution of relaxation times are included:

- Tikhonov regularization and non-negative least squares (see [DOI:10.1039/D0CP02094J](https://doi.org/10.1039/D0CP02094J)).
- Tikhonov regularization and either radial basis functions or piecewise linear discretization (see [DOI:10.1016/j.electacta.2015.09.097](https://doi.org/10.1016/j.electacta.2015.09.097)).
	An optional feature supported by this method is the calculation of the Bayesian credible intervals (see [DOI:10.1016/j.electacta.2015.03.123](https://doi.org/10.1016/j.electacta.2015.03.123) and [DOI:10.1016/j.electacta.2017.07.050](https://doi.org/10.1016/j.electacta.2017.07.050)).
- The Bayesian Hilbert transform (BHT) method (see [DOI:10.1016/j.electacta.2020.136864](https://doi.org/10.1016/j.electacta.2020.136864)).
	The results for this method include scores that can be used for assessing the quality of an impedance spectrum.

The results are contained in a `DRTResult` object that includes:

- The time constant and gamma values.
- The frequency values used in the process.
- The impedance values of the modeled response.
- The residuals of the real and imaginary parts of the impedances.


### Plotting

pyimpspec includes functions for visualizing `Circuit`, `DataSet`, `TestResult`, `DRTResult`, and `FitResult` objects.
The only backend that is currently supported is matplotlib.
The functions offer some room for customization of the figures, but they are primarily intended for quick visualization.


## Changelog

See [CHANGELOG.md](CHANGELOG.md) for details.


## Contributing

If you wish to contribute to the further development of pyimpspec, then there are several options available to you depending on your ability and the amount of time that you can spare.
If you find bugs, wish some feature was added, or find the documentation to be lacking, then please open an issue on [GitHub](https://github.com/vyrjana/pyimpspec/issues).
If you wish to contribute code, then clone the repository, create a new branch based on either the main branch or the most recent development branch, and submit your changes as a pull request.
Code contributions should, if it is applicable, also include unit tests, which should be implemented in files placed in the `tests` folder found in the root of the repository along with any assets required by the tests.
It should be possible to run the tests by executing the `run_tests.sh` script, which uses the test discovery built into the `unittest` module that is included with Python.

See [CONTRIBUTORS](CONTRIBUTORS) for a list of people who have contributed to the pyimpspec project.


## License

Copyright 2022 pyimpspec developers

pyimpspec is licensed under the [GPLv3 or later](https://www.gnu.org/licenses/gpl-3.0.html).

The licenses of pyimpspec's dependencies and/or sources of portions of code are included in the LICENSES folder.
