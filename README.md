# pyimpspec

A package for parsing, validating, analyzing, and simulating impedance spectra.

[![tests](https://github.com/vyrjana/pyimpspec/actions/workflows/test-package.yml/badge.svg)](https://github.com/vyrjana/pyimpspec/actions/workflows/test-package.yml)
[![build](https://github.com/vyrjana/pyimpspec/actions/workflows/test-wheel.yml/badge.svg)](https://github.com/vyrjana/pyimpspec/actions/workflows/test-wheel.yml)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pyimpspec)
[![GitHub](https://img.shields.io/github/license/vyrjana/pyimpspec)](https://www.gnu.org/licenses/gpl-3.0.html)
[![PyPI](https://img.shields.io/pypi/v/pyimpspec)](https://pypi.org/project/pyimpspec/)


## Table of contents

- [About](#about)
- [Contributing](#contributing)
- [License](#license)
- [Changelog](#changelog)


## About

Pyimpspec is a Python package that provides an application programming interface (API) and a command-line interface (CLI) for working with impedance spectra.
The target audience is researchers who use electrochemical impedance spectroscopy (EIS) though the program may also be useful in educational settings.
Those looking for a program with a graphical user interface may wish to instead use [DearEIS](https://github.com/vyrjana/DearEIS), which is based on pyimpspec.

The interfaces of pyimpspec implement features such as:

- reading certain data formats and parsing the experimental data contained within
- validation of impedance spectra using linear Kramers-Kronig tests or the Z-HIT algorithm
- construction of circuits by, e.g., parsing circuit description codes (CDC)
- support for user-defined circuit elements
- estimation of the distribution of relaxation times (DRT)
- complex non-linear least squares fitting of equivalent circuits
- simulation of the impedance spectra of circuits
- visualization of impedance spectra and various analysis results

See the [official documentation](https://vyrjana.github.io/pyimpspec/) for instructions on how to install pyimpspec, examples of how to use the API, and the API reference.


## Changelog

See [CHANGELOG.md](CHANGELOG.md) for details.


## Contributing

If you wish to contribute to the further development of pyimpspec, then there are several options available to you depending on your ability and the amount of time that you can spare.

If you find bugs, wish some feature was added, or find the documentation to be lacking, then please open an issue on [GitHub](https://github.com/vyrjana/pyimpspec/issues).

If you wish to contribute code, then start by cloning the repository:

`git clone https://github.com/vyrjana/pyimpspec.git`

The development dependencies can be installed from within the repository directory using pip:

`pip install -r ./dev-requirements.txt`

Create a new branch based on either the `main` branch or the most recent development branch (e.g., `dev-*`), and submit your changes as a pull request.

Code contributions should, if it is applicable, also include unit tests, which should be implemented in files placed in the `tests` folder found in the root of the repository along with any assets required by the tests.
It should be possible to run the tests by executing the `run_tests.sh` script, which uses the test discovery process built into the `unittest` module that is included with Python.

If it is possible, then delay the importing of dependencies until they are needed in order to reduce the overhead associated with just importing pyimpspec.
Ignoring this guideline may, e.g., make using the CLI slower than it needs to be.

See [CONTRIBUTORS](CONTRIBUTORS) for a list of people who have contributed to the pyimpspec project.


## License

Copyright 2024 pyimpspec developers

Pyimpspec is licensed under the [GPLv3 or later](https://www.gnu.org/licenses/gpl-3.0.html).

The licenses of pyimpspec's dependencies and/or sources of portions of code are included in the LICENSES folder.
