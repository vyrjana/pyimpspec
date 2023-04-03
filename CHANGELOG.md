# 4.1.0 (2023/04/03)

- Added the L-curve corner search algorithm described in DOI:10.1088/2633-1357/abad0d as an approach for automatically finding the optimal regularization parameter in DRT methods that use Tikhonov regularization.
- Added optional `custom_labels` keyword arguments to the `Circuit.to_circuitikz` and `Circuit.to_drawing` methods.
- Updated descriptions for the command-line arguments with multiple valid string values.
- Updated docstrings for the DRT calculation functions for the BHT and TR-RBF methods.
- Fixed a bug where large negative `num_procs` arguments were not handled properly.


# 4.0.1 (2023/03/22)

- Updated how `perform_exploratory_tests` handles cases where all mu values are greater than the mu-criterion.


# 4.0.0 (2023/03/20)

**THIS UPDATE CONTAINS SEVERAL CHANGES THAT ARE NOT BACKWARDS COMPATIBLE WITH CODE WRITTEN USING VERSION 3.x!**

- Added a command-line interface for quickly performing tasks via a terminal. The CLI also has support for a configuration file with user-definable default values.
- Added support for user-defined circuit elements, which can be registered using a `register_element` function along with some definition classes (e.g. `ElementDefinition` and `ParameterDefinition`).
- Added several transmission line models for blocking and non-blocking porous electrodes.
- Added support for parsing circuit description codes (CDCs) with additional metadata (e.g., version number). This can be used to help with parsing some older CDCs in the future if, e.g., the name of a circuit element's parameter has been changed.
- Added a Container class (subclass of Element) that can have parameters and subcircuits. 
- Added a general transmission line model that is implemented as a subclass of Container.
- Added support for parsing CDCs for elements based on the Container class. Subcircuits that are short circuits or open circuits can be specified with `short` (or `zero`) and `open` (or `inf`), respectively (e.g., `cdc = "Tlm{X_1=short, Z_B=open}"`).
- Added an implementation of the Z-HIT algorithm, which provides another means of data analysis/validation.
- Added statsmodels as an explicit dependency.
- Added `Circuit.serialize` method that returns a string that contains the corresponding CDC and some metadata (e.g., version number).
- Added a `get_label` method to the `FitResult` and `TestResult` classes.
- Added a `colored_axes` keyword argument to most of the plotting functions that can be used to set the colors of the y-axes to match that of the data plotted in those axes.
- Added the xdg package as an explicit dependency and use it for determining where to place the optional config file for the command-line interface.
- Added `low_pass` and `high_pass` methods to the `DataSet` class for quickly masking frequencies above or below a cutoff frequency, respectively.
- Added support for using `inf` to indicate an infinite limit for an element parameter's lower or upper limit in a circuit description code (interpreted as either negative or positive if provided as the limit for the lower or upper limit, respectively).
- Added type aliases for, e.g., frequencies, impedances, and residuals.
- Added `get_default_num_procs` and `set_default_num_procs` functions. The former is used internally but may also be of interest for debugging. The latter can be used to globally set the number of process to use instead of setting the value of `num_procs` in the various function calls (e.g., `fit_circuit`).
- Added numdifftools as explicit dependency.
- Updated the function signatures of most plotting functions (e.g., to make them more consistent).
- Updated title generation for `FitResult` objects in the `plot.mpl.plot_fit` function.
- Updated data parsing to handle files containing multiple impedance spectra and to have a more consistent API (e.g., the individual parsers all return `List[DataSet]`).
- Updated the `progress` module to support registration of multiple callbacks and the ability to unregister callbacks.
- Updated the implementation of `Circuit` objects.
- Updated the implementation of `Connection` objects:
	- These objects now include methods similar to those present in lists (`append`, `extend`, `remove`, etc.).
	- `Parallel` objects now properly handle paths that are short circuits or open circuits.
- Updated the implementation of `Element` objects:
	- Elements no longer have static identifiers assigned to them but rather they are assigned dynamically as needed, which improves support for adding connections and/or elements after a `Circuit` object has been created.
	- Defining a new circuit element (i.e., a subclass of `Element`) requires much less boilerplate code and implements automatic generation of the class docstring.
	- It is now possible to calculate the impedance of a circuit element when the excitation frequency approaches zero or infinity. Note that some elements may raise an exception about infinite impedance when attempting to calculate the impedance at a limit (e.g., `Capacitor` when f -> 0).
- Updated the DRT analysis module so that each method have their own result class based on a common DRTResult class.
- Updated the TR-NNLS method implementation (DRT analysis) to use a new approach to automatically suggest regularization parameters.
- Updated axis labels in plots.
- Updated LaTeX output for element equations.
- Updated parameter names/symbols of the de Levie element.
- Updated the function signature for the m(RQ)fit method (DRT analysis).
- Refactored circuit fitting, Kramers-Kronig testing, and m(RQ)fit (DRT analysis) to work with the dynamic identifiers.
- Refactored the BHT method implementation (DRT analysis).
- Refactored to use lazy imports, which avoids the overhead associated with importing dependencies until they are required.
- Improved the performance of impedance calculations (e.g., by utilizing NumPy better and by removing redundant assertions).
- Switched to Sphinx for documentation.
- Fixed bugs in the calculations of the TR-RBF method (DRT analysis) when using the 2nd derivative order in combination with some radial basis functions.
- Fixed a bug that prevented spreadsheet files from being detected as such when parsing data.
- Fixed a bug that raised an exception when calling `DataSet.to_dataframe` with `masked=True` or `masked=False` when there were masked data points.
- Fixed a bug that raised an exception when calling `Connection.get_connections` with `flattened=False`.
- Fixed a bug that caused the `Connection.get_elements` method to return elements in the wrong order with `flattened=False`.
- Fixed a bug that caused parallel connections to not properly handle short circuits.
- Fixed a bug that caused m(RQ)fit result to incorrectly calculate time constants.
- Fixed a bug that could cause an exception when calling `sympy.sympify` on an equation.

**NOTE!** The cvxopt package is now an optional dependency rather than an explicit one.
If that package is not available for a particular platform, then it should now not prevent installation of pyimpspec.
However, the TR-RBF method (DRT analysis) requires a convex optimizer package and attempting to use the method without one installed will raise an exception.
Currently, kvxopt and cvxpy can also be used if cvxopt is not available.
Kvxopt, which is a fork of cvxopt, may support a greater number of platforms (e.g., operating systems and/or hardware architectures).
Windows and MacOS users should carefully read the installation instructions for cvxpy if deciding to install that package as it requires certain development tools to be installed first.


# 3.2.4 (2022/12/14)

- Updated dependency versions.
- Fixed a few bugs in calculations in the TR-RBF method (DRT analysis) when using 2nd order derivatives and either the Cauchy or the inverse quadric radial basis function.
- Fixed a typo.


# 3.2.3 (2022/11/26)

- Updated the `parse_data` function to try other parsers when the parser that was chosen based on the file extension fails.
- Updated parsing of data stored as CSV.
- Improved support for files containing data with decimal commas.


# 3.2.2 (2022/11/25)

- Updated the .mpt parser to handle files without a metadata section.


# 3.2.1 (2022/11/22)

- Fixed the BHT method (DRT analysis) so that it works properly when `num_procs` is set to greater than one and NumPy is using OpenBLAS.


# 3.2.0 (2022/11/01)

- Added support for calculating the distribution of relaxation times using the `m(RQ)fit` method.
- Added `HavriliakNegamiAlternative` as an element with an alternative form of Havriliak-Negami relaxation.
- Added `ModifiedInductor` as an element for modeling non-ideal inductance.
- Updated the assertion message related to valid methods in the `calculate_drt` function.
- Updated the default lower and/or upper limits of some elements.
- Updated how `numpy.nan` is handled when creating `FittedParameter` objects.
- Refactored a minor portion of the TR-NNLS method (DRT analysis) code.
- Refactored the `pyimpspec.circuit.get_elements` function to ensure all circuit elements have unique symbols.
- Fixed a bug that caused the `Circuit.get_connections` method to return an empty list in some cases when invoked with `flattened=True`.
- Fixed bugs that caused the `Circuit.to_diagram` method to produce incorrect results.


# 3.1.3 (2022/10/28)

- Added support for `kvxopt` as an optional dependency as a drop-in replacement for `cvxopt`.
- Updated how import errors related to the convex optimizers required by the TR-RBF method (DRT analysis) are handled, which should allow the rest of pyimpspec to function even if no convex optimizers can be imported successfully.


# 3.1.2 (2022/09/15)

- Added the 3-sigma CI series to the legends of DRT plots.
- Updated the order that the mean and 3-sigma CI series are plotted in DRT plots.


# 3.1.1 (2022/09/13)

- Updated API documentation.


# 3.1.0 (2022/09/11)

- Added `Circuit.to_drawing` method for drawing circuit diagrams using the `schemdraw` package.
- Added `schemdraw` as an explicit dependency.
- Added support for using the `cvxpy` package as an optional solver in DRT calculations (TR-RBF method only).
- Added `cvxpy` as an optional dependency.
- Added `CircuitBuilder.__iadd__` method so that the `+=` operator can be used instead of the `CircuitBuilder.add` method.
- Updated `Element.set_label`, `Element.set_fixed`, `Element.set_lower_limit`, and `Element.set_upper_limit` methods to return the element so that the calls can be chained (e.g., `Resistor(R=50).set_label("ct").set_fixed("R", True)`).
- Updated the default terminal labels used in circuit diagrams.
- Updated how the title is generated in the `mpl.plot_fit` function.
- Updated minimum versions for dependencies.


# 3.0.0 (2022/09/05)

**Breaking changes in the API!**

- The `KramersKronigResult` class has been renamed to `TestResult`.
- The `FittingResult` class has been renamed to `FitResult`.
- `DataSet`, `TestResult`, and `FitResult` methods such as `get_bode_data` that previously returned base-10 logarithms of, e.g., frequencies now return the linear values.
- The `perform_exploratory_tests` function now returns a list of results that have already been sorted from best to worst.
- The `score_test_results` function has been removed.
- The `string_to_circuit` function has been renamed to `parse_cdc`.
- The `fit_circuit_to_data` function has been renamed to `fit_circuit`.
- Some of the plotting function signatures have changed:
	- Added `label`, `legend`, and `adjust_axes` keyword arguments.
	- Added color arguments to the `plot_circuit` function.
	- The `plot_mu_xps` and `plot_exploratory_test_results` functions now take a `List[TestResult]` as the first argument.
	- Some argument names have changed (e.g., `nyquist_color` in `plot_circuit` has changed to `color_nyquist`).

# 

- Added support for calculating the distribution of relaxation times using a few different methods.
	See the `calculate_drt` function, which returns a `DRTResult` object, for details.
- Added new functions to `plot.mpl` (`plot_drt`, `plot_complex_impedance`, etc.).
- Added `cvxopt` and `scipy` as explicit dependencies.
- Added `get_connections` and `substitute_element` methods to the `Circuit` and `Connection` classes.
- Added a `get_relative_error` method to the `FittedParameter` class.
- Added a `calculate_score` method to the `TestResult` class.
- Added progress system that can be provided a callback to obtain information about progress during, e.g., Kramers-Kronig testing.
- Added a `CircuitBuilder` class as an alternate means of creating circuits.
- Updated the `DataSet` class to sort data points in descending order of frequency when instantiated.
- Updated the `plot.mpl.plot_*` functions to make use of logarithmic scales instead of logarithmic values where applicable.
- Updated the appearance of the figures generated by `plot.mpl.plot_mu_xps`.
- Updated the `DataSet.set_mask` method to accept the use of `numpy.bool_` values in masks.
- Updated the return value of the `to_latex` method of circuits, elements, and connections.
- Refactored to use generator expressions when performing, e.g., Kramers-Kronig tests.


# 2.2.0 (2022/08/10)

- Added `num_per_decade` argument to the `pyimpspec.plot.mpl.plot_fit` function.
- Added sorting of elements to the `to_dataframe` method in the `FittingResult` class.
- Added `tabulate` package as explicit dependency.


# 2.1.0 (2022/08/04)

- Added support for `.mpt` data format.
- Refactored code.


# 2.0.1 (2022/08/01)

- Added GitHub Actions workflow for testing the package on Linux (Ubuntu), MacOS, and Windows.
- Fixed issues that prevented using the package with anything but Python 3.10.


# 2.0.0 (2022/07/31)

- Added support for parsing the `.i2b` data format.
- Added support for parsing the `.P00` data format.
- Updated minimum Python and dependency versions.
- Updated documentation.
- Refactored code.
- Removed support for parsing `.xls` files.


# 1.1.0 (2022/07/13)

- Added support for parsing the `.dfr` data format.
- Refactored `pyimpspec.data.formats`.
- Updated the API and its documentation.


# 1.0.1 (2022/07/05)

- Updated docstrings.


# 1.0.0 (2022/06/16)

- Added assertions to ensure that DataSet objects have at least one unmasked data point before performing a Kramers-Kronig test or a circuit fit.
- Updated the parsing of .dta files to support files generated with the THD (total harmonic distortion) setting enabled.
- Updated the validation of circuits prior to fitting.
- Updated some of the messages of type assertions to print the variable(s) being tested.
- Updated type assertions regarding DataSet objects to be more permissive in order to better support DearEIS.
- Refactored the code related to turning dictionaries into DataSet objects in order to better support DearEIS.
- Refactored the fitting and Kramers-Kronig testing modules to reduce code duplication.
- Refactored pyimpspec.plot.mpl so that `import *` doesn't pollute the namespace.


# 0.1.3 (2022/04/04)

- Updated the implementation of the `.idf/.ids` parser.
- Updated assertion messages in `.idf/.ids` and `.dta` parsers.


# 0.1.2 (2022/03/29)

- Added a missing trigger for raising an exception when encountering an unsupported file format.
- Fixed a bug that prevented parsing of some `.idf/.ids` data files.

# 0.1.1 (2022/03/28)

- Added a circuit validation step prior to fitting a circuit.


# 0.1.0 (2022/03/28)

- Initial public beta release.
