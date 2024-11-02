# 5.1.0 (2024/11/02)

- Added support for analyzing the peaks in DRT results by fitting skew normal distributions:
  - Added an `analyze_peaks` method to the `DRTResult` class.
  - Added a `DRTPeaks` class, which can be used to, e.g., calculate the area of a peak.
- Added an implementation of the Loewner method for calculating the distribution of relaxation times:
  - Added a `calculate_drt_lm` function.
  - Added an `LMResult` class.
  - Added `--model-order` and `--model-order-method` CLI arguments.
  - Updated the `plot.mpl.plot_gamma` function to support plotting `LMResult` instances.
- Added the ability to define constraints when fitting circuits:
  - Added a `generate_fit_identifiers` function.
  - Added a `FitIdentifiers` class.
- Added support for plotting DRT results as gamma vs f.
  - Added CLI argument for plotting DRT results as gamma vs f.
- Added an alternative form of the Gerischer element (`GerischerAlternative`) with the parameters `R`, `tau`, and `n`.
- Added more circuits for generating mock data.
- Added support for the modified Akima spline (`makima`) to the interpolation phase of Z-HIT and set it as the default.
- Added support for specifying multiple methods and/or weights to use when calling the `fit_circuit` function.
- Added a `max_iter` argument to the TR-NNLS method in case the default number of iterations is insufficient.
- Updated the TR-RBF implementation to be based off of a newer version of pyDRTtools.
- Updated plotting functions to support drawing smoother lines if the input result has a `circuit` property.
- Updated an exception message to provide more information if a variable that is being fitted is out of bounds.
- Updated documentation.
- Updated the `generate_mock_data` function so that it attempts to cast arguments to the appropriate type if some other types of values (e.g., integers) are provided instead.
- Updated the `circuit.registry.register_element` function to support a new `private` keyword argument.
  - Updated the `circuit.registry.get_elements` function to not include by default `Element` classes that were registered with `private=True`.
  - Updated the `KramersKronigRC` and `KramersKronigAdmittanceRC` classes to be registered with `private=True`.
- Updated some plotting functions (e.g., primitives such as `mpl.plot_nyquist`, `mpl.plot_gamma`) to support `None` as input so that the functions can be used to set up a blank plot.
- Updated the `get_default_num_procs` function to also support additional environment variables that may be supported by OpenBLAS (depends upon the settings used when OpenBLAS was compiled).
- Updated the `fit_circuit` function to ignore `RuntimeWarning`s during fitting.
- Updated the `Warburg` element class to include an `n` exponent (fixed at 0.5 by default).
- Updated the `plot_gamma` function to include a horizontal line marking zero on the y-axis.
- Updated the automatic adjustment of initial values of circuit parameters to be bypassed when a non-default value is detected in a `Circuit` that is provided to the m(RQ)-fit method without a corresponding `FitResult`.
- Updated the m(RQ)-fit method to support resistive-inductive `(RQ)` elements. The `n` parameter of the `ConstantPhaseElement` instance would then need to be `-1.0 <= n < 0.0`.
- Updated the parsing of data files to better support cases where a decimal comma is used instead of a decimal point.
- Fixed a bug that caused methods such as `DRTResult.get_peaks` to miss peaks at the time constant extremes.
- Fixed a bug that caused an element's parameters in `FitResult.to_parameters_dataframe` to not be in a sorted order.
- Fixed the previously unimplemented `FitResult.get_parameters` method.
- Fixed a bug that caused `FitResult.to_parameters_dataframe` to return negative standard errors when the fitted value was negative.
- Fixed a bug that could cause `FitResult.to_parameters_dataframe` to divide by zero.
- Fixed a bug that could cause `FittedParameter.get_relative_error` to divide by zero.
- Fixed a bug where an exception would be raised when whitespace was included between keyword arguments when passing circuit identifiers or CDCs via the CLI (e.g., `<R(RC):noise=5e-2, log_max_f=4>` would previously raise an exception whereas `<R(RC):noise=5e-2,log_max_f=4>` would not).
- Fixed a bug in the `perform_exploratory_kramers_kronig_tests` function that caused an exception to be raised when `admittance=True` or `admittance=False`.
- Fixed a bug where passing a list of `KramersKronigResult` objects corresponding to a noise-free `DataSet` to the `suggest_num_RC_limits` function could cause an exception to be raised because the lower limit of the number of RC elements was estimated to be greater than the highest tested number of RC elements.
- Fixed a bug where `get_default_num_procs` could raise an exception if an environment variable (e.g., `OPENBLAS_NUM_THREADS`) was assigned a non-numerical value.
- Fixed a bug that caused the range of time constants of the m(RQ)-fit and TR-NNLS results to be incorrect although the peaks were still at the right values.
- Refactored some of the code.


# 5.0.2 (2024/09/10)

- Updated the documentation (e.g., various function and method docstrings have been updated). Notably, the functions related to handling progress updates are now included in the API documentation.


# 5.0.1 (2024/09/03)

- Updated the docstring of the `generate_mock_data` function.
- Updated the documentation to include the functions for generating mock data and circuits.
- Fixed a bug in a function for interpolating frequencies.


# 5.0.0 (2024/08/29)

**THIS UPDATE CONTAINS SEVERAL CHANGES THAT ARE NOT BACKWARDS COMPATIBLE WITH CODE WRITTEN USING VERSION 4.x!**
**SOME OF THE ARGUMENTS IN THE COMMAND LINE INTERFACE HAVE ALSO CHANGED OR BEEN REMOVED!**


### Linear Kramers-Kronig tests

- Renamed the `TestResult` class to `KramersKronigResult`.
- Renamed the `perform_test` function to `perform_kramers_kronig_test`.
- Removed the `perform_exploratory_tests` function. The `pyimpspec.analysis.kramers_kronig.evaluate_log_F_ext` function should be used instead.
- The `complex`, `real`, and `imaginary` tests now use`numpy.linalg.lstsq`. The previous implementations based on matrix inversion are now accessible by appending `-inv` (e.g., `complex-inv`).
- Updated the `perform_test` and `perform_exploratory_tests` function signatures (e.g., `test='real'`, `add_capacitance=True`, and `add_inductance=True` are now the new defaults).
- Replaced the `--add-capacitance` and `--add-inductance` CLI arguments with `--no-capacitance` and `--no-inductance`. They new arguments have the same abbreviated forms, which means that `-C` and `-L` now have the opposite effect compared to previously.
- Added a variant of the mu-criterion algorithm that fits a logistic function to the mu-values (accessible via negative mu-criterion values).
- Added a `suggest_num_RC` function for suggesting the optimum number of RC elements to use when performing linear Kramers-Kronig tests:
    - Estimates the lower and upper limits for the range of numbers of RC elements to avoid under- and overfitting.
    - Uses one or more methods/algorithms for suggesting the optimum number of RC elements:
      - Method 1 (https://doi.org/10.1016/j.electacta.2014.01.034)
      - Method 2 (https://doi.org/10.1109/IWIS57888.2022.9975131)
      - Method 3 (https://doi.org/10.1109/IWIS57888.2022.9975131)
      - Method 4 (https://doi.org/10.1109/IWIS57888.2022.9975131)
      - Method 5 (https://doi.org/10.1016/j.electacta.2024.144951)
      - Method 6 (https://doi.org/10.1016/j.electacta.2024.144951)
    - Defaults to an approach that uses methods 4, 3, and 5 (in that order) to narrow the list of options.
    - Multiple methods can be combined in different ways:
      - mean of the suggested values
      - sum of scores obtained based on rankings
      - sum of relative scores
- Added support for performing linear Kramers-Kronig tests on the admittance representation:
  - Added `KramersKronigAdmittanceRC` element to represent the series RC element used in the equivalent circuit model.
  - Added a boolean `admittance` attribute to the `TestResult` class.
  - Added `get_parallel_resistance`, `get_parallel_capacitance`, and `get_parallel_inductance` methods to the `TestResult` class.
  - Added a variant of the mu-criterion algorithm that uses capacitance values instead of resistance values when operating on the admittance representation.
- Added `suggest_representation` function for suggesting either the impedance or admittance representation to use.
- Added `evaluate_time_constant_extensions` function for optimizing the extension of the range of time constants.
- Added the following arguments to the CLI:
  - `--admittance` and `--impedance` to only perform tests on the admittance and impedance representation, respectively.
  - `--suggestion-methods` to selecting one or more methods for suggesting the optimum number of RC elements.
  - `--mean`, `--ranking`, and `--sum` to specify how to combine multiple methods for suggesting the optimum number of RC elements.
  - `--num-F-ext-evaluations` to specify the number of evaluations to perform when optimizing the extension of the range of time constants.
  - `--min-log-F-ext` and `--max-log-F-ext` to specify the lower and upper limits for the number of decades to extend the range of time constants when `--num-F-ext-evaluations` is set to something else than zero.
  - `--log-F-ext` to specify the number of decades to extend the range of time constants when `--num-F-ext-evaluations` is set to zero.
  - `--no-rapid-F-ext-evaluations` to evaluate the full range of the number of time constants at each sampled extension of the range of time constants.
  - `--lower-limit`/`--upper-limit` to specify the lower/upper limit for the optimum number of RC elements to suggest.
  - `--limit-delta` as an alternative way of specifying the limits of the range of optimum number of RC elements to suggest.
  - `--plot-immittance` to automatically plot the corresponding immittance representation that was used when performing the linear Kramers-Kronig test.
  - `--plot-pseudo-chi-squared` to override the plot when a single suggestion method has been chosen.
  - `--plot-moving-average-width` to plot the moving averages of the residuals (the number of points must be provided).
  - `--plot-estimated-noise` to include the estimated standard deviation of the noise.
  - `--plot-log-F-ext-3d` and `--plot-log-F-ext-2d` to plot the pseudo chi-squared values as a function of the number of time constants and the extension of the range of time constants.
  - `--plot-auto-limited-residuals` to automatically adjust the limits of the y-axes when plotting the relative residuals.
- Added utility functions for subdividing frequency ranges and for calculating the curvatures of impedance spectra.
- Updated the `perform_test` function to make use of the `perform_exploratory_tests`, `suggest_num_RC`, and `suggest_representation` functions.
- Refactored the `perform_exploratory_tests` function to only perform tests with different numbers of RC elements.
- Removed the `--automatic` argument from the CLI.
- Updated the CLI to use similar plots both for exploratory results and when manually selecting a number of RC elements.
- Removed the `mu` attribute from the `TestResult` class.
- Fixed a bug in calculation of mu values that caused the series resistance to be included.
- Some functions are no longer available the top level of the package and must instead be accessed via the `pyimpspec.analysis.kramers_kronig` module.


### Z-HIT analysis

- Added support for performing Z-HIT analysis on admittance data.
- Added a CLI argument for performing analyses on admittance data (`--admittance` or `-A`).
- Added two smoothing algorithms (https://doi.org/10.1021/acsmeasuresciau.1c00054):
  - `whithend`: Whittaker-Henderson
  - `modsinc`: modified sinc kernel with linear extrapolation
- Updated the default smoothing algorithm to be `modsinc`.
- Added title to plot by default when performing analyses via the CLI.
- Changed `statsmodels` from a required dependency to an optional dependency.
- Added support for showing a plot of the residuals when using the CLI.


### Fitting

- Added an optional `timeout` argument to the `fit_circuit` function that can be used to set a time limit. This can be used to force the fitting process to timeout if it is taking a very long time to finish.
- Added `--timeout` argument to the CLI.
- Added `--type` argument to the CLI so that fit results can optionally be plotted as, e.g., just a Nyquist plot.


### Distribution of relaxation times

- Updated the TR-RBF implementation to be based off of a newer version of pyDRTtools:
  - `lambda_value` is now automatically determined using a cross-validation method unless the new `cross_validation` argument is an empty string (i.e., `cross_validation=""`).
  - If one of the cross-validation methods is chosen, then `lambda_value` is used as the initial value.
  - The currently supported cross-validation (CV) methods are:
    - `"gcv"` - generalized cross-validation (GCV)
    - `"mgcv"` - modified GCV
    - `"rgcv"` - robust GCV
    - `"re-im"` - real-imaginary CV
    - `"lc"` - L-curve
  - See https://doi.org/10.1149/1945-7111/acbca4 for more information about the CV methods.
- Removed the `maximum_symmetry` argument from the TR-RBF implementation.
- Changed how timeouts and progress updates are implemented when the TR-RBF method is called with `credible_intervals=True` and `timeout` is greater than zero.
- Some functions and classes are no longer available the top level of the package and must instead be accessed via the `pyimpspec.analysis.drt` module.


### Plotting

- Added support for plotting admittance data:
  - The affected plotting functions now have an optional, boolean `admittance` keyword argument.
- Added a CLI argument for plotting admittance data (`--plot-admittance` or `-pY`).
- Removed the `mpl.plot_mu_xps` function.
- Added an `mpl.plot_pseudo_chisqr` function for plotting the pseudo chi-squared values of `TestResult` instances.
- Updated the `mpl.plot_residuals` function to not use markers by default.
- Fixed a bug that caused `mpl.plot_residuals` to have empty legend labels when no markers were used.
- Updated how the limits are automatically determined by the `mpl.plot_residuals` function.
- Updated how the ticks are determined in the y-axes of the `mpl.plot_residuals` function.
- Added an `mpl.plot_suggestion` function that visualizes the suggested numbers of RC elements to use for linear Kramers-Kronig testing.
- Added an `mpl.plot_suggestion_method` function that visualizes the data that is used to suggest the number of RC elements to use for linear Kramers-Kronig testing.
- Removed support for colored axes from the `mpl.plot_nyquist` function.
- Updated the `mpl.plot_nyquist` function to switch to using a marker when using `line=True` if all points are approximately the same.
- Updated how the `--plot-type` CLI argument is handled when plotting, e.g., DRT results.
- Added an `mpl.show` function that acts as a wrapper for `matplotlib.pyplot.show`.
- Renamed the `plot_tests` function to `plot_kramers_kronig_tests`.
- Renamed the `plot_complex` function to `plot_real_imaginary`.


### Data parsing

- Added support for parsing ZView/ZPlot `.z` files.
- Added support for parsing PalmSens `.pssession` files.
- Added support for two more variants of column headers to parsers that attempt to automatically identify columns.
- Added support for using `pathlib.Path` in addition to `str` when passing paths to, e.g., the `parse_data` function.
- Added `--output-indices` argument to the CLI to include zero-based indices in text output.
- Added `--exclude-indices` argument to the CLI so that specific data points (e.g., outliers) can be excluded based on their zero-based indices.
- Added `--nth-data-set` argument to the CLI so that one or more data sets can be chosen from a file.
- Updated parsing of `.dta` files to support parsing the drift corrected impedances when it is available. The returned `List[DataSet]` is sorted so that the drift corrected impedance spectra have a lower index in the list than the uncorrected impedance spectra.
- Fixed a bug that caused an exception to be raised when parsing a spreadsheet that also contained at least one empty sheet.
- Fixed a bug that caused NumPy arrays in the dictionary returned by `DataSet.to_dict` to not be compatible with `json.dump` and `json.dumps` from the standard library.
- Fixed a bug where `DataSet.from_dict` was unable to handle mask dictionaries where the keys were strings instead of integers.
- Fixed a bug where the keyword arguments provided to `parse_data` were not being passed on to the different format parsers in a specific case.
- Fixed a bug where detecting columns in files would fail if an otherwise valid column name started with whitespace.


### Elements

- Added the ZARC element, which is represented by `Zarc` in circuit description codes.
- Added a `reset_default_parameter_values` function to reset the default parameter values of either all element types or specific element types.
- Added `remove_elements` and `reset` functions to the `pyimpspec.circuit.registry` module that can be used to remove user-defined elements, or to remove user-defined elements and reset the default parameter values, respectively.
- Added an optional `default_only` parameter to the `get_elements` function so that the function can return either all elements (including user-defined elements) or just the elements included in pyimpspec by default. The parameter is set to `False` by default, which means that all elements are returned.


### Miscellaneous

- Added `get` and `get_total` methods to the `Progress` class for obtaining the current step and the current total.
- Added `register_default_handler` and `clear_default_handler_output` functions to the `progress` module.
- Added mock data for circuit with negative differential resistance.
- Added mock data for Randles circuit with diffusion.
- Added noisy variants of mock data.
- Added a `set_default_values` class method to circuit elements.
- Refactored code.
- Updated minimum versions of dependencies and removed support for Python 3.9.
- Removed cvxpy from the list of supported optional dependencies.
- Added `canvas` argument to `Circuit.to_drawing` method.
- Changed some CLI argument names to improve consistency.


# 4.1.1 (2024/03/14)

- Maintenance release that updates the version requirements for dependencies.
- Support for Python 3.8 has been dropped due to minimum requirements set by one or more dependencies.
- Support for Python 3.11 and 3.12 has been added.
- Added Jinja2 as an explicit dependency.
- Updated the automated determination of how many parallel processes to use.


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
