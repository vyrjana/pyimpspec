---
layout: documentation
title: API - high-level functions
permalink: /api/high-level-functions/
---

The functions presented here are the recommended way of reading data, creating circuits, performing tests, etc.
Check the other pages for information about the objects returned by the functions presented here.
            

**Table of Contents**

- [pyimpspec](#pyimpspec)
	- [calculate_drt](#pyimpspeccalculate_drt)
	- [fit_circuit](#pyimpspecfit_circuit)
	- [get_elements](#pyimpspecget_elements)
	- [parse_cdc](#pyimpspecparse_cdc)
	- [parse_data](#pyimpspecparse_data)
	- [perform_exploratory_tests](#pyimpspecperform_exploratory_tests)
	- [perform_test](#pyimpspecperform_test)
	- [simulate_spectrum](#pyimpspecsimulate_spectrum)



## **pyimpspec**

### **pyimpspec.calculate_drt**

Calculates the distribution of relaxation times (DRT) for a given data set.

```python
def calculate_drt(data: DataSet, method: str = "tr-nnls", mode: str = "complex", lambda_value: float = -1.0, rbf_type: str = "gaussian", derivative_order: int = 1, rbf_shape: str = "fwhm", shape_coeff: float = 0.5, inductance: bool = False, credible_intervals: bool = False, num_samples: int = 2000, num_attempts: int = 10, maximum_symmetry: float = 0.5, num_procs: int = -1) -> DRTResult:
```


_Parameters_

- `data`: The data set to use in the calculations.
- `method`: Valid values include:
- "bht": Bayesian Hilbert Transform
- "tr-nnls": Tikhonov regularization with non-negative least squares
- "tr-rbf": Tikhonov regularization with radial basis function discretization
- `mode`: Which parts of the data are to be included in the calculations.
Used by the "tr-nnls" and "tr-rbf" methods.
Valid values include:
- "complex" ("tr-rbf" method only and the default for that method)
- "real" (default for the "tr-nnls" method)
- "imaginary"
- `lambda_value`: The Tikhonov regularization parameter.
Used by the "tr-nnls" and "tr-rbf" methods.
If the method is "tr-nnls" and this value is equal to or below zero, then an attempt will be made to automatically find a suitable value.
- `rbf_type`: The type of function to use for discretization.
Used by the "bht" and "tr-rbf" methods.
Valid values include:
- "gaussian"
- "c0-matern"
- "c2-matern"
- "c4-matern"
- "c6-matern"
- "inverse-quadratic"
- "inverse-quadric"
- "cauchy"
- `derivative_order`: The order of the derivative used during discretization.
Used by the "bht" and "tr-rbf" methods.
- `rbf_shape`: The shape control of the radial basis functions.
Used by the "bht" and "tr-rbf" methods.
Valid values include:
- "fwhm": full width at half maximum
- "factor": shape_coeff is used directly
- `shape_coeff`: The full width at half maximum (FWHM) coefficient affecting the chosen shape type.
Used by the "bht" and "tr-rbf" methods.
- `inductance`: If true, then an inductive element is included in the calculations.
Used by the "tr-rbf" method.
- `credible_intervals`: If true, then the credible intervals are also calculated for the DRT results according to Bayesian statistics.
Used by the "tr-rbf" method.
- `num_samples`: The number of samples drawn when calculating the Bayesian credible intervals ("tr-rbf" method) or the Jensen-Shannon distance ("bht" method).
A greater number provides better accuracy but requires more time.
Used by the "bht" and "tr-rbf" methods.
- `num_attempts`: The minimum number of attempts to make when trying to find suitable random initial values.
A greater number should provide better results at the expense of time.
Used by the "bht" method.
- `maximum_symmetry`: A maximum limit (between 0.0 and 1.0) for a descriptor of the vertical symmetry of the DRT.
A high degree of symmetry is common for results where the gamma value oscillates rather than forms distinct peaks.
A low value for the limit should improve the results but may cause the "bht" method to take longer to finish.
This limit is only used in the "tr-rbf" method when the regularization parameter (lambda) is not provided.
Used by the "bht" and "tr-rbf" methods.
- `num_procs`: The maximum number of processes to use.
A value below one results in using the total number of CPU cores present.


_Returns_

```python
DRTResult
```
### **pyimpspec.fit_circuit**

Fit a circuit to a data set.

```python
def fit_circuit(circuit: Circuit, data: DataSet, method: str = "auto", weight: str = "auto", max_nfev: int = -1, num_procs: int = -1) -> FitResult:
```


_Parameters_

- `circuit`: The circuit to fit to a data set.
- `data`: The data set that the circuit will be fitted to.
- `method`: The iteration method used during fitting.
See lmfit's documentation for valid method names.
Note that not all methods supported by lmfit are possible in the current implementation (e.g. some methods may require a function that calculates a Jacobian).
The "auto" value results in multiple methods being tested in parallel and the best result being returned based on the chi-squared values.
- `weight`: The weight function to use when calculating residuals.
Currently supported values: "modulus", "proportional", "unity", "boukamp", and "auto".
The "auto" value results in multiple weights being tested in parallel and the best result being returned based on the chi-squared values.
- `max_nfev`: The maximum number of function evaluations when fitting.
A value less than one equals no limit.
- `num_procs`: The maximum number of parallel processes to use when method and/or weight is "auto".


_Returns_

```python
FitResult
```
### **pyimpspec.get_elements**

Returns a mapping of element symbols to the element class.

```python
def get_elements() -> Dict[str, Type[Element]]:
```


_Returns_

```python
Dict[str, Type[Element]]
```
### **pyimpspec.parse_cdc**

Generate a Circuit instance from a string that contains a circuit description code (CDC).

```python
def parse_cdc(cdc: str) -> Circuit:
```


_Parameters_

- `cdc`: A circuit description code (CDC) corresponding to an equivalent circuit.


_Returns_

```python
Circuit
```
### **pyimpspec.parse_data**

Parse experimental data and return a list of DataSet instances.
One or more specific sheets can be specified by name when parsing spreadsheets (e.g., .xlsx or .ods) to only return DataSet instances for those sheets.
If no sheets are specified, then all sheets will be processed and the data from successfully parsed sheets will be returned as DataSet instances.

```python
def parse_data(path: str, file_format: Optional[str] = None, kwargs) -> List[DataSet]:
```


_Parameters_

- `path`: The path to a file containing experimental data that is to be parsed.
- `file_format`: The file format (or extension) that should be assumed when parsing the data.
If no file format is specified, then the file format will be determined based on the file extension.
If there is no file extension, then attempts will be made to parse the file as if it was one of the supported file formats.
- `kwargs`: Keyword arguments are passed to the parser.


_Returns_

```python
List[DataSet]
```
### **pyimpspec.perform_exploratory_tests**

Performs a batch of linear Kramers-Kronig tests, which are then scored and sorted from best to worst before they are returned.

```python
def perform_exploratory_tests(data: DataSet, test: str = "complex", num_RCs: List[int] = [], mu_criterion: float = 0.85, add_capacitance: bool = False, add_inductance: bool = False, method: str = "leastsq", max_nfev: int = -1, num_procs: int = -1) -> List[TestResult]:
```


_Parameters_

- `data`: The data set to be tested.
- `test`: See perform_test for details.
- `num_RCs`: A list of integers representing the various number of RC elements to test.
An empty list results in all possible numbers of RC elements up to the total number of frequencies being tested.
- `mu_criterion`: See perform_test for details.
- `add_capacitance`: See perform_test for details.
- `add_inductance`: See perform_test for details.
- `method`: See perform_test for details.
- `max_nfev`: See perform_test for details.
- `num_procs`: See perform_test for details.


_Returns_

```python
List[TestResult]
```
### **pyimpspec.perform_test**

Performs a linear Kramers-Kronig test as described by Boukamp (1995).
The results can be used to check the validity of an impedance spectrum before performing equivalent circuit fitting.
If the number of RC elements is less than two, then a suitable number of RC elements is determined using the procedure described by Schönleber et al. (2014) based on a criterion for the calculated mu-value (zero to one).
A mu-value of one represents underfitting and a mu-value of zero represents overfitting.

References:

- B.A. Boukamp, 1995, J. Electrochem. Soc., 142, 1885-1894 (https://doi.org/10.1149/1.2044210)
- M. Schönleber, D. Klotz, and E. Ivers-Tiffée, 2014, Electrochim. Acta, 131, 20-27 (https://doi.org/10.1016/j.electacta.2014.01.034)

```python
def perform_test(data: DataSet, test: str = "complex", num_RC: int = 0, mu_criterion: float = 0.85, add_capacitance: bool = False, add_inductance: bool = False, method: str = "leastsq", max_nfev: int = -1, num_procs: int = -1) -> TestResult:
```


_Parameters_

- `data`: The data set to be tested.
- `test`: Supported values include "cnls", "complex", "imaginary", and "real". The "cnls" test performs a complex non-linear least squares fit using lmfit.minimize, which usually provides a good fit but is also quite slow.
The "complex", "imaginary", and "real" tests perform the complex, imaginary, and real tests, respectively, according to Boukamp (1995).
- `num_RC`: The number of RC elements to use.
A value greater than or equal to one results in the specific number of RC elements being tested.
A value less than one results in the use of the procedure described by Schönleber et al. (2014) based on the chosen mu-criterion.
If the provided value is negative, then the maximum number of RC elements to test is equal to the absolute value of the provided value.
If the provided value is zero, then the maximum number of RC elements to test is equal to the number of frequencies in the data set.
- `mu_criterion`: The chosen mu-criterion. See Schönleber et al. (2014) for more information.
- `add_capacitance`: Add an additional capacitance in series with the rest of the circuit.
- `add_inductance`: Add an additional inductance in series with the rest of the circuit.
Applies only to the "cnls" test.
- `method`: The fitting method to use when performing a "cnls" test.
See the list of methods that are listed in the documentation for the lmfit package.
Methods that do not require providing bounds for all parameters or a function to calculate the Jacobian should work.
- `max_nfev`: The maximum number of function evaluations when fitting.
A value less than one equals no limit.
Applies only to the "cnls" test.
- `num_procs`: The maximum number of parallel processes to use when performing a test.
A value less than one results in using the number of cores returned by multiprocessing.cpu_count.
Applies only to the "cnls" test.


_Returns_

```python
TestResult
```
### **pyimpspec.simulate_spectrum**

Simulate the impedance spectrum generated by a circuit in a certain frequency range.

```python
def simulate_spectrum(circuit: Circuit, frequencies: Union[List[float], ndarray] = [], label: str = "") -> DataSet:
```


_Parameters_

- `circuit`: The circuit to use when calculating impedances at various frequencies.
- `frequencies`: A list of floats representing frequencies in Hz.
If no frequencies are provided, then a frequency range of 10 mHz to 100 kHz with 10 points per decade will be used.
- `label`: The label for the DataSet that is returned.


_Returns_

```python
DataSet
```