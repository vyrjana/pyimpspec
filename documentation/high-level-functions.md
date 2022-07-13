---
layout: default
title: API - high-level functions
permalink: /api/high-level-functions/
---

Check the other pages for information about the objects returned by the functions presented here.
            

**Table of Contents**

- [pyimpspec](#pyimpspec)
	- [fit_circuit_to_data](#pyimpspecfit_circuit_to_data)
	- [get_elements](#pyimpspecget_elements)
	- [parse_data](#pyimpspecparse_data)
	- [perform_exploratory_tests](#pyimpspecperform_exploratory_tests)
	- [perform_test](#pyimpspecperform_test)
	- [score_test_results](#pyimpspecscore_test_results)
	- [simulate_spectrum](#pyimpspecsimulate_spectrum)
	- [string_to_circuit](#pyimpspecstring_to_circuit)



## **pyimpspec**

### **pyimpspec.fit_circuit_to_data**

Fit a circuit to a data set.

```python
def fit_circuit_to_data(circuit: Circuit, data: DataSet, method: str = "auto", weight: str = "auto", max_nfev: int = -1, num_procs: int = -1) -> FittingResult:
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
FittingResult
```
### **pyimpspec.get_elements**

Returns a mapping of element symbols to the element class.

```python
def get_elements() -> Dict[str, Element]:
```


_Returns_

```python
Dict[str, Element]
```
### **pyimpspec.parse_data**

Parse experimental data and return a list of DataSet instances.
One or more specific sheets can be specified by name when parsing spreadsheets (e.g. .xlsx or .ods) to only return DataSet instances for those sheets.
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

Performs a batch of linear Kramers-Kronig tests.

```python
def perform_exploratory_tests(data: DataSet, test: str = "complex", num_RCs: List[int] = [], mu_criterion: float = 0.85, add_capacitance: bool = False, add_inductance: bool = False, method: str = "leastsq", max_nfev: int = -1, num_procs: int = -1) -> List[KramersKronigResult]:
```


_Parameters_

- `data`: The data set to be tested.
- `test`: See perform_test for details.
- `num_RCs`: A list of integers representing the various number of parallel RC circuits to test.
An empty list results in all possible numbers of parallel RC circuits up to the total number of frequencies being tested.
- `mu_criterion`: See perform_test for details.
- `add_capacitance`: See perform_test for details.
- `add_inductance`: See perform_test for details.
- `method`: See perform_test for details.
- `max_nfev`: See perform_test for details.
- `num_procs`: See perform_test for details.


_Returns_

```python
List[KramersKronigResult]
```
### **pyimpspec.perform_test**

Performs a linear Kramers-Kronig test as described by Boukamp (1995).
The results can be used to check the validity of an impedance spectrum before performing equivalent circuit fitting.
If the number of (RC) circuits is less than two, then a suitable number of (RC) circuits is determined using the procedure described by Schönleber et al.(2014) based on a criterion for the calculated mu-value (zero to one).
A mu-value of one represents underfitting and a mu-value of zero represents overfitting.

References:

- B.A. Boukamp, 1995, J. Electrochem. Soc., 142, 1885-1894 (https://doi.org/10.1149/1.2044210)
- M. Schönleber, D. Klotz, and E. Ivers-Tiffée, 2014, Electrochim. Acta, 131, 20-27 (https://doi.org/10.1016/j.electacta.2014.01.034)

```python
def perform_test(data: DataSet, test: str = "complex", num_RC: int = 0, mu_criterion: float = 0.85, add_capacitance: bool = False, add_inductance: bool = False, method: str = "leastsq", max_nfev: int = -1, num_procs: int = -1) -> KramersKronigResult:
```


_Parameters_

- `data`: The data set to be tested.
- `test`: Supported values include "cnls", "complex", "imaginary", and "real". The "cnls" test performs a complex non-linear least squares fit using lmfit.minimize, which usually provides a good fit but is also quite slow.
The "complex", "imaginary", and "real" tests perform the complex, imaginary, and real tests, respectively, according to Boukamp (1995).
- `num_RC`: The number of parallel RC circuits to use.
A value less than one results in the use of the procedure described by Schönleber et al. (2014) based on the chosen mu-criterion.
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
KramersKronigResult
```
### **pyimpspec.score_test_results**

Assign scores to test results as an alternative to just using the mu-value generated when using the procedure described by Schönleber et al. (2014).
The mu-value can in some cases fluctuate wildly at low numbers of parallel RC circuits and result in false positives (i.e. the mu-value briefly dips below the mu-criterion only to rises above it again).
The score is equal to -numpy.inf for results with mu-values greater than or equal to the mu-criterion.
For results with mu-values below the mu-criterion, the score is calculated based on the pseudo chi-squared value of the result and on the difference between the mu-criterion and the result's mu-value.
The results and their corresponding scores are returned as a list of tuples.
The list is sorted from the highest score to the lowest score.
The result with the highest score should be a good initial guess for a suitable candidate.

```python
def score_test_results(results: List[KramersKronigResult], mu_criterion: float) -> List[Tuple[float, KramersKronigResult]]:
```


_Parameters_

- `results`: The result to score.
- `mu_criterion`: The mu_criterion to use.


_Returns_

```python
List[Tuple[float, KramersKronigResult]]
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
### **pyimpspec.string_to_circuit**

Generate a Circuit instance from a string that contains a circuit description code (CDC).

```python
def string_to_circuit(cdc: str) -> Circuit:
```


_Parameters_

- `cdc`: A circuit description code (CDC) corresponding to an equivalent circuit.


_Returns_

```python
Circuit
```