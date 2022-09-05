---
layout: documentation
title: API - Kramers-Kronig testing
permalink: /api/kramers-kronig/
---

Check the page for [high-level functions](https://vyrjana.github.io/pyimpspec/api/high-level-functions) for the recommended ways to perform a Kramers-Kronig test to generate a `TestResult` object.

**Table of Contents**

- [TestResult](#pyimpspectestresult)
	- [calculate_score](#pyimpspectestresultcalculate_score)
	- [get_bode_data](#pyimpspectestresultget_bode_data)
	- [get_frequency](#pyimpspectestresultget_frequency)
	- [get_impedance](#pyimpspectestresultget_impedance)
	- [get_nyquist_data](#pyimpspectestresultget_nyquist_data)
	- [get_residual_data](#pyimpspectestresultget_residual_data)


### **pyimpspec.TestResult**

An object representing the results of a linear Kramers-Kronig test applied to a data set.

```python
class TestResult(object):
	circuit: Circuit
	num_RC: int
	mu: float
	pseudo_chisqr: float
	frequency: ndarray
	impedance: ndarray
	real_residual: ndarray
	imaginary_residual: ndarray
```

_Constructor parameters_

- `circuit`: The fitted circuit.
- `num_RC`: The final number of RC elements in the fitted model (Boukamp, 1995).
- `mu`: The mu-value of the final fit (eq. 21 in Schönleber et al., 2014).
- `pseudo_chisqr`: The pseudo chi-squared fit value (eq. 14 in Boukamp, 1995).
- `frequency`: The frequencies used to perform the test.
- `impedance`: The impedance produced by the fitted circuit at each of the tested frequencies.
- `real_residual`: The residuals for the real parts (eq. 15 in Schönleber et al., 2014).
- `imaginary_residual`: The residuals for the imaginary parts (eq. 16 in Schönleber et al., 2014).


_Functions and methods_

#### **pyimpspec.TestResult.calculate_score**

Calculate a score based on the provided mu-criterion and the statistics of the result.
A result with a mu-value greater than or equal to the mu-criterion will get a score of -numpy.inf.

```python
def calculate_score(self, mu_criterion: float) -> float:
```


_Parameters_

- `mu_criterion`: The mu-criterion to apply.
See perform_test for details.


_Returns_
```python
float
```

#### **pyimpspec.TestResult.get_bode_data**

Get the data necessary to plot this TestResult as a Bode plot: the frequencies, the absolute magnitudes of the impedances, and the negative phase angles/shifts of the impedances in degrees.

```python
def get_bode_data(self, num_per_decade: int = -1) -> Tuple[ndarray, ndarray, ndarray]:
```


_Parameters_

- `num_per_decade`


_Returns_
```python
Tuple[ndarray, ndarray, ndarray]
```

#### **pyimpspec.TestResult.get_frequency**


```python
def get_frequency(self, num_per_decade: int = -1) -> ndarray:
```


_Parameters_

- `num_per_decade`


_Returns_
```python
ndarray
```

#### **pyimpspec.TestResult.get_impedance**


```python
def get_impedance(self, num_per_decade: int = -1) -> ndarray:
```


_Parameters_

- `num_per_decade`


_Returns_
```python
ndarray
```

#### **pyimpspec.TestResult.get_nyquist_data**

Get the data necessary to plot this TestResult as a Nyquist plot: the real and the negative imaginary parts of the impedances.

```python
def get_nyquist_data(self, num_per_decade: int = -1) -> Tuple[ndarray, ndarray]:
```


_Parameters_

- `num_per_decade`


_Returns_
```python
Tuple[ndarray, ndarray]
```

#### **pyimpspec.TestResult.get_residual_data**

Get the data necessary to plot the relative residuals for this TestResult: the frequencies, the relative residuals for the real parts of the impedances in percents, and the relative residuals for the imaginary parts of the impedances in percents.

```python
def get_residual_data(self) -> Tuple[ndarray, ndarray, ndarray]:
```


_Returns_
```python
Tuple[ndarray, ndarray, ndarray]
```



