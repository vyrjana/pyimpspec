---
layout: documentation
title: API - fitting
permalink: /api/fitting/
---

Check the page for [high-level functions](https://vyrjana.github.io/pyimpspec/api/high-level-functions) for the recommended way to perform an equivalent circuit fit to generate a `FittingResult` object.

**Table of Contents**

- [FittedParameter](#pyimpspecfittedparameter)
	- [from_dict](#pyimpspecfittedparameterfrom_dict)
	- [to_dict](#pyimpspecfittedparameterto_dict)
- [FittingError](#pyimpspecfittingerror)
- [FittingResult](#pyimpspecfittingresult)
	- [get_bode_data](#pyimpspecfittingresultget_bode_data)
	- [get_frequency](#pyimpspecfittingresultget_frequency)
	- [get_impedance](#pyimpspecfittingresultget_impedance)
	- [get_nyquist_data](#pyimpspecfittingresultget_nyquist_data)
	- [get_residual_data](#pyimpspecfittingresultget_residual_data)
	- [to_dataframe](#pyimpspecfittingresultto_dataframe)


### **pyimpspec.FittedParameter**

An object representing a fitted parameter.

```python
class FittedParameter(object):
	value: float
	stderr: Optional[float] = None
	fixed: bool = False
```

_Constructor parameters_

- `value`: The fitted value.
- `stderr`: The estimated standard error of the fitted value.
- `fixed`: Whether or not this parameter had a fixed value during the circuit fitting.


_Functions and methods_

#### **pyimpspec.FittedParameter.from_dict**


```python
def from_dict(dictionary: dict) -> FittedParameter:
```


_Parameters_

- `dictionary`


_Returns_
```python
FittedParameter
```

#### **pyimpspec.FittedParameter.to_dict**


```python
def to_dict(self) -> dict:
```


_Returns_
```python
dict
```




### **pyimpspec.FittingError**

```python
class FittingError(Exception):
	args
	kwargs
```

_Constructor parameters_

- `args`
- `kwargs`




### **pyimpspec.FittingResult**

An object representing the results of fitting a circuit to a data set.

```python
class FittingResult(object):
	circuit: Circuit
	parameters: Dict[str, Dict[str, FittedParameter]]
	pseudo_chisqr: float
	minimizer_result: MinimizerResult
	frequency: ndarray
	impedance: ndarray
	real_residual: ndarray
	imaginary_residual: ndarray
	method: str
	weight: str
```

_Constructor parameters_

- `circuit`: The fitted circuit.
- `parameters`: Fitted parameters and their estimated standard errors (if possible to estimate).
- `pseudo_chisqr`: The pseudo chi-squared fit value (eq. 14 in Boukamp, 1995).
- `minimizer_result`: The results of the fit as provided by the lmfit.minimize function.
- `frequency`: The frequencies used to perform the fit.
- `impedance`: The impedance produced by the fitted circuit at each of the fitted frequencies.
- `real_residual`: The residuals for the real parts (eq. 15 in Schönleber et al., 2014).
- `imaginary_residual`: The residuals for the imaginary parts (eq. 16 in Schönleber et al., 2014).
- `method`: The iterative method used during the fitting process.
- `weight`: The weight function used during the fitting process.


_Functions and methods_

#### **pyimpspec.FittingResult.get_bode_data**

Get the data necessary to plot this FittingResult as a Bode plot: the base-10 logarithms of the frequencies, the base-10 logarithms of the absolute magnitudes of the impedances, and the negative phase angles/shifts of the impedances in degrees.

```python
def get_bode_data(self, num_per_decade: int = -1) -> Tuple[ndarray, ndarray, ndarray]:
```


_Parameters_

- `num_per_decade`: The number of points per decade.
A positive value results in data points being calculated using the fitted circuit within the original frequency range.
Otherwise, only the original frequencies are used.


_Returns_
```python
Tuple[ndarray, ndarray, ndarray]
```

#### **pyimpspec.FittingResult.get_frequency**


```python
def get_frequency(self, num_per_decade: int = -1) -> ndarray:
```


_Parameters_

- `num_per_decade`


_Returns_
```python
ndarray
```

#### **pyimpspec.FittingResult.get_impedance**


```python
def get_impedance(self, num_per_decade: int = -1) -> ndarray:
```


_Parameters_

- `num_per_decade`


_Returns_
```python
ndarray
```

#### **pyimpspec.FittingResult.get_nyquist_data**

Get the data necessary to plot this FittingResult as a Nyquist plot: the real and the negative imaginary parts of the impedances.

```python
def get_nyquist_data(self, num_per_decade: int = -1) -> Tuple[ndarray, ndarray]:
```


_Parameters_

- `num_per_decade`: The number of points per decade.
A positive value results in data points being calculated using the fitted circuit within the original frequency range.
Otherwise, only the original frequencies are used.


_Returns_
```python
Tuple[ndarray, ndarray]
```

#### **pyimpspec.FittingResult.get_residual_data**

Get the data necessary to plot the relative residuals for this FittingResult: the base-10 logarithms of the frequencies, the relative residuals for the real parts of the impedances in percents, and the relative residuals for the imaginary parts of the impedances in percents.

```python
def get_residual_data(self) -> Tuple[ndarray, ndarray, ndarray]:
```


_Returns_
```python
Tuple[ndarray, ndarray, ndarray]
```

#### **pyimpspec.FittingResult.to_dataframe**


```python
def to_dataframe(self) -> DataFrame:
```


_Returns_
```python
DataFrame
```


