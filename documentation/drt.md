---
layout: documentation
title: API - drt
permalink: /api/drt/
---

Check the page for [high-level functions](https://vyrjana.github.io/pyimpspec/api/high-level-functions) for the recommended way to calculate the distribution of relaxation times to generate a `DRTResult` object.

**Table of Contents**

- [DRTError](#pyimpspecdrterror)
- [DRTResult](#pyimpspecdrtresult)
	- [get_bode_data](#pyimpspecdrtresultget_bode_data)
	- [get_drt_credible_intervals](#pyimpspecdrtresultget_drt_credible_intervals)
	- [get_drt_data](#pyimpspecdrtresultget_drt_data)
	- [get_frequency](#pyimpspecdrtresultget_frequency)
	- [get_gamma](#pyimpspecdrtresultget_gamma)
	- [get_impedance](#pyimpspecdrtresultget_impedance)
	- [get_label](#pyimpspecdrtresultget_label)
	- [get_nyquist_data](#pyimpspecdrtresultget_nyquist_data)
	- [get_peaks](#pyimpspecdrtresultget_peaks)
	- [get_residual_data](#pyimpspecdrtresultget_residual_data)
	- [get_score_dataframe](#pyimpspecdrtresultget_score_dataframe)
	- [get_scores](#pyimpspecdrtresultget_scores)
	- [get_tau](#pyimpspecdrtresultget_tau)
	- [to_dataframe](#pyimpspecdrtresultto_dataframe)


### **pyimpspec.DRTError**

```python
class DRTError(Exception):
	args
	kwargs
```

_Constructor parameters_

- `args`
- `kwargs`




### **pyimpspec.DRTResult**

An object representing the results of calculating the distribution of relaxation times in a  data set.

```python
class DRTResult(object):
	label: str
	tau: ndarray
	gamma: ndarray
	frequency: ndarray
	impedance: ndarray
	real_residual: ndarray
	imaginary_residual: ndarray
	mean_gamma: ndarray
	lower_bound: ndarray
	upper_bound: ndarray
	imaginary_gamma: ndarray
	scores: Dict[str, complex]
	chisqr: float
	lambda_value: float
```

_Constructor parameters_

- `label`: Includes information such as the method used.
- `tau`: The time constants.
- `gamma`: The gamma values.
These values are the real gamma values when the BHT method has been used.
- `frequency`: The frequencies of the impedance spectrum.
- `impedance`: The impedance produced by the model.
- `real_residual`: The residuals of the real parts of the model and the data set.
- `imaginary_residual`: The residuals of the imaginary parts of the model and the data set.
- `mean_gamma`: The mean gamma values of the Bayesian credible intervals.
- `lower_bound`: The lower bound gamma values of the Bayesian credible intervals.
- `upper_bound`: The upper bound gamma values of the Bayesian credible intervals.
- `imaginary_gamma`: The imaginary gamma values produced by the BHT method.
- `scores`: The scores calculated by the BHT method.
- `chisqr`: The chi-squared value of the modeled impedance.
- `lambda_value`: The lambda value that was ultimately used.


_Functions and methods_

#### **pyimpspec.DRTResult.get_bode_data**

Get the data necessary to plot this DataSet as a Bode plot: the frequencies, the absolute magnitudes of the impedances, and the negative phase angles/shifts of the impedances in degrees.

```python
def get_bode_data(self) -> Tuple[ndarray, ndarray, ndarray]:
```


_Returns_
```python
Tuple[ndarray, ndarray, ndarray]
```

#### **pyimpspec.DRTResult.get_drt_credible_intervals**

Get the data necessary to plot the Bayesian credible intervals for this DRTResult: the time constants, the mean gamma values, the lower bound gamma values, and the upper bound gamma values.

```python
def get_drt_credible_intervals(self) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
```


_Returns_
```python
Tuple[ndarray, ndarray, ndarray, ndarray]
```

#### **pyimpspec.DRTResult.get_drt_data**

Get the data necessary to plot this DRTResult as a DRT plot: the time constants and the corresponding gamma values.

```python
def get_drt_data(self, imaginary: bool = False) -> Tuple[ndarray, ndarray]:
```


_Parameters_

- `imaginary`: Get the imaginary gamma (non-empty only when using the BHT method).


_Returns_
```python
Tuple[ndarray, ndarray]
```

#### **pyimpspec.DRTResult.get_frequency**

Get the frequencies (in hertz) of the data set.

```python
def get_frequency(self) -> ndarray:
```


_Returns_
```python
ndarray
```

#### **pyimpspec.DRTResult.get_gamma**

Get the gamma values.

```python
def get_gamma(self, imaginary: bool = False) -> ndarray:
```


_Parameters_

- `imaginary`: Get the imaginary gamma (non-empty only when using the BHT method).


_Returns_
```python
ndarray
```

#### **pyimpspec.DRTResult.get_impedance**

Get the complex impedance of the model.

```python
def get_impedance(self) -> ndarray:
```


_Returns_
```python
ndarray
```

#### **pyimpspec.DRTResult.get_label**

The label includes information such as the method that was used.

```python
def get_label(self) -> str:
```


_Returns_
```python
str
```

#### **pyimpspec.DRTResult.get_nyquist_data**

Get the data necessary to plot this DataSet as a Nyquist plot: the real and the negative imaginary parts of the impedances.

```python
def get_nyquist_data(self) -> Tuple[ndarray, ndarray]:
```


_Returns_
```python
Tuple[ndarray, ndarray]
```

#### **pyimpspec.DRTResult.get_peaks**

Get the time constants (in seconds) and gamma (in ohms) of peaks with magnitudes greater than the threshold.
The threshold and the magnitudes are all relative to the magnitude of the highest peak.

```python
def get_peaks(self, threshold: float = 0.0, imaginary: bool = False) -> Tuple[ndarray, ndarray]:
```


_Parameters_

- `threshold`: The threshold for the relative magnitude (0.0 to 1.0).
- `imaginary`: Use the imaginary gamma (non-empty only when using the BHT method).


_Returns_
```python
Tuple[ndarray, ndarray]
```

#### **pyimpspec.DRTResult.get_residual_data**

Get the data necessary to plot the relative residuals for this DRTResult: the frequencies, the relative residuals for the real parts of the impedances in percents, and the relative residuals for the imaginary parts of the impedances in percents.

```python
def get_residual_data(self) -> Tuple[ndarray, ndarray, ndarray]:
```


_Returns_
```python
Tuple[ndarray, ndarray, ndarray]
```

#### **pyimpspec.DRTResult.get_score_dataframe**

Get the scores (BHT) method for the data set as a pandas.DataFrame object that can be used to generate, e.g., a Markdown table.

```python
def get_score_dataframe(self, latex_labels: bool = False) -> Optional[DataFrame]:
```


_Parameters_

- `latex_labels`: Whether or not to use LaTeX macros in the labels.


_Returns_
```python
Optional[DataFrame]
```

#### **pyimpspec.DRTResult.get_scores**

Get the scores (BHT method) for the data set.
The scores are represented as complex values where the real and imaginary parts have magnitudes ranging from 0.0 to 1.0.
A consistent impedance spectrum should score high.

```python
def get_scores(self) -> Dict[str, complex]:
```


_Returns_
```python
Dict[str, complex]
```

#### **pyimpspec.DRTResult.get_tau**

Get the time constants.

```python
def get_tau(self) -> ndarray:
```


_Returns_
```python
ndarray
```

#### **pyimpspec.DRTResult.to_dataframe**

Get the peaks as a pandas.DataFrame object that can be used to generate, e.g., a Markdown table.

```python
def to_dataframe(self, threshold: float = 0.0, imaginary: bool = False, latex_labels: bool = False, include_frequency: bool = False) -> DataFrame:
```


_Parameters_

- `threshold`: The threshold for the peaks (0.0 to 1.0 relative to the highest peak).
- `imaginary`: Use the imaginary gamma (non-empty only when using the BHT method).
- `latex_labels`: Whether or not to use LaTeX macros in the labels.
- `include_frequency`: Whether or not to also include a column with the frequencies corresponding to the time constants.


_Returns_
```python
DataFrame
```



