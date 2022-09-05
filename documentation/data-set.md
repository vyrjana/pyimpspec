---
layout: documentation
title: API - data set
permalink: /api/data-set/
---

Check the page for [high-level functions](https://vyrjana.github.io/pyimpspec/api/high-level-functions) for the recommended way of reading data to generate a `DataSet` object.

**Table of Contents**

- [DataSet](#pyimpspecdataset)
	- [average](#pyimpspecdatasetaverage)
	- [copy](#pyimpspecdatasetcopy)
	- [from_dict](#pyimpspecdatasetfrom_dict)
	- [get_bode_data](#pyimpspecdatasetget_bode_data)
	- [get_frequency](#pyimpspecdatasetget_frequency)
	- [get_imaginary](#pyimpspecdatasetget_imaginary)
	- [get_impedance](#pyimpspecdatasetget_impedance)
	- [get_label](#pyimpspecdatasetget_label)
	- [get_magnitude](#pyimpspecdatasetget_magnitude)
	- [get_mask](#pyimpspecdatasetget_mask)
	- [get_num_points](#pyimpspecdatasetget_num_points)
	- [get_nyquist_data](#pyimpspecdatasetget_nyquist_data)
	- [get_path](#pyimpspecdatasetget_path)
	- [get_phase](#pyimpspecdatasetget_phase)
	- [get_real](#pyimpspecdatasetget_real)
	- [set_label](#pyimpspecdatasetset_label)
	- [set_mask](#pyimpspecdatasetset_mask)
	- [set_path](#pyimpspecdatasetset_path)
	- [subtract_impedance](#pyimpspecdatasetsubtract_impedance)
	- [to_dataframe](#pyimpspecdatasetto_dataframe)
	- [to_dict](#pyimpspecdatasetto_dict)
- [UnsupportedFileFormat](#pyimpspecunsupportedfileformat)


### **pyimpspec.DataSet**

A class that represents an impedance spectrum.
The data points can be masked, which results in those data points being omitted from any analyses and visualization.

```python
class DataSet(object):
	frequency: ndarray
	impedance: ndarray
	mask: Dict[int, bool] = {}
	path: str = ""
	label: str = ""
	uuid: str = ""
```

_Constructor parameters_

- `frequency`: A 1-dimensional array of frequencies in hertz.
- `impedance`: A 1-dimensional array of complex impedances in ohms.
- `mask`: A mapping of integer indices to boolean values where a value of True means that the data point is to be omitted.
- `path`: The path to the file that has been parsed to generate this DataSet instance.
- `label`: The label assigned to this DataSet instance.
- `uuid`: The universally unique identifier assigned to this DataSet instance.
If empty, then one will be automatically assigned.


_Functions and methods_

#### **pyimpspec.DataSet.average**

Create a DataSet by averaging the impedances of multiple DataSet instances.

```python
def average(data_sets: List[DataSet], label: str = "Average") -> DataSet:
```


_Parameters_

- `data_sets`: The DataSet instances to average.
- `label`: The label that the new DataSet should have.


_Returns_
```python
DataSet
```

#### **pyimpspec.DataSet.copy**

Create a copy of an existing DataSet.

```python
def copy(data: DataSet, label: Optional[str] = None) -> DataSet:
```


_Parameters_

- `data`: The existing DataSet to make a copy of.
- `label`: The label that the copy should have.


_Returns_
```python
DataSet
```

#### **pyimpspec.DataSet.from_dict**

Create a DataSet from a dictionary.

```python
def from_dict(dictionary: dict) -> DataSet:
```


_Parameters_

- `dictionary`: A dictionary containing at least the frequencies, and the real and the imaginary parts of the impedances.


_Returns_
```python
DataSet
```

#### **pyimpspec.DataSet.get_bode_data**

Get the data necessary to plot this DataSet as a Bode plot: the frequencies, the absolute magnitudes of the impedances, and the negative phase angles/shifts of the impedances in degrees.

```python
def get_bode_data(self, masked: Optional[bool] = False) -> Tuple[ndarray, ndarray, ndarray]:
```


_Parameters_

- `masked`: None means that all impedances are returned.
True means that only impedances that are to be omitted are returned.
False means that only impedances that are to be included are returned.


_Returns_
```python
Tuple[ndarray, ndarray, ndarray]
```

#### **pyimpspec.DataSet.get_frequency**

Get the frequencies in this DataSet.

```python
def get_frequency(self, masked: Optional[bool] = False) -> ndarray:
```


_Parameters_

- `masked`: None means that all frequencies are returned.
True means that only frequencies that are to be omitted are returned.
False means that only frequencies that are to be included are returned.


_Returns_
```python
ndarray
```

#### **pyimpspec.DataSet.get_imaginary**

Get the imaginary parts of the impedances in this DataSet.

```python
def get_imaginary(self, masked: Optional[bool] = False) -> ndarray:
```


_Parameters_

- `masked`: None means that all impedances are returned.
True means that only impedances that are to be omitted are returned.
False means that only impedances that are to be included are returned.


_Returns_
```python
ndarray
```

#### **pyimpspec.DataSet.get_impedance**

Get the complex impedances in this DataSet.

```python
def get_impedance(self, masked: Optional[bool] = False) -> ndarray:
```


_Parameters_

- `masked`: None means that all impedances are returned.
True means that only impedances that are to be omitted are returned.
False means that only impedances that are to be included are returned.


_Returns_
```python
ndarray
```

#### **pyimpspec.DataSet.get_label**

Get the label assigned to this DataSet.

```python
def get_label(self) -> str:
```


_Returns_
```python
str
```

#### **pyimpspec.DataSet.get_magnitude**

Get the absolute magnitudes of the impedances in this DataSet.

```python
def get_magnitude(self, masked: Optional[bool] = False) -> ndarray:
```


_Parameters_

- `masked`: None means that all impedances are returned.
True means that only impedances that are to be omitted are returned.
False means that only impedances that are to be included are returned.


_Returns_
```python
ndarray
```

#### **pyimpspec.DataSet.get_mask**

Get the mask for this DataSet.
The keys are zero-based indices and the values are booleans.
True means that the data point is to be omitted and False means that the data point is to be included.

```python
def get_mask(self) -> Dict[int, bool]:
```


_Returns_
```python
Dict[int, bool]
```

#### **pyimpspec.DataSet.get_num_points**

Get the number of data points in this DataSet

```python
def get_num_points(self, masked: Optional[bool] = False) -> int:
```


_Parameters_

- `masked`: None means that all impedances are returned.
True means that only impedances that are to be omitted are returned.
False means that only impedances that are to be included are returned.


_Returns_
```python
int
```

#### **pyimpspec.DataSet.get_nyquist_data**

Get the data necessary to plot this DataSet as a Nyquist plot: the real and the negative imaginary parts of the impedances.

```python
def get_nyquist_data(self, masked: Optional[bool] = False) -> Tuple[ndarray, ndarray]:
```


_Parameters_

- `masked`: None means that all impedances are returned.
True means that only impedances that are to be omitted are returned.
False means that only impedances that are to be included are returned.


_Returns_
```python
Tuple[ndarray, ndarray]
```

#### **pyimpspec.DataSet.get_path**

Get the path to the file that was parsed to generate this DataSet.

```python
def get_path(self) -> str:
```


_Returns_
```python
str
```

#### **pyimpspec.DataSet.get_phase**

Get the phase angles/shifts of the impedances in this DataSet in degrees.

```python
def get_phase(self, masked: Optional[bool] = False) -> ndarray:
```


_Parameters_

- `masked`: None means that all impedances are returned.
True means that only impedances that are to be omitted are returned.
False means that only impedances that are to be included are returned.


_Returns_
```python
ndarray
```

#### **pyimpspec.DataSet.get_real**

Get the real parts of the impedances in this DataSet.

```python
def get_real(self, masked: Optional[bool] = False) -> ndarray:
```


_Parameters_

- `masked`: None means that all impedances are returned.
True means that only impedances that are to be omitted are returned.
False means that only impedances that are to be included are returned.


_Returns_
```python
ndarray
```

#### **pyimpspec.DataSet.set_label**

Set the label assigned to this DataSet.

```python
def set_label(self, label: str):
```


_Parameters_

- `label`: The new label.

#### **pyimpspec.DataSet.set_mask**

Set the mask for this DataSet.

```python
def set_mask(self, mask: Dict[int, bool]):
```


_Parameters_

- `mask`: The new mask.
The keys must be zero-based indices and the values must be boolean values.
True means that the data point is to be omitted and False means that the data point is to be included.

#### **pyimpspec.DataSet.set_path**

Set the path to the file that was parsed to generate this DataSet.

```python
def set_path(self, path: str):
```


_Parameters_

- `path`: The path.

#### **pyimpspec.DataSet.subtract_impedance**

Subtract either the same complex value from all data points or a unique complex value for each data point in this DataSet.

```python
def subtract_impedance(self, Z: Union[complex, List[complex], ndarray]):
```


_Parameters_

- `Z`: The complex value(s) to subtract from this DataSet's impedances.

#### **pyimpspec.DataSet.to_dataframe**

Create a pandas.DataFrame instance from this DataSet.

```python
def to_dataframe(self, masked: Optional[bool] = False, frequency_label: str = "f (Hz)", real_label: Optional[str] = "Zre (ohm)", imaginary_label: Optional[str] = "Zim (ohm)", magnitude_label: Optional[str] = "|Z| (ohm)", phase_label: Optional[str] = "phase angle (deg.)", negative_imaginary: bool = False, negative_phase: bool = False) -> DataFrame:
```


_Parameters_

- `masked`: None means that all impedances are returned.
True means that only impedances that are to be omitted are returned.
False means that only impedances that are to be included are returned.
- `frequency_label`: The label assigned to the frequency data.
- `real_label`: The label assigned to the real part of the impedance data.
- `imaginary_label`: The label assigned to the imaginary part of the impedance data.
- `magnitude_label`: The label assigned to the magnitude of the impedance data.
- `phase_label`: The label assigned to the phase of the imedance data.
- `negative_imaginary`: Whether or not the sign of the imaginary part of the impedance data should be inverted.
- `negative_phase`: Whether or not the sign of the phase of the impedance data should be inverted.


_Returns_
```python
DataFrame
```

#### **pyimpspec.DataSet.to_dict**

Get a dictionary that represents this DataSet, can be used to serialize the DataSet (e.g. as a JSON file), and then used to recreate this DataSet.

```python
def to_dict(self) -> dict:
```


_Returns_
```python
dict
```




### **pyimpspec.UnsupportedFileFormat**

```python
class UnsupportedFileFormat(Exception):
	args
	kwargs
```

_Constructor parameters_

- `args`
- `kwargs`



