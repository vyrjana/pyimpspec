---
layout: documentation
title: API - base element
permalink: /api/base-element/
---

This is the page for the base class for all [elements](https://vyrjana.github.io/pyimpspec/api/elements).

**Table of Contents**

- [Element](#pyimpspecelement)
	- [get_default_fixed](#pyimpspecelementget_default_fixed)
	- [get_default_label](#pyimpspecelementget_default_label)
	- [get_default_lower_limits](#pyimpspecelementget_default_lower_limits)
	- [get_default_upper_limits](#pyimpspecelementget_default_upper_limits)
	- [get_defaults](#pyimpspecelementget_defaults)
	- [get_description](#pyimpspecelementget_description)
	- [get_extended_description](#pyimpspecelementget_extended_description)
	- [get_identifier](#pyimpspecelementget_identifier)
	- [get_label](#pyimpspecelementget_label)
	- [get_lower_limit](#pyimpspecelementget_lower_limit)
	- [get_parameters](#pyimpspecelementget_parameters)
	- [get_symbol](#pyimpspecelementget_symbol)
	- [get_upper_limit](#pyimpspecelementget_upper_limit)
	- [impedance](#pyimpspecelementimpedance)
	- [impedances](#pyimpspecelementimpedances)
	- [is_fixed](#pyimpspecelementis_fixed)
	- [reset_parameters](#pyimpspecelementreset_parameters)
	- [set_fixed](#pyimpspecelementset_fixed)
	- [set_label](#pyimpspecelementset_label)
	- [set_lower_limit](#pyimpspecelementset_lower_limit)
	- [set_parameters](#pyimpspecelementset_parameters)
	- [set_upper_limit](#pyimpspecelementset_upper_limit)
	- [to_latex](#pyimpspecelementto_latex)
	- [to_string](#pyimpspecelementto_string)
	- [to_sympy](#pyimpspecelementto_sympy)


### **pyimpspec.Element**

```python
class Element(object):
	keys: List[str]
```

_Constructor parameters_

- `keys`


_Functions and methods_

#### **pyimpspec.Element.get_default_fixed**

Get whether or not the element's parameters are fixed by default.

```python
def get_default_fixed() -> Dict[str, bool]:
```


_Returns_
```python
Dict[str, bool]
```

#### **pyimpspec.Element.get_default_label**

Get the default label for this element.

```python
def get_default_label(self) -> str:
```


_Returns_
```python
str
```

#### **pyimpspec.Element.get_default_lower_limits**

Get the default lower limits for the element's parameters.

```python
def get_default_lower_limits() -> Dict[str, float]:
```


_Returns_
```python
Dict[str, float]
```

#### **pyimpspec.Element.get_default_upper_limits**

Get the default upper limits for the element's parameters.

```python
def get_default_upper_limits() -> Dict[str, float]:
```


_Returns_
```python
Dict[str, float]
```

#### **pyimpspec.Element.get_defaults**

Get the default values for the element's parameters.

```python
def get_defaults() -> Dict[str, float]:
```


_Returns_
```python
Dict[str, float]
```

#### **pyimpspec.Element.get_description**

Get a brief description of the element and its symbol.

```python
def get_description() -> str:
```


_Returns_
```python
str
```

#### **pyimpspec.Element.get_extended_description**


```python
def get_extended_description() -> str:
```


_Returns_
```python
str
```

#### **pyimpspec.Element.get_identifier**

Get the internal identifier that is unique in the context of a circuit.
Used internally when generating unique names for parameters when fitting a circuit to a
data set.

```python
def get_identifier(self) -> int:
```


_Returns_
```python
int
```

#### **pyimpspec.Element.get_label**

Get the label assigned to a specific instance of the element.

```python
def get_label(self) -> str:
```


_Returns_
```python
str
```

#### **pyimpspec.Element.get_lower_limit**

Get the lower limit for the value of an element parameter when fitting a circuit to a data
set.
The absence of a limit is represented by -numpy.inf.

```python
def get_lower_limit(self, key: str) -> float:
```


_Parameters_

- `key`: A key corresponding to an element parameter.


_Returns_
```python
float
```

#### **pyimpspec.Element.get_parameters**

Get the current parameters of the element.

```python
def get_parameters(self) -> OrderedDict[str, float]:
```


_Returns_
```python
OrderedDict[str, float]
```

#### **pyimpspec.Element.get_symbol**

Get the symbol representing the element.

```python
def get_symbol() -> str:
```


_Returns_
```python
str
```

#### **pyimpspec.Element.get_upper_limit**

Get the upper limit for the value of an element parameter when fitting a circuit to a data
set.
The absence of a limit is represented by numpy.inf.

```python
def get_upper_limit(self, key: str) -> float:
```


_Parameters_

- `key`: A key corresponding to an element parameter.


_Returns_
```python
float
```

#### **pyimpspec.Element.impedance**

Calculates the complex impedance of the element at a specific frequency.

```python
def impedance(self, f: float) -> complex:
```


_Parameters_

- `f`: Frequency in hertz.


_Returns_
```python
complex
```

#### **pyimpspec.Element.impedances**

Calculates the complex impedance of the element at specific frequencies.

```python
def impedances(self, freq: Union[list, ndarray]) -> ndarray:
```


_Parameters_

- `freq`: Frequencies in hertz.


_Returns_
```python
ndarray
```

#### **pyimpspec.Element.is_fixed**

Check if an element parameter should have a fixed value when fitting a circuit to a data
set.
True if fixed and False if not fixed.

```python
def is_fixed(self, key: str) -> bool:
```


_Parameters_

- `key`: A key corresponding to an element parameter.


_Returns_
```python
bool
```

#### **pyimpspec.Element.reset_parameters**

Resets the value, lower limit, upper limit, and fixed state of one or more parameters.

```python
def reset_parameters(self, keys: List[str]):
```


_Parameters_

- `keys`: Names of the parameters to reset.

#### **pyimpspec.Element.set_fixed**

Set whether or not an element parameter should have a fixed value when fitting a circuit
to a data set.

```python
def set_fixed(self, key: str, value: bool):
```


_Parameters_

- `key`: A key corresponding to an element parameter.
- `value`: True if the value should be fixed.

#### **pyimpspec.Element.set_label**

Set the label assigned to a specific instance of the element.

```python
def set_label(self, label: str):
```


_Parameters_

- `label`: The new label.

#### **pyimpspec.Element.set_lower_limit**

Set the upper limit for the value of an element parameter when fitting a circuit to a data
set.

```python
def set_lower_limit(self, key: str, value: float):
```


_Parameters_

- `key`: A key corresponding to an element parameter.
- `value`: The new limit for the element parameter. The limit can be removed by setting the limit
to be -numpy.inf.

#### **pyimpspec.Element.set_parameters**

Set new values for the parameters of the element.

```python
def set_parameters(self, parameters: Dict[str, float]):
```


_Parameters_

- `parameters`

#### **pyimpspec.Element.set_upper_limit**

Set the upper limit for the value of an element parameter when fitting a circuit to a data
set.

```python
def set_upper_limit(self, key: str, value: float):
```


_Parameters_

- `key`: A key corresponding to an element parameter.
- `value`: The new limit for the element parameter. The limit can be removed by setting the limit
to be numpy.inf.

#### **pyimpspec.Element.to_latex**


```python
def to_latex(self) -> str:
```


_Returns_
```python
str
```

#### **pyimpspec.Element.to_string**

Generates a string representation of the element.

```python
def to_string(self, decimals: int = -1) -> str:
```


_Parameters_

- `decimals`: The number of decimals used when formatting the current value and the limits for the element's parameters.
-1 corresponds to no values being included in the output.


_Returns_
```python
str
```

#### **pyimpspec.Element.to_sympy**


```python
def to_sympy(self, substitute: bool = False) -> Expr:
```


_Parameters_

- `substitute`


_Returns_
```python
Expr
```



