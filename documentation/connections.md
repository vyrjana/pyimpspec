---
layout: documentation
title: API - connections
permalink: /api/connections/
---

These are used inside of [`Circuit`](https://vyrjana.github.io/pyimpspec/api/circuit) objects.

**Table of Contents**

- [Connection](#pyimpspecconnection)
	- [contains](#pyimpspecconnectioncontains)
	- [get_element](#pyimpspecconnectionget_element)
	- [get_elements](#pyimpspecconnectionget_elements)
	- [get_label](#pyimpspecconnectionget_label)
	- [get_parameters](#pyimpspecconnectionget_parameters)
	- [impedance](#pyimpspecconnectionimpedance)
	- [impedances](#pyimpspecconnectionimpedances)
	- [set_parameters](#pyimpspecconnectionset_parameters)
	- [to_latex](#pyimpspecconnectionto_latex)
	- [to_stack](#pyimpspecconnectionto_stack)
	- [to_string](#pyimpspecconnectionto_string)
	- [to_sympy](#pyimpspecconnectionto_sympy)
- [Parallel](#pyimpspecparallel)
- [Series](#pyimpspecseries)


### **pyimpspec.Connection**

```python
class Connection(object):
	elements: List[Union[Element, Connection]]
```

_Constructor parameters_

- `elements`


_Functions and methods_

#### **pyimpspec.Connection.contains**

Check if this connection contains a specific Element or Connection instance.

```python
def contains(self, element: Union[Element, Connection], top_level: bool = False) -> bool:
```


_Parameters_

- `element`: The Element or Connection instance to check for.
- `top_level`: Whether to only check in the current Connection instance instead of also checking in any nested Connection instances.


_Returns_
```python
bool
```

#### **pyimpspec.Connection.get_element**

Get a specific element based on its unique identifier.

```python
def get_element(self, ident: int) -> Optional[Element]:
```


_Parameters_

- `ident`: The integer identifier that should be unique in the context of the circuit.


_Returns_
```python
Optional[Element]
```

#### **pyimpspec.Connection.get_elements**

Get a list of elements and connections nested inside this connection.

```python
def get_elements(self, flattened: bool = True) -> List[Union[Element, Connection]]:
```


_Parameters_

- `flattened`: Whether the returned list should only contain elements or a combination of elements and connections.


_Returns_
```python
List[Union[Element, Connection]]
```

#### **pyimpspec.Connection.get_label**


```python
def get_label(self) -> str:
```


_Returns_
```python
str
```

#### **pyimpspec.Connection.get_parameters**

Get the current element parameters of all elements nested inside this connection.
The outer key is the unique identifier assigned to an element.
The inner key is the symbol corresponding to an element parameter.

```python
def get_parameters(self) -> Dict[int, OrderedDict[str, float]]:
```


_Returns_
```python
Dict[int, OrderedDict[str, float]]
```

#### **pyimpspec.Connection.impedance**

Calculates the impedance of the connection at a single frequency.

```python
def impedance(self, f: float) -> complex:
```


_Parameters_

- `f`: Frequency in Hz


_Returns_
```python
complex
```

#### **pyimpspec.Connection.impedances**

Calculates the impedance of the element at multiple frequencies.

```python
def impedances(self, freq: Union[list, ndarray]) -> ndarray:
```


_Parameters_

- `freq`: Frequencies in Hz


_Returns_
```python
ndarray
```

#### **pyimpspec.Connection.set_parameters**

Set new element parameters to some/all elements nested inside this connection.

```python
def set_parameters(self, parameters: Dict[int, Dict[str, float]]):
```


_Parameters_

- `parameters`: The outer key is the unique identifier assigned to an element.
The inner key is the symbol corresponding to an element parameter.

#### **pyimpspec.Connection.to_latex**


```python
def to_latex(self) -> str:
```


_Returns_
```python
str
```

#### **pyimpspec.Connection.to_stack**


```python
def to_stack(self, stack: List[Tuple[str, Union[Element, Connection]]]):
```


_Parameters_

- `stack`

#### **pyimpspec.Connection.to_string**


```python
def to_string(self, decimals: int = -1) -> str:
```


_Parameters_

- `decimals`


_Returns_
```python
str
```

#### **pyimpspec.Connection.to_sympy**


```python
def to_sympy(self, substitute: bool = False) -> Expr:
```


_Parameters_

- `substitute`


_Returns_
```python
Expr
```




### **pyimpspec.Parallel**

Elements connected in parallel.

```python
class Parallel(Connection):
	elements: List[Union[Element, Connection]]
```

_Constructor parameters_

- `elements`: List of elements (and connections) that are connected in parallel.




### **pyimpspec.Series**

Elements connected in series.

```python
class Series(Connection):
	elements: List[Union[Element, Connection]]
```

_Constructor parameters_

- `elements`: List of elements (and connections) that are connected in series.



