---
layout: default
title: API - circuit
permalink: /api/circuit/
---

[Check the page for high-level functions for relevant functions.](https://vyrjana.github.io/pyimpspec/api/high-level-functions)

**Table of Contents**

- [Circuit](#pyimpspeccircuit)
	- [get_element](#pyimpspeccircuitget_element)
	- [get_elements](#pyimpspeccircuitget_elements)
	- [get_label](#pyimpspeccircuitget_label)
	- [get_parameters](#pyimpspeccircuitget_parameters)
	- [impedance](#pyimpspeccircuitimpedance)
	- [impedances](#pyimpspeccircuitimpedances)
	- [set_label](#pyimpspeccircuitset_label)
	- [set_parameters](#pyimpspeccircuitset_parameters)
	- [to_circuitikz](#pyimpspeccircuitto_circuitikz)
	- [to_latex](#pyimpspeccircuitto_latex)
	- [to_stack](#pyimpspeccircuitto_stack)
	- [to_string](#pyimpspeccircuitto_string)
	- [to_sympy](#pyimpspeccircuitto_sympy)


### **pyimpspec.Circuit**

A class that represents an equivalent circuit.

```python
class Circuit(object):
	elements: Series
	label: str = ""
```

_Constructor parameters_

- `elements`: The elements of the circuit wrapped in a Series connection.
- `label`: The label assigned to the circuit.


_Functions and methods_

#### **pyimpspec.Circuit.get_element**

Get the circuit element with a given integer identifier.

```python
def get_element(self, ident: int) -> Optional[Element]:
```


_Parameters_

- `ident`: The integer identifier corresponding to an element in the circuit.


_Returns_
```python
Optional[Element]
```

#### **pyimpspec.Circuit.get_elements**

Get the elements in this circuit.

```python
def get_elements(self, flattened: bool = True) -> List[Union[Element, Connection]]:
```


_Parameters_

- `flattened`: Whether or not the elements should be returned as a list of only elements or as a list of elements and connections.


_Returns_
```python
List[Union[Element, Connection]]
```

#### **pyimpspec.Circuit.get_label**

Get the label assigned to this circuit.

```python
def get_label(self) -> str:
```


_Returns_
```python
str
```

#### **pyimpspec.Circuit.get_parameters**

Get a mapping of each circuit element's integer identifier to an OrderedDict representing that element's parameters.

```python
def get_parameters(self) -> Dict[int, OrderedDict[str, float]]:
```


_Returns_
```python
Dict[int, OrderedDict[str, float]]
```

#### **pyimpspec.Circuit.impedance**

Calculate the impedance of this circuit at a single frequency.

```python
def impedance(self, f: float) -> complex:
```


_Parameters_

- `f`: The frequency in hertz.


_Returns_
```python
complex
```

#### **pyimpspec.Circuit.impedances**

Calculate the impedance of this circuit at multiple frequencies.

```python
def impedances(self, f: Union[list, ndarray]) -> ndarray:
```


_Parameters_

- `f`


_Returns_
```python
ndarray
```

#### **pyimpspec.Circuit.set_label**

Set the label assigned to this circuit.

```python
def set_label(self, label: str):
```


_Parameters_

- `label`: The new label.

#### **pyimpspec.Circuit.set_parameters**

Assign new parameters to the circuit elements.

```python
def set_parameters(self, parameters: Dict[int, Dict[str, float]]):
```


_Parameters_

- `parameters`: A mapping of circuit element integer identifiers to an OrderedDict mapping the parameter symbol to the new value.

#### **pyimpspec.Circuit.to_circuitikz**

Get the LaTeX source needed to draw a circuit diagram for this circuit using the circuitikz package.

```python
def to_circuitikz(self, node_width: float = 3.0, node_height: float = 1.5, working_label: str = "WE+WS", counter_label: str = "CE+RE", hide_labels: bool = False) -> str:
```


_Parameters_

- `node_width`: The width of each node.
- `node_height`: The height of each node.
- `working_label`: The label assigned to the terminal representing the working and working sense electrodes.
- `counter_label`: The label assigned to the terminal representing the counter and reference electrodes.
- `hide_labels`: Whether or not to hide element and terminal labels.


_Returns_
```python
str
```

#### **pyimpspec.Circuit.to_latex**

Get the LaTeX math expression corresponding to this circuit's impedance.

```python
def to_latex(self) -> str:
```


_Returns_
```python
str
```

#### **pyimpspec.Circuit.to_stack**


```python
def to_stack(self) -> List[Tuple[str, Union[Element, Connection]]]:
```


_Returns_
```python
List[Tuple[str, Union[Element, Connection]]]
```

#### **pyimpspec.Circuit.to_string**

Generate the circuit description code (CDC) that represents this circuit.

```python
def to_string(self, decimals: int = -1) -> str:
```


_Parameters_

- `decimals`: The number of decimals to include for the current element parameter values and limits.
-1 means that the CDC is generated using the basic syntax, which omits element labels, parameter values, and parameter limits.


_Returns_
```python
str
```

#### **pyimpspec.Circuit.to_sympy**

Get the SymPy expression corresponding to this circuit's impedance.

```python
def to_sympy(self, substitute: bool = False) -> Expr:
```


_Parameters_

- `substitute`: Whether or not the variables should be substituted with the current values.


_Returns_
```python
Expr
```



