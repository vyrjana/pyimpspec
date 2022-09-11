---
layout: documentation
title: API - circuit
permalink: /api/circuit/
---

Circuits can be generated in one of two ways:
- by parsing a circuit description code (CDC)
- by using the `CircuitBuilder` class

The basic syntax for CDCs is fairly straighforward:

```python
# A resistor connected in series with a resistor and a capacitor connected in parallel
circuit: pyimpspec.Circuit = pyimpspec.parse_cdc("[R(RC)]")
```

An extended syntax, which allows for defining initial values, lower/upper limits, and labels, is also supported:

```python
circuit: pyimpspec.Circuit = pyimpspec.parse_cdc("[R{R=50:sol}(R{R=250f:ct}C{C=1.5e-6/1e-6/2e-6:dl})]")
```

Alternatively, the `CircuitBuilder` class can be used:

```python
with pyimpspec.CircuitBuilder() as builder:
    builder += (
        pyimpspec.Resistor(R=50)
        .set_label("sol")
    )
    with builder.parallel() as parallel:
        parallel += (
            pyimpspec.Resistor(R=250)
            .set_fixed("R", True)
        )
        parallel += (
            pyimpspec.Capacitor(C=1.5e-6)
            .set_label("dl")
            .set_lower_limit("C", 1e-6)
            .set_upper_limit("C", 2e-6)
        )
circuit: pyimpspec.Circuit = builder.to_circuit()
```

Information about the supported circuit elements can be found [here](https://vyrjana.github.io/pyimpspec/api/elements).

**Table of Contents**

- [Circuit](#pyimpspeccircuit)
	- [get_connections](#pyimpspeccircuitget_connections)
	- [get_element](#pyimpspeccircuitget_element)
	- [get_elements](#pyimpspeccircuitget_elements)
	- [get_label](#pyimpspeccircuitget_label)
	- [get_parameters](#pyimpspeccircuitget_parameters)
	- [impedance](#pyimpspeccircuitimpedance)
	- [impedances](#pyimpspeccircuitimpedances)
	- [set_label](#pyimpspeccircuitset_label)
	- [set_parameters](#pyimpspeccircuitset_parameters)
	- [substitute_element](#pyimpspeccircuitsubstitute_element)
	- [to_circuitikz](#pyimpspeccircuitto_circuitikz)
	- [to_drawing](#pyimpspeccircuitto_drawing)
	- [to_latex](#pyimpspeccircuitto_latex)
	- [to_stack](#pyimpspeccircuitto_stack)
	- [to_string](#pyimpspeccircuitto_string)
	- [to_sympy](#pyimpspeccircuitto_sympy)
- [CircuitBuilder](#pyimpspeccircuitbuilder)
	- [add](#pyimpspeccircuitbuilderadd)
	- [parallel](#pyimpspeccircuitbuilderparallel)
	- [series](#pyimpspeccircuitbuilderseries)
	- [to_circuit](#pyimpspeccircuitbuilderto_circuit)
	- [to_string](#pyimpspeccircuitbuilderto_string)
- [ParsingError](#pyimpspecparsingerror)
- [UnexpectedCharacter](#pyimpspecunexpectedcharacter)


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

#### **pyimpspec.Circuit.get_connections**

Get the connections in this circuit.

```python
def get_connections(self, flattened: bool = True) -> List[Connection]:
```


_Parameters_

- `flattened`: Whether or not the connections should be returned as a list of all connections or as a list connections that may also contain more connections.


_Returns_
```python
List[Connection]
```

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

- `flattened`: Whether or not the elements should be returned as a list of only elements or as a list of connections containing elements.


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

#### **pyimpspec.Circuit.substitute_element**

Substitute the element with the given integer identifier in the circuit with another element.

```python
def substitute_element(self, ident: int, element: Element):
```


_Parameters_

- `ident`: The integer identifier corresponding to an element in the circuit.
- `element`: The new element that will substitute the old element.

#### **pyimpspec.Circuit.to_circuitikz**

Get the LaTeX source needed to draw a circuit diagram for this circuit using the circuitikz package.

```python
def to_circuitikz(self, node_width: float = 3.0, node_height: float = 1.5, working_label: str = "WE", counter_label: str = "CE+RE", hide_labels: bool = False) -> str:
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

#### **pyimpspec.Circuit.to_drawing**

Get a schemdraw.Drawing object to draw a circuit diagram using the matplotlib backend.

```python
def to_drawing(self, node_height: float = 1.5, working_label: str = "WE", counter_label: str = "CE+RE", hide_labels: bool = False) -> Drawing:
```


_Parameters_

- `node_height`: The height of each node.
- `working_label`: The label assigned to the terminal representing the working and working sense electrodes.
- `counter_label`: The label assigned to the terminal representing the counter and reference electrodes.
- `hide_labels`: Whether or not to hide element and terminal labels.


_Returns_
```python
Drawing
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




### **pyimpspec.CircuitBuilder**

A class for building circuits using context managers

```python
class CircuitBuilder(object):
	parallel: bool = False
```

_Constructor parameters_

- `parallel`: Whether or not this context/connection is a parallel connection.


_Functions and methods_

#### **pyimpspec.CircuitBuilder.add**

Add an element to the current context (i.e., connection).

```python
def add(self, element: Element):
```


_Parameters_

- `element`: The element to add to the current series or parallel connection.

#### **pyimpspec.CircuitBuilder.parallel**

Create a parallel connection.

```python
def parallel(self) -> CircuitBuilder:
```


_Returns_
```python
CircuitBuilder
```

#### **pyimpspec.CircuitBuilder.series**

Create a series connection.

```python
def series(self) -> CircuitBuilder:
```


_Returns_
```python
CircuitBuilder
```

#### **pyimpspec.CircuitBuilder.to_circuit**

Generate a circuit.

```python
def to_circuit(self) -> Circuit:
```


_Returns_
```python
Circuit
```

#### **pyimpspec.CircuitBuilder.to_string**

Generate a circuit description code.

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




### **pyimpspec.ParsingError**

```python
class ParsingError(Exception):
	msg: str
```

_Constructor parameters_

- `msg`




### **pyimpspec.UnexpectedCharacter**

```python
class UnexpectedCharacter(Exception):
	args
	kwargs
```

_Constructor parameters_

- `args`
- `kwargs`



