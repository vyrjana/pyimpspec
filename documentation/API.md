# pyimpspec - API reference (3.2.0)



**Table of Contents**

- [pyimpspec](#pyimpspec)
	- [Capacitor](#pyimpspeccapacitor)
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
	- [Connection](#pyimpspecconnection)
		- [contains](#pyimpspecconnectioncontains)
		- [get_connections](#pyimpspecconnectionget_connections)
		- [get_element](#pyimpspecconnectionget_element)
		- [get_elements](#pyimpspecconnectionget_elements)
		- [get_label](#pyimpspecconnectionget_label)
		- [get_parameters](#pyimpspecconnectionget_parameters)
		- [impedance](#pyimpspecconnectionimpedance)
		- [impedances](#pyimpspecconnectionimpedances)
		- [set_parameters](#pyimpspecconnectionset_parameters)
		- [substitute_element](#pyimpspecconnectionsubstitute_element)
		- [to_latex](#pyimpspecconnectionto_latex)
		- [to_stack](#pyimpspecconnectionto_stack)
		- [to_string](#pyimpspecconnectionto_string)
		- [to_sympy](#pyimpspecconnectionto_sympy)
	- [ConstantPhaseElement](#pyimpspecconstantphaseelement)
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
	- [DeLevieFiniteLength](#pyimpspecdeleviefinitelength)
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
	- [FitResult](#pyimpspecfitresult)
		- [get_bode_data](#pyimpspecfitresultget_bode_data)
		- [get_frequency](#pyimpspecfitresultget_frequency)
		- [get_impedance](#pyimpspecfitresultget_impedance)
		- [get_nyquist_data](#pyimpspecfitresultget_nyquist_data)
		- [get_residual_data](#pyimpspecfitresultget_residual_data)
		- [to_dataframe](#pyimpspecfitresultto_dataframe)
	- [FittedParameter](#pyimpspecfittedparameter)
		- [from_dict](#pyimpspecfittedparameterfrom_dict)
		- [get_relative_error](#pyimpspecfittedparameterget_relative_error)
		- [to_dict](#pyimpspecfittedparameterto_dict)
	- [FittingError](#pyimpspecfittingerror)
	- [Gerischer](#pyimpspecgerischer)
	- [HavriliakNegami](#pyimpspechavriliaknegami)
	- [HavriliakNegamiAlternative](#pyimpspechavriliaknegamialternative)
	- [Inductor](#pyimpspecinductor)
	- [ModifiedInductor](#pyimpspecmodifiedinductor)
	- [Parallel](#pyimpspecparallel)
	- [ParsingError](#pyimpspecparsingerror)
	- [Resistor](#pyimpspecresistor)
	- [Series](#pyimpspecseries)
	- [TestResult](#pyimpspectestresult)
		- [calculate_score](#pyimpspectestresultcalculate_score)
		- [get_bode_data](#pyimpspectestresultget_bode_data)
		- [get_frequency](#pyimpspectestresultget_frequency)
		- [get_impedance](#pyimpspectestresultget_impedance)
		- [get_nyquist_data](#pyimpspectestresultget_nyquist_data)
		- [get_residual_data](#pyimpspectestresultget_residual_data)
	- [UnexpectedCharacter](#pyimpspecunexpectedcharacter)
	- [UnsupportedFileFormat](#pyimpspecunsupportedfileformat)
	- [Warburg](#pyimpspecwarburg)
	- [WarburgOpen](#pyimpspecwarburgopen)
	- [WarburgShort](#pyimpspecwarburgshort)
	- [calculate_drt](#pyimpspeccalculate_drt)
	- [fit_circuit](#pyimpspecfit_circuit)
	- [get_elements](#pyimpspecget_elements)
	- [parse_cdc](#pyimpspecparse_cdc)
	- [parse_data](#pyimpspecparse_data)
	- [perform_exploratory_tests](#pyimpspecperform_exploratory_tests)
	- [perform_test](#pyimpspecperform_test)
	- [simulate_spectrum](#pyimpspecsimulate_spectrum)
- [pyimpspec.plot.mpl](#pyimpspecplotmpl)
	- [plot_bode](#pyimpspecplotmplplot_bode)
	- [plot_circuit](#pyimpspecplotmplplot_circuit)
	- [plot_complex_impedance](#pyimpspecplotmplplot_complex_impedance)
	- [plot_data](#pyimpspecplotmplplot_data)
	- [plot_drt](#pyimpspecplotmplplot_drt)
	- [plot_exploratory_tests](#pyimpspecplotmplplot_exploratory_tests)
	- [plot_fit](#pyimpspecplotmplplot_fit)
	- [plot_gamma](#pyimpspecplotmplplot_gamma)
	- [plot_imaginary_impedance](#pyimpspecplotmplplot_imaginary_impedance)
	- [plot_impedance_magnitude](#pyimpspecplotmplplot_impedance_magnitude)
	- [plot_impedance_phase](#pyimpspecplotmplplot_impedance_phase)
	- [plot_mu_xps](#pyimpspecplotmplplot_mu_xps)
	- [plot_nyquist](#pyimpspecplotmplplot_nyquist)
	- [plot_real_impedance](#pyimpspecplotmplplot_real_impedance)
	- [plot_residual](#pyimpspecplotmplplot_residual)



## **pyimpspec**

### **pyimpspec.Capacitor**

Capacitor

    Symbol: C

    Z = 1/(j*2*pi*f*C)

    Variables
    ---------
    C: float = 1E-6 (F)

```python
class Capacitor(Element):
	kwargs
```

_Constructor parameters_

- `kwargs`




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

#### **pyimpspec.Connection.get_connections**

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

#### **pyimpspec.Connection.substitute_element**


```python
def substitute_element(self, ident: int, element: Element) -> bool:
```


_Parameters_

- `ident`
- `element`


_Returns_
```python
bool
```

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




### **pyimpspec.ConstantPhaseElement**

Constant phase element

    Symbol: Q

    Z = 1/(Y*(j*2*pi*f)^n)

    Variables
    ---------
    Y: float = 1E-6 (F*s^(n-1))
    n: float = 0.95

```python
class ConstantPhaseElement(Element):
	kwargs
```

_Constructor parameters_

- `kwargs`




### **pyimpspec.DRTError**

```python
class DRTError(Exception):
```



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




### **pyimpspec.DeLevieFiniteLength**

de Levie pore (finite length)

    Symbol: Ls

    Z = (Ri*Rr)^(1/2)*coth(d*(Ri/Rr)^(1/2)*(1+Y*(2*pi*f*j)^n)^(1/2))/(1+Y*(2*pi*f*j)^n)^(1/2)

    Variables
    ---------
    Ri: float = 10.0 (ohm/cm)
    Rr: float = 1.0 (ohm*cm)
    Y: float = 0.01 (F*s^(n-1)/cm)
    n: float = 0.8
    d: float = 0.2 (cm)

```python
class DeLevieFiniteLength(Element):
	kwargs
```

_Constructor parameters_

- `kwargs`




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
def set_fixed(self, key: str, value: bool) -> Element:
```


_Parameters_

- `key`: A key corresponding to an element parameter.
- `value`: True if the value should be fixed.


_Returns_
```python
Element
```

#### **pyimpspec.Element.set_label**

Set the label assigned to a specific instance of the element.

```python
def set_label(self, label: str) -> Element:
```


_Parameters_

- `label`: The new label.


_Returns_
```python
Element
```

#### **pyimpspec.Element.set_lower_limit**

Set the upper limit for the value of an element parameter when fitting a circuit to a data
set.

```python
def set_lower_limit(self, key: str, value: float) -> Element:
```


_Parameters_

- `key`: A key corresponding to an element parameter.
- `value`: The new limit for the element parameter. The limit can be removed by setting the limit
to be -numpy.inf.


_Returns_
```python
Element
```

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
def set_upper_limit(self, key: str, value: float) -> Element:
```


_Parameters_

- `key`: A key corresponding to an element parameter.
- `value`: The new limit for the element parameter. The limit can be removed by setting the limit
to be numpy.inf.


_Returns_
```python
Element
```

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




### **pyimpspec.FitResult**

An object representing the results of fitting a circuit to a data set.

```python
class FitResult(object):
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

#### **pyimpspec.FitResult.get_bode_data**

Get the data necessary to plot this FitResult as a Bode plot: the frequencies, the absolute magnitudes of the impedances, and the negative phase angles/shifts of the impedances in degrees.

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

#### **pyimpspec.FitResult.get_frequency**


```python
def get_frequency(self, num_per_decade: int = -1) -> ndarray:
```


_Parameters_

- `num_per_decade`


_Returns_
```python
ndarray
```

#### **pyimpspec.FitResult.get_impedance**


```python
def get_impedance(self, num_per_decade: int = -1) -> ndarray:
```


_Parameters_

- `num_per_decade`


_Returns_
```python
ndarray
```

#### **pyimpspec.FitResult.get_nyquist_data**

Get the data necessary to plot this FitResult as a Nyquist plot: the real and the negative imaginary parts of the impedances.

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

#### **pyimpspec.FitResult.get_residual_data**

Get the data necessary to plot the relative residuals for this FitResult: the frequencies, the relative residuals for the real parts of the impedances in percents, and the relative residuals for the imaginary parts of the impedances in percents.

```python
def get_residual_data(self) -> Tuple[ndarray, ndarray, ndarray]:
```


_Returns_
```python
Tuple[ndarray, ndarray, ndarray]
```

#### **pyimpspec.FitResult.to_dataframe**


```python
def to_dataframe(self) -> DataFrame:
```


_Returns_
```python
DataFrame
```




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

#### **pyimpspec.FittedParameter.get_relative_error**


```python
def get_relative_error(self) -> float:
```


_Returns_
```python
float
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
```



### **pyimpspec.Gerischer**

Gerischer

    Symbol: G

    Z = 1/(Y*(k+j*2*pi*f)^n)

    Variables
    ---------
    Y: float = 1.0 (S*s^n)
    k: float = 1.0 (s^-1)
    n: float = 0.5

```python
class Gerischer(Element):
	kwargs
```

_Constructor parameters_

- `kwargs`




### **pyimpspec.HavriliakNegami**

Havriliak-Negami relaxation

    Symbol: H

    Z = (((1+(j*2*pi*f*t)^a)^b)/(j*2*pi*f*dC))

    Variables
    ---------
    dC: float = 1E-6 (F)
    t: float = 1.0 (s)
    a: float = 0.9
    b: float = 0.9

```python
class HavriliakNegami(Element):
	kwargs
```

_Constructor parameters_

- `kwargs`




### **pyimpspec.HavriliakNegamiAlternative**

Havriliak-Negami relaxation (alternative form)

    Symbol: Ha

    Z = R / ((1 + (I*2*pi*f*t)^b)^g)

    Variables
    ---------
    R: float = 1 (ohm)
    t: float = 1.0 (s)
    b: float = 0.7
    g: float = 0.8

```python
class HavriliakNegamiAlternative(Element):
	kwargs
```

_Constructor parameters_

- `kwargs`




### **pyimpspec.Inductor**

Inductor

    Symbol: L

    Z = j*2*pi*f*L

    Variables
    ---------
    L: float = 1E-6 (H)

```python
class Inductor(Element):
	kwargs
```

_Constructor parameters_

- `kwargs`




### **pyimpspec.ModifiedInductor**

Modified inductor

    Symbol: La

    Z = L*(j*2*pi*f)^n

    Variables
    ---------
    L: float = 1E-6 (H*s^(n-1))
    n: float = 0.95

```python
class ModifiedInductor(Element):
	kwargs
```

_Constructor parameters_

- `kwargs`




### **pyimpspec.Parallel**

Elements connected in parallel.

```python
class Parallel(Connection):
	elements: List[Union[Element, Connection]]
```

_Constructor parameters_

- `elements`: List of elements (and connections) that are connected in parallel.




### **pyimpspec.ParsingError**

```python
class ParsingError(Exception):
```



### **pyimpspec.Resistor**

Resistor

    Symbol: R

    Z = R

    Variables
    ---------
    R: float = 1E+3 (ohm)

```python
class Resistor(Element):
	kwargs
```

_Constructor parameters_

- `kwargs`




### **pyimpspec.Series**

Elements connected in series.

```python
class Series(Connection):
	elements: List[Union[Element, Connection]]
```

_Constructor parameters_

- `elements`: List of elements (and connections) that are connected in series.




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
See `perform_test` for details.


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




### **pyimpspec.UnexpectedCharacter**

```python
class UnexpectedCharacter(Exception):
```



### **pyimpspec.UnsupportedFileFormat**

```python
class UnsupportedFileFormat(Exception):
```



### **pyimpspec.Warburg**

Warburg (semi-infinite diffusion)

    Symbol: W

    Z = 1/(Y*(2*pi*f*j)^(1/2))

    Variables
    ---------
    Y: float = 1.0 (S*s^(1/2))

```python
class Warburg(Element):
	kwargs
```

_Constructor parameters_

- `kwargs`




### **pyimpspec.WarburgOpen**

Warburg, finite space or open (finite length diffusion with reflective boundary)

    Symbol: Wo

    Z = coth((B*j*2*pi*f)^n)/((Y*j*2*pi*f)^n)

    Variables
    ---------
    Y: float = 1.0 (S)
    B: float = 1.0 (s^n)
    n: float = 0.5

```python
class WarburgOpen(Element):
	kwargs
```

_Constructor parameters_

- `kwargs`




### **pyimpspec.WarburgShort**

Warburg, finite length or short (finite length diffusion with transmissive boundary)

    Symbol: Ws

    Z = tanh((B*j*2*pi*f)^n)/((Y*j*2*pi*f)^n)

    Variables
    ---------
    Y: float = 1.0 (S)
    B: float = 1.0 (s^n)
    n: float = 0.5

```python
class WarburgShort(Element):
	kwargs
```

_Constructor parameters_

- `kwargs`




### **pyimpspec.calculate_drt**

Calculates the distribution of relaxation times (DRT) for a given data set.

References:

- Kulikovsky, A., 2020, Phys. Chem. Chem. Phys., 22, 19131-19138 (https://doi.org/10.1039/D0CP02094J)
- Wan, T. H., Saccoccio, M., Chen, C., and Ciucci, F., 2015, Electrochim. Acta, 184, 483-499 (https://doi.org/10.1016/j.electacta.2015.09.097).
- Ciucci, F. and Chen, C., 2015, Electrochim. Acta, 167, 439-454 (https://doi.org/10.1016/j.electacta.2015.03.123)
- Effat, M. B. and Ciucci, F., 2017, Electrochim. Acta, 247, 1117-1129 (https://doi.org/10.1016/j.electacta.2017.07.050)
- Liu, J., Wan, T. H., and Ciucci, F., 2020, Electrochim. Acta, 357, 136864 (https://doi.org/10.1016/j.electacta.2020.136864)
- Boukamp, B.A., 2015, Electrochim. Acta, 154, 35-46, (https://doi.org/10.1016/j.electacta.2014.12.059)
- Boukamp, B.A. and Rolle, A, 2017, Solid State Ionics, 302, 12-18 (https://doi.org/10.1016/j.ssi.2016.10.009)

```python
def calculate_drt(data: DataSet, method: str = "tr-nnls", mode: str = "complex", lambda_value: float = -1.0, rbf_type: str = "gaussian", derivative_order: int = 1, rbf_shape: str = "fwhm", shape_coeff: float = 0.5, inductance: bool = False, credible_intervals: bool = False, num_samples: int = 2000, num_attempts: int = 10, maximum_symmetry: float = 0.5, circuit: Optional[Circuit] = None, W: float = 0.15, num_per_decade: int = 100, num_procs: int = -1) -> DRTResult:
```


_Parameters_

- `data`: The data set to use in the calculations.
- `method`: Valid values include:
    - "bht": Bayesian Hilbert Transform
    - "m(RQ)fit": m(RQ)fit for calculating the DRT based on a fitted circuit
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
    - "fwhm": full width half maximum
    - "factor": `shape_coeff` is used directly
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
- `circuit`: A circuit that contains one or more "(RQ)" or "(RC)" elements connected in series.
An optional series resistance may also be included.
For example, a circuit with a CDC representation of "R(RQ)(RQ)(RC)" would be a valid circuit.
It is highly recommended that the provided circuit has already been fitted.
However, if all of the various parameters of the provided circuit are at their default values, then an attempt will be made to fit the circuit to the data.
Used by the "m(RQ)fit" method.
- `W`: The width of the Gaussian curve that is used to approximate the DRT of an "(RC)" element.
Used by the "m(RQ)fit" method.
- `num_per_decade`: The number of points per decade to use when calculating a DRT.
Used by the "m(RQ)fit" method.
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
- `test`: See `perform_test` for details.
- `num_RCs`: A list of integers representing the various number of RC elements to test.
An empty list results in all possible numbers of RC elements up to the total number of frequencies being tested.
- `mu_criterion`: See `perform_test` for details.
- `add_capacitance`: See `perform_test` for details.
- `add_inductance`: See `perform_test` for details.
- `method`: See `perform_test` for details.
- `max_nfev`: See `perform_test` for details.
- `num_procs`: See `perform_test` for details.


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
A value less than one results in using the number of cores returned by `multiprocessing.cpu_count`.
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



## **pyimpspec.plot.mpl**

### **pyimpspec.plot.mpl.plot_bode**

Plot some data as a Bode plot (\|Z\| and phi vs f).

```python
def plot_bode(data: Union[DataSet, TestResult, FitResult, DRTResult], color_magnitude: str = "black", color_phase: str = "black", marker_magnitude: str = "o", marker_phase: str = "s", line: bool = False, label: Optional[str] = None, legend: bool = True, fig: Optional[Figure] = None, axes: List[Axes] = [], num_per_decade: int = 100, adjust_axes: bool = True) -> Tuple[Figure, List[Axes]]:
```


_Parameters_

- `data`: The data to plot.
DataSet instances are plotted using markers (optionally as a line) while all other types of data are plotted as lines.
- `color_magnitude`: The color of the marker or line for the absolute magnitude of the impedance.
- `color_phase`: The color of the marker or line) for the phase shift of the impedance.
- `marker_magnitude`: The marker for the absolute magnitude of the impedance.
- `marker_phase`: The marker for the phase shift of the impedance.
- `line`: Whether or not a DataSet instance should be plotted as a line instead.
- `label`: The optional label to use in the legend.
- `legend`: Whether or not to add a legend.
- `fig`: The matplotlib.figure.Figure instance to use when plotting the data.
- `axes`: A list of matplotlib.axes.Axes instances to use when plotting the data.
- `num_per_decade`: If the data being plotted is not a DataSet instance (e.g. a TestResult instance), then this parameter can be used to change how many points are used to draw the line (i.e. how smooth or angular the line looks).
- `adjust_axes`: Whether or not to adjust the axes (label, scale, limits, etc.).


_Returns_

```python
Tuple[Figure, List[Axes]]
```
### **pyimpspec.plot.mpl.plot_circuit**

Plot the simulated impedance response of a circuit as both a Nyquist and a Bode plot.

```python
def plot_circuit(circuit: Circuit, f: Union[List[float], ndarray] = [], min_f: float = 0.1, max_f: float = 100000.0, color_nyquist: str = "#CC3311", color_bode_magnitude: str = "#CC3311", color_bode_phase: str = "#009988", data: Optional[DataSet] = None, visible_data: bool = False, title: Optional[str] = None, label: Optional[str] = None, legend: bool = True, fig: Optional[Figure] = None, axes: List[Axes] = []) -> Tuple[Figure, List[Axes]]:
```


_Parameters_

- `circuit`: The circuit to use when simulating the impedance response.
- `f`: The frequencies (in hertz) to use when simulating the impedance response.
If no frequencies are provided, then the range defined by the min_f and max_f parameters will be used instead.
Alternatively, a DataSet instance can be provided via the data parameter.
- `min_f`: The lower limit of the frequency range to use if a list of frequencies is not provided.
- `max_f`: The upper limit of the frequency range to use if a list of frequencies is not provided.
- `color_nyquist`: The color to use in the Nyquist plot.
- `color_bode_magnitude`: The color to use for the magnitude in the Bode plot.
- `color_bode_phase`: The color to use for the phase shift in the Bode plot.
- `data`: An optional DataSet instance.
If provided, then the frequencies of this instance will be used when simulating the impedance spectrum of the circuit.
- `visible_data`: Whether or not the optional DataSet instance should also be plotted alongside the simulated impedance spectrum of the circuit.
- `title`: The title of the figure.
If not title is provided, then the circuit description code of the circuit is used instead.
- `label`: The optional label to use in the legend.
- `legend`: Whether or not to add a legend.
- `fig`: The matplotlib.figure.Figure instance to use when plotting the data.
- `axes`: A list of matplotlib.axes.Axes instances to use when plotting the data.


_Returns_

```python
Tuple[Figure, List[Axes]]
```
### **pyimpspec.plot.mpl.plot_complex_impedance**

Plot the real and imaginary parts of the impedance of some data.

```python
def plot_complex_impedance(data: Union[DataSet, TestResult, FitResult, DRTResult], color_real: str = "black", color_imaginary: str = "black", marker_real: str = "o", marker_imaginary: str = "s", line: bool = False, label: Optional[str] = None, legend: bool = True, fig: Optional[Figure] = None, axes: List[Axes] = [], num_per_decade: int = 100, adjust_axes: bool = True) -> Tuple[Figure, List[Axes]]:
```


_Parameters_

- `data`: The data to plot.
DataSet instances are plotted using markers (optionally as a line) while all other types of data are plotted as lines.
- `color_real`: The color of the marker or line for the real part of the impedance.
- `color_imaginary`: The color of the marker or line for the imaginary part of the impedance.
- `marker_real`: The marker for the real part of the impedance.
- `marker_imaginary`: The marker for the imaginary part of the impedance.
- `line`: Whether or not a DataSet instance should be plotted as a line instead.
- `label`: The optional label to use in the legend.
- `legend`: Whether or not to add a legend.
- `fig`: The matplotlib.figure.Figure instance to use when plotting the data.
- `axes`: A list of matplotlib.axes.Axes instances to use when plotting the data.
- `num_per_decade`: If the data being plotted is not a DataSet instance (e.g. a TestResult instance), then this parameter can be used to change how many points are used to draw the line (i.e. how smooth or angular the line looks).
- `adjust_axes`: Whether or not to adjust the axes (label, scale, limits, etc.).


_Returns_

```python
Tuple[Figure, List[Axes]]
```
### **pyimpspec.plot.mpl.plot_data**

Plot a DataSet instance as both a Nyquist and a Bode plot.

```python
def plot_data(data: DataSet, title: Optional[str] = None, label: Optional[str] = None, legend: bool = True, fig: Optional[Figure] = None, axes: List[Axes] = []) -> Tuple[Figure, List[Axes]]:
```


_Parameters_

- `data`: The DataSet instance to plot.
- `title`: The title of the figure.
If not title is provided, then the label of the DataSet is used instead.
- `label`: The optional label to use in the legend.
- `legend`: Whether or not to add a legend.
- `fig`: The matplotlib.figure.Figure instance to use when plotting the data.
- `axes`: A list of matplotlib.axes.Axes instances to use when plotting the data.


_Returns_

```python
Tuple[Figure, List[Axes]]
```
### **pyimpspec.plot.mpl.plot_drt**

Plot the result of calculating the distribution of relaxation times (DRT) as a Bode plot, a DRT plot, and a plot of the residuals.

```python
def plot_drt(drt: DRTResult, data: Optional[DataSet] = None, peak_threshold: float = -1.0, title: Optional[str] = None, label: Optional[str] = None, legend: bool = True, fig: Optional[Figure] = None, axes: List[Axes] = []) -> Tuple[Figure, List[Tuple[Axes]]]:
```


_Parameters_

- `drt`: The result to plot.
- `data`: The DataSet instance that was used in the DRT calculations.
- `peak_threshold`: The threshold to use for identifying and marking peaks (0.0 to 1.0, relative to the highest peak).
Negative values disable marking peaks.
- `title`: The title of the figure.
If no title is provided, then the circuit description code (and label of the DataSet) is used instead.
- `label`: The optional label to use in the legend.
- `legend`: Whether or not to add a legend.
- `fig`: The matplotlib.figure.Figure instance to use when plotting the data.
- `axes`: A list of matplotlib.axes.Axes instances to use when plotting the data.


_Returns_

```python
Tuple[Figure, List[Tuple[Axes]]]
```
### **pyimpspec.plot.mpl.plot_exploratory_tests**

Plot the results of an exploratory Kramers-Kronig test as a Nyquist plot, a Bode plot, a plot of the residuals, and a plot of the mu- and pseudo chi-squared values.

```python
def plot_exploratory_tests(tests: List[TestResult], mu_criterion: float, data: DataSet, title: Optional[str] = None, legend: bool = True, fig: Optional[Figure] = None, axes: List[Axes] = []) -> Tuple[Figure, List[Axes]]:
```


_Parameters_

- `tests`: The results to plot.
- `mu_criterion`: The mu-criterion to apply.
- `data`: The DataSet instance that was tested.
- `title`: The title of the figure.
If no title is provided, then the label of the DataSet is used instead.
- `legend`: Whether or not to add a legend.
- `fig`: The matplotlib.figure.Figure instance to use when plotting the data.
- `axes`: A list of matplotlib.axes.Axes instances to use when plotting the data.


_Returns_

```python
Tuple[Figure, List[Axes]]
```
### **pyimpspec.plot.mpl.plot_fit**

Plot the result of a fit as a Nyquist plot, a Bode plot, and a plot of the residuals.

```python
def plot_fit(fit: Union[TestResult, FitResult, DRTResult], data: Optional[DataSet] = None, title: Optional[str] = None, label: Optional[str] = None, legend: bool = True, fig: Optional[Figure] = None, axes: List[Axes] = [], num_per_decade: int = 100) -> Tuple[Figure, List[Tuple[Axes]]]:
```


_Parameters_

- `fit`: The circuit fit or test result.
- `data`: The DataSet instance that a circuit was fitted to.
- `title`: The title of the figure.
If no title is provided, then the circuit description code (and label of the DataSet) is used instead.
- `label`: The optional label to use in the legend.
- `legend`: Whether or not to add a legend.
- `fig`: The matplotlib.figure.Figure instance to use when plotting the data.
- `axes`: A list of matplotlib.axes.Axes instances to use when plotting the data.
- `num_per_decade`: If the data being plotted is not a DataSet instance (e.g. a TestResult instance), then this parameter can be used to change how many points are used to draw the line (i.e. how smooth or angular the line looks).


_Returns_

```python
Tuple[Figure, List[Tuple[Axes]]]
```
### **pyimpspec.plot.mpl.plot_gamma**

Plot the distribution of relaxation times (gamma vs tau).

```python
def plot_gamma(drt: DRTResult, peak_threshold: float = -1.0, color: Any = "black", bounds_alpha: float = 0.3, label: Optional[str] = None, legend: bool = True, fig: Optional[Figure] = None, axis: Optional[Axes] = None, adjust_axes: bool = True) -> Tuple[Figure, Axes]:
```


_Parameters_

- `drt`: The result to plot.
- `peak_threshold`: The threshold to use for identifying and marking peaks (0.0 to 1.0, relative to the highest peak).
Negative values disable marking peaks.
- `color`: The color to use to plot the data.
- `bounds_alpha`: The alpha to use when plotting the bounds of the Bayesian credible intervals (if they are included in the data).
- `label`: The optional label to use in the legend.
- `legend`: Whether or not to add a legend.
- `fig`: The matplotlib.figure.Figure instance to use when plotting the data.
- `axis`: The matplotlib.axes.Axes instance to use when plotting the data.
- `adjust_axes`: Whether or not to adjust the axes (label, scale, limits, etc.).


_Returns_

```python
Tuple[Figure, Axes]
```
### **pyimpspec.plot.mpl.plot_imaginary_impedance**

Plot the imaginary impedance of some data (-Zim vs f).

```python
def plot_imaginary_impedance(data: Union[DataSet, TestResult, FitResult, DRTResult], color: Any = "black", marker: str = "s", line: bool = False, label: Optional[str] = None, legend: bool = True, fig: Optional[Figure] = None, axis: Optional[Axes] = None, num_per_decade: int = 100, adjust_axes: bool = True) -> Tuple[Figure, Axes]:
```


_Parameters_

- `data`: The data to plot.
DataSet instances are plotted using markers (optionally as a line) while all other types of data are plotted as lines.
- `color`: The color of the marker or line.
- `marker`: The marker.
- `line`: Whether or not a DataSet instance should be plotted as a line instead.
- `label`: The optional label to use in the legend.
- `legend`: Whether or not to add a legend.
- `fig`: The matplotlib.figure.Figure instance to use when plotting the data.
- `axis`: The matplotlib.axes.Axes instance to use when plotting the data.
- `num_per_decade`: If the data being plotted is not a DataSet instance (e.g. a TestResult instance), then this parameter can be used to change how many points are used to draw the line (i.e. how smooth or angular the line looks).
- `adjust_axes`: Whether or not to adjust the axes (label, scale, limits, etc.).


_Returns_

```python
Tuple[Figure, Axes]
```
### **pyimpspec.plot.mpl.plot_impedance_magnitude**

Plot the absolute magnitude of the impedance of some data (\|Z\| vs f).

```python
def plot_impedance_magnitude(data: Union[DataSet, TestResult, FitResult, DRTResult], color: Any = "black", marker: str = "o", line: bool = False, label: Optional[str] = None, legend: bool = True, fig: Optional[Figure] = None, axis: Optional[Axes] = None, num_per_decade: int = 100, adjust_axes: bool = True) -> Tuple[Figure, Axes]:
```


_Parameters_

- `data`: The data to plot.
DataSet instances are plotted using markers (optionally as a line) while all other types of data are plotted as lines.
- `color`: The color of the marker or line.
- `marker`: The marker.
- `line`: Whether or not a DataSet instance should be plotted as a line instead.
- `label`: The optional label to use in the legend.
- `legend`: Whether or not to add a legend.
- `fig`: The matplotlib.figure.Figure instance to use when plotting the data.
- `axis`: The matplotlib.axes.Axes instance to use when plotting the data.
- `num_per_decade`: If the data being plotted is not a DataSet instance (e.g. a TestResult instance), then this parameter can be used to change how many points are used to draw the line (i.e. how smooth or angular the line looks).
- `adjust_axes`: Whether or not to adjust the axes (label, scale, limits, etc.).


_Returns_

```python
Tuple[Figure, Axes]
```
### **pyimpspec.plot.mpl.plot_impedance_phase**

Plot the phase shift of the impedance of some data (phi vs f).

```python
def plot_impedance_phase(data: Union[DataSet, TestResult, FitResult, DRTResult], color: Any = "black", marker: str = "o", line: bool = False, label: Optional[str] = None, legend: bool = True, fig: Optional[Figure] = None, axis: Optional[Axes] = None, num_per_decade: int = 100, adjust_axes: bool = True) -> Tuple[Figure, Axes]:
```


_Parameters_

- `data`: The data to plot.
DataSet instances are plotted using markers (optionally as a line) while all other types of data are plotted as lines.
- `color`: The color of the marker or line.
- `marker`: The marker.
- `line`: Whether or not a DataSet instance should be plotted as a line instead.
- `label`: The optional label to use in the legend.
- `legend`: Whether or not to add a legend.
- `fig`: The matplotlib.figure.Figure instance to use when plotting the data.
- `axis`: The matplotlib.axes.Axes instance to use when plotting the data.
- `num_per_decade`: If the data being plotted is not a DataSet instance (e.g. a TestResult instance), then this parameter can be used to change how many points are used to draw the line (i.e. how smooth or angular the line looks).
- `adjust_axes`: Whether or not to adjust the axes (label, scale, limits, etc.).


_Returns_

```python
Tuple[Figure, Axes]
```
### **pyimpspec.plot.mpl.plot_mu_xps**

Plot the mu-values and pseudo chi-squared values of Kramers-Kronig test results.

```python
def plot_mu_xps(tests: List[TestResult], mu_criterion: float, color_mu: str = "black", color_xps: str = "black", color_criterion: str = "black", legend: bool = True, fig: Optional[Figure] = None, axes: List[Axes] = [], adjust_axes: bool = True) -> Tuple[Figure, List[Axes]]:
```


_Parameters_

- `tests`: The results to plot.
- `mu_criterion`: The mu-criterion to apply.
- `color_mu`: The color of the markers and line for the mu-values.
- `color_xps`: The color of the markers and line for the pseudo chi-squared values.
- `color_criterion`: The color of the line for the mu-criterion.
- `legend`: Whether or not to add a legend.
- `fig`: The matplotlib.figure.Figure instance to use when plotting the data.
- `axes`: A list of matplotlib.axes.Axes instances to use when plotting the data.
- `adjust_axes`: Whether or not to adjust the axes (label, scale, limits, etc.).


_Returns_

```python
Tuple[Figure, List[Axes]]
```
### **pyimpspec.plot.mpl.plot_nyquist**

Plot some data as a Nyquist plot (-Z" vs Z').

```python
def plot_nyquist(data: Union[DataSet, TestResult, FitResult, DRTResult], color: Any = "black", marker: str = "o", line: bool = False, label: Optional[str] = None, legend: bool = True, fig: Optional[Figure] = None, axis: Optional[Axes] = None, num_per_decade: int = 100, adjust_axes: bool = True) -> Tuple[Figure, Axes]:
```


_Parameters_

- `data`: The data to plot.
DataSet instances are plotted using markers (optionally as a line) while all other types of data are plotted as lines.
- `color`: The color of the marker or line.
- `marker`: The marker.
- `line`: Whether or not a DataSet instance should be plotted as a line instead.
- `label`: The optional label to use in the legend.
- `legend`: Whether or not to add a legend.
- `fig`: The matplotlib.figure.Figure instance to use when plotting the data.
- `axis`: The matplotlib.axes.Axes instance to use when plotting the data.
- `num_per_decade`: If the data being plotted is not a DataSet instance (e.g. a TestResult instance), then this parameter can be used to change how many points are used to draw the line (i.e. how smooth or angular the line looks).
- `adjust_axes`: Whether or not to adjust the axes (label, scale, limits, etc.).


_Returns_

```python
Tuple[Figure, Axes]
```
### **pyimpspec.plot.mpl.plot_real_impedance**

Plot the real impedance of some data (Zre vs f).

```python
def plot_real_impedance(data: Union[DataSet, TestResult, FitResult, DRTResult], color: Any = "black", marker: str = "o", line: bool = False, label: Optional[str] = None, legend: bool = True, fig: Optional[Figure] = None, axis: Optional[Axes] = None, num_per_decade: int = 100, adjust_axes: bool = True) -> Tuple[Figure, Axes]:
```


_Parameters_

- `data`: The data to plot.
DataSet instances are plotted using markers (optionally as a line) while all other types of data are plotted as lines.
- `color`: The color of the marker or line.
- `marker`: The marker.
- `line`: Whether or not a DataSet instance should be plotted as a line instead.
- `label`: The optional label to use in the legend.
- `legend`: Whether or not to add a legend.
- `fig`: The matplotlib.figure.Figure instance to use when plotting the data.
- `axis`: The matplotlib.axes.Axes instance to use when plotting the data.
- `num_per_decade`: If the data being plotted is not a DataSet instance (e.g. a TestResult instance), then this parameter can be used to change how many points are used to draw the line (i.e. how smooth or angular the line looks).
- `adjust_axes`: Whether or not to adjust the axes (label, scale, limits, etc.).


_Returns_

```python
Tuple[Figure, Axes]
```
### **pyimpspec.plot.mpl.plot_residual**

Plot the residuals of a result.

```python
def plot_residual(result: Union[TestResult, FitResult, DRTResult], color_real: str = "black", color_imaginary: str = "black", legend: bool = True, fig: Optional[Figure] = None, axes: List[Axes] = [], adjust_axes: bool = True) -> Tuple[Figure, List[Axes]]:
```


_Parameters_

- `result`: The result to plot.
- `color_real`: The color of the markers and line for the residuals of the real parts of the impedances.
- `color_imaginary`: The color of the markers and line for the residuals of the imaginary parts of the impedances.
- `legend`: Whether or not to add a legend.
- `fig`: The matplotlib.figure.Figure instance to use when plotting the data.
- `axes`: A list of matplotlib.axes.Axes instances to use when plotting the data.
- `adjust_axes`: Whether or not to adjust the axes (label, scale, limits, etc.).


_Returns_

```python
Tuple[Figure, List[Axes]]
```