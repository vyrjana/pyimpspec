---
layout: documentation
title: API - plotting - matplotlib
permalink: /api/plot-mpl/
---

These functions are for basic visualization of various objects (e.g., `DataSet`, `KramersKronigResult`, and `FittingResult`) using the [matplotlib](https://matplotlib.org/) package.
            

**Table of Contents**

- [pyimpspec.plot.mpl](#pyimpspecplotmpl)
	- [plot_bode](#pyimpspecplotmplplot_bode)
	- [plot_circuit](#pyimpspecplotmplplot_circuit)
	- [plot_data](#pyimpspecplotmplplot_data)
	- [plot_exploratory_tests](#pyimpspecplotmplplot_exploratory_tests)
	- [plot_fit](#pyimpspecplotmplplot_fit)
	- [plot_mu_xps](#pyimpspecplotmplplot_mu_xps)
	- [plot_nyquist](#pyimpspecplotmplplot_nyquist)
	- [plot_residual](#pyimpspecplotmplplot_residual)



## **pyimpspec.plot.mpl**

### **pyimpspec.plot.mpl.plot_bode**

Plot some data as a Bode plot (log |Z| and phi vs log f).

```python
def plot_bode(data: Union[DataSet, KramersKronigResult, FittingResult], color_mag: str = "black", color_phase: str = "black", line: bool = False, fig: Optional[Figure] = None, axes: List[Axes] = [], num_per_decade: int = 100) -> Tuple[Figure, List[Axes]]:
```


_Parameters_

- `data`: The data to plot.
DataSet instances are plotted using markers (optionally as a line) while all other types of data are plotted as lines.
- `color_mag`: The color of the marker (and line) for the logarithm of the absolute magnitude.
- `color_phase`: The color of the marker (and line) for the logarithm of the phase shift.
- `line`: Whether or not a DataSet instance should be plotted as a line instead.
- `fig`: The matplotlib.figure.Figure instance to use when plotting the data.
- `axes`: A list of matplotlib.axes.Axes instances to use when plotting the data.
- `num_per_decade`: If the data being plotted is not a DataSet instance (e.g. a KramersKronigResult instance), then this parameter can be used to change how many points are used to draw the line (i.e. how smooth or angular the line looks).


_Returns_

```python
Tuple[Figure, List[Axes]]
```
### **pyimpspec.plot.mpl.plot_circuit**

Plot the simulated impedance response of a circuit as both a Nyquist and a Bode plot.

```python
def plot_circuit(circuit: Circuit, f: Union[List[float], ndarray] = [], min_f: float = 0.1, max_f: float = 100000.0, data: Optional[DataSet] = None, visible_data: bool = False, title: Optional[str] = None, fig: Optional[Figure] = None, axes: List[Axes] = []) -> Tuple[Figure, List[Axes]]:
```


_Parameters_

- `circuit`: The circuit to use when simulating the impedance response.
- `f`: The frequencies (in hertz) to use when simulating the impedance response.
If no frequencies are provided, then the range defined by the min_f and max_f parameters will be used instead.
Alternatively, a DataSet instance can be provided via the data parameter.
- `min_f`: The lower limit of the frequency range to use if a list of frequencies is not provided.
- `max_f`: The upper limit of the frequency range to use if a list of frequencies is not provided.
- `data`: An optional DataSet instance.
If provided, then the frequencies of this instance will be used when simulating the impedance spectrum of the circuit.
- `visible_data`: Whether or not the optional DataSet instance should also be plotted alongside the simulated impedance spectrum of the circuit.
- `title`: The title of the figure.
If not title is provided, then the circuit description code of the circuit is used instead.
- `fig`: The matplotlib.figure.Figure instance to use when plotting the data.
- `axes`: A list of matplotlib.axes.Axes instances to use when plotting the data.


_Returns_

```python
Tuple[Figure, List[Axes]]
```
### **pyimpspec.plot.mpl.plot_data**

Plot a DataSet instance as both a Nyquist and a Bode plot.

```python
def plot_data(data: DataSet, title: Optional[str] = None, fig: Optional[Figure] = None, axes: List[Axes] = []) -> Tuple[Figure, List[Axes]]:
```


_Parameters_

- `data`: The DataSet instance to plot.
- `title`: The title of the figure.
If not title is provided, then the label of the DataSet is used instead.
- `fig`: The matplotlib.figure.Figure instance to use when plotting the data.
- `axes`: A list of matplotlib.axes.Axes instances to use when plotting the data.


_Returns_

```python
Tuple[Figure, List[Axes]]
```
### **pyimpspec.plot.mpl.plot_exploratory_tests**

Plot the results of an exploratory Kramers-Kronig test and the tested DataSet as a Nyquist plot, a Bode plot, a plot of the residuals, and a plot of the mu-values and pseudo chi-squared values.

```python
def plot_exploratory_tests(scored_tests: List[Tuple[float, KramersKronigResult]], mu_criterion: float, data: DataSet, title: Optional[str] = None, fig: Optional[Figure] = None, axes: List[Axes] = []) -> Tuple[Figure, List[Axes]]:
```


_Parameters_

- `scored_tests`: The scored test results as returned by the pyimpspec.analysis.kramers_kronig.score_test_results function.
- `mu_criterion`: The mu-criterion that was used when performing the tests.
- `data`: The DataSet instance that was tested.
- `title`: The title of the figure.
If no title is provided, then the label of the DataSet is used instead.
- `fig`: The matplotlib.figure.Figure instance to use when plotting the data.
- `axes`: A list of matplotlib.axes.Axes instances to use when plotting the data.


_Returns_

```python
Tuple[Figure, List[Axes]]
```
### **pyimpspec.plot.mpl.plot_fit**

Plot a the result of a circuit fit as a Nyquist plot, a Bode plot, and a plot of the residuals.

```python
def plot_fit(fit: Union[FittingResult, KramersKronigResult], data: Optional[DataSet] = None, title: Optional[str] = None, fig: Optional[Figure] = None, axes: List[Axes] = []) -> Tuple[Figure, List[Tuple[Axes]]]:
```


_Parameters_

- `fit`: The circuit fit or test result.
- `data`: The DataSet instance that a circuit was fitted to.
- `title`: The title of the figure.
If no title is provided, then the circuit description code (and label of the DataSet) is used instead.
- `fig`: The matplotlib.figure.Figure instance to use when plotting the data.
- `axes`: A list of matplotlib.axes.Axes instances to use when plotting the data.


_Returns_

```python
Tuple[Figure, List[Tuple[Axes]]]
```
### **pyimpspec.plot.mpl.plot_mu_xps**

Plot the mu-values and pseudo chi-squared values of exploratory Kramers-Kronig test results.

```python
def plot_mu_xps(scored_tests: List[Tuple[float, KramersKronigResult]], mu_criterion: float, color_mu: str = "black", color_xps: str = "black", color_criterion: str = "black", fig: Optional[Figure] = None, axes: List[Axes] = []) -> Tuple[Figure, List[Axes]]:
```


_Parameters_

- `scored_tests`: The scored test results as returned by the pyimpspec.analysis.kramers_kronig.score_test_results function.
- `mu_criterion`: The mu-criterion that was used when performing the tests.
- `color_mu`: The color of the markers and line for the mu-values.
- `color_xps`: The color of the markers and line for the pseudo chi-squared values.
- `color_criterion`: The color of the line for the mu-criterion.
- `fig`: The matplotlib.figure.Figure instance to use when plotting the data.
- `axes`: A list of matplotlib.axes.Axes instances to use when plotting the data.


_Returns_

```python
Tuple[Figure, List[Axes]]
```
### **pyimpspec.plot.mpl.plot_nyquist**

Plot some data as a Nyquist plot (-Z" vs Z').

```python
def plot_nyquist(data: Union[DataSet, KramersKronigResult, FittingResult], color: str = "black", line: bool = False, fig: Optional[Figure] = None, axis: Optional[Axes] = None, num_per_decade: int = 100) -> Tuple[Figure, Axes]:
```


_Parameters_

- `data`: The data to plot.
DataSet instances are plotted using markers (optionally as a line) while all other types of data are plotted as lines.
- `color`: The color of the marker (and line).
- `line`: Whether or not a DataSet instance should be plotted as a line instead.
- `fig`: The matplotlib.figure.Figure instance to use when plotting the data.
- `axis`: The matplotlib.axes.Axes instance to use when plotting the data.
- `num_per_decade`: If the data being plotted is not a DataSet instance (e.g. a KramersKronigResult instance), then this parameter can be used to change how many points are used to draw the line (i.e. how smooth or angular the line looks).


_Returns_

```python
Tuple[Figure, Axes]
```
### **pyimpspec.plot.mpl.plot_residual**

Plot the residuals of a test or fit result.

```python
def plot_residual(result: Union[KramersKronigResult, FittingResult], color_re: str = "black", color_im: str = "black", fig: Optional[Figure] = None, axes: List[Axes] = []) -> Tuple[Figure, List[Axes]]:
```


_Parameters_

- `result`: The result to plot.
- `color_re`: The color of the markers and line for the residuals of the real parts of the impedances.
- `color_im`: The color of the markers and line for the residuals of the imaginary parts of the impedances.
- `fig`: The matplotlib.figure.Figure instance to use when plotting the data.
- `axes`: A list of matplotlib.axes.Axes instances to use when plotting the data.


_Returns_

```python
Tuple[Figure, List[Axes]]
```