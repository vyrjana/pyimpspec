.. include:: ./substitutions.rst

Circuit fitting
===============

Equivalent circuit fitting is a very common way of analyzing impedance spectra.
The aim is to extract quantitative information regarding physical characteristics by modeling a system using an equivalent circuit where each element corresponds to some aspect of the system that is being investigated (solution resistance, double-layer capacitance, cable inductance, etc.).
Each element in the circuit has a known expression for its impedance and so does the resulting circuit.

.. note::

   Impedance spectra should be validated using, e.g., Kramers-Kronig tests prior to proceeding with circuit fitting.
   The instrument control software for your potentiostat/galvanostat may include tools for analyzing the excitation and response signals for indications of non-linear behavior.


Performing a fit
------------------

The |fit_circuit| function performs the fitting and returns a |FitResult| object.

.. doctest::

   >>> from pyimpspec import (
   ...   Circuit,
   ...   DataSet,
   ...   FitResult,
   ...   fit_circuit,
   ...   parse_cdc,
   ...   generate_mock_data,
   ... )
   >>>
   >>> data: DataSet = generate_mock_data("CIRCUIT_1", noise=5e-2, seed=42)[0]
   >>> circuit: Circuit = parse_cdc("R(RC)(RW)")
   >>> fit: FitResult = fit_circuit(circuit, data)

|fit_circuit| tries various combinations of iteration methods and weights by default to achieve the best fit.
It may still be necessary to adjust the initial values and/or the limits of the various parameters of the circuit elements.
The |FitResult| object contains, e.g., the fitted |Circuit| object and the `lmfit.MinimizerResult <https://lmfit.github.io/lmfit-py/fitting.html#lmfit.minimizer.MinimizerResult>`_.

The two figures below show the impedance spectrum and the fitted circuit as a Nyquist plot, and the residuals of the fit, respectively.

.. plot::

   from pyimpspec import (
     FitResult,
     fit_circuit,
     parse_cdc,
     generate_mock_data,
   )
   from pyimpspec import mpl
   data = generate_mock_data("CIRCUIT_1", noise=5e-2, seed=42)[0]
   circuit = parse_cdc("R(RC)(RW)")
   fit = fit_circuit(circuit, data)

   figure, axes = mpl.plot_nyquist(data, colors={"impedance": "black"})
   mpl.plot_nyquist(fit, line=True, figure=figure, axes=axes)
   figure.tight_layout()
   
   figure, axes = mpl.plot_residuals(fit)
   figure.tight_layout()

.. raw:: latex

    \clearpage


Generating tables and diagrams
------------------------------

|FitResult| objects have methods that generate pandas.DataFrame_ objects, which can in turn be used to generate, e.g., Markdown or LaTeX tables (pandas.DataFrame.to_markdown_ or pandas.DataFrame.to_latex_, respectively).
Thus, it is quite easy to generate a table containing the fitted parameter values or the statistics related to the fit.
|Circuit| objects have methods for generating circuit diagrams either in the form of a string containing CircuiTikZ_-compatible commands or as a `schemdraw.Drawing <https://schemdraw.readthedocs.io/en/latest/classes/drawing.html#schemdraw.Drawing>`_ object.

.. doctest::

   >>> from schemdraw import Drawing
   >>> from pandas import DataFrame
   >>> from pyimpspec import (
   ...   Circuit,
   ...   DataSet,
   ...   FitResult,
   ...   fit_circuit,
   ...   generate_mock_data,
   ... )
   >>>
   >>> data: DataSet = generate_mock_data("CIRCUIT_1", noise=5e-2, seed=42)[0]
   >>> circuit: Circuit = parse_cdc("R(RC)(RW)")
   >>> fit: FitResult = fit_circuit(circuit, data)
   >>>
   >>> fit.circuit.to_sympy()
   R_0 + 1/(2*I*pi*C_2*f + 1/R_1) + 1/(sqrt(2)*sqrt(pi)*Y_4*sqrt(I*f) + 1/R_3)
   >>>
   >>> drawing: Drawing = fit.circuit.to_drawing()
   >>>
   >>> df: DataFrame = fit.to_parameters_dataframe()
   >>> parameters: str = df.to_markdown(index=False)
   >>>
   >>> df = fit.to_statistics_dataframe()
   >>> statistics: str = df.to_markdown(index=False)

The circuit diagram would look like this:

.. plot::

   from pyimpspec import Circuit, parse_cdc
   circuit: Circuit = parse_cdc("R(RC)(RW)")
   drawing = circuit.to_drawing()
   drawing.draw()

The contents of ``parameters`` and ``statistics`` in the example above would be along the lines of:

.. code::

   | Element   | Parameter   |         Value |   Std. err. (%) | Unit      | Fixed   |
   |:----------|:------------|--------------:|----------------:|:----------|:--------|
   | R_1       | R           |  99.9526      |      0.0270415  | ohm       | No      |
   | R_2       | R           | 200.295       |      0.0161802  | ohm       | No      |
   | C_1       | C           |   7.98617e-07 |      0.00251256 | F         | No      |
   | R_3       | R           | 499.93        |      0.0228977  | ohm       | No      |
   | W_1       | Y           |   0.000400664 |      0.0303443  | S*s^(1/2) | No      |

   | Label                          | Value               |
   |:-------------------------------|:--------------------|
   | Log pseudo chi-squared         | -5.334885584145635  |
   | Log chi-squared                | -10.8934043180641   |
   | Log chi-squared (reduced)      | -12.617680187664888 |
   | Akaike info. criterion         | -1680.3191375061115 |
   | Bayesian info. criterion       | -1670.0169224533793 |
   | Degrees of freedom             | 53                  |
   | Number of data points          | 58                  |
   | Number of function evaluations | 45                  |
   | Method                         | least_squares       |
   | Weight                         | proportional        |

.. note::

   It may not always be possible to estimate errors for fitted parameters.
   Common causes include:

   - A parameter's fitted value is close to the parameter's lower or upper limit.
   - An inappropriate equivalent circuit has been chosen.
   - The maximum number of function evaluations is set too low.
   - The data contains no noise and the equivalent circuit is very good at reproducing the data.


.. note::

   As was mentioned in :doc:`/guide_circuit`, circuit elements and their variables are by default represented in different ways in SymPy_ expressions compared to circuit drawings and in this case also tables of fitted parameters.
   Calling ``fit.to_parameters_dataframe`` with ``running=True`` can be done to have the same running count as in the SymPy expression of the fitted circuit.

.. doctest::

   >>> from schemdraw import Drawing
   >>> from pandas import DataFrame
   >>> from pyimpspec import (
   ...   Circuit,
   ...   DataSet,
   ...   FitResult,
   ...   fit_circuit,
   ...   generate_mock_data,
   ... )
   >>>
   >>> data: DataSet = generate_mock_data("CIRCUIT_1", noise=5e-2, seed=42)[0]
   >>> circuit: Circuit = parse_cdc("R(RC)(RW)")
   >>> fit: FitResult = fit_circuit(circuit, data)
   >>>
   >>> fit.circuit.to_sympy()
   R_0 + 1/(2*I*pi*C_2*f + 1/R_1) + 1/(sqrt(2)*sqrt(pi)*Y_4*sqrt(I*f) + 1/R_3)
   >>>
   >>> drawing: Drawing = fit.circuit.to_drawing(running=True)
   >>>
   >>> df: DataFrame = fit.to_parameters_dataframe(running=True)
   >>> parameters: str = df.to_markdown(index=False)

.. plot::

   from pyimpspec import Circuit, parse_cdc
   circuit: Circuit = parse_cdc("R(RC)(RW)")
   drawing = circuit.to_drawing(running=True)
   drawing.draw()

.. code::

   | Element   | Parameter   |         Value |   Std. err. (%) | Unit      | Fixed   |
   |:----------|:------------|--------------:|----------------:|:----------|:--------|
   | R_0       | R           |  99.9526      |      0.0270415  | ohm       | No      |
   | R_1       | R           | 200.295       |      0.0161802  | ohm       | No      |
   | C_2       | C           |   7.98617e-07 |      0.00251256 | F         | No      |
   | R_3       | R           | 499.93        |      0.0228977  | ohm       | No      |
   | W_4       | Y           |   0.000400664 |      0.0303443  | S*s^(1/2) | No      |


.. raw:: latex

    \clearpage
