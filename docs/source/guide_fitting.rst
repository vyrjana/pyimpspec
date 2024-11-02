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
----------------

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
   R_0 + 1/(2*I*pi*C_2*f + 1/R_1) + 1/(Y_4*(2*I*pi*f)**n_4 + 1/R_3)
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
   R_0 + 1/(2*I*pi*C_2*f + 1/R_1) + 1/(Y_4*(2*I*pi*f)**n_4 + 1/R_3)
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



Using mathematical constraints
------------------------------

It is possible to make use of `lmfit`_'s support for `mathemical constraints <https://lmfit.github.io/lmfit-py/constraints.html>`_.
We will fit a circuit to the following impedance spectrum.


.. plot::

   from pyimpspec import mpl
   from pyimpspec import generate_mock_data

   data = generate_mock_data("CIRCUIT_5", noise=5e-2, seed=42)[0]
   figure, axes = mpl.plot_nyquist(data, colors={"impedance": "black"})
   figure.tight_layout()


The circuit that we will use is given by the following CDC: ``R(RQ)(RQ)(RQ)``.


.. plot::

   from pyimpspec import Circuit, parse_cdc
   circuit: Circuit = parse_cdc("R(RQ)(RQ)(RQ)")
   drawing = circuit.to_drawing()
   drawing.draw()


If we simply fit this circuit to the impedance spectrum, then we will get a fit that is not great but manages to somewhat resemble the data.
Also, the fitted ``(RQ)`` units may not align nicely with what we can see in the Nyquist plot from left to right.
For example, ``R4`` and ``Q3`` correspond to the semicircle in the middle based on the values of their parameters when compared to the values of the parameters of the other ``(RQ)`` units.


.. plot::

   from pyimpspec import (
     Circuit,
     DataSet,
     Element,
     FitResult,
     fit_circuit,
     generate_mock_data,
     mpl,
     parse_cdc,
   )
   from typing import (
     Dict,
     List,
   )
   
   data: DataSet = generate_mock_data("CIRCUIT_5", noise=5e-2, seed=42)[0]

   circuit: Circuit = parse_cdc("R(RQ)(RQ)(RQ)")
   
   fit: FitResult = fit_circuit(
     circuit,
     data,
     method="least_squares",
     weight="boukamp",
   )
   print(fit.to_parameters_dataframe().to_markdown(index=False))
   print(fit.to_statistics_dataframe().to_markdown(index=False))
   
   figure, axes = mpl.plot_fit(
     fit,
     data=data,
     colored_axes=True,
     legend=False,
     title="Without constraints",
   )
   figure.tight_layout()

.. code::

   | Element   | Parameter   |         Value |   Std. err. (%) | Unit   | Fixed   |
   |:----------|:------------|--------------:|----------------:|:-------|:--------|
   | R_1       | R           | 151.126       |       0.387059  | ohm    | No      |
   | R_2       | R           | 259.693       |       0.944897  | ohm    | No      |
   | Q_1       | Y           |   1.07068e-07 |       0.126065  | S*s^n  | No      |
   | Q_1       | n           |   0.950525    |       0.0555558 |        | No      |
   | R_3       | R           | 581.496       |       1.97511   | ohm    | No      |
   | Q_2       | Y           |   1.8216e-05  |       6.33997   | S*s^n  | No      |
   | Q_2       | n           |   0.85963     |       1.65839   |        | No      |
   | R_4       | R           | 726.502       |       0.695414  | ohm    | No      |
   | Q_3       | Y           |   4.18597e-07 |       0.325163  | S*s^n  | No      |
   | Q_3       | n           |   0.940476    |       0.115395  |        | No      |
   
   | Label                          | Value               |
   |:-------------------------------|:--------------------|
   | Log pseudo chi-squared         | -2.6523611472485586 |
   | Log chi-squared                | -6.941315663341954  |
   | Log chi-squared (reduced)      | -8.798648159773222  |
   | Akaike info. criterion         | -1651.9545159942043 |
   | Bayesian info. criterion       | -1627.8873235215617 |
   | Degrees of freedom             | 72                  |
   | Number of data points          | 82                  |
   | Number of function evaluations | 209                 |
   | Method                         | least_squares       |
   | Weight                         | boukamp             |


However, we can use constraints to enforce the following orders: :math:`R2 \le R4 \le R3` and :math:`Q1 \le Q2 \le Q3`. To accomplish this, we need to provide the |fit_circuit| function with two additional arguments: ``constraint_expressions`` and ``constraint_variables``.

The ``constraint_expressions`` argument maps the names/identifiers, which are provided to `lmfit`_ during the fitting process, of the constrained parameters to strings that contain the expressions that define the constraints.
The |generate_fit_identifiers| function can be used to obtain the names/identifiers of an element's parameters that will be available within the context where the constraint expressions are parsed and evaluated by `lmfit`_.

The ``constraint_variables`` argument can be used to define any additional variables that would be used by the constraint expressions.
See `lmfit's documentation about Parameters <https://lmfit.github.io/lmfit-py/parameters.html#lmfit.parameter.Parameter>`_ for information about valid keyword arguments.

.. doctest::

   >>> from pyimpspec import (
   ...   Circuit,
   ...   DataSet,
   ...   Element,
   ...   FitIdentifiers,
   ...   FitResult,
   ...   fit_circuit,
   ...   generate_fit_identifiers,
   ...   generate_mock_data,
   ...   parse_cdc,
   ... )
   >>> from typing import (
   ...   Dict,
   ...   List,
   ... )
   >>> 
   >>> data: DataSet = generate_mock_data("CIRCUIT_5", noise=5e-2, seed=42)[0]
   >>>
   >>> circuit: Circuit = parse_cdc("R(RQ)(RQ)(RQ)")
   >>> identifiers: Dict[Element, FitIdentifiers] = generate_fit_identifiers(circuit)
   >>> elements: List[Element] = circuit.get_elements()
   >>> R1, R2, Q1, R3, Q2, R4, Q3 = elements
   >>> 
   >>> fit: FitResult = fit_circuit(
   ...   circuit,
   ...   data,
   ...   method="least_squares",
   ...   weight="boukamp",
   ...   constraint_expressions={
   ...     identifiers[R3].R: f"{identifiers[R2].R} + alpha",
   ...     identifiers[R4].R: f"{identifiers[R3].R} - beta",
   ...     identifiers[Q2].Y: f"{identifiers[Q1].Y} + gamma",
   ...     identifiers[Q3].Y: f"{identifiers[Q2].Y} + delta",
   ...   },
   ...   constraint_variables=dict(
   ...     alpha=dict(
   ...       value=500,
   ...       min=0,
   ...     ),
   ...     beta=dict(
   ...       value=300,
   ...       min=0,
   ...     ),
   ...     gamma=dict(
   ...       value=1e-8,
   ...       min=0,
   ...     ),
   ...     delta=dict(
   ...       value=2e-7,
   ...       min=0,
   ...     ),
   ...   ),
   ... )
   >>>
   >>> R1, R2, Q1, R3, Q2, R4, Q3 = fit.circuit.get_elements()
   >>> (
   ...   (R2.get_value("R") < R4.get_value("R") < R3.get_value("R"))
   ...   and (Q1.get_value("Y") < Q2.get_value("Y") < Q3.get_value("Y"))
   ... )
   True


Now we can obtain a better fit and also have all of the elements in an order that matches what we can observe in the Nyquist plot from left to right.

.. note::

   Your results may vary depending on the platform and/or BLAS and LAPACK libraries.


.. plot::

   from pyimpspec import (
     Circuit,
     DataSet,
     Element,
     FitIdentifiers,
     FitResult,
     fit_circuit,
     generate_fit_identifiers,
     generate_mock_data,
     mpl,
     parse_cdc,
   )
   from typing import (
     Dict,
     List,
   )
   
   data: DataSet = generate_mock_data("CIRCUIT_5", noise=5e-2, seed=42)[0]

   circuit: Circuit = parse_cdc("R(RQ)(RQ)(RQ)")
   identifiers: Dict[Element, FitIdentifiers] = generate_fit_identifiers(circuit)
   elements: List[Element] = circuit.get_elements()
   R1, R2, Q1, R3, Q2, R4, Q3 = elements
   
   fit: FitResult = fit_circuit(
     circuit,
     data,
     method="least_squares",
     weight="boukamp",
     constraint_expressions={
       identifiers[R3].R: f"{identifiers[R2].R} + alpha",
       identifiers[R4].R: f"{identifiers[R3].R} - beta",
       identifiers[Q2].Y: f"{identifiers[Q1].Y} + gamma",
       identifiers[Q3].Y: f"{identifiers[Q2].Y} + delta",
     },
     constraint_variables=dict(
       alpha=dict(
         value=500,
         min=0,
       ),
       beta=dict(
         value=300,
         min=0,
       ),
       gamma=dict(
         value=1e-8,
         min=0,
       ),
       delta=dict(
         value=2e-7,
         min=0,
       ),
     ),
   )
   print(fit.to_parameters_dataframe().to_markdown(index=False))
   print(fit.to_statistics_dataframe().to_markdown(index=False))
   
   R1, R2, Q1, R3, Q2, R4, Q3 = fit.circuit.get_elements()
   assert R2.get_value("R") < R4.get_value("R") < R3.get_value("R")
   assert Q1.get_value("Y") < Q2.get_value("Y") < Q3.get_value("Y")

   refined_fit: FitResult = fit_circuit(
     fit.circuit,
     data,
     method="least_squares",
     weight="boukamp",
   )

   print(refined_fit.to_parameters_dataframe().to_markdown(index=False))
   print(refined_fit.to_statistics_dataframe().to_markdown(index=False))

   figure, axes = mpl.plot_fit(
     fit,
     data=data,
     colored_axes=True,
     legend=False,
     title="With constraints",
   )
   figure.tight_layout()


.. code::

   | Element   | Parameter   |         Value |   Std. err. (%) | Unit   | Fixed   |
   |:----------|:------------|--------------:|----------------:|:-------|:--------|
   | R_1       | R           | 143.461       |      0.0917482  | ohm    | No      |
   | R_2       | R           | 229.453       |      0.288944   | ohm    | No      |
   | Q_1       | Y           |   1.78147e-07 |      0.00643312 | S*s^n  | No      |
   | Q_1       | n           |   0.912666    |      0.0138234  |        | No      |
   | R_3       | R           | 854.409       |    nan          | ohm    | Yes     |
   | Q_2       | Y           |   7.82691e-07 |    nan          | S*s^n  | Yes     |
   | Q_2       | n           |   0.856438    |      0.0285731  |        | No      |
   | R_4       | R           | 475.654       |    nan          | ohm    | Yes     |
   | Q_3       | Y           |   1.60541e-05 |    nan          | S*s^n  | Yes     |
   | Q_3       | n           |   0.952466    |      0.19436    |        | No      |
   
   | Label                          | Value               |
   |:-------------------------------|:--------------------|
   | Log pseudo chi-squared         | -4.2481804461232695 |
   | Log chi-squared                | -10.082848279294568 |
   | Log chi-squared (reduced)      | -11.940180775725837 |
   | Akaike info. criterion         | -2245.113501987264  |
   | Bayesian info. criterion       | -2221.0463095146215 |
   | Degrees of freedom             | 72                  |
   | Number of data points          | 82                  |
   | Number of function evaluations | 569                 |
   | Method                         | least_squares       |
   | Weight                         | boukamp             |


.. note::

   Keep in mind that an estimate for the error of a constrained parameter's value will not be available. However, this can be remedied by using the obtained fitted circuit to perform yet another fit without any constraints.


.. code::

   | Element   | Parameter   |         Value |   Std. err. (%) | Unit   | Fixed   |
   |:----------|:------------|--------------:|----------------:|:-------|:--------|
   | R_1       | R           | 143.48        |      0.0892796  | ohm    | No      |
   | R_2       | R           | 229.69        |      0.283135   | ohm    | No      |
   | Q_1       | Y           |   1.78048e-07 |      0.00759739 | S*s^n  | No      |
   | Q_1       | n           |   0.912652    |      0.013558   |        | No      |
   | R_3       | R           | 853.869       |      0.0916479  | ohm    | No      |
   | Q_2       | Y           |   7.80619e-07 |      0.0126453  | S*s^n  | No      |
   | Q_2       | n           |   0.856809    |      0.0280145  |        | No      |
   | R_4       | R           | 475.967       |      0.230118   | ohm    | No      |
   | Q_3       | Y           |   1.60529e-05 |      0.697925   | S*s^n  | No      |
   | Q_3       | n           |   0.952274    |      0.190482   |        | No      |

   | Label                          | Value               |
   |:-------------------------------|:--------------------|
   | Log pseudo chi-squared         | -4.262336239351527  |
   | Log chi-squared                | -10.111392253122428 |
   | Log chi-squared (reduced)      | -11.968724749553697 |
   | Akaike info. criterion         | -2250.5029461349936 |
   | Bayesian info. criterion       | -2226.435753662351  |
   | Degrees of freedom             | 72                  |
   | Number of data points          | 82                  |
   | Number of function evaluations | 79                  |
   | Method                         | least_squares       |
   | Weight                         | boukamp             |


.. raw:: latex

    \clearpage
