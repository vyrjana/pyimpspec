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
   >>> data: DataSet = generate_mock_data("CIRCUIT_1", noise=0.1, seed=42)[0]
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
   data = generate_mock_data("CIRCUIT_1", noise=0.1, seed=42)[0]
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
   >>> data: DataSet = generate_mock_data("CIRCUIT_1", noise=0.1, seed=42)[0]
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

   | Element   | Parameter   |         Value |   Std. err. (%) | Unit   | Fixed   |
   |:----------|:------------|--------------:|----------------:|:-------|:--------|
   | R_1       | R           | 100.019       |       0.0510552 | ohm    | No      |
   | R_2       | R           | 199.805       |       0.0792938 | ohm    | No      |
   | C_1       | C           |   7.99849e-07 |       0.0882405 | F      | No      |
   | R_3       | R           | 500.029       |       0.117923  | ohm    | No      |
   | W_1       | Y           |   0.00039991  |       0.198213  | S*s^n  | No      |
   | W_1       | n           |   0.5         |     nan         |        | Yes     |
   
   | Label                          | Value               |
   |:-------------------------------|:--------------------|
   | Log pseudo chi-squared         | -4.160334813189715  |
   | Log residual sum of squares    | 1.122377404804515   |
   | Akaike info. criterion         | -139.43268083233875 |
   | Bayesian info. criterion       | -127.39908459601747 |
   | Degrees of freedom             | 77                  |
   | Number of data points          | 82                  |
   | Number of function evaluations | 204                 |
   | Method                         | least_squares       |
   | Weight                         | boukamp             |

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

   | Element   | Parameter   |         Value |   Std. err. (%) | Unit   | Fixed   |
   |:----------|:------------|--------------:|----------------:|:-------|:--------|
   | R_0       | R           | 100.009       |       0.0272264 | ohm    | No      |
   | R_1       | R           | 199.904       |       0.0760636 | ohm    | No      |
   | C_2       | C           |   7.99918e-07 |       0.453359  | F      | No      |
   | R_3       | R           | 500.016       |       0.0796571 | ohm    | No      |
   | W_4       | Y           |   0.000399959 |       0.187019  | S*s^n  | No      |
   | W_4       | n           |   0.5         |     nan         |        | Yes     |



Using mathematical constraints
------------------------------

It is possible to make use of `lmfit`_'s support for `mathemical constraints <https://lmfit.github.io/lmfit-py/constraints.html>`_.
For example, let's say that we have a series of impedance spectra that we wish to analyze using the same equivalent circuit model where there are three repeating ``(RC)`` units that should correspond to the high-, mid-, and low-frequency ranges.


.. plot:: example_fitting_constraints_circuit.py


Fitting this circuit to the first impedance spectrum yields a good fit but the fitted parameters of the ``(RC)`` units do not correspond to the expected order.
The time constants of the high-, mid-, and low-frequency ``(RC)`` units should have the following trend: :math:`\tau_{\rm hf} < \tau_{\rm mf} < \tau_{\rm lf}`.
However, this is not the case since :math:`\tau_{\rm hf} \approx 10^{-3.1}\ {\rm s}`, :math:`\tau_{\rm mf} \approx 10^{-4.5}\ {\rm s}`, and :math:`\tau_{\rm lf} \approx 10^{-2.0}\ {\rm s}`.

We could set the initial values to coax the ``(RC)`` units into the desired order.
Or we could programmatically rearrange and label the ``(RC)`` units ourselves after the fit.
Alternatively, we could use constraints.
In the following example, we express the mid-frequency resistance as :math:`R_{\rm mf} = 10^\alpha \times R_{\rm hf} C_{\rm hf} / C_{\rm mf}` where :math:`\alpha \geq 0`.
Similarly, the low-frequency resistance is expressed as :math:`R_{\rm lf} = 10^\beta \times R_{\rm mf} C_{\rm mf} / C_{\rm lf}` where :math:`\beta \geq 0`.
To accomplish this, we need to provide the |fit_circuit| function with two additional arguments: ``constraint_expressions`` and ``constraint_variables``.

The ``constraint_expressions`` argument maps the names/identifiers, which are provided to `lmfit`_ during the fitting process, of the constrained parameters to strings that contain the expressions that define the constraints.
The |generate_fit_identifiers| function can be used to obtain the names/identifiers of an element's parameters that will be available within the context where the constraint expressions are parsed and evaluated by `lmfit`_.
See `lmfit's documentation about constraints <https://lmfit.github.io/lmfit-py/constraints.html>`_ for information about operators, functions, and constants that are available for use in expressions.

The ``constraint_variables`` argument can be used to define any additional variables that would be used by the constraint expressions.
See `lmfit's documentation about Parameters <https://lmfit.github.io/lmfit-py/parameters.html#lmfit.parameter.Parameter>`_ for information about valid keyword arguments.


.. plot:: example_fitting_constraints_1.py
   :caption: Example data based on a ``R(RC)(RC)(RC)`` circuit (:math:`R_{\rm ser} = 120\ \Omega`, :math:`R_1 = 120\ \Omega`, :math:`R_2 = 840\ \Omega`, :math:`R_3 = 470\ \Omega`, :math:`C_1 = 10^{-7}\ {\rm F}`, :math:`C_2 = 10^{-6}\ {\rm F}`, and :math:`C_3 = 2 \times 10^{-5}\ {\rm F}`).


.. literalinclude:: example_fitting_constraints_1.txt


As shown in the next two examples, the use of constraints can also help to achieve a good fit without first setting the initial values of the circuit elements themselves.
Letting |fit_circuit| try different combinations of fitting methods and weights can also achieve a good fit without adjusting the initial values since another combination of method and weight may be able to converge on a solution despite the suboptimal starting conditions.


.. plot:: example_fitting_constraints_2.py
   :caption: Example data based on a ``R(RC)(RC)(RC)`` circuit (:math:`R_{\rm ser} = 120\ \Omega`, :math:`R_1 = 630\ \Omega`, :math:`R_2 = 210\ \Omega`, :math:`R_3 = 1230\ \Omega`, :math:`C_1 = 2 \times 10^{-7}\ {\rm F}`, :math:`C_2 = 3 \times 10^{-6}\ {\rm F}`, and :math:`C_3 = 3 \times 10^{-5}\ {\rm F}`).


.. literalinclude:: example_fitting_constraints_2.txt


.. plot:: example_fitting_constraints_3.py
   :caption: Example data based on a ``R(RC)(RC)(RC)`` circuit (:math:`R_{\rm ser} = 120\ \Omega`, :math:`R_1 = 350\ \Omega`, :math:`R_2 = 30\ \Omega`, :math:`R_3 = 5000\ \Omega`, :math:`C_1 = 10^{-7}\ {\rm F}`, :math:`C_2 = 4 \times 10^{-5}\ {\rm F}`, and :math:`C_3 = 5 \times 10^{-6}\ {\rm F}`).


.. literalinclude:: example_fitting_constraints_3a.txt


.. note::

   Keep in mind that an estimate for the error of a constrained parameter's value will not be available as indicated by the ``nan`` values. However, this can be remedied by using the obtained fitted circuit to perform yet another fit without any constraints.


.. literalinclude:: example_fitting_constraints_3b.txt


.. raw:: latex

    \clearpage
