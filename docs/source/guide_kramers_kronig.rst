.. include:: ./substitutions.rst

Kramers-Kronig testing
======================

One method for validating impedance spectra involves the use of Kramers-Kronig_ (KK) transforms.
Implementations of the three variants of the linear KK transform tests described by `Boukamp (1995)`_ are included in pyimpspec.
An implementation that uses complex non-linear least squares fitting is also included.
These tests attempt to fit a generally applicable equivalent circuit (see circuit diagram below) where there can be an arbitrary number of parallel RC elements connected in series.
This equivalent circuit is KK transformable, which means that if it can be fitted to the data, then the data should also be KK transformable.
The capacitor and inductor connected in series are necessary for impedance spectra where the imaginary parts of the impedances at the low- and/or high-frequency limits, respectively, do not approach zero.

.. _`Boukamp (1995)`: https://doi.org/10.1149/1.2044210

.. plot::

   from pyimpspec import parse_cdc
   circuit = parse_cdc("R(RC)CL")
   circuit.to_drawing(hide_labels=True).draw()

A few things to keep in mind about this approach to KK testing:

- The fitted circuit has no physical significance.
- An appropriate number of parallel RC elements (i.e., the number of time constants) should be chosen to avoid over- and underfitting (i.e., fitting to the noise or not fitting at all, respectively).
- Each parallel RC element is replaced with an element where the time constant, :math:`\tau=RC`, is fixed but the resistance, :math:`R`, is still variable (i.e., from :math:`Z=\frac{R}{1+j 2 \pi f R C}` to :math:`Z=\frac{R}{1+j 2 \pi f \tau}`).

There are three approaches available for selecting the number of parallel RC elements in pyimpspec:

- Manually specifying the number.
- Using the algorithm described by `Schönleber et al. (2014)`_, which requires choosing a |mu|-criterion value (0.0 to 1.0 where the limits represent over- and underfitting, respectively). See `Lin-KK Tool`_ for an implementation released by that group.
- An alternative implementation of the algorithm above with additional weighting to help avoid false negatives in some circumstances.

.. _`Schönleber et al. (2014)`: https://doi.org/10.1016/j.electacta.2014.01.034

There are two functions for performing KK tests: |perform_test| and |perform_exploratory_tests|.
The |perform_test| function returns a single |TestResult| object and can be used to perform the test with either of the first two approaches to choosing the number of parallel RC elements.
The |perform_exploratory_tests| returns a list of |TestResult| objects, which are sorted from the highest to the lowest scoring result based on the distance of |mu| from the |mu|-criterion and how good the fit is.

.. doctest::

   >>> from pyimpspec import (
   ...   TestResult,
   ...   perform_exploratory_tests,
   ...   perform_test,
   ... )
   >>> from pyimpspec.mock_data import EXAMPLE
   >>> from typing import List
   >>>
   >>> mu_criterion: float = 0.85
   >>> test: TestResult = perform_test(EXAMPLE, mu_criterion=mu_criterion)
   >>> tests: List[TestResult] = perform_exploratory_tests(
   ...   EXAMPLE,
   ...   mu_criterion=mu_criterion,
   ... )  # tests[0] is the highest-scoring result

The three figures below present the results of using |perform_exploratory_tests| (i.e., the last of the three approaches listed above) with the example data.
The first figure plots |mu| and |pseudo chi-squared| as a function of the number of parallel RC elements.
From this plot one can see that 19 seems to be an appropriate number of parallel RC elements to use in this case.
The second figure plots the relative residuals of the real and imaginary parts of the impedances of the fitted circuit and the example data.
From this plot one can see that the residuals are small and randomly distributed around zero, which is what one would hope to see for an impedance spectrum with low noise (or none at all).
The third figure plots the impedance spectrum and the fitted circuit as a Nyquist plot.
From this plot one can see that the fit is indeed good.

.. plot::

   from pyimpspec import perform_exploratory_tests
   from pyimpspec import mpl
   from pyimpspec.mock_data import EXAMPLE
   mu_criterion = 0.85
   tests = perform_exploratory_tests(EXAMPLE, mu_criterion=mu_criterion, add_capacitance=True)

   figure, axes = mpl.plot_mu_xps(tests, mu_criterion=mu_criterion)
   figure.tight_layout()
   
   figure, axes = mpl.plot_residuals(tests[0])
   figure.tight_layout()
   
   figure, axes = mpl.plot_nyquist(tests[0], line=True)
   _ = mpl.plot_nyquist(EXAMPLE, figure=figure, axes=axes, colors={"impedance": "black"})
   figure.tight_layout()

References:

- Boukamp, B.A., 1995, J. Electrochem. Soc., 142, 1885-1894 (https://doi.org/10.1149/1.2044210)
- Schönleber, M., Klotz, D., and Ivers-Tiffée, E., 2014, Electrochim. Acta, 131, 20-27 (https://doi.org/10.1016/j.electacta.2014.01.034)

.. raw:: latex

    \clearpage
