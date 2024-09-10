.. include:: ./substitutions.rst

Kramers-Kronig testing
======================

One method for validating immittance spectra involves the use of Kramers-Kronig_ (KK) transforms.
Implementations of the three variants of the linear KK tests (complex, real, and imaginary) described by `Boukamp (1995) <https://doi.org/10.1149/1.2044210>`_ are included in pyimpspec.
These three types of tests have been implemented using least squares fitting.
Alternative implementations, which were the default implementation before version 5.0.0, based on matrix inversion are also included.
An implementation that uses complex non-linear least squares fitting is also included, but this tends to be significantly slower than any of the other implementations.
These tests attempt to fit generally applicable equivalent circuit models (ECM, see the two circuit diagrams below).
These ECMs are KK transformable, which means that if they can be fitted to the data with small, random residuals, then the data should also be KK transformable.

The ECM that is used for the impedance representation of the immittance data is shown below and it can contain many parallel RC elements connected in series.
The capacitor and inductor connected in series may be necessary for impedance spectra where the imaginary parts do not approach zero at the low- and high-frequency limits, respectively.

.. plot::
   :alt: The type of circuit that is used to check impedance data for Kramers-Kronig compliance: a resistance, capacitance, and inductor connected in series to an arbitrary number of parallel RC elements that are connected in series.

   from pyimpspec import parse_cdc
   circuit = parse_cdc("R(RC)(RC)CL")
   elements = circuit.get_elements()
   custom_labels = {
       elements[0]: r"$R_{\rm ser}$",
       elements[1]: r"$R_1$",
       elements[2]: r"$C_1$",
       elements[3]: r"$R_k$",
       elements[4]: r"$C_k$",
       elements[5]: r"$C_{\rm ser}$",
       elements[6]: r"$L_{\rm ser}$",
   }
   circuit.to_drawing(custom_labels=custom_labels).draw()


For the admittance representation of the immittance data, the following equivalent circuit is used instead.
Here, many series RC elements are connected in parallel.
Similarly to the circuit shown above, a parallel capacitor and/or a parallel inductor may be needed for some admittance spectra.

.. plot::
   :alt: The type of circuit that is used to check admittance data for Kramers-Kronig compliance: a resistance, capacitance, and inductor connected in parallel to an arbitrary number of series RC elements that are connected in parallel.

   from pyimpspec import parse_cdc
   circuit = parse_cdc("(R[RC][RC]CL)")
   elements = circuit.get_elements()
   custom_labels = {
       elements[0]: r"$R_{\rm par}$",
       elements[1]: r"$R_1$",
       elements[2]: r"$C_1$",
       elements[3]: r"$R_k$",
       elements[4]: r"$C_k$",
       elements[5]: r"$C_{\rm par}$",
       elements[6]: r"$L_{\rm par}$",
   }
   circuit.to_drawing(custom_labels=custom_labels).draw()

A few things to keep in mind about this approach to KK testing:

- The fitted circuits have no physical significance and some of the fitted parameters may end up with negative values.
- Each parallel/series RC element is replaced with an element where the time constant, :math:`\tau=RC`, is fixed but either the resistance, :math:`R`, or the capacitance, :math:`C`, is still variable for the impedance and admittance representations, respectively.

  - :math:`Z(\omega)=\frac{R}{1+j \omega R C}` becomes :math:`Z(\omega)=\frac{R}{1+j \omega \tau}` when operating on the impedance representation.
  - :math:`Y(\omega)=\frac{C \omega}{\omega R C - j}` becomes :math:`Y(\omega)=\frac{C \omega}{\omega \tau - j}` when operating on the admittance representation.


Either the complex or the real test should be used.
Obtaining good fits with the imaginary test can be challenging even when the immittance spectrum is known to be valid.
The error is spread out across the real and imaginary parts when using the complex test, which can make it more difficult to spot issues compared to the real test where the error is concentrated on the imaginary part.
However, the real test can be overly sensitive by comparison and one should keep in mind that, e.g., slight increases in the magnitudes of the residuals at the frequency extremes might be resolved by choosing a more appropriate data representation or optimizing the range of time constants.

Many immittance spectra can be validated using the range of time constants within the bounds defined by the inverse of the maximum and the minimum excitation frequencies.
However, in some cases it is necessary to extend the range of time constants by some factor :math:`F_{\rm ext} > 1` so that :math:`\tau \in [\frac{1}{F_{\rm ext} \omega_{\rm max}}, \frac{F_{\rm ext}}{\omega_{\rm min}}]` where :math:`\omega_{\rm max}` and :math:`\omega_{\rm min}` are the maximum and minimum, respectively, of the measured angular frequencies.
The range may also need to be contracted (i.e., :math:`F_{\rm ext} < 1`).
Pyimpspec includes an implementation for automatically optimizing |F_ext| and whether or not the suggested |F_ext| is appropriate can be assessed with the help of a 3D plot of |log pseudo chi-squared| as a function of |N_tau| and |log F_ext|.

.. plot::

   from pyimpspec import (
     generate_mock_data,
     mpl,
   )
   from pyimpspec.analysis.kramers_kronig import evaluate_log_F_ext

   data = generate_mock_data("CIRCUIT_4", noise=5e-2, seed=42)[0]
   evaluations = evaluate_log_F_ext(data, min_log_F_ext=-1.0, max_log_F_ext=1.0, num_F_ext_evaluations=20)
   figure, axes = mpl.plot_log_F_ext(evaluations)
   figure.tight_layout()

   figure, axes = mpl.plot_log_F_ext(evaluations, projection="2d", legend=False)
   figure.tight_layout()

In the eaxmple above, the default range of time constants (:math:`\log{F_{\rm ext}} = 0`) exhibits a wide range of |N_tau| (:math:`8 < N_\tau < 45`) with a gradual decrease of |pseudo chi-squared|.
An extended range of time constants (:math:`\log{F_{\rm ext}} = 0.394`, purple markers) is found to be optimal since it achieves a similarly low |pseudo chi-squared| with a lower |N_tau|.

An optimum number of parallel/series RC elements (i.e., the number of time constants or |N_tauopt|) should be chosen to avoid over- and underfitting (i.e., fitting to the noise or not fitting to the data, respectively).
Pyimpspec implements multiple methods for suggesting |N_tauopt|:

.. list-table:: Methods for suggesting the optimum number of time constants (i.e., the number of parallel/series RC elements).
   :header-rows: 1

   * - Method
     - Reference
   * - 1: |mu|-criterion
     - `Schönleber et al. (2014) <https://doi.org/10.1016/j.electacta.2014.01.034>`_
   * - 2: norm of fitted variables
     - `Plank et al. (2022) <https://doi.org/10.1109/IWIS57888.2022.9975131>`_
   * - 3: norm of curvatures
     - `Plank et al. (2022) <https://doi.org/10.1109/IWIS57888.2022.9975131>`_
   * - 4: number of sign changes among curvatures
     - `Plank et al. (2022) <https://doi.org/10.1109/IWIS57888.2022.9975131>`_
   * - 5: mean distance between sign changes among curvatures
     - `Yrjänä and Bobacka (2024) <https://doi.org/10.1016/j.electacta.2024.144951>`_
   * - 6: apex of |log sum abs tau R| (or |log sum abs tau C|) *versus* |N_tau|
     - `Yrjänä and Bobacka (2024) <https://doi.org/10.1016/j.electacta.2024.144951>`_

.. note::

   The implementations of methods 1, 3, and 4 include some `modifications <https://doi.org/10.1016/j.electacta.2024.144951>`_ to make them more robust, but these modifications can be disabled.


The default approach combines the three methods that are based on the curvatures of the immittance spectrum of the fitted ECM in order to:

- minimize the number of sign changes of the curvatures (method 4)
- minimize the norm of the curvatures (method 3)
- maximize the mean distance between sign changes of the curvatures (method 5)

Each method represents a stage that is used to narrow down suitable |N_tau| until one remains.
It is also possible to either choose which method(s) to use or to pick a specific number of time constants manually.

Pyimpspec also includes automatic estimation of the lower and upper limits for |N_tauopt| in order to reduce the probability of suggesting an |N_tauopt| that is either too small or too large.
The lower limit is estimated using a plot of |log pseudo chi-squared| as a function of |N_tau| while the upper limit is estimated with the help of method 5 (i.e., the mean distances between sign changes of the curvature of the impedance spectra of the fitted ECMs).
Either limit, both limits, and/or the difference between the limits can also be specified manually.


How to use
----------

A KK test can be performed by calling the |perform_kramers_kronig_test| function, which returns a |KramersKronigResult| object. This function acts as a wrapper for several other functions that can also be called individually: |evaluate_log_F_ext|, |suggest_num_RC_limits|, |suggest_num_RC|, and |suggest_representation|.

The |evaluate_log_F_ext| function attempts to optimize the range of time constants (i.e., optimize |F_ext|), but the value of |F_ext| can also be specified explicitly.
A list of |KramersKronigResult| can be supplied to the |suggest_num_RC_limits| and |suggest_num_RC| functions.
The former function will return the estimated lower and upper limits (|N_taumin| and |N_taumax|, respectively) of |N_tau| where |N_tauopt| is likely to exist.
The latter function will return a tuple containing the suggested |KramersKronigResult| instance (i.e., the one that corresponds to |N_tauopt|), a dictionary that maps the numbers of time constants to the scores that were used to suggest |N_tauopt|, and the estimated |N_taumin| and |N_taumax|.
A list of these tuples, where each tuple corresponds to a KK test that was performed on either the impedance or the admittance representation, can then be provided to the |suggest_representation| function.
If |perform_kramers_kronig_test| is called with ``admittance=None``, then both the impedance and the admittance representation are tested.
Otherwise, only either the impedance (``admittance=False``) or the admittance (``admittance=True``) is tested.

.. doctest::

   >>> from pyimpspec import (
   ...   DataSet,
   ...   KramersKronigResult,
   ...   generate_mock_data,
   ...   perform_kramers_kronig_test,
   ... )
   >>> from pyimpspec.analysis.kramers_kronig import (
   ...   evaluate_log_F_ext,
   ...   suggest_num_RC,
   ...   suggest_representation,
   ... )
   >>> from typing import Dict, List, Tuple
   >>>
   >>>
   >>> data: DataSet = generate_mock_data("CIRCUIT_1", noise=5e-2, seed=42)[0]
   >>>
   >>> test: KramersKronigResult  # The suggested result
   >>> test = perform_kramers_kronig_test(data)
   >>> # The line above is equivalent to the lines below
   >>> # in terms of the work that is performed
   >>>
   >>> Z_evaluations: List[Tuple[float, List[KramersKronigResult], float]]
   >>> Z_evaluations = evaluate_log_F_ext(data, admittance=False)
   >>>
   >>> Z_suggested_F_ext: float
   >>> Z_tests: List[KramersKronigResult]
   >>> Z_minimized_statistic: float
   >>> Z_suggested_F_ext, Z_tests, Z_minimized_statistic = Z_evaluations[0]
   >>>
   >>> Z_suggestion: Tuple[KramersKronigResult, Dict[int, float], int, int]
   >>> Z_suggestion = suggest_num_RC(Z_tests)
   >>>
   >>> Y_evaluations: List[Tuple[float, List[KramersKronigResult], float]]
   >>> Y_evaluations = evaluate_log_F_ext(data, admittance=True)
   >>>
   >>> Y_tests: List[KramersKronigResult] = Y_evaluations[0][1]
   >>>
   >>> Y_suggestion: Tuple[KramersKronigResult, Dict[int, float], int, int]
   >>> Y_suggestion = suggest_num_RC(Y_tests)
   >>>
   >>> suggestion: Tuple[KramersKronigResult, Dict[int, float], int, int]
   >>> suggestion = suggest_representation([Z_suggestion, Y_suggestion])
   >>>
   >>> scores: Dict[int, float]  # Scores for various numbers of RC elements
   >>> lower_limit: int
   >>> upper_limit: int
   >>> test, scores, lower_limit, upper_limit = suggestion


A single |KramersKronigResult| can be plotted on its own, but it is also possible to plot the suggested |KramersKronigResult| along with the |pseudo chi-squared| values of all |KramersKronigResult| instances so that one can see if the suggested |KramersKronigResult| is indeed the best choice.


.. plot::

   from pyimpspec import generate_mock_data
   from pyimpspec.analysis.kramers_kronig import evaluate_log_F_ext, suggest_num_RC
   from pyimpspec import mpl

   data = generate_mock_data("CIRCUIT_1", noise=5e-2, seed=42)[0]
   tests = evaluate_log_F_ext(data)[0][1]
   suggestion = suggest_num_RC(tests)

   figure, axes = mpl.plot_kramers_kronig_tests(
     tests,
     suggestion,
     data,
     legend=False,
     colored_axes=True,
   )
   figure.tight_layout()


From the top-left plot one can see that the estimated lower and upper limits define a range of |N_tau| values (filled circles, the y-axis on the left-hand side) where |N_tauopt| value is estimated to exist.
The y-axis on the right-hand side shows the scores assigned based on an approach that makes use of methods 3, 4, and 5.
These scores are then used to suggest |N_tauopt| (dashed line).

The |perform_kramers_kronig_test| function takes keyword arguments that can be passed on to the |suggest_num_RC| function.
This can be used to, e.g., select which method(s) to use or to adjust any method-specific settings such as the |mu|-criterion of method 1.

.. doctest::

   >>> from pyimpspec import (
   ...   DataSet,
   ...   KramersKronigResult,
   ...   generate_mock_data,
   ...   perform_kramers_kronig_test,
   ... )
   >>> from pyimpspec.analysis.kramers_kronig import (
   ...   evaluate_log_F_ext,
   ...   suggest_num_RC,
   ... )
   >>> from typing import List, Tuple
   >>>
   >>> mu_criterion: float = 0.85
   >>>
   >>> data: DataSet = generate_mock_data("CIRCUIT_1", noise=5e-2, seed=42)[0]
   >>>
   >>> test: KramersKronigResult
   >>> test = perform_kramers_kronig_test(data, mu_criterion=mu_criterion)
   >>> # The above is equivalent to the following lines
   >>> # in terms of the work that is performed
   >>>
   >>> evaluations: List[Tuple[float, List[KramersKronigResult], float]]
   >>> evaluations = evaluate_log_F_ext(data)
   >>>
   >>> optimum_log_Fext: Tuple[float, List[KramersKronigResult], float]
   >>> optimum_log_Fext = evaluations[0]
   >>>
   >>> tests: List[KramersKronigResult] = optimum_log_Fext[1]
   >>> suggestion: Tuple[KramersKronigResult, Dict[int, float], int, int] = suggest_num_RC(
   ...   tests,
   ...   mu_criterion=mu_criterion,
   ... )
   >>>
   >>> scores: Dict[int, float]
   >>> lower_limit: int
   >>> upper_limit: int
   >>> test, scores, lower_limit, upper_limit = suggestion


The plot of relative residuals is typically used to interpret the validity of the immittance spectrum that was tested. Alternatively, statistical tests performed on the residuals can also be used.

.. doctest::

   >>> from pyimpspec import (
   ...   DataSet,
   ...   KramersKronigResult,
   ...   generate_mock_data,
   ...   perform_kramers_kronig_test,
   ... )
   >>> from pyimpspec.analysis.kramers_kronig import (
   ...   evaluate_log_F_ext,
   ...   suggest_num_RC,
   ...   suggest_representation,
   ... )
   >>> from typing import List, Tuple
   >>>
   >>>
   >>> data: DataSet = generate_mock_data("CIRCUIT_1", noise=5e-2, seed=42)[0]
   >>>
   >>> test: KramersKronigResult  # The suggested result
   >>> test = perform_kramers_kronig_test(data)
   >>> statistics: str = test.to_statistics_dataframe().to_markdown(index=False)


The contents of ``statistics`` would look something like:

.. code::

   | Label                                               |         Value |
   |:----------------------------------------------------|--------------:|
   | Log pseudo chi-squared                              |  -4.46966     |
   | Number of RC elements                               |  13           |
   | Log Fext (extension factor for time constant range) |  -0.100386    |
   | Series resistance (ohm)                             | 103.525       |
   | Series capacitance (F)                              |   0.0120676   |
   | Series inductance (H)                               |  -2.24707e-06 |
   | Mean of residuals, real (% of |Z|)                  |  -5.25249e-06 |
   | Mean of residuals, imag. (% of |Z|)                 |   0.00220288  |
   | SD of residuals, real (% of |Z|)                    |   0.0523757   |
   | SD of residuals, imag. (% of |Z|)                   |   0.075694    |
   | Residuals within 1 SD, real (%)                     |  68.2927      |
   | Residuals within 1 SD, imag. (%)                    |  58.5366      |
   | Residuals within 2 SD, real (%)                     |  97.561       |
   | Residuals within 2 SD, imag. (%)                    |  95.122       |
   | Residuals within 3 SD, real (%)                     | 100           |
   | Residuals within 3 SD, imag. (%)                    | 100           |
   | Lilliefors test p-value, real                       |   0.252269    |
   | Lilliefors test p-value, imag.                      |   0.513698    |
   | Shapiro-Wilk test p-value, real                     |   0.591578    |
   | Shapiro-Wilk test p-value, imag.                    |   0.168292    |
   | Estimated SD of Gaussian noise (% of |Z|)           |   0.0643079   |
   | One-sample Kolmogorov-Smirnov test p-value, real    |   0.871214    |
   | One-sample Kolmogorov-Smirnov test p-value, imag.   |   0.60763     |


All three statistical tests (`Lilliefors <https://www.statsmodels.org/dev/generated/statsmodels.stats.diagnostic.lilliefors.html>`_, `Shapiro-Wilk <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html>`_, and `Kolmogorov-Smirnov <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kstest.html>`_) return :math:`p`-values greater than 0.05 (our chosen threshold) for the residuals of both the real and the imaginary parts. The means of the residuals are close to zero as well. All of this indicates that the tested immittance spectrum is likely to be valid. This is also in agreement with the interpretation based on inspecting the plot of the relative residuals.


Some immittance spectra might not be possible to validate based on testing the impedance representation.
For example, the Nyquist plot below shows a synthetic impedance spectrum that includes a negative differential resistance (the larger, outer loop that goes from the right-hand side to the left-hand side as the frequency is decreased).
Similar impedance spectra have been reported when measuring, e.g., `in the passive region of a system with a tantalum working electrode in hydrofluoric acid (Fig. 3b in the reference) <https://iopscience.iop.org/article/10.1149/2.0691805jes>`_.

.. plot::

   from pyimpspec import mpl, generate_mock_data
   
   data = generate_mock_data("CIRCUIT_8", noise=5e-2, seed=42)[0]
   figure, axes = mpl.plot_nyquist(data)
   figure.tight_layout()


Attempting to perform KK tests on this impedance data as shown in the previous example incorrectly indicates that the spectrum is not linear, causal, and stable.

.. plot::

   from pyimpspec import generate_mock_data
   from pyimpspec.analysis.kramers_kronig import evaluate_log_F_ext, suggest_num_RC
   from pyimpspec import mpl

   data = generate_mock_data("CIRCUIT_8", noise=5e-2, seed=42)[0]
   tests = evaluate_log_F_ext(data)[0][1]
   suggestion = suggest_num_RC(tests)
   
   figure, axes = mpl.plot_kramers_kronig_tests(tests, suggestion, data, legend=False, colored_axes=True)
   figure.tight_layout()


However, there are two approaches that can be used to successfully validate this impedance spectrum.
The first approach is to perform the KK tests on the admittance data either explicitly (i.e., by specifying ``admittance=True`` when calling the |perform_kramers_kronig_test| and |evaluate_log_F_ext| functions) or by calling |perform_kramers_kronig_test| with ``admittance=None`` (default value). The latter should then test both the impedance and the admittance representation before ultimately suggesting the result for the most appropriate represention, which in this case is the admittance representation.

.. plot::

   from pyimpspec import generate_mock_data
   from pyimpspec.analysis.kramers_kronig import evaluate_log_F_ext, suggest_num_RC
   from pyimpspec import mpl

   data = generate_mock_data("CIRCUIT_8", noise=5e-2, seed=42)[0]
   tests = evaluate_log_F_ext(data, admittance=True)[0][1]
   suggestion = suggest_num_RC(tests)
   
   figure, axes = mpl.plot_kramers_kronig_tests(tests, suggestion, data, legend=False, colored_axes=True)
   figure.tight_layout()


The second approach is to add a parallel resistance of a suitable magnitude to the impedance data and to perform the KK tests on the resulting impedance data.

.. plot::

   from pyimpspec import parse_cdc
   # A Warburg impedance is used here just to have two different symbols
   circuit = parse_cdc("(WR)")
   elements = circuit.get_elements()
   custom_labels = {
       elements[0]: r"$Z_{\rm data}$",
       elements[1]: r"$R_{\rm par}$",
   }
   circuit.to_drawing(custom_labels=custom_labels).draw()


The resistance, :math:`R_{\rm par}`, is known *a priori* to be KK transformable.
Adding the resistance in parallel to the experimental data, which is represented in this circuit diagram as :math:`Z_{\rm data}`, does not negatively affect the compliance of the resulting circuit.
Thus, the KK compliance of the resulting circuit is dependent on whether or not :math:`Z_{\rm data}` is KK compliant.

.. note::

   The magnitude of the resistance to choose depends on the original impedance data.
   In this example, the real part of the impedance at the lowest frequency in the original data is approximately :math:`-100` |ohm|.
   A value of 50 |ohm| was chosen for the parallel resistance after testing a few different values.

As can be seen from the results below, the new, and thus also the original, impedance data has been validated successfully.

.. plot::

   from pyimpspec import Resistor, DataSet, generate_mock_data
   from pyimpspec.analysis.kramers_kronig import evaluate_log_F_ext, suggest_num_RC
   from pyimpspec import mpl
   
   data = generate_mock_data("CIRCUIT_8", noise=5e-2, seed=42)[0]
   f = data.get_frequencies()
   Z_data = data.get_impedances()
   R = Resistor(R=50)
   Z_res = R.get_impedances(f)
   data = DataSet(frequencies=f, impedances=1/(1/Z_data + 1/Z_res), label=f"With parallel R={R.get_value('R'):.0f} $\Omega$")

   tests = evaluate_log_F_ext(data)[0][1]
   suggestion = suggest_num_RC(tests)

   figure, axes = mpl.plot_kramers_kronig_tests(tests, suggestion, data, legend=False, colored_axes=True)
   figure.tight_layout()


References:

- `Boukamp, B.A., 1995, J. Electrochem. Soc., 142, 1885-1894 <https://doi.org/10.1149/1.2044210>`_
- `Schönleber, M., Klotz, D., and Ivers-Tiffée, E., 2014, Electrochim. Acta, 131, 20-27 <https://doi.org/10.1016/j.electacta.2014.01.034>`_
- `Plank, C., Rüther, T., and Danzer, M.A., 2022, 2022 International Workshop on Impedance Spectroscopy (IWIS), 1-6 <https://doi.org/10.1109/IWIS57888.2022.9975131>`_
- `Yrjänä, V. and Bobacka, J., 2024, Electrochim. Acta, 504, 144951 <https://doi.org/10.1016/j.electacta.2024.144951>`_

.. raw:: latex

    \clearpage
