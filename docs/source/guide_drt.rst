.. include:: ./substitutions.rst

Distribution of relaxation times
================================

Calculating the distribution of relaxation times (DRT) is one way of analyzing impedance spectra.
Several different approaches have been described (see `Boukamp (2020) <https://doi.org/10.1088/2515-7655/aba9e0>`_ for a review).
The DRT results can be used to, e.g., develop a suitable equivalent circuit.

.. note::

   Validation of impedance spectra using, e.g., Kramers-Kronig tests prior to proceeding with DRT analyses is highly recommended.
   The instrument control software for your potentiostat/galvanostat may include tools for analyzing the excitation and response signals for indications of non-linear behavior.


The supported methods
---------------------

Implementations based on the following approaches are included in pyimpspec:

- **BHT**: The Bayesian Hilbert transform method (see `Liu et al. (2020) <https://doi.org/10.1016/j.electacta.2020.136864>`_) that was originally implemented in `DRTtools <https://github.com/ciuccislab/DRTtools>`_ and `pyDRTtools <https://github.com/ciuccislab/pyDRTtools>`_.
- **LM**: The Loewner method (see `Mayo and Antoulas (2007) <https://doi.org/10.1016/j.laa.2007.03.008>`_, `Sorrentino et al. (2022) <http://doi.org/10.2139/ssrn.4217752>`_, `Rüther et al. (2023) <https://doi.org/10.3390/batteries9020132>`_, and `Sorrentino et al. (2023) <https://doi.org/10.1016/j.jpowsour.2023.233575>`_) that was implemented in `DRT-from-Loewner-framework <https://github.com/projectsEECandDRI/DRT-from-Loewner-framework>`_.
- **m(RQ)fit**: The multi-(RQ)-fit method (see `Boukamp (2015) <https://doi.org/10.1016/j.electacta.2014.12.059>`_ and `Boukamp and Rolle (2017) <https://doi.org/10.1016/j.ssi.2016.10.009>`_).
- **TR-NNLS**: Tikhonov regularization and non-negative least squares (see `Kulikovsky (2021) <https://doi.org/10.1149/1945-7111/abf508>`_) that was originally implemented in `DRT-python-code <https://github.com/akulikovsky/DRT-python-code>`_.
- **TR-RBF**: Tikhonov regularization and radial basis function (or piecewise linear) discretization (see `Wan et al. (2015) <https://doi.org/10.1016/j.electacta.2015.09.097>`_, `Ciucci and Chen (2015) <https://doi.org/10.1016/j.electacta.2015.03.123>`_, `Effat and Ciucci (2017) <https://doi.org/10.1016/j.electacta.2017.07.050>`_, and `Maradesa et al. (2023) <https://doi.org/10.1149/1945-7111/acbca4>`_) that was originally implemented in DRTtools_ and pyDRTtools_.


.. note::

  The methods using Tikhonov regularization require selection of a suitable regularization parameter, |lambda|.
  The implementations of the TR-NNLS and TR-RBF methods in pyimpspec include approaches to automatically selecting this parameter.
  However, this is not guaranteed to work in all cases.
  An unsuitable regularization parameter can lead to, e.g., sharp peaks that should not be there or broad peaks that should be two or more separate sharper peaks.

.. note::

   The BHT method makes use of random initial values for some of its calculations, which can produce different results when repeated with the same impedance spectrum.


How to use
----------

Each method has its own function that can be used but there is also a wrapper function (|calculate_drt|) that takes a ``method`` argument.

.. doctest::

   >>> from pyimpspec import (
   ...   DataSet,
   ...   DRTResult,              # An abstract class for DRT results
   ...   calculate_drt,          # Wrapper function for all methods
   ...   generate_mock_data,
   ... )
   >>> from pyimpspec.analysis.drt import (
   ...   BHTResult,              # Result of the BHT method
   ...   LMResult,               # Result of the Loewner method
   ...   MRQFitResult,           # Result of the m(RQ)fit method
   ...   TRNNLSResult,           # Result of the TR-NNLS method
   ...   TRRBFResult,            # Result of the TR-RBF method
   ...   calculate_drt_bht,      # BHT method
   ...   calculate_drt_lm,       # Loewner method
   ...   calculate_drt_mrq_fit,  # m(RQ)fit method
   ...   calculate_drt_tr_nnls,  # TR-NNLS method
   ...   calculate_drt_tr_rbf,   # TR-RBF method
   ... )
   >>>
   >>> data: DataSet = generate_mock_data("CIRCUIT_1", noise=5e-2, seed=42)[0]
   >>> drt: TRNNLSResult = calculate_drt_tr_nnls(data, lambda_value=1e-4)
   >>> drt: DRTResult = calculate_drt(data, method="tr-nnls", lambda_value=1e-4)
   >>> assert isinstance(drt, TRNNLSResult)

Below are some figures demonstrating the results of the methods listed above when applied to the example data.


.. plot:: example_drt_data.py
   :caption: An impedance spectrum corresponding to a ``R(RQ)(RQ)(RQ)`` circuit.


.. plot:: example_drt_bht.py
   :caption: The distribution of relaxation times obtained with the Bayesian Hilbert transform method.


.. plot:: example_drt_lm.py
   :caption: The distribution of relaxation times obtained with the Loewner method.


.. plot:: example_drt_mRQfit.py
   :caption: The distribution of relaxation times obtained with the m(RQ)-fit method.


.. plot:: example_drt_tr_nnls.py
   :caption: The distribution of relaxation times obtained with the Tikhonov-regularized non-negative least squares method.


.. plot:: example_drt_tr_rbf.py
   :caption: The distribution of relaxation times obtained with the Tikhonov-regularized radial basis function discretization method.



.. raw:: latex

    \clearpage



.. plot:: example_drt_lm_analysis_a.py
   :caption: The equivalent circuit model used to simulate the example data (:math:`R_1 = 140\ \Omega`, :math:`R_2 = 230\ \Omega`, :math:`R_3 = 576\ \Omega`, :math:`R_4 = 150\ \Omega`, :math:`C_1 = 10^{-6}\ {\rm F}`, :math:`C_2 = 10^{-4}\ {\rm F}`, and :math:`L_1 = 40\ {\rm H}`).


.. plot:: example_drt_lm_analysis_b.py
   :caption: Nyquist plot of the example data and the impedance reconstructed based on the Loewner method's result.



.. plot:: example_drt_lm_analysis_c.py
   :caption: The distribution of relaxation times obtained with the Loewner method without any noise added to the example data.


.. literalinclude:: example_drt_lm_analysis_c.txt


This example shows that some additional processing of the obtained values may be necessary.
In this case, the high-frequency resistive-capacitive peak does not directly tell us the value of :math:`R_1` and instead it is :math:`R_1 + R_4`.
However, we can still obtain an estimated value: :math:`R_1 \approx 289.999\ \Omega - 150.012\ \Omega \approx 140\ \Omega`.
Other estimated values such as capacitances and inductances can also be obtained:

- :math:`R_2 \approx 230\ \Omega` and :math:`C_1 \approx 0.00023\ {\rm s} / 230\ \Omega \approx 10^{-6}\ {\rm F}`
- :math:`R_3 \approx 576\ \Omega` and :math:`C_2 \approx 0.0576\ {\rm s} / 576.007\ \Omega \approx 10^{-4}\ {\rm F}`
- :math:`R_4 \approx 150\ \Omega` and :math:`L_2 \approx 0.266667\ {\rm s} \times 150.012\ \Omega \approx 40\ {\rm H}`

The above is a best-case scenario, but even with a bit of added noise (:math:`\sigma = 0.05\ \% \times |Z|`) one can still obtain decent estimates despite the additional peaks:

- :math:`R1 \approx 289.925\ \Omega - 149.085\ \Omega \approx 141\ \Omega`.
- :math:`R_2 \approx 230\ \Omega` and :math:`C_1 \approx 0.000230559\ {\rm s} / 230.233\ \Omega \approx 10^{-6}\ {\rm F}`
- :math:`R_3 \approx 574\ \Omega` and :math:`C_2 \approx 0.0574022\ {\rm s} / 574.438\ \Omega \approx 10^{-4}\ {\rm F}`
- :math:`R_4 \approx 149\ \Omega` and :math:`L_2 \approx 0.271608\ {\rm s} \times 149.085\ \Omega \approx 40\ {\rm H}`


.. plot:: example_drt_lm_analysis_d.py
   :caption: The distribution of relaxation times obtained with the Loewner method when noise has been added to the example data.


.. literalinclude:: example_drt_lm_analysis_d.txt


.. raw:: latex

    \clearpage


Peak analysis
-------------

DRT results (aside from those obtained using the Loewner method) can be analyzed using skew normal distributions (see `Danzer (2019) <https://doi.org/10.3390/batteries5030053>`_ and `Plank et al. (2024) <https://doi.org/10.1016/j.jpowsour.2023.233845>`_).
This facilitates the estimation of the polarization contribution of each peak based on the area of that peak.


.. plot:: example_drt_peak_analysis_a.py
   :caption: The equivalent circuit model used to simulate the example data (:math:`R_1 = 100\ \Omega`, :math:`R_2 = 300\ \Omega`, :math:`R_3 = 450\ \Omega`, :math:`C_1 = 5 \times 10^{-6}\ {\rm F}`, and :math:`C_2 = 10^{-5}\ {\rm F}`).


.. plot:: example_drt_peak_analysis_b.py
   :caption: The example data a small amount of added noise and the impedance reconstructed from the TR-RBF method result.


.. plot:: example_drt_peak_analysis_c.py
   :caption: The distribution of relaxation times obtained with the TR-RBF method.



.. plot:: example_drt_peak_analysis_d.py
   :caption: The results of the peak analysis applied to the distribution of relaxation times obtained with the TR-RBF method.


.. literalinclude:: example_drt_peak_analysis_d.txt


.. raw:: latex

    \clearpage


References:

- `Boukamp, B.A. and Rolle, A, 2017, Solid State Ionics, 302, 12-18 <https://doi.org/10.1016/j.ssi.2016.10.009>`_
- `Boukamp, B.A., 2015, Electrochim. Acta, 154, 35-46 <https://doi.org/10.1016/j.electacta.2014.12.059>`_
- `Ciucci, F. and Chen, C., 2015, Electrochim. Acta, 167, 439-454 <https://doi.org/10.1016/j.electacta.2015.03.123>`_
- `Danzer, M.A., 2019, Batteries, 5, 53 <https://doi.org/10.3390/batteries5030053>`_
- `Effat, M. B. and Ciucci, F., 2017, Electrochim. Acta, 247, 1117-1129 <https://doi.org/10.1016/j.electacta.2017.07.050>`_
- `Kulikovsky, A., 2021, J. Electrochem. Soc., 168, 044512 <https://doi.org/10.1149/1945-7111/abf508>`_
- `Liu, J., Wan, T. H., and Ciucci, F., 2020, Electrochim. Acta, 357, 136864 <https://doi.org/10.1016/j.electacta.2020.136864>`_
- `Maradesa, A., Py, B., Wan, T.H., Effat, M.B., and Ciucci F., 2023, J. Electrochem. Soc, 170, 030502 <https://doi.org/10.1149/1945-7111/acbca4>`_
- `Mayo, A.J. and Antoulas, A.C., 2007, Linear Algebra Its Appl. 425, 634–662 <https://doi.org/10.1016/j.laa.2007.03.008>`_,
- `Plank, C., Rüther, T., Jahn, L., Schamel, M., Schmidt, J.P., Ciucci, F., and Danzer, M.A., 2024, Journal of Power Sources, 594, 233845 <https://doi.org/10.1016/j.jpowsour.2023.233845>`_
- `Rüther, T., Gosea, I.V., Jahn, L., Antoulas, A.C., and Danzer, M.A., 2023, Batteries, 9, 132 <https://doi.org/10.3390/batteries9020132>`_
- `Sorrentino, A., Patel, B., Gosea, I.V., Antoulas, A.C., and Vidaković-Koch, T., 2022, SSRN <http://doi.org/10.2139/ssrn.4217752>`_
- `Sorrentino, A., Patel, B., Gosea, I.V., Antoulas, A.C., and Vidaković-Koch, T., 2023, J. Power Sources, 585, 223575 <https://doi.org/10.1016/j.jpowsour.2023.233575>`_
- `Wan, T. H., Saccoccio, M., Chen, C., and Ciucci, F., 2015, Electrochim. Acta, 184, 483-499 <https://doi.org/10.1016/j.electacta.2015.09.097>`_

.. raw:: latex

    \clearpage
