.. include:: ./substitutions.rst

Distribution of relaxation times
================================

Calculating the distribution of relaxation times (DRT) is one way of analyzing impedance spectra.
Several different approaches have been described (see `Boukamp (2020) <https://doi.org/10.1088/2515-7655/aba9e0>`_ for a review).
The DRT results can be used to, e.g., develop a suitable equivalent circuit.

.. note::

   Validation of impedance spectra using, e.g., Kramers-Kronig tests prior to proceeding with DRT analyses is highly recommended.
   The instrument control software for your potentiostat/galvanostat may include tools for analyzing the excitation and response signals for indications of non-linear behavior.

Implementations based on the following approaches are included in pyimpspec:

- **BHT**: The Bayesian Hilbert transform method (see `Liu et al. (2020) <https://doi.org/10.1016/j.electacta.2020.136864>`_) that was originally implemented in DRTtools_ and pyDRTtools_.
- **m(RQ)fit**: The multi-(RQ)-fit method (see `Boukamp (2015) <https://doi.org/10.1016/j.electacta.2014.12.059>`_ and `Boukamp and Rolle (2017) <https://doi.org/10.1016/j.ssi.2016.10.009>`_).
- **TR-NNLS**: Tikhonov regularization and non-negative least squares (see `Kulikovsky (2021) <https://doi.org/10.1149/1945-7111/abf508>`_) that was originally implemented in `DRT-python-code <https://github.com/akulikovsky/DRT-python-code>`_.
- **TR-RBF**: Tikhonov regularization and radial basis function (or piecewise linear) discretization (see `Wan et al. (2015) <https://doi.org/10.1016/j.electacta.2015.09.097>`_, `Ciucci and Chen (2015) <https://doi.org/10.1016/j.electacta.2015.03.123>`_, and `Effat and Ciucci (2017) <https://doi.org/10.1016/j.electacta.2017.07.050>`_) that was originally implemented in DRTtools_ and pyDRTtools_.

.. _DRTtools: https://github.com/ciuccislab/DRTtools
.. _pyDRTtools: https://github.com/ciuccislab/pyDRTtools

.. note::

  The methods using Tikhonov regularization require selection of a suitable regularization parameter, |lambda|.
  The implementations of the TR-NNLS and TR-RBF methods in pyimpspec include approaches to automatically selecting this parameter.
  However, this is not guaranteed to work in all cases.
  An unsuitable regularization parameter can lead to, e.g., sharp peaks that should not be there or broad peaks that should be two or more separate sharper peaks.

.. note::

   The BHT method makes use of random initial values for some of its calculations, which can produce different results when repeated with the same impedance spectrum.

.. doctest::

   >>> from pyimpspec import (
   ...   DRTResult,              # An abstract class for DRT results
   ...   BHTResult,              # Result of the BHT method
   ...   MRQFitResult,           # Result of the m(RQ)fit method
   ...   TRNNLSResult,           # Result of the TR-NNLS method
   ...   TRRBFResult,            # Result of the TR-RBF method
   ...   calculate_drt,          # Wrapper function for all methods
   ...   calculate_drt_bht,      # BHT method
   ...   calculate_drt_mrq_fit,  # m(RQ)fit method
   ...   calculate_drt_tr_nnls,  # TR-NNLS method
   ...   calculate_drt_tr_rbf,   # TR-RBF method
   ... )
   >>> from pyimpspec.mock_data import EXAMPLE
   >>>
   >>> drt: TRNNLSResult = calculate_drt_tr_nnls(EXAMPLE, lambda_value=1e-4)
   >>> drt: DRTResult = calculate_drt(EXAMPLE, method="tr-nnls", lambda_value=1e-4)
   >>> assert isinstance(drt, TRNNLSResult)

Below are some figures demonstrating the results of the methods listed above when applied to the example data.

.. plot::

   from pyimpspec import (
     calculate_drt_bht,
     calculate_drt_mrq_fit,
     calculate_drt_tr_nnls,
     calculate_drt_tr_rbf,
     fit_circuit,
     parse_cdc,
   )
   from pyimpspec import mpl
   from pyimpspec.mock_data import EXAMPLE

   def adjust_limits(ax):
     ax.set_xlim(1e-5, 1e1)
     ax.set_ylim(-100, 900)
   
   drt = calculate_drt_bht(EXAMPLE)
   figure, axes = mpl.plot_gamma(drt)
   adjust_limits(axes[0])
   figure.tight_layout()

   circuit = parse_cdc("R(RQ)(RQ)")
   fit = fit_circuit(circuit, EXAMPLE)
   drt = calculate_drt_mrq_fit(EXAMPLE, fit.circuit, fit=fit)
   figure, axes = mpl.plot_gamma(drt)
   adjust_limits(axes[0])
   figure.tight_layout()

   drt = calculate_drt_tr_nnls(EXAMPLE)
   figure, axes = mpl.plot_gamma(drt)
   adjust_limits(axes[0])
   figure.tight_layout()

   drt = calculate_drt_tr_rbf(EXAMPLE)
   figure, axes = mpl.plot_gamma(drt)
   adjust_limits(axes[0])
   figure.tight_layout()


.. raw:: latex

    \clearpage


References:

- Boukamp, B.A., 2015, Electrochim. Acta, 154, 35-46, (https://doi.org/10.1016/j.electacta.2014.12.059)
- Boukamp, B.A. and Rolle, A, 2017, Solid State Ionics, 302, 12-18 (https://doi.org/10.1016/j.ssi.2016.10.009)
- Ciucci, F. and Chen, C., 2015, Electrochim. Acta, 167, 439-454 (https://doi.org/10.1016/j.electacta.2015.03.123)
- Effat, M. B. and Ciucci, F., 2017, Electrochim. Acta, 247, 1117-1129 (https://doi.org/10.1016/j.electacta.2017.07.050)
- Kulikovsky, A., 2021, J. Electrochem. Soc., 168, 044512 (https://doi.org/10.1149/1945-7111/abf508)
- Liu, J., Wan, T. H., and Ciucci, F., 2020, Electrochim. Acta, 357, 136864 (https://doi.org/10.1016/j.electacta.2020.136864)
- Wan, T. H., Saccoccio, M., Chen, C., and Ciucci, F., 2015, Electrochim. Acta, 184, 483-499 (https://doi.org/10.1016/j.electacta.2015.09.097)

.. raw:: latex

    \clearpage
