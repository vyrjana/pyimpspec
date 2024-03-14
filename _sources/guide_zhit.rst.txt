.. include:: ./substitutions.rst

Z-HIT analysis
==============

The `Z-HIT algorithm <https://en.wikipedia.org/wiki/Z-HIT>`_ (Ehm et al., 2000) reconstructs the modulus data of an impedance spectrum based on the phase data of that impedance spectrum.
This algorithm can be used to, e.g., validate impedance spectra.
Drifting at low frequencies and mutual induction at high frequencies may be detectable based on the results of the algorithm.

The algorith is based on the following steps:

- Smoothing and interpolation of the phase data using, e.g., `LOWESS <https://en.wikipedia.org/wiki/Local_regression>`_ and an `Akima spline <https://en.wikipedia.org/wiki/Akima_spline>`_, respectively.
- Approximation of the modulus data according to

  :math:`\ln{|Z(\omega_O)|} \approx \frac{2}{\pi} \int_{\omega_S}^{\omega_O} \varphi(\omega) \,{\rm d}\ln{\omega} + \gamma \frac{{\rm d}\varphi(\omega_O)}{{\rm d}\ln{\omega}} + C`

  where :math:`\omega_S` is the starting frequency, :math:`\omega_O` is the frequency of interest, :math:`\varphi` is the interpolated phase data, :math:`\gamma = \frac{\pi}{6}`, and :math:`C` is a constant.

- The constant :math:`C` is determined by fitting the approximated modulus data to a portion of the experimental modulus data.
  The frequency range from 1 Hz to 1000 Hz is typically less affected by drift or mutual induction.

.. note::

   The reconstruction of the modulus data is not perfect and there can be minor deviations even when analyzing ideal data.


.. doctest::

   >>> from pyimpspec import ZHITResult, perform_zhit
   >>> from pyimpspec.mock_data import DRIFTING_RANDLES
   >>>
   >>> zhit: ZHITResult = perform_zhit(DRIFTING_RANDLES)

Below is an example where simplified Randles circuits with or without drifting have been plotted as "Drifting" and "Valid", respectively.
The reconstructed impedance spectrum ("Z-HIT") has also been plotted and it is a close match to the impedance spectrum of the circuit without drift.


.. plot::

   from pyimpspec import perform_zhit
   from pyimpspec import mpl
   from pyimpspec.mock_data import DRIFTING_RANDLES, VALID_RANDLES
   zhit = perform_zhit(DRIFTING_RANDLES)
   figure, axes = mpl.plot_bode(
     VALID_RANDLES,
     legend=False,
     colors={"magnitude": "black", "phase": "black"},
     markers={"magnitude": "o", "phase": "s"},
   )
   mpl.plot_bode(
     DRIFTING_RANDLES,
     legend=False,
     colors={"magnitude": "black", "phase": "black"},
     markers={"magnitude": "x", "phase": "+"},
     figure=figure,
     axes=axes,
   )
   mpl.plot_bode(
     zhit,
     line=True,
     legend=False,
     figure=figure,
     axes=axes,
   )
   lines = []
   labels = []
   for ax in axes:
     lin, lab = ax.get_legend_handles_labels()
     lines.extend(lin)
     labels.extend(lab)
   axes[1].legend(lines, labels, loc=(0.03, 0.13))
   figure.tight_layout()


References:

- Ehm, W., Göhr, H., Kaus, R., Röseler, B., and Schiller, C.A., 2000, Acta Chimica Hungarica, 137 (2-3), 145-157.
- Ehm, W., Kaus, R., Schiller, C.A., and Strunz, W., 2001, in "New Trends in Electrochemical Impedance Spectroscopy and Electrochemical Noise Analysis".
- Schiller, C.A., Richter, F., Gülzow, E., and Wagner, N., 2001, 3, 374-378 (https://doi.org/10.1039/B007678N)

.. raw:: latex

    \clearpage

