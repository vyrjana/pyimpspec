.. include:: ./substitutions.rst

Z-HIT analysis
==============

The `Z-HIT algorithm <https://en.wikipedia.org/wiki/Z-HIT>`_ (Ehm et al., 2000) reconstructs the modulus data of an immittance spectrum based on the phase data of that immittance spectrum.
This algorithm can be used to help validate immittance spectra.
Drifting at low frequencies and mutual induction at high frequencies may be detectable based on the results of the algorithm.

The algorithm is based on the following steps:

- Smoothing and interpolation of the phase data using, e.g., a `Savitzky-Golay filter <https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter>`_ and an `Akima spline <https://en.wikipedia.org/wiki/Akima_spline>`_, respectively.
- Approximation of the modulus data according to

  :math:`\ln{|Z(\omega_O)|} \approx \frac{2}{\pi} \int_{\omega_S}^{\omega_O} \varphi(\omega) \,{\rm d}\ln{\omega} + \gamma \frac{{\rm d}\varphi(\omega_O)}{{\rm d}\ln{\omega}} + C`

  where :math:`\omega_S` is the starting frequency, :math:`\omega_O` is the frequency of interest, :math:`\varphi` is the interpolated phase data, :math:`\gamma = \frac{\pi}{6}`, and :math:`C` is a constant.

- The constant :math:`C` is determined by fitting the approximated modulus data to a portion of the experimental modulus data.
  The frequency range from 1 Hz to 1000 Hz is a good starting point since it is typically less affected by drift and/or mutual induction.

.. note::

   The reconstruction of the modulus data is not likely to be perfect and there can be deviations even when analyzing ideal data.


How to use
----------

.. doctest::

   >>> from pyimpspec import DataSet, ZHITResult, generate_mock_data, perform_zhit
   >>>
   >>> data: DataSet = generate_mock_data("CIRCUIT_2_INVALID", noise=5e-2, seed=42)[0]
   >>> zhit: ZHITResult = perform_zhit(data)

Below is an example where simplified Randles circuits with or without drifting have been plotted as "Drifting" and "Valid", respectively.
The impedance spectrum ("Z-HIT") reconstructed from the phase data of the drifting spectrum has also been plotted and it is a close match to the impedance spectrum without drift.


.. plot::

   from pyimpspec import perform_zhit, generate_mock_data
   from pyimpspec import mpl

   valid = generate_mock_data("CIRCUIT_2", noise=5e-2, seed=42)[0]
   invalid = generate_mock_data("CIRCUIT_2_INVALID", noise=5e-2, seed=42)[0]
   zhit = perform_zhit(invalid)

   figure, axes = mpl.plot_bode(
     valid,
     legend=False,
     colors={"magnitude": "black", "phase": "black"},
     markers={"magnitude": "o", "phase": "s"},
   )
   mpl.plot_bode(
     invalid,
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
     li, la = ax.get_legend_handles_labels()
     lines.extend(li)
     labels.extend(la)

   axes[1].legend(lines, labels, loc=(0.03, 0.13))
   figure.tight_layout()


.. note::

   Pyimpspec's implementation of the algorithm also supports operating on the admittance representation of the immittance data, which can be done by setting ``admittance=True`` when calling |perform_zhit|.


References:

- Ehm, W., Göhr, H., Kaus, R., Röseler, B., and Schiller, C.A., 2000, Acta Chimica Hungarica, 137 (2-3), 145-157.
- Ehm, W., Kaus, R., Schiller, C.A., and Strunz, W., 2001, in "New Trends in Electrochemical Impedance Spectroscopy and Electrochemical Noise Analysis".
- Schiller, C.A., Richter, F., Gülzow, E., and Wagner, N., 2001, 3, 374-378 (https://doi.org/10.1039/B007678N)

.. raw:: latex

    \clearpage

