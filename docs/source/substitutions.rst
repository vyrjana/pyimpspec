.. |alpha| replace:: :math:`\alpha`
.. |beta| replace:: :math:`\beta`
.. |mu| replace:: :math:`\mu`
.. |mu crit| replace:: :math:`\mu_{\rm crit}`
.. |kappa| replace:: :math:`\kappa`
.. |lambda| replace:: :math:`\lambda`
.. |chi-squared| replace:: :math:`\chi^2`
.. |pseudo chi-squared| replace:: :math:`\chi^2_{\rm ps}`
.. |log pseudo chi-squared| replace:: :math:`\log{\chi^2_{\rm ps}}`
.. |N_tau| replace:: :math:`N_\tau`
.. |N_tauopt| replace:: :math:`N_{\tau\rm,opt}`
.. |N_taumin| replace:: :math:`N_{\tau\rm,min}`
.. |N_taumax| replace:: :math:`N_{\tau\rm,max}`
.. |F_ext| replace:: :math:`F_{\rm ext}`
.. |log F_ext| replace:: :math:`\log{F_{\rm ext}}`
.. |ohm| replace:: :math:`\Omega`
.. |log sum abs tau R| replace:: :math:`\log{\Sigma_{k=1}^{N_\tau} |\tau_k / R_k|}`
.. |log sum abs tau C| replace:: :math:`\log{\Sigma_{k=1}^{N_\tau} |\tau_k / C_k|}`

.. classes
   || replace:: :class:`~pyimpspec.`
.. |BHTResult| replace:: :class:`~pyimpspec.BHTResult`
.. |Capacitor| replace:: :class:`~pyimpspec.circuit.elements.Capacitor`
.. |CircuitBuilder| replace:: :class:`~pyimpspec.CircuitBuilder`
.. |Circuit| replace:: :class:`~pyimpspec.Circuit`
.. |Connection| replace:: :class:`~pyimpspec.Connection`
.. |ContainerDefinition| replace:: :class:`~pyimpspec.ContainerDefinition`
.. |Container| replace:: :class:`~pyimpspec.Container`
.. |DRTResult| replace:: :class:`~pyimpspec.DRTResult`
.. |DRTPeaks| replace:: :class:`~pyimpspec.DRTPeaks`
.. |DRTPeak| replace:: :class:`~pyimpspec.DRTPeak`
.. |DataSet| replace:: :class:`~pyimpspec.DataSet`
.. |ElementDefinition| replace:: :class:`~pyimpspec.ElementDefinition`
.. |Element| replace:: :class:`~pyimpspec.Element`
.. |FitIdentifiers| replace:: :class:`~pyimpspec.FitIdentifiers`
.. |FitResult| replace:: :class:`~pyimpspec.FitResult`
.. |FittedParameter| replace:: :class:`~pyimpspec.FittedParameter`
.. |KramersKronigResult| replace:: :class:`~pyimpspec.KramersKronigResult`
.. |LMResult| replace:: :class:`~pyimpspec.LMResult`
.. |MRQFitResult| replace:: :class:`~pyimpspec.MRQFitResult`
.. |ParameterDefinition| replace:: :class:`~pyimpspec.ParameterDefinition`
.. |Resistor| replace:: :class:`~pyimpspec.circuit.elements.Resistor`
.. |Series| replace:: :class:`~pyimpspec.Series`
.. |SubcircuitDefinition| replace:: :class:`~pyimpspec.SubcircuitDefinition`
.. |TRNNLSResult| replace:: :class:`~pyimpspec.TRNNLSResult`
.. |TRRBFResult| replace:: :class:`~pyimpspec.TRRBFResult`
.. |TransmissionLineModel| replace:: :class:`~pyimpspec.circuit.elements.TransmissionLineModel`
.. |ZHITResult| replace:: :class:`~pyimpspec.ZHITResult`

.. type hints
.. |ComplexImpedances| replace:: :class:`~pyimpspec.ComplexImpedances`
.. |ComplexImpedance| replace:: :class:`~pyimpspec.ComplexImpedance`
.. |ComplexResiduals| replace:: :class:`~pyimpspec.ComplexResiduals`
.. |ComplexResidual| replace:: :class:`~pyimpspec.ComplexResidual`
.. |Frequencies| replace:: :class:`~pyimpspec.Frequencies`
.. |Frequency| replace:: :class:`~pyimpspec.Frequency`
.. |Gammas| replace:: :class:`~pyimpspec.Gammas`
.. |Gamma| replace:: :class:`~pyimpspec.Gamma`
.. |Impedances| replace:: :class:`~pyimpspec.Impedances`
.. |Impedance| replace:: :class:`~pyimpspec.Impedance`
.. |Indices| replace:: :class:`~pyimpspec.Indices`
.. |Phases| replace:: :class:`~pyimpspec.Phases`
.. |Phase| replace:: :class:`~pyimpspec.Phase`
.. |Residuals| replace:: :class:`~pyimpspec.Residuals`
.. |Residual| replace:: :class:`~pyimpspec.Residual`
.. |TimeConstants| replace:: :class:`~pyimpspec.TimeConstants`
.. |TimeConstant| replace:: :class:`~pyimpspec.TimeConstant`

.. methods
   || replace:: :func:`~pyimpspec.`
.. |Circuit.to_circuitikz| replace:: :func:`~pyimpspec.Circuit.to_circuitikz`
.. |Circuit.to_drawing| replace:: :func:`~pyimpspec.Circuit.to_drawing`
.. |DataSet.get_frequencies| replace:: :func:`~pyimpspec.DataSet.get_frequencies`
.. |DataSet.get_phases| replace:: :func:`~pyimpspec.DataSet.get_phases`
.. |Element.get_impedances| replace:: :func:`~pyimpspec.Element.get_impedances`

.. functions
   || replace:: :func:`~pyimpspec.`
.. |calculate_drt| replace:: :func:`~pyimpspec.calculate_drt`
.. |dataframe_to_data_sets| replace:: :func:`~pyimpspec.dataframe_to_data_sets`
.. |evaluate_log_F_ext| replace:: :func:`~pyimpspec.analysis.kramers_kronig.evaluate_log_F_ext`
.. |fit_circuit| replace:: :func:`~pyimpspec.fit_circuit`
.. |generate_fit_identifiers| replace:: :func:`~pyimpspec.generate_fit_identifiers`
.. |generate_mock_circuits| replace:: :func:`~pyimpspec.mock_data.generate_mock_circuits`
.. |generate_mock_data| replace:: :func:`~pyimpspec.mock_data.generate_mock_data`
.. |get_default_num_procs| replace:: :func:`~pyimpspec.get_default_num_procs`
.. |parse_cdc| replace:: :func:`~pyimpspec.parse_cdc`
.. |parse_data| replace:: :func:`~pyimpspec.parse_data`
.. |perform_exploratory_kramers_kronig_tests| replace:: :func:`~pyimpspec.perform_exploratory_kramers_kronig_tests`
.. |perform_kramers_kronig_test| replace:: :func:`~pyimpspec.perform_kramers_kronig_test`
.. |perform_zhit| replace:: :func:`~pyimpspec.perform_zhit`
.. |plot_circuit| replace:: :func:`~pyimpspec.plot.mpl.plot_circuit`
.. |register_element| replace:: :func:`~pyimpspec.register_element`
.. |set_default_num_procs| replace:: :func:`~pyimpspec.set_default_num_procs`
.. |simulate_spectrum| replace:: :func:`~pyimpspec.simulate_spectrum`
.. |suggest_num_RC_limits| replace:: :func:`~pyimpspec.analysis.kramers_kronig.suggest_num_RC_limits`
.. |suggest_num_RC| replace:: :func:`~pyimpspec.analysis.kramers_kronig.suggest_num_RC`
.. |suggest_representation| replace:: :func:`~pyimpspec.analysis.kramers_kronig.suggest_representation`

.. links
.. _circuitikz: https://github.com/circuitikz/circuitikz
.. _github: https://github.com/vyrjana/pyimpspec
.. _gplv3: https://www.gnu.org/licenses/gpl-3.0.en.html
.. _kramers-kronig: https://en.wikipedia.org/wiki/Kramers%E2%80%93Kronig_relations
.. _lin-kk tool: https://www.iam.kit.edu/et/english/Lin-KK.php
.. _lmfit.MinimizerResult: https://lmfit.github.io/lmfit-py/fitting.html#lmfit.minimizer.MinimizerResult
.. _lmfit.minimize: https://lmfit.github.io/lmfit-py/fitting.html#lmfit.minimizer.minimize
.. _lmfit: https://lmfit.github.io/lmfit-py/index.html
.. _matplotlib: https://matplotlib.org
.. _numpy.inf: https://numpy.org/doc/stable/reference/constants.html#numpy.inf
.. _numpy.nan: https://numpy.org/doc/stable/reference/constants.html#numpy.nan
.. _pandas.dataframe.to_latex: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_latex.html#pandas.DataFrame.to_latex
.. _pandas.dataframe.to_markdown: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_markdown.html#pandas.DataFrame.to_markdown
.. _pandas.dataframe: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html#pandas.DataFrame
.. _randles: https://en.wikipedia.org/wiki/Randles_circuit
.. _schemdraw.drawing: https://schemdraw.readthedocs.io/en/latest/classes/drawing.html#schemdraw.Drawing
.. _schemdraw: https://schemdraw.readthedocs.io/en/latest/
.. _scipy.signal.savgol_filter: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html
.. _scipy.optimize.nnls: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.nnls.html
.. _statsmodels.nonparametric.smoothers_lowess.lowess: https://www.statsmodels.org/dev/generated/statsmodels.nonparametric.smoothers_lowess.lowess.html
.. _sympy: https://www.sympy.org/en/index.html
