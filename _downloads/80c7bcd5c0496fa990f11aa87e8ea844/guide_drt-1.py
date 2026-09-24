from pyimpspec import (
  fit_circuit,
  parse_cdc,
  generate_mock_data,
)
from pyimpspec.analysis.drt import (
  calculate_drt_bht,
  calculate_drt_lm,
  calculate_drt_mrq_fit,
  calculate_drt_tr_nnls,
  calculate_drt_tr_rbf,
)
from pyimpspec import mpl

def adjust_limits(ax):
  ax.set_xlim(1e-5, 1e1)
  ax.set_ylim(-100, 900)

data = generate_mock_data("CIRCUIT_1", noise=5e-2, seed=42)[0]
figure, axes = mpl.plot_nyquist(data, colors=dict(impedance="black"), markers=dict(impedance="o"))
figure.tight_layout()

drt = calculate_drt_bht(data)
figure, axes = mpl.plot_gamma(drt)
adjust_limits(axes[0])
figure.tight_layout()

drt = calculate_drt_lm(data)
figure, axes = mpl.plot_gamma(drt)
figure.tight_layout()

circuit = parse_cdc("R(RQ)(RQ)")
fit = fit_circuit(circuit, data)
drt = calculate_drt_mrq_fit(data, fit.circuit, fit=fit)
figure, axes = mpl.plot_gamma(drt)
adjust_limits(axes[0])
figure.tight_layout()

drt = calculate_drt_tr_nnls(data)
figure, axes = mpl.plot_gamma(drt)
adjust_limits(axes[0])
figure.tight_layout()

drt = calculate_drt_tr_rbf(data)
figure, axes = mpl.plot_gamma(drt)
adjust_limits(axes[0])
figure.tight_layout()