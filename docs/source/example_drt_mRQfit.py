#!/usr/bin/env python
from pyimpspec import (
    fit_circuit,
    parse_cdc,
    generate_mock_data,
)
from pyimpspec.analysis.drt import (
    calculate_drt_mrq_fit,
)
from pyimpspec import mpl

def adjust_limits(ax):
    ax.set_xlim(1e-6, 1e0)
    ax.set_ylim(-100, 1200)

data = generate_mock_data(
    "CIRCUIT_5",
    noise=5e-2,
    seed=42,
)[0]

circuit = parse_cdc("R(RQ)(RQ)(RQ)")
fit = fit_circuit(circuit, data)
drt = calculate_drt_mrq_fit(
    data,
    fit.circuit,
    fit=fit,
)
figure, axes = mpl.plot_gamma(drt)
adjust_limits(axes[0])
figure.tight_layout()
