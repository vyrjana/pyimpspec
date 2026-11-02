#!/usr/bin/env python
from pyimpspec import (
    fit_circuit,
    parse_cdc,
    generate_mock_data,
)
from pyimpspec.analysis.drt import (
    calculate_drt_tr_nnls,
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

drt = calculate_drt_tr_nnls(data)
figure, axes = mpl.plot_gamma(
    drt,
    label=r"$\text{TR-NNLS},\ \lambda = " + f"{drt.lambda_value:.2g}" + r"$",
)
adjust_limits(axes[0])
figure.tight_layout()
