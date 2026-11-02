#!/usr/bin/env python
from pyimpspec import (
    fit_circuit,
    parse_cdc,
    generate_mock_data,
)
from pyimpspec.analysis.drt import (
    calculate_drt_bht,
)
from pyimpspec import mpl

def adjust_limits(ax):
    ax.set_xlim(1e-6, 1e0)
    ax.set_ylim(-500, 3500)

data = generate_mock_data(
    "CIRCUIT_5",
    noise=5e-2,
    seed=42,
)[0]

drt = calculate_drt_bht(data)
figure, axes = mpl.plot_gamma(drt)
adjust_limits(axes[0])
figure.tight_layout()
