#!/usr/bin/env python
from pyimpspec import (
    fit_circuit,
    parse_cdc,
    generate_mock_data,
)
from pyimpspec.analysis.drt import (
    calculate_drt_lm,
)
from pyimpspec import mpl

data = generate_mock_data(
    "CIRCUIT_5",
    noise=5e-2,
    seed=42,
)[0]

drt = calculate_drt_lm(data)
figure, axes = mpl.plot_gamma(drt)
figure.tight_layout()
