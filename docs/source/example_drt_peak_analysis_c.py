#!/usr/bin/env python
from pyimpspec import (
    generate_mock_data,
)
from pyimpspec.analysis.drt import calculate_drt_tr_rbf
from pyimpspec import mpl

cdc = "R{R=100}(R{R=300}C{C=5e-6})(R{R=450}C{C=1e-5})"

data = generate_mock_data(
    cdc,
    noise=5e-2,
    seed=42,
)[0]
drt = calculate_drt_tr_rbf(data)

figure, axes = mpl.plot_gamma(drt)
figure.tight_layout()
