from pyimpspec import (
  FitResult,
  fit_circuit,
  parse_cdc,
  generate_mock_data,
)
from pyimpspec import mpl
data = generate_mock_data("CIRCUIT_1", noise=5e-2, seed=42)[0]
circuit = parse_cdc("R(RC)(RW)")
fit = fit_circuit(circuit, data)

figure, axes = mpl.plot_nyquist(data, colors={"impedance": "black"})
mpl.plot_nyquist(fit, line=True, figure=figure, axes=axes)
figure.tight_layout()

figure, axes = mpl.plot_residuals(fit)
figure.tight_layout()