from pyimpspec import (
  Circuit,
  DataSet,
  Element,
  FitResult,
  fit_circuit,
  generate_mock_data,
  mpl,
  parse_cdc,
)
from typing import (
  Dict,
  List,
)

data: DataSet = generate_mock_data("CIRCUIT_5", noise=5e-2, seed=42)[0]

circuit: Circuit = parse_cdc("R(RQ)(RQ)(RQ)")

fit: FitResult = fit_circuit(
  circuit,
  data,
  method="least_squares",
  weight="boukamp",
)
print(fit.to_parameters_dataframe().to_markdown(index=False))
print(fit.to_statistics_dataframe().to_markdown(index=False))

figure, axes = mpl.plot_fit(
  fit,
  data=data,
  colored_axes=True,
  legend=False,
  title="Without constraints",
)
figure.tight_layout()