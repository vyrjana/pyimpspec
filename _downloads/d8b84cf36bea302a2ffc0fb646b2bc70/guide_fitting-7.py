from pyimpspec import (
  Circuit,
  DataSet,
  Element,
  FitIdentifiers,
  FitResult,
  fit_circuit,
  generate_fit_identifiers,
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
identifiers: Dict[Element, FitIdentifiers] = generate_fit_identifiers(circuit)
elements: List[Element] = circuit.get_elements()
R1, R2, Q1, R3, Q2, R4, Q3 = elements

fit: FitResult = fit_circuit(
  circuit,
  data,
  method="least_squares",
  weight="boukamp",
  constraint_expressions={
    identifiers[R3].R: f"{identifiers[R2].R} + alpha",
    identifiers[R4].R: f"{identifiers[R3].R} - beta",
    identifiers[Q2].Y: f"{identifiers[Q1].Y} + gamma",
    identifiers[Q3].Y: f"{identifiers[Q2].Y} + delta",
  },
  constraint_variables=dict(
    alpha=dict(
      value=500,
      min=0,
    ),
    beta=dict(
      value=300,
      min=0,
    ),
    gamma=dict(
      value=1e-8,
      min=0,
    ),
    delta=dict(
      value=2e-7,
      min=0,
    ),
  ),
)
print(fit.to_parameters_dataframe().to_markdown(index=False))
print(fit.to_statistics_dataframe().to_markdown(index=False))

R1, R2, Q1, R3, Q2, R4, Q3 = fit.circuit.get_elements()
assert R2.get_value("R") < R4.get_value("R") < R3.get_value("R")
assert Q1.get_value("Y") < Q2.get_value("Y") < Q3.get_value("Y")

refined_fit: FitResult = fit_circuit(
  fit.circuit,
  data,
  method="least_squares",
  weight="boukamp",
)

print(refined_fit.to_parameters_dataframe().to_markdown(index=False))
print(refined_fit.to_statistics_dataframe().to_markdown(index=False))

figure, axes = mpl.plot_fit(
  fit,
  data=data,
  colored_axes=True,
  legend=False,
  title="With constraints",
)
figure.tight_layout()