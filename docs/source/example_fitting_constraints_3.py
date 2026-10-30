#!/usr/bin/env python
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

data: DataSet = generate_mock_data("R{R=120}(R{R=350}C{C=1e-7})(R{R=30}C{C=4e-5})(R{R=5e3}C{C=5e-6})", noise=0.1, seed=42)[0]

circuit: Circuit = parse_cdc("R{:ser}(R{:hf}C{:hf})(R{:mf}C{:mf})(R{:lf}C{:lf})")

unconstrained_fit: FitResult = fit_circuit(
    circuit,
    data,
    method="least_squares",
    weight="boukamp",
)

identifiers: Dict[Element, FitIdentifiers] = generate_fit_identifiers(circuit)
elements: List[Element] = circuit.get_elements()
R_ser, R_hf, C_hf, R_mf, C_mf, R_lf, C_lf = elements

constrained_fit: FitResult = fit_circuit(
    circuit,
    data,
    method="least_squares",
    weight="boukamp",
    constraint_expressions={
        identifiers[R_mf].R: f"{identifiers[R_hf].R} * {identifiers[C_hf].C} / {identifiers[C_mf].C} * 10**alpha",
        identifiers[R_lf].R: f"{identifiers[R_mf].R} * {identifiers[C_mf].C} / {identifiers[C_lf].C} * 10**beta",
    },
    constraint_variables=dict(
        alpha=dict(
            value=0.3,
            min=0.0,
            max=6.0,
        ),
        beta=dict(
            value=0.8,
            min=0.0,
            max=6.0,
        ),
    ),
)

refined_fit: FitResult = fit_circuit(
    constrained_fit.circuit,
    data,
    method="least_squares",
    weight="boukamp",
)

text_output = f"""
# Unconstrained fit

{unconstrained_fit.to_parameters_dataframe().to_markdown(index=False)}

{unconstrained_fit.to_statistics_dataframe().to_markdown(index=False)}


# Constrained fit

{constrained_fit.to_parameters_dataframe().to_markdown(index=False)}

{constrained_fit.to_statistics_dataframe().to_markdown(index=False)}
""".strip()

with open("example_fitting_constraints_3a.txt", "w") as fp:
    fp.write(text_output)


text_output = f"""
# Refined fit

{refined_fit.to_parameters_dataframe().to_markdown(index=False)}

{refined_fit.to_statistics_dataframe().to_markdown(index=False)}
""".strip()

with open("example_fitting_constraints_3b.txt", "w") as fp:
    fp.write(text_output)


figure, axes = mpl.plot_nyquist(
    data,
    legend=False,
    label="Data",
    colors=dict(impedance="black"),
)
mpl.plot_nyquist(
    unconstrained_fit,
    legend=False,
    label="Unconstrained fit",
    line=True,
    colors=dict(impedance="red"),
    figure=figure,
    axes=axes,
)
mpl.plot_nyquist(
    constrained_fit,
    legend=True,
    label="Constrained fit",
    line=True,
    colors=dict(impedance="blue"),
    figure=figure,
    axes=axes,
)
figure.tight_layout()

if __name__ == "__main__":
    mpl.show()
