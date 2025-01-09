# pyimpspec is licensed under the GPLv3 or later (https://www.gnu.org/licenses/gpl-3.0.html).
# Copyright 2024 pyimpspec developers
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# The licenses of pyimpspec's dependencies and/or sources of portions of code are included in
# the LICENSES folder.

from typing import Tuple
from numpy import (
    abs,
    complex128,
    float64,
    inf,
    pi,
    sum as array_sum,
    zeros,
)
from numpy.linalg import (
    inv,
    pinv,
)
from numpy.typing import NDArray
from pyimpspec.circuit.base import Element
from pyimpspec.circuit.circuit import Circuit
from pyimpspec.circuit.elements import (
    Capacitor,
    Inductor,
    KramersKronigAdmittanceRC,
    KramersKronigRC,
    Resistor,
)
from pyimpspec.exceptions import KramersKronigError
from pyimpspec.typing import (
    ComplexImpedances,
    Frequencies,
    Impedances,
)
from pyimpspec.typing.helpers import (
    List,
    _is_boolean,
    _is_complex_array,
    _is_floating,
    _is_floating_array,
    _is_integer,
)
from .utility import (
    _generate_circuit,
    _generate_time_constants,
)


def _update_circuit(
    circuit: Circuit,
    variables: NDArray[float64],
    add_capacitance: bool,
    admittance: bool,
):
    elements: List[Element] = circuit.get_elements(recursive=True)
    if len(elements) != len(variables):
        raise ValueError(
            f"Expected the circuit to contain as many elements ({len(elements)=}) as there are variables ({len(variables)=})"
        )

    # Series or parallel R
    R: float64
    R, variables = variables[0], variables[1:]
    if admittance:
        if R == 0.0:
            R = 1e18
        else:
            R = 1 / R

    for element in elements:
        if isinstance(element, Resistor):
            element.set_values(R=R)
            break
    else:
        raise KramersKronigError("Failed to update series/parallel R!")

    # Series or parallel L
    L: float64
    L, variables = variables[-1], variables[:-1]
    if admittance:
        if L == 0.0:
            L = 1e18
        else:
            L = 1 / L

        L *= -1

    if add_capacitance:
        C: float64
        C, variables = variables[-1], variables[:-1]
        if C == 0.0:
            C = 1e-50

        if not admittance:
            C = 1 / C

    for element in elements:
        if isinstance(element, Inductor):
            element.set_values(L=L)
            element.set_lower_limits(L=-inf)
            element.set_upper_limits(L=inf)
            break
    else:
        raise KramersKronigError("Failed to update series/parallel L!")

    # Series or parallel C
    if add_capacitance:
        for element in elements:
            if isinstance(element, Capacitor):
                element.set_values(C=C)
                element.set_lower_limits(C=-inf)
                element.set_upper_limits(C=inf)
                break
        else:
            raise KramersKronigError("Failed to update series/parallel C!")

    # Fitted R_k or C_k
    for i, element in enumerate(
        filter(
            lambda e: isinstance(
                e,
                KramersKronigAdmittanceRC if admittance else KramersKronigRC,
            ),
            elements,
        )
    ):
        if admittance:
            element.set_values(C=variables[i])
        else:
            element.set_values(R=variables[i])


def _initialize_A_matrices(
    w: NDArray[float64],
    taus: NDArray[float64],
    add_capacitance: bool,
) -> Tuple[NDArray[float64], NDArray[float64]]:
    # Generate matrices with the following columns
    # (top to bottom is left to right)
    # - R0, resistance
    # - Ri or Ci associated with taus[i - 1] (0 < i <= num_RC)
    # - C, optional capacitance
    # - L, inductance
    shape: Tuple[int, int] = (w.size, len(taus) + (3 if add_capacitance else 2))

    A_re: NDArray[float64] = zeros(shape, dtype=float64)
    A_im: NDArray[float64] = zeros(shape, dtype=float64)

    return (A_re, A_im)


def _add_resistance_to_A_matrix(A_re: NDArray[float64]):
    # See Fig. 1 (impedance) and 13 (admittance) in Boukamp (1995)
    A_re[:, 0] = 1


def _add_capacitance_to_A_matrix(
    A_im: NDArray[float64],
    w: NDArray[float64],
    admittance: bool,
):
    if admittance:
        A_im[:, -2] = w
    else:
        A_im[:, -2] = -1 / w


def _add_inductance_to_A_matrix(
    A_im: NDArray[float64],
    w: NDArray[float64],
    admittance: bool,
):
    if admittance:
        A_im[:, -1] = 1 / w
    else:
        A_im[:, -1] = w


def _add_kth_variables_to_A_matrices(
    A_re: NDArray[float64],
    A_im: NDArray[float64],
    w: NDArray[float64],
    taus: NDArray[float64],
    admittance: bool,
):
    i: int
    tau: float64

    if admittance:
        for i, tau in enumerate(taus):
            A_re[:, i + 1] = w**2 * tau / (1 + (w * tau) ** 2)
            A_im[:, i + 1] = w / (1 + (w * tau) ** 2)
    else:
        for i, tau in enumerate(taus):
            k: NDArray[complex128] = 1 / (1 + 1j * w * tau)
            A_re[:, i + 1] = k.real
            A_im[:, i + 1] = k.imag


def _scale_A_matrices(
    A_re: NDArray[float64],
    A_im: NDArray[float64],
    abs_X_exp: Impedances,
):
    i: int
    for i in range(A_re.shape[1]):
        A_re[:, i] /= abs_X_exp
        A_im[:, i] /= abs_X_exp


def _generate_A_matrices(
    w: NDArray[float64],
    taus: NDArray[float64],
    add_capacitance: bool,
    admittance: bool,
    abs_X_exp: Impedances,
) -> Tuple[NDArray[float64], NDArray[float64]]:
    A_re: NDArray[float64]
    A_im: NDArray[float64]
    A_re, A_im = _initialize_A_matrices(
        w=w,
        taus=taus,
        add_capacitance=add_capacitance,
    )

    _add_resistance_to_A_matrix(A_re)

    if add_capacitance:
        _add_capacitance_to_A_matrix(
            A_im=A_im,
            w=w,
            admittance=admittance,
        )

    _add_inductance_to_A_matrix(
        A_im=A_im,
        w=w,
        admittance=admittance,
    )

    _add_kth_variables_to_A_matrices(
        A_re=A_re,
        A_im=A_im,
        w=w,
        taus=taus,
        admittance=admittance,
    )

    _scale_A_matrices(
        A_re=A_re,
        A_im=A_im,
        abs_X_exp=abs_X_exp,
    )

    return (
        A_re,
        A_im,
    )


def _real_test(
    A_re: NDArray[float64],
    X_exp: NDArray[complex128],
    w: NDArray[float64],
    f: Frequencies,
    taus: NDArray[float64],
    add_capacitance: bool,
    admittance: bool,
    circuit: Circuit,
) -> NDArray[float64]:
    abs_X_exp: NDArray[float64] = abs(X_exp)

    # Fit using the real part
    variables: NDArray[float64] = pinv(A_re).dot(X_exp.real / abs_X_exp)
    if add_capacitance:
        # Nullifies the capacitance without dividing by 0
        variables[-2] = 1e-18

    # Fit using the imaginary part to fix the series/parallel
    # inductance (and capacitance)
    A_im: NDArray[float64] = zeros((w.size, 2), dtype=float64)
    A_im[:, -1] = (1 / w) if admittance else w
    if add_capacitance:
        A_im[:, -2] = w if admittance else (-1 / w)

    # Scaling
    for i in range(A_im.shape[1]):
        A_im[:, i] /= abs_X_exp

    # Update the circuit and calculate the impedance or admittance
    _update_circuit(
        circuit=circuit,
        variables=variables,
        add_capacitance=add_capacitance,
        admittance=admittance,
    )
    X_fit: NDArray[complex128] = circuit.get_impedances(f) ** (-1 if admittance else 1)

    # Extract the corrected series/parallel inductance (and capacitance)
    coefs: NDArray[float64] = pinv(A_im).dot((X_exp.imag - X_fit.imag) / abs_X_exp)
    if add_capacitance:
        variables[-2:] = coefs
    else:
        variables[-1] = coefs[-1]

    return variables


def _imaginary_test(
    A_im: NDArray[float64],
    X_exp: NDArray[complex128],
    f: Frequencies,
    taus: NDArray[float64],
    add_capacitance: bool,
    admittance: bool,
    weight: NDArray[float64],
    circuit: Circuit,
) -> NDArray[float64]:
    abs_X_exp: NDArray[float64] = abs(X_exp)

    # Fit using the imaginary part
    variables: NDArray[float64] = pinv(A_im).dot(X_exp.imag / abs_X_exp)

    # Update the circuit and calculate the impedance or admittance
    _update_circuit(
        circuit=circuit,
        variables=variables,
        add_capacitance=add_capacitance,
        admittance=admittance,
    )
    X_fit: NDArray[complex128] = circuit.get_impedances(f) ** (-1 if admittance else 1)

    # Estimate the series or parallel resistance
    variables[0] = array_sum(weight * (X_exp.real - X_fit.real)) / array_sum(weight)

    return variables


def _complex_test(
    A_re: NDArray[float64],
    A_im: NDArray[float64],
    X_exp: NDArray[complex128],
    taus: NDArray[float64],
    add_capacitance: bool,
    admittance: bool,
    circuit: Circuit,
) -> NDArray[float64]:
    abs_X_exp: NDArray[float64] = abs(X_exp)

    # Fit using the complex impedance
    x: NDArray[float64] = inv(A_re.T.dot(A_re) + A_im.T.dot(A_im))

    y: NDArray[float64] = A_re.T.dot(X_exp.real / abs_X_exp) + A_im.T.dot(
        X_exp.imag / abs_X_exp
    )

    variables: NDArray[float64] = x.dot(y)

    return variables


def _test_wrapper(args: tuple) -> Tuple[int, Circuit]:
    test: str
    f: Frequencies
    Z_exp: ComplexImpedances
    weight: NDArray[float64]
    num_RC: int
    add_capacitance: bool
    admittance: bool
    log_F_ext: float
    (
        test,
        f,
        Z_exp,
        weight,
        num_RC,
        add_capacitance,
        admittance,
        log_F_ext,
    ) = args

    if not isinstance(test, str):
        raise TypeError(f"Expected a string instead of {test=}")
    elif test not in ("complex", "real", "imaginary"):
        raise ValueError(
            f"Expected 'complex', 'real', or 'imaginary' instead of {test=}"
        )

    if not _is_floating_array(f):
        raise TypeError(f"Expected an array of floats instead of {f=}")

    if not _is_complex_array(Z_exp):
        raise TypeError(f"Expected an array of complex values instead of {Z_exp=}")

    if not _is_floating_array(weight):
        raise TypeError(f"Expected an array of floats instead of {weight=}")

    if not _is_integer(num_RC):
        raise TypeError(f"Expected an integer instead of {num_RC=}")

    if not _is_boolean(add_capacitance):
        raise TypeError(f"Expected a boolean instead of {add_capacitance=}")

    if not _is_boolean(admittance):
        raise TypeError(f"Expected a boolean instead of {admittance=}")

    if not _is_floating(log_F_ext):
        raise TypeError(f"Expected a float instead of {log_F_ext=}")

    X_exp: NDArray[complex128] = Z_exp ** (-1 if admittance else 1)
    w: NDArray[float64] = 2 * pi * f
    taus: NDArray[float64] = _generate_time_constants(w, num_RC, log_F_ext)

    A_re: NDArray[float64]
    A_im: NDArray[float64]
    A_re, A_im = _generate_A_matrices(
        w,
        taus,
        add_capacitance,
        admittance,
        abs(X_exp),
    )

    circuit: Circuit = _generate_circuit(
        taus,
        add_capacitance,
        True,
        admittance,
    )

    # Solve the set of linear equations and update the circuit's parameters
    variables: NDArray[float64]
    if test == "real":
        variables = _real_test(
            A_re,
            X_exp,
            w,
            f,
            taus,
            add_capacitance,
            admittance,
            circuit,
        )

    elif test == "imaginary":
        variables = _imaginary_test(
            A_im,
            X_exp,
            f,
            taus,
            add_capacitance,
            admittance,
            weight,
            circuit,
        )

    elif test == "complex":
        variables = _complex_test(
            A_re,
            A_im,
            X_exp,
            taus,
            add_capacitance,
            admittance,
            circuit,
        )

    else:
        raise ValueError(f"Unsupported test type {test=}")

    # Update the circuit
    _update_circuit(
        circuit=circuit,
        variables=variables,
        add_capacitance=add_capacitance,
        admittance=admittance,
    )

    return (
        num_RC,
        circuit,
    )
