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

from typing import (
    List,
    Tuple,
)
from numpy import (
    complex128,
    float64,
    inf,
    pi,
    sum as array_sum,
    zeros,
)
from numpy.linalg import lstsq
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
)
from pyimpspec.typing.helpers import (
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


def _initialize_A_matrix(
    test: str,
    w: NDArray[float64],
    taus: NDArray[float64],
    add_capacitance: bool,
    add_inductance: bool,
) -> NDArray[float64]:
    m: int = len(w) * (2 if test == "complex" else 1)
    n: int = len(taus) + 1

    if add_capacitance:
        n += 1

    if add_inductance:
        n += 1

    return zeros((m, n), dtype=float64)


def _add_resistance_to_A_matrix(
    A: NDArray[float64],
    test: str,
):
    m: int = A.shape[0]
    i: int = 0

    if test == "complex":
        A[0:m // 2, i] = 1
    elif test == "real":
        A[0:m, i] = 1


def _calculate_kth_A_matrix_variables(
    w: NDArray,
    tau: float64,
    admittance: bool,
) -> NDArray[complex128]:
    if admittance:
        return w / (w * tau - 1j)
    else:
        return 1 / (1 + 1j * w * tau)


def _add_kth_variables_to_A_matrix(
    A: NDArray[float64],
    test: str,
    w: NDArray[float64],
    tau: float64,
    i: int,
    admittance: bool,
):
    c: NDArray[complex128] = _calculate_kth_A_matrix_variables(
        w=w,
        tau=tau,
        admittance=admittance,
    )

    m: int = A.shape[0]
    if test == "complex":
        A[0:m // 2, i] = c.real
        A[m // 2:, i] = c.imag
    elif test == "real":
        A[0:m, i] = c.real
    else:
        A[0:m, i] = c.imag


def _add_capacitance_to_A_matrix(
    A: NDArray[float64],
    test: str,
    w: NDArray[float64],
    i: int,
    admittance: bool,
):
    m: int = A.shape[0]

    if test == "complex":
        A[m // 2:, i] = w if admittance else (-1 / w)
    elif test == "imaginary":
        A[:, i] = w if admittance else (-1 / w)


def _add_inductance_to_A_matrix(
    A: NDArray[float64],
    test: str,
    w: NDArray[float64],
    i: int,
    admittance: bool,
):
    m: int = A.shape[0]

    if test == "complex":
        A[m // 2:, i] = (1 / w) if admittance else w
    elif test == "imaginary":
        A[:, i] = (1 / w) if admittance else w


def _generate_A_matrix(
    test: str,
    w: NDArray[float64],
    taus: NDArray[float64],
    add_capacitance: bool,
    add_inductance: bool,
    admittance: bool,
) -> NDArray[float64]:
    A: NDArray[float64] = _initialize_A_matrix(
        test,
        w,
        taus,
        add_capacitance,
        add_inductance,
    )

    # Series or parallel R
    _add_resistance_to_A_matrix(A=A, test=test)

    # R_k or C_k
    i: int
    tau: float64
    for i, tau in enumerate(taus, start=1):
        _add_kth_variables_to_A_matrix(
            A=A,
            test=test,
            w=w,
            tau=tau,
            i=i,
            admittance=admittance,
        )

    # Series or parallel C
    if add_capacitance:
        i += 1
        _add_capacitance_to_A_matrix(
            A=A,
            test=test,
            w=w,
            i=i,
            admittance=admittance,
        )

    # Series or parallel L
    if add_inductance:
        i += 1
        _add_inductance_to_A_matrix(
            A=A,
            test=test,
            w=w,
            i=i,
            admittance=admittance,
        )

    return A


def _initialize_b_vector(test: str, Z_exp: ComplexImpedances) -> NDArray[float64]:
    m: int = len(Z_exp) * (2 if test == "complex" else 1)

    return zeros(m, dtype=float64)


def _add_values_to_b_vector(
    b: NDArray[float64],
    test: str,
    Z_exp: ComplexImpedances,
    admittance: bool,
):
    m: int = b.shape[0]

    if test == "complex":
        b[0:m // 2] = (Z_exp ** (-1 if admittance else 1)).real
        b[m // 2:] = (Z_exp ** (-1 if admittance else 1)).imag
    elif test == "real":
        b[0:m] = (Z_exp ** (-1 if admittance else 1)).real
    else:
        b[0:m] = (Z_exp ** (-1 if admittance else 1)).imag


def _generate_b_vector(
    test: str,
    Z_exp: ComplexImpedances,
    admittance: bool,
) -> NDArray[float64]:
    b: NDArray[float64] = _initialize_b_vector(test, Z_exp)
    _add_values_to_b_vector(b, test, Z_exp, admittance)

    return b


def _update_circuit(
    circuit: Circuit,
    variables: NDArray[float64],
    add_capacitance: bool,
    add_inductance: bool,
    admittance: bool,
):
    elements: List[Element] = circuit.get_elements(recursive=True)
    if len(elements) != len(variables):
        raise ValueError(f"Expected the circuit to contain as many elements ({len(elements)=}) as there are variables ({len(variables)=})")

    # Series or parallel R
    R: float64
    R, variables = variables[0], variables[1:]
    for element in elements:
        if isinstance(element, Resistor):
            if admittance:
                if R == 0.0:
                    R = inf
                else:
                    R = 1 / R
            element.set_values(R=R)
            break
    else:
        raise KramersKronigError("Failed to update series/parallel R!")

    # Series or parallel L
    if add_inductance:
        L: float64
        L, variables = variables[-1], variables[:-1]
        for element in elements:
            if isinstance(element, Inductor):
                element.set_values(L=(-1 / L) if admittance else L)
                element.set_lower_limits(L=-inf)
                element.set_upper_limits(L=inf)
                break
        else:
            raise KramersKronigError("Failed to update series/parallel L!")

    # Series or parallel C
    if add_capacitance:
        C: float64
        C, variables = variables[-1], variables[:-1]
        for element in elements:
            if isinstance(element, Capacitor):
                element.set_values(C=C if admittance else 1 / C)
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


def _real_test(
    Z_exp: ComplexImpedances,
    f: Frequencies,
    w: NDArray[float64],
    taus: NDArray[float64],
    add_capacitance,
    add_inductance,
    admittance,
) -> Circuit:
    test: str = "real"
    A: NDArray[float64] = _generate_A_matrix(
        test,
        w,
        taus,
        add_capacitance=False,
        add_inductance=False,
        admittance=admittance,
    )
    b: NDArray[float64] = _generate_b_vector(test, Z_exp, admittance)
    x: NDArray[float64] = lstsq(A, b, rcond=None)[0]

    circuit: Circuit = _generate_circuit(
        taus,
        add_capacitance=False,
        add_inductance=False,
        admittance=admittance,
    )
    _update_circuit(
        circuit,
        x,
        add_capacitance=False,
        add_inductance=False,
        admittance=admittance,
    )

    if add_capacitance or add_inductance:
        for element in circuit.get_elements(recursive=True):
            if isinstance(element, Resistor):
                break
        else:
            raise ValueError("Failed to find series/parallel resistance!")

        for connection in circuit.get_connections(recursive=True):
            if connection.contains(element, top_level=True):
                break
        else:
            raise ValueError(
                "Failed to find series/parallel connection containing the resistance!"
            )

        A = zeros((w.size, int(add_capacitance) + int(add_inductance)), dtype=float64)
        if add_capacitance:
            A[:, 0] = w if admittance else (-1 / w)

        if add_inductance:
            A[:, -1] = (1 / w) if admittance else w

        b = _generate_b_vector(
            "imaginary",
            (
                Z_exp ** (-1 if admittance else 1)
                - circuit.get_impedances(f) ** (-1 if admittance else 1)
            )
            ** (-1 if admittance else 1),
            admittance,
        )

        corrections: NDArray[float64] = lstsq(A, b, rcond=None)[0]

        i: int = len(x) - 1
        tmp: NDArray[float64] = x
        x = zeros(len(x) + int(add_capacitance) + int(add_inductance), dtype=float64)
        x[: len(tmp)] = tmp

        if add_capacitance:
            i += 1
            x[i] = corrections[0]
            connection.append(Capacitor())

        if add_inductance:
            i += 1
            x[i] = corrections[-1]
            connection.append(Inductor())

        _update_circuit(
            circuit,
            x,
            add_capacitance=add_capacitance,
            add_inductance=add_inductance,
            admittance=admittance,
        )

    return circuit


def _imaginary_test(
    Z_exp: ComplexImpedances,
    f: Frequencies,
    w: NDArray[float64],
    taus: NDArray[float64],
    weight: NDArray[float64],
    add_capacitance,
    add_inductance,
    admittance,
) -> Circuit:
    test: str = "imaginary"
    A: NDArray[float64] = _generate_A_matrix(
        test,
        w,
        taus,
        add_capacitance,
        add_inductance,
        admittance,
    )
    b: NDArray[float64] = _generate_b_vector(test, Z_exp, admittance)
    x: NDArray[float64] = lstsq(A, b, rcond=None)[0]

    circuit: Circuit = _generate_circuit(
        taus,
        add_capacitance,
        add_inductance,
        admittance,
    )
    _update_circuit(circuit, x, add_capacitance, add_inductance, admittance)

    X_exp: NDArray[complex128] = Z_exp ** (-1 if admittance else 1)
    X_fit: NDArray[complex128] = circuit.get_impedances(f) ** (-1 if admittance else 1)
    x[0] = array_sum(weight * (X_exp.real - X_fit.real)) / array_sum(weight)
    _update_circuit(circuit, x, add_capacitance, add_inductance, admittance)

    return circuit


def _complex_test(
    Z_exp: ComplexImpedances,
    f: Frequencies,
    w: NDArray[float64],
    taus: NDArray[float64],
    add_capacitance,
    add_inductance,
    admittance,
) -> Circuit:
    test: str = "complex"
    A: NDArray[float64] = _generate_A_matrix(
        test,
        w,
        taus,
        add_capacitance,
        add_inductance,
        admittance,
    )
    b: NDArray[float64] = _generate_b_vector(test, Z_exp, admittance)
    x: NDArray[float64] = lstsq(A, b, rcond=None)[0]

    circuit: Circuit = _generate_circuit(
        taus,
        add_capacitance,
        add_inductance,
        admittance,
    )
    _update_circuit(circuit, x, add_capacitance, add_inductance, admittance)

    return circuit


def _test_wrapper(args: tuple) -> Tuple[int, Circuit]:
    test: str
    f: Frequencies
    Z_exp: ComplexImpedances
    weight: NDArray[float64]
    num_RC: int
    add_capacitance: bool
    add_inductance: bool
    admittance: bool
    log_F_ext: float
    (
        test,
        f,
        Z_exp,
        weight,
        num_RC,
        add_capacitance,
        add_inductance,
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
        raise TypeError(f"Expected an an array of floats instead of {weight=}")

    if not _is_integer(num_RC):
        raise TypeError(f"Expected an integer instead of {num_RC=}")

    if not _is_boolean(add_capacitance):
        raise TypeError(f"Expected a boolean instead of {add_capacitance=}")

    if not _is_boolean(add_inductance):
        raise TypeError(f"Expected a boolean instead of {add_inductance=}")

    if not _is_boolean(admittance):
        raise TypeError(f"Expected a boolean instead of {admittance=}")

    if not _is_floating(log_F_ext):
        raise TypeError(f"Expected a float instead of {log_F_ext=}")

    w: NDArray[float64] = 2 * pi * f
    taus: NDArray[float64] = _generate_time_constants(w, num_RC, log_F_ext)
    circuit: Circuit
    if test == "real":
        circuit = _real_test(
            Z_exp=Z_exp,
            f=f,
            w=w,
            taus=taus,
            add_capacitance=add_capacitance,
            add_inductance=add_inductance,
            admittance=admittance,
        )

    elif test == "imaginary":
        circuit = _imaginary_test(
            Z_exp=Z_exp,
            f=f,
            w=w,
            taus=taus,
            add_capacitance=add_capacitance,
            add_inductance=add_inductance,
            admittance=admittance,
            weight=weight,
        )

    elif test == "complex":
        circuit = _complex_test(
            Z_exp=Z_exp,
            f=f,
            w=w,
            taus=taus,
            add_capacitance=add_capacitance,
            add_inductance=add_inductance,
            admittance=admittance,
        )

    else:
        raise ValueError(f"Unsupported test type ({test=})")

    return (
        num_RC,
        circuit,
    )
