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
    Dict,
    Tuple,
)
from numpy import (
    array,
    complex128,
    float64,
    pi,
)
from pyimpspec.analysis.fitting import (
    FitIdentifiers,
    _from_lmfit,
    _to_lmfit,
    generate_fit_identifiers,
)
from numpy.typing import NDArray
from pyimpspec.circuit.base import Element
from pyimpspec.circuit.circuit import Circuit
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


def _complex_residual(
    params: "Parameters",  # noqa: F821
    circuit: Circuit,
    f: Frequencies,
    X_exp: NDArray[complex128],
    weight: NDArray[float64],
    identifiers: Dict[int, Element],
    admittance: bool,
) -> NDArray[float64]:
    _from_lmfit(params, identifiers)
    X_fit: ComplexImpedances = circuit.get_impedances(f) ** (-1 if admittance else 1)

    return weight * array(
        [
            (X_exp.real - X_fit.real) ** 2,
            (X_exp.imag - X_fit.imag) ** 2,
        ]
    )


def _test_wrapper(args: tuple) -> Tuple[int, Circuit]:
    from lmfit import minimize
    from lmfit.minimizer import MinimizerResult

    f: Frequencies
    Z_exp: ComplexImpedances
    weight: NDArray[float64]
    num_RC: int
    add_capacitance: bool
    add_inductance: bool
    admittance: bool
    log_F_ext: float
    method: str
    max_nfev: int
    (
        f,
        Z_exp,
        weight,
        num_RC,
        add_capacitance,
        add_inductance,
        admittance,
        log_F_ext,
        method,
        max_nfev,
    ) = args

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

    if not isinstance(method, str):
        raise TypeError(f"Expected a string instead of {method=}")

    if not _is_integer(num_RC):
        raise TypeError(f"Expected an integer instead of {num_RC=}")

    if not _is_integer(max_nfev):
        raise TypeError(f"Expected an integer instead of {max_nfev=}")

    w: NDArray[float64] = 2 * pi * f
    taus: NDArray[float64] = _generate_time_constants(w, num_RC, log_F_ext)
    circuit: Circuit = _generate_circuit(
        taus,
        add_capacitance,
        add_inductance,
        admittance,
    )

    identifiers: Dict[Element, FitIdentifiers]
    identifiers = generate_fit_identifiers(circuit)

    fit: MinimizerResult
    fit = minimize(
        _complex_residual,
        _to_lmfit(
            identifiers,
            constraint_expressions={},
            constraint_variables={},
        ),
        method,
        args=(
            circuit,
            f,
            Z_exp ** (-1 if admittance else 1),
            weight,
            identifiers,
            admittance,
        ),
        max_nfev=None if max_nfev < 1 else max_nfev,
    )
    _from_lmfit(fit.params, identifiers)

    return (
        num_RC,
        circuit,
    )
