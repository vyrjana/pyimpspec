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

from numpy import (
    array,
    complex128,
    float64,
    inf,
    int64,
    log10 as log,
    max,
    min,
    sqrt,
)
from numpy.typing import NDArray
from pyimpspec.circuit import parse_cdc
from pyimpspec.circuit.base import Element
from pyimpspec.circuit.circuit import (
    Circuit,
    Series,
    Parallel,
)
from pyimpspec.circuit.resistor import Resistor
from pyimpspec.circuit.capacitor import Capacitor
from pyimpspec.circuit.inductor import Inductor
from pyimpspec.circuit.kramers_kronig import (
    KramersKronigAdmittanceRC,
    KramersKronigRC,
)
from pyimpspec.typing import ComplexImpedances
from pyimpspec.typing.helpers import List


def _generate_time_constants(
    w: NDArray[float64],
    num_RC: int,
    log_F_ext: float,
) -> NDArray[float64]:
    # Calculate time constants according to eq. 12 from
    # Schönleber et al. (2014) and eq. 18 in Boukamp (1995).
    if num_RC < 2:
        raise ValueError(
            f"Expected an integer greater than or equal to 2 instead of {num_RC=}"
        )

    F_ext: float = 10**log_F_ext
    tau_min: float64 = 1 / (max(w) * F_ext)
    tau_max: float64 = F_ext / min(w)
    k: NDArray[int64] = array(list(range(1, num_RC + 1)))

    return 10 ** (log(tau_min) + (k - 1) / (num_RC - 1) * log(tau_max / tau_min))


def _generate_circuit(
    taus: NDArray[float64],
    add_capacitance: bool,
    add_inductance: bool,
    admittance: bool,
) -> Circuit:
    elements: List[Element] = []
    elements.append(Resistor(R=1).set_lower_limits(R=-inf).set_upper_limits(R=inf))

    t: float
    for t in taus:
        if admittance:
            elements.append(KramersKronigAdmittanceRC(tau=t))
        else:
            elements.append(KramersKronigRC(tau=t))

    if add_capacitance:
        elements.append(Capacitor(C=1e-6).set_lower_limits(C=-inf).set_upper_limits(C=inf))

    if add_inductance:
        elements.append(Inductor(L=1e-3).set_lower_limits(L=-inf).set_upper_limits(L=inf))

    if admittance:
        return Circuit(Parallel(elements))

    return Circuit(Series(elements))


def _boukamp_weight(
    Z: ComplexImpedances,
    admittance: bool,
) -> NDArray[float64]:
    # See eq. 13 in Boukamp (1995)
    if admittance:
        Y: NDArray[complex128] = 1 / Z
        return (Y.real**2 + Y.imag**2) ** -1

    return (Z.real**2 + Z.imag**2) ** -1  # type: ignore


def _estimate_pseudo_chisqr(Z: ComplexImpedances, pct_noise: float) -> float:
    # Assumes that two uncorrelated and normally distributed noises with the
    # same standard deviation were added to the real and imaginary parts of
    # the impedance spectrum.
    return len(Z) * pct_noise**2 / 5000


def _estimate_pct_noise(Z: ComplexImpedances, pseudo_chisqr: float) -> float:
    # See _estimate_pseudo_chisqr
    return sqrt(5000 * pseudo_chisqr / len(Z))


def _format_log_F_ext_for_latex(log_F_ext: float) -> str:
    value: str = f"{log_F_ext:.3g}"
    if "e" in value:
        coefficient: str
        exponent: str
        coefficient, exponent = value.split("e")
        value = coefficient + r" \times 10^{" + str(int(exponent)) + "}"

    return value
