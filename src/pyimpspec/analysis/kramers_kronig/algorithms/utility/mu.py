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

from typing import Dict
from numpy import nan

from pyimpspec.circuit.base import Element
from pyimpspec.circuit.circuit import Circuit
from pyimpspec.circuit.kramers_kronig import (
    KramersKronigAdmittanceRC,
    KramersKronigRC,
)
from .common import _is_admittance_test_circuit


def _calculate_mu(circuit: Circuit) -> float:
    r"""
    Calculate :math:`\mu \in [0.0, 1.0]` for a |KramersKronigResult|.
    Over- and underfitting are represented by 0.0 and 1.0, respectively.
    Based on Eq. 21 on page 25 of Schönleber et al. (2014):

    :math:`\mu = 1 - \frac{\Sigma_{R_k < 0} |R_k|}{\Sigma_{R_k \geq 0} |R_k|}`

    with some modifications:

    - The return value is clamped to the range [0.0, 1.0] (see the point below for an exception).
    - If the denominator is zero, then |numpy.nan| is returned.
    - If the test was performed on admittance data, then :math:`C_k` is substituted for :math:`R_k`.

    References:

    - M. Schönleber, D. Klotz, and E. Ivers-Tiffée, 2014, Electrochim. Acta, 131, 20-27 (https://doi.org/10.1016/j.electacta.2014.01.034)

    Parameters
    ----------
    circuit: |Circuit|
        The circuit that was fitted as part of the Kramers-Kronig test.

    Returns
    -------
    float
    """
    key: str = "C" if _is_admittance_test_circuit(circuit) else "R"
    mass_of_negatives: float = 0.0
    mass_of_positives: float = 0.0

    element: Element
    for element in circuit.get_elements(recursive=True):
        if type(element) not in (KramersKronigAdmittanceRC, KramersKronigRC):
            continue

        parameters: Dict[str, float] = element.get_values()
        value: float = parameters[key]
        if value >= 0.0:
            mass_of_positives += value
        else:
            mass_of_negatives += abs(value)

    if mass_of_positives == 0.0:
        return nan

    mu: float = 1.0 - mass_of_negatives / mass_of_positives
    return min((1.0, max((0.0, mu))))
