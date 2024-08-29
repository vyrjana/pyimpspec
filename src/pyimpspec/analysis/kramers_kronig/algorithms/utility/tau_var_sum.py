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
from numpy import (
    float64,
    log10 as log,
)
from pyimpspec.circuit.base import Element
from pyimpspec.circuit.circuit import Circuit
from pyimpspec.circuit.kramers_kronig import (
    KramersKronigAdmittanceRC,
    KramersKronigRC,
)
from .common import _is_admittance_test_circuit


def _calculate_log_sum_abs_tau_var(circuit: Circuit) -> float64:
    key: str = "C" if _is_admittance_test_circuit(circuit) else "R"
    total: float = 0.0

    element: Element
    for element in circuit.get_elements():
        if type(element) not in (KramersKronigRC, KramersKronigAdmittanceRC):
            continue

        parameters: Dict[str, float] = element.get_values()
        total += abs(parameters["tau"] / parameters[key])

    return log(total)
