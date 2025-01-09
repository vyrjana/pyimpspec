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


from numpy.linalg import norm
from pyimpspec.analysis.kramers_kronig.result import KramersKronigResult
from pyimpspec.circuit.base import Element
from pyimpspec.circuit.circuit import Circuit
from pyimpspec.circuit.kramers_kronig import (
    KramersKronigAdmittanceRC,
    KramersKronigRC,
)
from pyimpspec.typing.helpers import (
    Dict,
    List,
)
from .utility.common import (
    _is_admittance_test_circuit,
    _truncate_circuits,
)


def _calculate_zeta(circuit: Circuit) -> float:
    theta: List[float] = []
    admittance: bool = _is_admittance_test_circuit(circuit)
    num_RC: int = 0

    # Extract the original fitted values obtained by solving the set of
    # linear equations.
    element: Element
    parameters: Dict[str, float]
    if admittance:
        for element in circuit.get_elements(recursive=True):
            parameters = element.get_values()
            if isinstance(element, KramersKronigAdmittanceRC):
                num_RC += 1

            if "tau" in parameters:
                v = parameters["C"]
            elif "R" in parameters:
                v = 1 / parameters["R"]
            elif "C" in parameters:
                v = parameters["C"]
            elif "L" in parameters:
                v = 1 / parameters["L"]

            theta.append(v)

    else:
        for element in circuit.get_elements(recursive=True):
            parameters = element.get_values()
            if isinstance(element, KramersKronigRC):
                num_RC += 1

            if "tau" in parameters:
                v = parameters["R"]
            elif "R" in parameters:
                v = parameters["R"]
            elif "C" in parameters:
                v = 1 / parameters["C"]
            elif "L" in parameters:
                v = parameters["L"]

            theta.append(v)

    return norm(theta) / num_RC


def suggest(
    tests: List[KramersKronigResult],
    lower_limit: int = 0,
    upper_limit: int = 0,
    relative_scores: bool = True,
) -> Dict[int, float]:
    """
    Suggest the optimal number of RC elements to use based on the norm of the fitted variables divided by the number of RC elements.
    Growing norms are used as indications of underfitting and overfitting.
    Thus, a minimum of the norm of the fitted variables should coincide with the desired optimum.

    References:

    - `C. Plank, T. Rüther, and M.A. Danzer, 2022, 2022 International Workshop on Impedance Spectroscopy (IWIS), 1-6 <https://doi.org/10.1109/IWIS57888.2022.9975131>`_

    Parameters
    ----------
    tests: List[|KramersKronigResult|]
        The test results to evaluate.

    lower_limit: int, optional
        The lower limit to enforce for the number of RC elements.
        If this value is less than one, then no limit is enforced.
        If both the lower and upper limit are greater than zero, then the lower limit must have a smaller value than the upper limit.

    upper_limit: int, optional
        The upper limit to enforce for the number of RC elements.
        If this value is less than one, then no limit is enforced.
        If both the lower and upper limit are greater than zero, then the upper limit must have a greater value than the lower limit.

    relative_scores: bool, optional
        Return relative scores ranging from 0.0 to 1.0 (from worst to best) rather than the raw values.

    Returns
    -------
    Dict[int, float]

        A dictionary mapping the number of RC elements to its corresponding score.
    """
    circuits: Dict[int, Circuit] = {t.num_RC: t.circuit for t in tests}
    circuits = _truncate_circuits(circuits, lower_limit, upper_limit)
    scores: Dict[int, float] = {
        num_RC: _calculate_zeta(circuit) for num_RC, circuit in circuits.items()
    }

    if not relative_scores:
        return scores

    min_score: float = min(scores.values())
    max_score: float = max(scores.values()) - min_score
    if max_score == 0.0:
        return {num_RC: 1.0 for num_RC in scores.keys()}

    return {
        num_RC: float(1.0 - (value - min_score) / max_score)
        for num_RC, value in scores.items()
    }
