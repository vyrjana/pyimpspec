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


from numpy import float64
from numpy.linalg import norm
from numpy.typing import NDArray
from pyimpspec.analysis.kramers_kronig.result import KramersKronigResult
from pyimpspec.circuit.circuit import Circuit
from pyimpspec.typing.aliases import Frequencies
from pyimpspec.typing.helpers import (
    Dict,
    List,
    Optional,
)
from .utility.osculating_circle import calculate_curvatures
from .utility.common import (
    _truncate_circuits,
    subdivide_frequencies,
)


def suggest(
    tests: List[KramersKronigResult],
    lower_limit: int = 0,
    upper_limit: int = 0,
    subdivision: int = 4,
    subdivided_frequencies: Optional[Frequencies] = None,
    curvatures: Optional[Dict[int, NDArray[float64]]] = None,
    relative_scores: bool = True,
) -> Dict[int, float]:
    """
    Suggest the optimal number of RC elements to use based on the norm of the curvatures of the fitted impedance spectrum.
    The curvatures at each point of the fitted impedance spectrum is approximated using an osculating circle.
    A minimum of the norm of these curvatures should coincide with the desired optimum.

    A modification whereby the frequency intervals are subdivided before determining curvatures is used to make the method less prone to suggesting circuits that produce oscillating impedance spectra.
    This modified approach is used by default, but the original approach can be used by setting ``subdivision = 0``.

    References:

    - `C. Plank, T. Rüther, and M.A. Danzer, 2022, 2022 International Workshop on Impedance Spectroscopy (IWIS), 1-6 <https://doi.org/10.1109/IWIS57888.2022.9975131>`_
    - `V. Yrjänä and J. Bobacka, 2024, Electrochim. Acta, 504, 144951 <https://doi.org/10.1016/j.electacta.2024.144951>`_

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

    subdivision: int, optional
        If greater than zero, then that number of additional frequencies are inserted into each frequency interval.

    subdivided_frequencies: Optional[Frequencies], optional
        Frequencies that have already been subdivided. If multiple methods that require subdividing frequencies will be used, then this provides a means of reusing those frequencies.

    curvatures: Optional[Dict[int, NDArray[float64]]], optional
        Curvatures that have already been estimated. If multiple methods that require curvatures will be used, then this provides a means of reusing those curvatures.

    relative_scores: bool, optional
        Return relative scores ranging from 0.0 to 1.0 (from worst to best) rather than the raw values.

    Returns
    -------
    Dict[int, float]

        A dictionary mapping the number of RC elements to its corresponding score.
    """
    f: Frequencies = tests[0].get_frequencies()
    if subdivided_frequencies is not None:
        f = subdivided_frequencies
    elif subdivision > 0:
        f = subdivide_frequencies(f, subdivision=subdivision)

    circuits: Dict[int, Circuit] = {t.num_RC: t.circuit for t in tests}
    circuits = _truncate_circuits(circuits, lower_limit, upper_limit)

    scores: Dict[int, float] = {}
    if curvatures is not None:
        for num_RC, circuit in circuits.items():
            if num_RC in curvatures:
                scores[num_RC] = norm(curvatures[num_RC])
            else:
                scores[num_RC] = norm(
                    calculate_curvatures(
                        circuit.get_impedances(f)
                    )
                )
    else:
        for num_RC, circuit in circuits.items():
            scores[num_RC] = norm(
                calculate_curvatures(
                    circuit.get_impedances(f)
                )
            )

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
