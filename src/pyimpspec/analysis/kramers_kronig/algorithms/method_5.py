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
    float64,
    mean,
)
from numpy.typing import NDArray
from pyimpspec.analysis.kramers_kronig.result import KramersKronigResult
from pyimpspec.circuit.circuit import Circuit
from pyimpspec.typing.aliases import Frequencies
from pyimpspec.typing.helpers import (
    Dict,
    List,
    Optional,
)
from .utility.osculating_circle import (
    _calculate_sign_change_distances,
    calculate_curvatures,
)
from .utility.common import (
    _truncate_circuits,
    subdivide_frequencies,
)


def _suggest(
    circuits: Dict[int, Circuit],
    f: Frequencies,
    lower_limit: int,
    upper_limit: int,
    curvatures: Optional[Dict[int, NDArray[float64]]],
    relative_scores: bool,
) -> Dict[int, float]:
    circuits = _truncate_circuits(circuits, lower_limit, upper_limit)

    scores: Dict[int, float] = {}
    if curvatures is not None:
        for num_RC, circuit in circuits.items():
            if num_RC in curvatures:
                scores[num_RC] = mean(
                    _calculate_sign_change_distances(
                        curvatures[num_RC]
                    )
                )
            else:
                scores[num_RC] = mean(
                    _calculate_sign_change_distances(
                        calculate_curvatures(
                            circuit.get_impedances(f)
                        )
                    )
                )
    else:
        for num_RC, circuit in circuits.items():
            scores[num_RC] = mean(
                _calculate_sign_change_distances(
                    calculate_curvatures(
                        circuit.get_impedances(f)
                    )
                )
            )

    if not relative_scores:
        return scores

    min_score: float = min(scores.values())
    max_score: float = max(scores.values()) - min_score
    if max_score == 0.0:
        return {num_RC: 1.0 for num_RC in scores.keys()}

    return {
        num_RC: float((value - min_score) / max_score)
        for num_RC, value in scores.items()
    }


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
    Suggest the optimal number of RC elements to use based on the average distance between sign changes of the curvatures of the fitted impedance spectrum.
    The curvatures at each point of the fitted impedance spectrum is approximated using an osculating circle.
    The largest average distance should occur at the lowest number of RC elements, but the optimum coincides with a local maximum at an intermediate number of RC elements.
    The average distance will tend towards one as the number of RC elements is incremented further.

    References:

    - `V. Yrjänä and J. Bobacka, 2024, Electrochim. Acta, 504, 144951 <https://doi.org/10.1016/j.electacta.2024.144951>`_
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
    m: int = len(f)
    if subdivided_frequencies is not None:
        f = subdivided_frequencies
    elif subdivision > 0:
        f = subdivide_frequencies(f, subdivision=subdivision)
    n: int = len(f)

    scores: Dict[int, float] = _suggest(
        circuits={t.num_RC: t.circuit for t in tests},
        f=f,
        lower_limit=lower_limit,
        upper_limit=upper_limit,
        curvatures=curvatures,
        relative_scores=relative_scores,
    )

    return {num_RC: value * m / n for num_RC, value in scores.items()}
