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
    sign,
)
from numpy.typing import NDArray
from pyimpspec.analysis.kramers_kronig.result import KramersKronigResult
from pyimpspec.circuit.circuit import Circuit
from pyimpspec.typing.aliases import Frequencies
from pyimpspec.typing.helpers import (
    Dict,
    List,
    Optional,
    _is_floating,
)
from .utility.common import (
    _generate_pseudo_chisqr_offsets,
    _truncate_circuits,
    subdivide_frequencies,
)
from .utility.osculating_circle import calculate_curvatures


def _count_sign_changes(kappas: NDArray[float64]) -> int:
    previous_sign: int = sign(kappas[0])
    n: int = 1

    for i in range(1, len(kappas)):
        if kappas[i] == 0.0:
            continue

        current_sign: int = sign(kappas[i])
        if current_sign != previous_sign:
            n += 1
            previous_sign = current_sign

    return n


def _suggest(
    circuits: Dict[int, Circuit],
    pseudo_chisqrs: Dict[int, float],
    f: Frequencies,
    offset_factor: float,
    lower_limit: int,
    upper_limit: int,
    curvatures: Optional[Dict[int, NDArray[float64]]],
    relative_scores: bool,
) -> Dict[int, float]:
    if not _is_floating(offset_factor):
        raise TypeError(f"Expected a float instead of {offset_factor=}")
    elif not (0.0 <= offset_factor < 1.0):
        raise ValueError(
            f"Expected a value in the range [0.0, 1.0) instead of {offset_factor=}"
        )

    circuits = _truncate_circuits(circuits, lower_limit, upper_limit)

    pseudo_chisqrs = {
        num_RC: pseudo_chisqr
        for num_RC, pseudo_chisqr in pseudo_chisqrs.items()
        if num_RC in circuits
    }

    offsets: Dict[int, float] = _generate_pseudo_chisqr_offsets(
        pseudo_chisqrs,
        factor=offset_factor,
    )

    scores: Dict[int, float] = {}
    if curvatures is not None:
        for num_RC, circuit in circuits.items():
            if num_RC in curvatures:
                scores[num_RC] = (
                    _count_sign_changes(
                        curvatures[num_RC]
                    ) + offsets[num_RC]
                )
            else:
                scores[num_RC] = (
                    _count_sign_changes(
                        calculate_curvatures(
                            circuit.get_impedances(f)
                        )
                    ) + offsets[num_RC]
                )
    else:
        for num_RC, circuit in circuits.items():
            scores[num_RC] = (
                _count_sign_changes(
                    calculate_curvatures(
                        circuit.get_impedances(f)
                    )
                ) + offsets[num_RC]
            )

    if not relative_scores:
        return scores

    min_score: float = min(scores.values())
    max_score: float = max(scores.values()) - min_score
    if max_score == 0.0:
        return {num_RC: 1.0 for num_RC in scores.keys()}

    return {
        num_RC: 1.0 - float((value - min_score) / max_score)
        for num_RC, value in scores.items()
    }


def suggest(
    tests: List[KramersKronigResult],
    lower_limit: int = 0,
    upper_limit: int = 0,
    subdivision: int = 4,
    subdivided_frequencies: Optional[Frequencies] = None,
    curvatures: Optional[Dict[int, NDArray[float64]]] = None,
    offset_factor: float = 1e-1,
    relative_scores: bool = True,
) -> Dict[int, float]:
    """
    Suggest the optimal number of RC elements to use based on the number of sign changes of curvatures of the fitted impedance spectrum.
    The curvatures at each point of the fitted impedance spectrum is approximated using an osculating circle.
    An increasing number of sign changes of these curvatures results from oscillations brought on by overfitting.
    Thus, a minimum of the number of sign changes should coincide with the desired optimum.

    The method is modified by subdividing the frequency intervals, which makes the method less prone to suggesting circuits that produce oscillating impedance spectra.
    Small offsets are also added to the number of sign changes based on the corresponding pseudo chi-squared values in order to act as tiebreakers in case there are multiple numbers of RC elements that correspond to the same number of sign changes.
    This modified approach is used by default, but the original approach can be used by setting ``subdivision = 0`` and ``offset_factor = 0.0``.

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

    offset_factor: float, optional
        The factor that an offset is multiplied by when it is being added to a number of sign changes.
        Must be in the range [0.0, 1.0).

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

    return _suggest(
        circuits={t.num_RC: t.circuit for t in tests},
        pseudo_chisqrs={t.num_RC: t.pseudo_chisqr for t in tests},
        f=f,
        offset_factor=offset_factor,
        lower_limit=lower_limit,
        upper_limit=upper_limit,
        curvatures=curvatures,
        relative_scores=relative_scores,
    )
