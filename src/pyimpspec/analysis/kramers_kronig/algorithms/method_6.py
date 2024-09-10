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
    fromiter,
    int64,
)
from numpy.typing import NDArray
from pyimpspec.analysis.kramers_kronig.result import KramersKronigResult
from pyimpspec.circuit.circuit import Circuit
from pyimpspec.typing.helpers import (
    Dict,
    List,
    Tuple,
)
from .utility.common import _truncate_circuits
from .utility.cubic import (
    _cubicish_function,
    _fit_cubic_function,
    _fit_cubicish_function,
)
from .utility.tau_var_sum import _calculate_log_sum_abs_tau_var


def _approximate_trend(circuits: Dict[int, Circuit]) -> Tuple[NDArray[float64], Tuple[float, ...]]:
    x: NDArray[int64] = fromiter(
        (num_RC for num_RC in circuits.keys()),
        dtype=int64,
        count=len(circuits),
    )
    raw_y: NDArray[float64] = fromiter(
        map(_calculate_log_sum_abs_tau_var, circuits.values()),
        dtype=float64,
        count=len(circuits),
    )

    num_points: int = 5
    threshold: float64 = min(raw_y[:num_points])

    n: int
    y: float64
    for n, y in enumerate(raw_y):
        if n < num_points:
            continue
        elif y <= threshold:
            break

    p: Tuple[float64, ...] = _fit_cubic_function(x[:n], raw_y[:n])
    n += 10
    p = _fit_cubicish_function(x[:n], raw_y[:n], p0=p)

    return (_cubicish_function(x, *p), p)


def _suggest(
    circuits: Dict[int, Circuit],
    lower_limit: int,
    upper_limit: int,
    relative_scores: bool,
) -> Dict[int, float]:
    smooth_y: Dict[int, float64] = {
        num_RC: v
        for num_RC, v in zip(
            circuits.keys(),
            _approximate_trend(circuits)[0],
        )
    }

    if not relative_scores:
        return smooth_y

    threshold: float64 = smooth_y[min(smooth_y.keys())]
    scores: Dict[int, float64] = {num_RC: threshold for num_RC in circuits.keys()}
    circuits = _truncate_circuits(circuits, lower_limit, upper_limit)
    scores.update(
        {
            num_RC: (smooth_y[num_RC] if smooth_y[num_RC] > threshold else threshold)
            for num_RC in circuits.keys()
        }
    )

    min_score: float = min(scores.values())
    max_score: float = max(scores.values()) - min_score
    if max_score == 0.0:
        return {num_RC: 1.0 for num_RC in scores.keys()}

    return {
        num_RC: float((scores[num_RC] - min_score) / max_score)
        for num_RC in circuits.keys()
    }


def suggest(
    tests: List[KramersKronigResult],
    lower_limit: int = 0,
    upper_limit: int = 0,
    relative_scores: bool = True,
) -> Dict[int, float]:
    """
    Suggest the optimal number of RC elements to use based on the approximate position of the apex in a plot of |log sum abs tau R| versus the number of RC elements.
    If the tests were performed on the admittance representation of the immittance data, then :math:`C_k` is substituted for :math:`R_k`.
    The sum grows initially as the number of RC elements increases.
    However, the magnitudes of the fitted :math:`R_k` (or :math:`C_k`) also tend to increase, which causes the magnitudes of the corresponding :math:`C_k` (or :math:`R_k`) to decrease.
    Thus, the sum begins to decline despite the increasing number of RC elements and the fitted impedance spectrum begins to oscillate (i.e., overfitting takes place).
    The apex should coincide with or be near the optimum.

    References:

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

    relative_scores: bool, optional
        Return relative scores ranging from 0.0 to 1.0 (from worst to best) rather than the raw values.

    Returns
    -------
    Dict[int, float]

        A dictionary mapping the number of RC elements to its corresponding score.
    """
    return _suggest(
        circuits={t.num_RC: t.circuit for t in tests},
        lower_limit=lower_limit,
        upper_limit=upper_limit,
        relative_scores=relative_scores,
    )
