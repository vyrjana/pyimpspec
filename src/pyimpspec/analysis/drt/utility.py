# pyimpspec is licensed under the GPLv3 or later (https://www.gnu.org/licenses/gpl-3.0.html).
# Copyright 2023 pyimpspec developers
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
    Callable,
    List,
    Tuple,
)
from warnings import (
    catch_warnings,
    filterwarnings,
)
from numpy import (
    float64,
    inf,
    sqrt,
    log10 as log,
)
from numpy.typing import NDArray


def _l_curve_corner_search(
    l_curve_P: Callable,
    minimum: float,
    maximum: float,
    epsilon: float = 0.01,
) -> float:
    """
    Implementation of algorithm 1 in DOI:10.1088/2633-1357/abad0d

    l_curve_P must be a function that takes a regularization parameter (or
    lambda value)and returns a tuple containing the base-10 log of the
    residual norm squared and the base-10 log of the solution norm squared.

    Returns
    -------
    float
        The optimal regularization parameter
    """
    # 'Line N' comments refer to the line numbers in algorithm 1
    # in DOI:10.1088/2633-1357/abad0d
    phi = (1 + sqrt(5)) / 2  # Line 3

    def update_lambda(lambdas: List[float], index: int):
        assert index == 1 or index == 2, index
        xs: NDArray[float64] = log(lambdas)
        if index == 1:
            lambdas[1] = 10 ** ((xs[3] + phi * xs[0]) / (1 + phi))
        elif index == 2:
            lambdas[2] = 10 ** (xs[0] + (xs[3] - xs[1]))

    def menger(Ps: List[Tuple[float64, float64]]) -> float64:
        xi: List[float64] = [_[0] for _ in Ps]
        eta: List[float64] = [_[1] for _ in Ps]
        j: int = 0
        k: int = 1
        l: int = 2
        with catch_warnings():
            filterwarnings("ignore", "divide by zero encountered in scalar divide")
            filterwarnings("ignore", "invalid value encountered in scalar divide")
            filterwarnings("ignore", "invalid value encountered in scalar subtract")
            num: float64 = 2 * (
                xi[j] * eta[k]
                + xi[k] * eta[l]
                + xi[l] * eta[j]
                - xi[j] * eta[l]
                - xi[k] * eta[j]
                - xi[l] * eta[k]
            )
            den: float64 = (
                ((xi[k] - xi[j]) ** 2 + (eta[k] - eta[j]) ** 2)
                * ((xi[l] - xi[k]) ** 2 + (eta[l] - eta[k]) ** 2)
                * ((xi[j] - xi[l]) ** 2 + (eta[j] - eta[l]) ** 2)
            ) ** (1 / 2)
            return num / den

    lambdas: List[float] = [minimum, 1.0, 1.0, maximum]  # Line 1
    update_lambda(lambdas, index=1)  # Line 4
    update_lambda(lambdas, index=2)  # Line 5
    Ps: List[Tuple[float64, float64]] = []
    lm: float
    for lm in lambdas:  # Line 6
        Ps.append(l_curve_P(lm))  # Line 7
    optimal_lambda: float = -1.0
    while (lambdas[3] - lambdas[0]) / lambdas[3] >= epsilon:
        C_2: float64 = menger(Ps[:-1])  # Line 10
        C_3: float64 = menger(Ps[1:])  # Line 11
        while C_3 <= 0.0:  # Line 18
            lambdas[3], Ps[3] = lambdas[2], Ps[2]  # Line 13
            lambdas[2], Ps[2] = lambdas[1], Ps[1]  # Line 14
            update_lambda(lambdas, index=1)  # Line 15
            Ps[1] = l_curve_P(lambdas[1])  # Line 16
            C_3 = menger(Ps[1:])  # Line 17
        if C_2 > C_3:  # Line 19
            optimal_lambda = lambdas[1]  # Line 20
            lambdas[3], Ps[3] = lambdas[2], Ps[2]  # Line  21
            lambdas[2], Ps[2] = lambdas[1], Ps[1]  # Line 22
            update_lambda(lambdas, index=1)  # Line 23
            Ps[1] = l_curve_P(lambdas[1])  # Line 24
        else:  # Line
            optimal_lambda = lambdas[2]  # Line 26
            lambdas[0], Ps[0] = lambdas[1], Ps[1]  # Line 27
            lambdas[1], Ps[1] = lambdas[2], Ps[2]  # Line 28
            update_lambda(lambdas, index=2)  # Line 29
            Ps[2] = l_curve_P(lambdas[2])  # Line 30
    return optimal_lambda  # Line 33
