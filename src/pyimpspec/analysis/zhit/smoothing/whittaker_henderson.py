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

# Whittaker-Henderson smoothing that was included in the article:
# https://doi.org/10.1021/acsmeasuresciau.1c00054
#
# Ported from the Java source code included in the correction:
# https://doi.org/10.1021/acsmeasuresciau.3c00017
#
# The original Java source code was written by Michael Schmid
# and was licensed under GPLv3.

from numpy import (
    cos,
    float64,
    pi,
    sqrt,
    zeros,
)
from numpy.typing import NDArray
from pyimpspec.typing.helpers import (
    List,
    _is_floating_array,
    _is_integer,
)


def _bandwidth_to_lambda(order: int, bandwidth: float) -> float:
    """
    Calculates the lambda smoothing parameter for a given penalty derivative
    order, given the desired band width, i.e., the frequency where the response
    decreases to -3 dB, i.e., 1/sqrt(2). This band width is valid for points
    far from the boundaries of the data.
    """
    if not (0 < bandwidth < 0.5):
        raise ValueError(f"Expected 0 < {bandwidth=} < 0.5")

    omega: float = 2 * pi * bandwidth
    cos_term: float = 2 * (1 - cos(omega))
    cos_power: float = cos_term

    i: int
    for i in range(1, order):
        # finally results in (2-2*cos(omega))^order
        cos_power *= cos_term

    return (sqrt(2) - 1) / cos_power


def _savitzky_golay_bandwidth(degree: int, m: int) -> float:
    """
    Calculates the bandwidth of a traditional Savitzky-Golay filter.

    Returns the -3 dB-bandwidth of the Savitzky-Golay filter, i.e. the
    frequency where the response is 1/sqrt(2). The sampling frequency
    is defined as f = 1. For degree up to 10, the accuracy is typically
    much better than 1%; higher errors occur only for the lowest m values
    where the Savitzky-Golay filter is defined (worst case: 4% error at
    degree = 10, m = 6).
    """
    return 1.0 / (
        6.352 * (m + 0.5) / (degree + 1.379) - (0.513 + 0.316 * degree) / (m + 0.5)
    )


def _make_D_prime_D_matrix(order: int, size: int) -> List[List[float]]:
    """
    Creates a symmetric band-diagonal matrix D'*D where D is the n-th
    derivative matrix and D' its transpose.
    """
    # Maximum penalty derivative order and corresponding coefficients
    max_order: int = 5
    if not (1 <= order <= max_order):
        raise ValueError(
            f"Expected the order to be a positive value less than or equal to {max_order}"
        )

    if not (size >= order):
        raise ValueError(f"Expected {size=} >= {order=}")

    coeffs: List[int] = [
        [-1, 1],
        [1, -2, 1],
        [-1, 3, -3, 1],
        [1, -4, 6, -4, 1],
        [-1, 5, -10, 10, -5, 1],
    ][order - 1]

    out: List[List[float]] = [[]] * (order + 1)

    d: int  # Distance from diagonal
    for d in range(0, order + 1):
        out[d] = [0.0] * (size - d)

    for d in range(0, order + 1):
        length: int = len(out[d])

        i: int
        for i in range(0, (length + 1) // 2):
            total: float = 0.0

            j: int = max((0, i - length + len(coeffs) - d))
            while j < i + 1 and j < len(coeffs) - d:
                total += coeffs[j] * coeffs[j + d]
                j += 1

            out[d][i] = total
            out[d][length - 1 - i] = total

    return out


def _times_lambda_plus_identity(b: List[List[float]], lmbd: float) -> List[List[float]]:
    """
    Modifies a symmetric band-diagonal matrix b so that the output is
    1 + lambda*b where 1 is the identity matrix.
    """
    i: int
    for i in range(0, len(b[0])):
        b[0][i] = 1.0 + b[0][i] * lmbd

    d: int
    for d in range(1, len(b)):
        for i in range(0, len(b[d])):
            b[d][i] = b[d][i] * lmbd

    return b


def _cholesky_L(b: List[List[float]]) -> List[List[float]]:
    """
    Cholesky decomposition of a symmetric band-diagonal matrix b.
    The input is replaced by the lower left trianglar matrix.
    """
    n: int = len(b[0])
    dmax = len(b) - 1

    i: int
    for i in range(0, n):
        j: int
        for j in range(max((0, i - dmax)), i + 1):
            total: float = 0.0

            k: int
            for k in range(max((0, i - dmax)), j):
                total += b[i - k][k] * b[j - k][k]

            if i == j:
                sqrtArg: float = b[0][i] - total
                if not (sqrtArg > 0.0):
                    raise ValueError("Matrix is not positive definite")

                b[0][i] = sqrt(sqrtArg)
            else:
                dAij: int = i - j
                b[dAij][j] = 1.0 / b[0][j] * (b[dAij][j] - total)

    return b


def _solve(b: List[List[float]], vec: NDArray[float64]) -> NDArray[float64]:
    """
    Solves the equation b*y = vec for y (forward substitution) and
    thereafter b'*x = y, where b' is the transposed (back substitution)

    Returns vector x resulting from forward and back subsitution. If b is the
    result of Cholesky decomposition, then x is the solution for A*x = vec.
    For data smoothing, x holds the smoothed data.
    """
    out: NDArray[float64] = zeros(vec.shape, dtype=float64)
    n: int = len(b[0])
    dmax: int = len(b) - 1

    i: int
    for i in range(0, n):
        total: float = 0.0

        j: int
        for j in range(max((0, i - dmax)), i):
            total += b[i - j][j] * out[j]

        out[i] = (vec[i] - total) / b[0][i]

    i = n - 1
    while i >= 0:
        total = 0.0

        for j in range(i + 1, min((i + dmax + 1, n))):
            total += b[j - i][i] * out[j]

        out[i] = (out[i] - total) / b[0][i]

        i -= 1

    return out


def _smooth(data: NDArray[float64], degree: int, m: int) -> NDArray[float64]:
    """
    Interface for using the Whittaker-Henderson smoothing algorithm with
    parameters associated with Savitzky-Golay filters.

    Minimizes

       sum(f - y)^2 + sum(lambda * f'(p))

    where y are the data, f are the smoothed data, and f'(p) is the p-th
    derivative of the smoothed function evaluated numerically. In other words,
    the filter imposes a penalty on the p-th derivative of the data, which is
    taken as a measure of non-smoothness. Smoothing increases with increasing
    value of lambda. The current implementation works up to p = 5; usually one
    should use p = 2 or 3.

    For points far from the boundaries of the data series, the frequency
    response of the smoother is given by

      1/(1+lambda*(2-2*cos(omega))^2p)

    where n is the order of the penalized derivative and omega = 2*pi*f/fs,
    with fs being the sampling frequency (reciprocal of the distance between
    the data points).

    Note that strong smoothing leads to numerical noise (which is smoothed
    similar to the input data, thus not obvious in the output). For
    lambda = 1e9, the noise is about 1e-6 times the magnitude of the data.
    Since higher p values require a higher value of lambda for the same extent
    of smoothing (the same band width), numerical noise is increasingly
    bothersome for large p, not for p <= 2.

    Parameters
    ----------
    data: NDArray[float64]
        The data to be smoothed.

    degree: int
        The degree of the polynomial fit used in the Savitzky-Golay filter.

    m: int
        Half-width of the Savitzky-Golay kernel. Kernel size of the
        Savitzky-Golay filter is 2*m + 1. Schmid recommends the following
        limits:

        degree |  m
           2   | 700
           4   | 190
           6   | 100
           8   | 75

    Returns
    -------
    NDArray[float64]
        The smoothed data.
    """
    if not _is_floating_array(data):
        raise TypeError(f"Expected an array of floats instead of {data=}")
    elif not (len(data.shape) == 1):
        raise ValueError(f"Expected {len(data.shape)=} == 1")

    if not _is_integer(degree):
        raise TypeError(f"Expected an integer instead of {degree=}")

    if not _is_integer(m):
        raise TypeError(f"Expected an integer instead of {m=}")

    order: int = degree // 2 + 1
    matrix: List[List[float]] = _make_D_prime_D_matrix(order, len(data))

    bandwidth: float = _savitzky_golay_bandwidth(degree, m)
    lmbd: float = _bandwidth_to_lambda(order, bandwidth)
    matrix = _times_lambda_plus_identity(matrix, lmbd)

    matrix = _cholesky_L(matrix)
    if not (len(data) == len(matrix[0])):
        raise ValueError(f"Expected {len(data)=} == {len(matrix[0])=}")

    return _solve(matrix, data)
