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

from warnings import (
    catch_warnings,
    filterwarnings,
)
from numpy import (
    arccos,
    argmin,
    argwhere,
    array,
    complex128,
    dot,
    float64,
    int64,
    nan,
    ones,
    sign,
    sin,
    sqrt,
    zeros,
)
from numpy.linalg import (
    det,
    norm,
)
from numpy.typing import NDArray
from pyimpspec.circuit.circuit import Circuit
from pyimpspec.typing.aliases import (
    ComplexImpedance,
    ComplexImpedances,
)
from pyimpspec.typing.helpers import (
    List,
    Tuple,
)


def _find_x_axis_intersections(
    x_center: float,
    y_center: float,
    radius: float,
) -> List[Tuple[float, float]]:
    from sympy.geometry import (
        Circle,
        Line,
        Point,
    )

    center = Point(x_center, y_center)
    circle = Circle(center, radius)
    x_axis = Line(Point(0.0, 0.0), slope=0.0)

    return [(float(point.x), float(point.y)) for point in circle.intersection(x_axis)]


def _get_osculating_circle(
    circuit: Circuit,
    f: List[float],
    admittance: bool,
) -> Tuple[float, float, float]:
    if not (len(f) == 3):
        raise ValueError(f"Expected a list with three items instead of {f=}")

    X_circle: NDArray[complex128] = circuit.get_impedances(f) ** (
        -1 if admittance else 1
    )
    kappa: float64 = _fit_osculating_circle(*X_circle)
    if kappa == 0.0:
        return (nan, nan, 0.0)

    radius: float64 = abs(1 / kappa)

    X_tangent: NDArray[complex128] = circuit.get_impedances(
        [f[1] * (1 + 1e-6), f[1] * (1 - 1e-6)]
    ) ** (-1 if admittance else 1)
    slope: float64 = (X_tangent[1] - X_tangent[0]).imag / (
        X_tangent[1] - X_tangent[0]
    ).real
    normal: float64 = -1 / slope

    dx: float64 = radius / sqrt(1 + normal**2)
    dy: float64 = normal * dx

    x_center: float64 = X_circle[1].real
    y_center: float64 = X_circle[1].imag

    centers: NDArray[complex128] = zeros(2, dtype=complex128)
    centers[0] = complex(x_center + dx, y_center + dy)
    centers[1] = complex(x_center - dx, y_center - dy)

    vectors: NDArray[complex128] = zeros((2, 3), dtype=complex128)
    distances: NDArray[float64] = zeros((2, 3), dtype=float64)

    for i in range(0, 2):
        for j in range(0, 3):
            vectors[i][j] = X_circle[j] - centers[i]
            distances[i][j] = abs(vectors[i][j])

    sums_of_distances: List[float] = [abs(sum(row) - radius * 3) for row in distances]
    i: int = argmin(sums_of_distances)

    x_center, y_center = centers[i].real, centers[i].imag

    return (x_center, y_center, kappa)


def _fit_osculating_circle(
    Z_i: ComplexImpedance,
    Z_j: ComplexImpedance,
    Z_k: ComplexImpedance,
) -> float64:
    # The initial slower implementation that is still used as part of
    # suggesting a representation via the _get_osculating_circle function.
    # This implementation is also used to verify the correctness of the faster
    # implementation (i.e., the calculate_curvatures function).
    a: NDArray[float64] = array([(Z_j - Z_i).real, (Z_j - Z_i).imag])
    b: NDArray[float64] = array([(Z_k - Z_j).real, (Z_k - Z_j).imag])
    c: NDArray[float64] = array([(Z_k - Z_i).real, (Z_k - Z_i).imag])

    a_dot_b: float64 = dot(a, b)
    a_norm: float64 = norm(a)
    b_norm: float64 = norm(b)
    a_norm_dot_b_norm: float64 = dot(a_norm, b_norm)
    cos_alpha: float64 = a_dot_b / a_norm_dot_b_norm

    # To handle potential floating point inaccuracy
    # that could cause issues when cos_alpha is provided as
    # input to numpy.arccos
    if cos_alpha < -1.0:
        cos_alpha = -1.0
    elif cos_alpha > 1.0:
        cos_alpha = 1.0
    try:
        with catch_warnings():
            filterwarnings("error", category=RuntimeWarning)
            alpha: float64 = arccos(cos_alpha)
    except RuntimeWarning as e:
        print(a_dot_b, a_norm, b_norm, a_norm_dot_b_norm)
        raise e

    abs_kappa: float64 = 2 * sin(alpha) / norm(c)

    # Determine the direction of the curvature using the sign of the following
    # determinant (<0 means clockwise, >0 means counter-clockwise in a
    # typical Nyquist plot where -Z" is plotted versus Z').
    matrix: NDArray[float64] = ones((3, 3), dtype=float64)
    matrix[:, 0] = [Z_i.real, Z_j.real, Z_k.real]
    matrix[:, 1] = [-Z_i.imag, -Z_j.imag, -Z_k.imag]

    return abs_kappa * sign(det(matrix))


def calculate_curvatures(Z: ComplexImpedances) -> NDArray[float64]:
    r"""
    Estimate the curvatures of an impedance spectrum.

    References:

    - `C. Plank, T. Rüther, and M.A. Danzer, 2022, 2022 International Workshop on Impedance Spectroscopy (IWIS), 1-6 <https://doi.org/10.1109/IWIS57888.2022.9975131>`_
    - `V. Yrjänä and J. Bobacka, 2024, Electrochim. Acta, 504, 144951 <https://doi.org/10.1016/j.electacta.2024.144951>`_

    Parameters
    ----------
    Z: ComplexImpedances
        The impedances to use when estimating the curvatures using osculating circles.

    Returns
    -------
    NDArray[float64]

        An array of curvatures.
        If the impedances were sorted in order of decreasing frequency, then a curvature with a negative sign indicates a clockwise motion within the context of a plot of :math:`-{\rm Im}(Z)\ {\rm versus}\ {\rm Re}(Z)`.
    """
    kappa: NDArray[float64] = zeros((Z.size - 2, 3), dtype=float64)

    a: NDArray[float64] = zeros((Z.size - 2, 2), dtype=float64)
    a[:, 0] = (Z[1:-1] - Z[:-2]).real
    a[:, 1] = (Z[1:-1] - Z[:-2]).imag

    b: NDArray[float64] = zeros((Z.size - 2, 2), dtype=float64)
    b[:, 0] = (Z[2:] - Z[1:-1]).real
    b[:, 1] = (Z[2:] - Z[1:-1]).imag

    c: NDArray[float64] = zeros((Z.size - 2, 2), dtype=float64)
    c[:, 0] = (Z[2:] - Z[:-2]).real
    c[:, 1] = (Z[2:] - Z[:-2]).imag

    a_dot_b: NDArray[float64] = zeros(Z.size - 2, dtype=float64)
    a_norm: NDArray[float64] = zeros(Z.size - 2, dtype=float64)
    b_norm: NDArray[float64] = zeros(Z.size - 2, dtype=float64)
    a_norm_dot_b_norm: NDArray[float64] = zeros(Z.size - 2, dtype=float64)
    cos_alpha: NDArray[float64] = zeros(Z.size - 2, dtype=float64)
    c_norm: NDArray[float64] = zeros(Z.size - 2, dtype=float64)

    i: int
    for i in range(0, a.shape[0]):
        a_dot_b[i] = dot(a[i, :], b[i, :])
        a_norm[i] = norm(a[i, :])
        b_norm[i] = norm(b[i, :])
        a_norm_dot_b_norm[i] = dot(a_norm[i], b_norm[i])
        cos_alpha[i] = a_dot_b[i] / a_norm_dot_b_norm[i]
        c_norm[i] = norm(c[i, :])

    indices: NDArray[int64] = argwhere(cos_alpha < -1.0).flatten()
    if indices.size > 0:
        cos_alpha[indices] = -1.0

    indices = argwhere(cos_alpha > 1.0).flatten()
    if indices.size > 0:
        cos_alpha[indices] = 1.0

    try:
        with catch_warnings():
            filterwarnings("error", category=RuntimeWarning)
            alpha: NDArray[float64] = arccos(cos_alpha)
    except RuntimeWarning as e:
        print(a_dot_b, a_norm, b_norm, a_norm_dot_b_norm)
        raise e

    abs_kappa: NDArray[float64] = 2 * sin(alpha) / c_norm

    matrix: NDArray[float64] = ones((Z.size, 3), dtype=float64)
    matrix[:, 0] = Z.real
    matrix[:, 1] = -Z.imag
    determinant: NDArray[float64] = zeros(Z.size - 2, dtype=float64)

    for i in range(0, a.shape[0]):
        determinant[i] = det(matrix[i:i+3, :])

    kappa: NDArray[float64] = abs_kappa * sign(determinant)

    return kappa


def _calculate_sign_change_distances(kappas: NDArray[float64]) -> NDArray[int64]:
    distances: List[int] = []
    previous_sign: int = sign(kappas[0])
    previous_sign_index: int = 0

    i: int
    for i in range(1, len(kappas)):
        if kappas[i] == 0.0:
            continue

        current_sign: int = sign(kappas[i])
        if current_sign != previous_sign:
            distances.append(i - previous_sign_index)
            previous_sign = current_sign
            previous_sign_index = i

    if len(distances) == 0:
        distances.append(i)

    return array(distances)
