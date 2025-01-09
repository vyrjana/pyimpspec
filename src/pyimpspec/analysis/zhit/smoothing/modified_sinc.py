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

# Modified sinc kernel with linear extrapolation that was included in the article:
# https://doi.org/10.1021/acsmeasuresciau.1c00054
#
# Ported from the Java source code included in the correction:
# https://doi.org/10.1021/acsmeasuresciau.3c00017
#
# The original Java source code was written by Michael Schmid
# and was licensed under GPLv3.

from dataclasses import dataclass
from typing import (
    List,
    Optional,
    Tuple,
)
from numpy import (
    exp,
    float64,
    ceil,
    cos,
    isnan,
    nan,
    pi,
    sin,
    zeros,
)
from numpy.typing import NDArray


def _smooth_except_boundaries(
    data: NDArray[float64],
    kernel: NDArray[float64],
) -> NDArray[float64]:
    """
    Smooths the data with the parameters passed with the constructor,
    except for the near-end points.
    """
    out: NDArray[float64] = zeros(data.shape, dtype=float64)
    radius: int = len(kernel) - 1  # how many additional points we need

    for i in range(radius, len(data) - radius):
        total: float = kernel[0] * data[i]

        for j in range(1, len(kernel)):
            total += kernel[j] * (data[i - j] + data[i + j])

        out[i] = total

    return out


def _bandwidth_to_m(is_MS1: bool, degree: int, bandwidth: float) -> int:
    """
    Calculates the kernel halfwidth m that comes closest to the desired
    band width, i.e., the frequency where the response decreases to
    -3 dB, i.e., 1/sqrt(2).
    """
    if not (0 < bandwidth < 0.5):
        raise ValueError(
            f"Calculated bandwidth is out of bounds: 0 < {bandwidth=:.3g} < 0.5"
        )

    radius: float = (
        ((0.27037 + 0.24920 * degree) / bandwidth - 1.0)
        if is_MS1
        else ((0.74548 + 0.24943 * degree) / bandwidth - 1.0)
    )

    return int(round(radius))


def _make_kernel(
    is_MS1: bool,
    degree: int,
    m: int,
    coeffs: Optional[NDArray[float64]],
) -> NDArray[float64]:
    """
    Creates an MS or MS1 kernel and returns it.
    """
    kernel: NDArray[float64] = zeros(m + 1, dtype=float64)
    num_coeffs: int = 0 if coeffs is None else len(coeffs)
    total: float = 0.0

    i: int
    for i in range(0, m + 1):
        # x=0 at center, x=1 at zero
        x: float = i * (1.0 / (m + 1))
        sinc_arg: float = pi * 0.5 * (degree + (2 if is_MS1 else 4)) * x
        k: float = 1 if i == 0 else sin(sinc_arg) / sinc_arg

        if coeffs is not None:
            j: int
            for j in range(0, num_coeffs):
                if is_MS1:
                    # shorter kernel version, needs more correction terms
                    k += coeffs[j] * x * sin((j + 1) * pi * x)
                else:
                    # start at 1 for degree 6, 10; at 2 for degree 8
                    nu: int = 2 if ((degree // 2) & 0x1) == 0 else 1
                    k += coeffs[j] * x * sin((2 * j + nu) * pi * x)

        # decay alpha=2: 13.5% at end without correction, 2sqrt2 sigma
        decay: float = 2 if is_MS1 else 4
        k *= (
            exp(-x * x * decay)
            + exp(-(x - 2) * (x - 2) * decay)
            + exp(-(x + 2) * (x + 2) * decay)
            - 2 * exp(-decay)
            - exp(-9 * decay)
        )

        kernel[i] = k

        total += k
        if i > 0:
            # off-center kernel elements appear twice
            total += k

    for i in range(0, m + 1):
        # normalize the kernel to total=1
        kernel[i] *= 1.0 / total

    return kernel


def _get_correction_data(is_MS1, degree: int) -> Optional[List[List[float]]]:
    """
    Returns degree-specific correction data or None if not required.
    """
    if is_MS1:
        if degree < 4:
            return None
        elif degree == 4:
            return [
                [0.021944195, 0.050284006, 0.765625],
            ]
        elif degree == 6:
            return [
                [0.0018977303, 0.008476806, 1.2625],
                [0.023064667, 0.13047926, 1.2265625],
            ]
        elif degree == 8:
            return [
                [0.0065903002, 0.057929456, 1.915625],
                [0.0023234477, 0.010298849, 2.2726562],
                [0.021046653, 0.16646601, 1.98125],
            ]
        elif degree == 10:
            return [
                [9.749618e-4, 0.0020742896, 3.74375],
                [0.008975366, 0.09902466, 2.7078125],
                [0.0024195414, 0.010064855, 3.296875],
                [0.019185117, 0.18953617, 2.784961],
            ]

    else:
        if degree < 6:
            return None
        elif degree == 6:
            return [
                [0.001717576, 0.02437382, 1.64375],
            ]
        elif degree == 8:
            return [
                [0.0043993373, 0.088211164, 2.359375],
                [0.006146815, 0.024715371, 3.6359375],
            ]
        elif degree == 10:
            return [
                [0.0011840032, 0.04219344, 2.746875],
                [0.0036718843, 0.12780383, 2.7703125],
            ]

    raise NotImplementedError(f"Unsupported degree: {degree}")


def _get_coefficients(is_MS1: bool, degree: int, m: int) -> Optional[NDArray[float64]]:
    """
    Returns the correction coefficients for a Sinc*Gaussian kernel
    to flatten the passband. Coefficients z for the x*sin((j+1)*pi*x) terms,
    or null if no correction is required.
    """
    correction_data: Optional[List[List[float]]] = _get_correction_data(is_MS1, degree)
    if correction_data is None:
        return None

    coeffs: NDArray[float64] = zeros(len(correction_data), dtype=float64)
    for i, (a, b, c) in enumerate(correction_data):
        coeffs[i] = a + b / (c - m) ** 3

    return coeffs


def _make_fit_weights(is_MS1: bool, degree: int, m: int) -> NDArray[float64]:
    """
    Returns the weights for the linear fit used for linear extrapolation
    at the end. The weight function is a Hann (cos^2) function. For beta=1
    (the beta value for n=4), it decays to zero at the position of the
    first zero of the sinc function in the kernel. Larger beta values lead
    to stronger noise suppression near the edges, but the smoothed curve
    does not follow the input as well as for lower beta (for high degrees,
    also leading to more ringing near the boundaries).
    """
    first_zero: float = (m + 1) / ((1.0 if is_MS1 else 1.5) + 0.5 * degree)
    beta: float = (
        (0.65 + 0.35 * exp(-0.55 * (degree - 4)))
        if is_MS1
        else 0.70 + 0.14 * exp(-0.60 * (degree - 4))
    )
    fit_length: int = int(ceil(first_zero * beta))
    weights: NDArray[float64] = zeros(fit_length, dtype=float64)

    p: int
    for p in range(0, fit_length):
        weights[p] = cos(0.5 * pi / (first_zero * beta) * p) ** 2

    return weights


def _savitzky_golay_bandwidth(degree: int, m: int) -> float:
    """
    Calculates the bandwidth of a traditional Savitzky-Golay filter.
    The -3 dB-bandwidth of the Savitzky-Golay filter, i.e. the frequency where
    the response is 1/sqrt(2). The sampling frequency is defined as f = 1.
    For degree up to 10, the accuracy is typically much better than 1%;
    higher errors occur only for the lowest m values where the Savitzky-Golay
    filter is defined (worst case: 4% error at degree = 10, m = 6).
    """
    return 1.0 / (
        6.352 * (m + 0.5) / (degree + 1.379) - (0.513 + 0.316 * degree) / (m + 0.5)
    )


def _extend_data(
    data: NDArray[float64],
    degree: int,
    m: int,
    fit_weights: NDArray[float64],
) -> NDArray[float64]:
    """
    Extends the data by a weighted fit to a linear function (linear regression).
    At each end, m extrapolated points are appended.
    """
    extended_data: NDArray[float64] = zeros(len(data) + 2 * m, dtype=float64)
    extended_data[m:m + len(data)] = data

    lin_reg: LinearRegression = LinearRegression()

    # Linear fit of first points and extrapolate
    fit_length: int = min((len(fit_weights), len(data)))

    p: int
    for p in range(0, fit_length):
        lin_reg.add_point(p, data[p], fit_weights[p])

    slope: float
    offset: float
    slope, offset = lin_reg.calculate()

    p = -1
    while p >= -m:
        extended_data[m + p] = offset + slope * p
        p -= 1

    # Linear fit of last points and extrapolate
    lin_reg.clear()

    for p in range(0, fit_length):
        lin_reg.add_point(p, data[len(data) - 1 - p], fit_weights[p])

    slope, offset = lin_reg.calculate()

    p = -1
    while p >= -m:
        extended_data[len(data) + m - 1 - p] = offset + slope * p
        p -= 1

    return extended_data


@dataclass
class LinearRegression:
    sum_weights: float = 0.0
    sum_x: float = 0.0
    sum_y: float = 0.0
    sum_xy: float = 0.0
    sum_x2: float = 0.0
    # sum_y2: float = 0.0

    def clear(self):
        self.sum_weights = 0.0
        self.sum_x = 0.0
        self.sum_y = 0.0
        self.sum_xy = 0.0
        self.sum_x2 = 0.0
        # self.sum_y2 = 0.0

    def add_point(self, x: float, y: float, weight: float):
        self.sum_weights += weight
        self.sum_x += weight * x
        self.sum_y += weight * y
        self.sum_xy += weight * x * y
        self.sum_x2 += weight * x**2
        # self.sum_y2 += weight * y**2

    def calculate(self) -> Tuple[float, float]:
        std_x2_times_N: float = self.sum_x2 - self.sum_x * self.sum_x * (
            1 / self.sum_weights
        )
        # std_y2_times_N: float = self.sum_y2 - self.sum_y * self.sum_y * (1 / self.sum_weights)

        slope: float
        if self.sum_weights > 0:
            slope = (
                self.sum_xy - self.sum_x * self.sum_y * (1 / self.sum_weights)
            ) / std_x2_times_N
            if isnan(slope):
                slope = 0.0  # slope 0 if only one x value
        else:
            slope = nan

        offset = (self.sum_y - slope * self.sum_x) / self.sum_weights

        return (
            slope,
            offset,
        )


def _smooth(
    data: NDArray[float64],
    degree: int,
    m: int,
    is_MS1: bool = False,
) -> NDArray[float64]:
    max_degree: int = 10
    if not (2 <= degree <= max_degree and (degree & 0x1) == 0):
        raise ValueError(
            f"Only the following degrees are supported: {', '.join(map(str, range(2, max_degree+1, 2)))}"
        )

    min_m: int = degree // 2 + (1 if is_MS1 else 2)
    if not (m >= min_m):
        raise ValueError(
            f"The kernel half-width must be greater than or equal to {min_m}"
        )

    coeffs: Optional[NDArray[float64]] = _get_coefficients(is_MS1, degree, m)
    kernel: NDArray[float64] = _make_kernel(is_MS1, degree, m, coeffs)
    fit_weights: NDArray[float64] = _make_fit_weights(is_MS1, degree, m)

    radius: int = len(kernel) - 1
    extended_data: NDArray[float64] = _extend_data(data, degree, radius, fit_weights)
    extended_smoothed: NDArray[float64] = _smooth_except_boundaries(
        extended_data,
        kernel,
    )

    return extended_smoothed[radius:radius + len(data)]


def _smooth_like_savgol(
    data: NDArray[float64],
    degree: int,
    m: int,
    is_MS1: bool = False,
) -> NDArray[float64]:
    """
    Smooths the data in a way comparable to a traditional Savitzky-Golay
    filter with the given parameters degree and m.

    Parameters
    ----------
    data: NDArray[float64]
        The data to be smoothed.

    degree: int
        The degree of the polynomial fit used in the Savitzky-Golay filter.
        Must be an even number less than or equal to ten.

    m: int
        Half-width of the Savitzky-Golay kernel.

    is_MS1: bool, optional
        Use the MS1 variant, which has a smaller kernel size, at the cost
        of reduced stopband suppression and more gradual cutoff for degree=2.
        Otherwise, standard MS kernels are used.

    Returns
    -------
    NDArray[float64]
        The smoothed data.
    """
    bandwidth: float = _savitzky_golay_bandwidth(degree, m)
    m = _bandwidth_to_m(is_MS1, degree, bandwidth)

    return _smooth(data, degree, m, is_MS1)


def _test():
    from numpy import (
        array,
        isclose,
    )

    data = array(
        [0, 1, -2, 3, -4, 5, -6, 7, -8, 9, 10, 6, 3, 1, 0],
        dtype=float64,
    )
    smoothed = _smooth(
        data=data,
        degree=6,
        m=7,
        is_MS1=False,
    )

    control = array(
        [
            0.1583588453161306,
            0.11657466389491726,
            -0.09224721042380793,
            0.031656885544917315,
            -0.054814729808335835,
            -0.054362188355910813,
            0.5105482655952578,
            -0.5906786605713916,
            -1.2192869459451745,
            5.286105202110525,
            10.461619519603234,
            6.82674246410578,
            2.4923674303784833,
            1.0422038091960153,
            0.032646599192913656,
        ]
    )

    for s, c in zip(smoothed, control):
        if not isclose(s, c):
            raise ValueError(f"Expected {s=} to be almost equal to {c=}")
