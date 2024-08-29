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

from typing import (
    Optional,
    Tuple,
)
from warnings import (
    catch_warnings,
    filterwarnings,
)
from numpy import (
    float64,
    int64,
    ndarray,
)
from numpy.random import normal
from numpy.typing import NDArray


def _cubic_function(x, a: float, b: float, c: float, d: float):
    return a * x**3 + b * x**2 + c * x + d


def _cubicish_function(x, a: float, b: float, c: float, d: float):
    y = _cubic_function(x, a, b, c, d)
    if a > 0.0:
        # 2nd derivative
        x_sign_change: float = (-2 * b) / (6 * a)
        # 1st derivative
        slope: float = 3 * a * x_sign_change**2 + 2 * b * x_sign_change + c
        offset: float = (
            _cubic_function(x_sign_change, a, b, c, d) - slope * x_sign_change
        )

        if isinstance(x, ndarray):
            i: int = abs(x - x_sign_change).argmin()
            y[i:] = slope * x[i:] + offset
        elif x >= x_sign_change:
            y = slope * x + offset

    return y


def _fit_cubic_function(
    x: NDArray[int64],
    y: NDArray[float64],
    p0: Optional[Tuple[float64, ...]] = None,
) -> Tuple[float64, ...]:
    from scipy.optimize import (
        OptimizeWarning,
        curve_fit,
    )

    for _ in range(10):
        with catch_warnings():
            filterwarnings("ignore", category=RuntimeWarning)
            filterwarnings("ignore", category=OptimizeWarning)
            try:
                p: NDArray[float64] = curve_fit(
                    _cubic_function,
                    xdata=x,
                    ydata=y,
                    p0=normal(0.0, 1.0, 4) if p0 is None else p0,
                    maxfev=100000,
                )[0]
            except RuntimeError:
                continue
            except TypeError:
                break

            return tuple(p)

    raise ValueError("Failed to fit cubic function!")


def _fit_cubicish_function(
    x: NDArray[int64],
    y: NDArray[float64],
    p0: Optional[Tuple[float64, ...]] = None,
) -> Tuple[float64, ...]:
    from scipy.optimize import (
        OptimizeWarning,
        curve_fit,
    )

    for _ in range(10):
        with catch_warnings():
            filterwarnings("ignore", category=RuntimeWarning)
            filterwarnings("ignore", category=OptimizeWarning)
            try:
                p: NDArray[float64] = curve_fit(
                    _cubicish_function,
                    xdata=x,
                    ydata=y,
                    p0=normal(0.0, 1.0, 4) if p0 is None else p0,
                    maxfev=100000,
                )[0]
            except RuntimeError:
                continue
            except TypeError:
                break

            return tuple(p)

    raise ValueError("Failed to fit cubic function!")
