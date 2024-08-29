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
    array,
    ceil,
    float64,
    log10 as log,
    nan,
)
from pyimpspec.typing.helpers import (
    Dict,
    List,
    NDArray,
    Tuple,
    Union,
    floating,
)


def _ideal_model_function(
    x: NDArray[floating],
    a: floating,
    b: floating,
    c: floating,
) -> NDArray[floating]:
    return -a * log(10**-x + 10**-b) + c


def _ideal_model_residual(
    p: "Parameters",  # noqa: F821
    x: NDArray[floating],
    y: NDArray[floating],
) -> NDArray[floating]:
    return y - _ideal_model_function(x, **p.valuesdict())


def _fit_ideal_model(
    x: NDArray[floating],
    y: NDArray[floating],
) -> Tuple[floating, floating, floating]:
    from lmfit import (
        Parameters,
        minimize,
    )
    from lmfit.minimizer import MinimizerResult

    if len(x) != len(y):
        raise ValueError("Expected the same number of x and y points!")
    elif len(x) < 3:
        raise ValueError("Expected at least three data points!")

    i: int = max((1, len(y) // 2))
    a = (y[i] - y[0]) / (x[i] - x[0])
    b = x[i]
    c = y[i] + a * log(10 ** -x[i] + 10**-b)

    parameters = Parameters()
    parameters.add("a", value=a)
    parameters.add("b", value=b, min=1, max=max(x))
    parameters.add("c", value=c)

    fit: MinimizerResult = minimize(
        _ideal_model_residual,
        parameters,
        method="leastsq",
        args=(x, y),
    )
    p: Dict[str, floating] = fit.params.valuesdict()

    return (
        p["a"],
        p["b"],
        p["c"],
    )


def _calculate_intercept_of_lines(
    s1: floating,
    o1: floating,
    s2: floating,
    o2: floating,
) -> floating:
    if s1 - s2 == 0.0:
        raise ZeroDivisionError()

    return (o2 - o1) / (s1 - s2)


def _intersecting_lines_function(
    x: NDArray[floating],
    s1: floating,
    o1: floating,
    s2: floating,
    o2: floating,
) -> Union[floating, NDArray[floating]]:
    y1 = s1 * x + o1
    y2 = s2 * x + o2
    intercept = _calculate_intercept_of_lines(s1, o1, s2, o2)

    y: List[floating] = []
    i: int
    _x: floating
    for i, _x in enumerate(x):
        y.append((y1 if _x < intercept else y2)[i])

    return array(y)


def _fit_intersecting_lines(
    x: NDArray[floating],
    y: NDArray[floating],
) -> Tuple[floating, floating, floating, floating]:
    from scipy.optimize import (
        OptimizeWarning,
        curve_fit,
    )

    i: int = len(x) // 2
    s1: floating = (y[i] - y[0]) / (x[i] - x[0])
    o1: floating = y[0] - s1 * x[0]
    s2: floating = (y[-1] - y[i]) / (x[-1] - x[i])
    o2: floating = y[-1] - s2 * x[-1]

    for _ in range(10):
        with catch_warnings():
            filterwarnings("ignore", category=RuntimeWarning)
            filterwarnings("ignore", category=OptimizeWarning)
            try:
                p: NDArray[floating] = curve_fit(
                    _intersecting_lines_function,
                    xdata=x,
                    ydata=y,
                    p0=(s1, o1, s2, o2),
                    maxfev=1000,
                )[0]
            except RuntimeError:
                continue
            except TypeError:
                break

            return tuple(p)

    raise ValueError("Failed to fit the intersecting lines function!")


def _approximate_transition_and_end_point(
    num_RCs: NDArray[float64],
    log_pseudo_chisqrs: NDArray[float64],
) -> Tuple[int, int, Tuple[float, float, float, float]]:
    # First try to determine the maximum num_RC. If the log(X²ps) vs num_RC
    # plot exhibits numerical instability, which is common for tests like
    # 'complex-inv', 'real', or 'real-inv', then that will hopefully be
    # handled appropriately.
    maximum_distance: int = 10
    lowest_value: float64 = 0.0
    lowest_index: int = -1

    i: int = 0
    y: float64 = log_pseudo_chisqrs[i]
    for i, y in enumerate(log_pseudo_chisqrs):
        if lowest_index < 0 or y <= lowest_value:
            lowest_index = i
            lowest_value = y
        elif (i - lowest_index) >= maximum_distance:
            break

    while (y - lowest_value) > 0.5:
        i -= 1
        y = log_pseudo_chisqrs[i]

    # Some test implementations may have a very significant drop in
    # X²ps at high num_RC when the number of points per decade is low.
    while (log_pseudo_chisqrs[i-1] - log_pseudo_chisqrs[i]) > 1.0:
        i -= 1

    max_num_RC: int = int(max(num_RCs[:i]))

    # Try to determine the point where the log(X²ps) vs num_RC plot
    # transitions from a rapid decrease in error to a more gradual
    # rate of decrease in error. This point can be considered the lower
    # limit for the optimal num_RC.
    try:
        p: Tuple[float, float, float, float] = _fit_intersecting_lines(
            num_RCs[:i],
            log_pseudo_chisqrs[:i],
        )
    except ValueError:
        return (
            min(num_RCs),
            max_num_RC,
            (nan, nan, nan, nan),
        )

    intercept: int = int(ceil(_calculate_intercept_of_lines(*p)))

    return (
        intercept,
        max_num_RC,
        p,
    )
