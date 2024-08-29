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

from inspect import (
    Signature,
    signature,
)
from typing import (
    Callable,
    Dict,
    List,
    Optional,
)
from numpy import (
    ceil,
    float64,
    floor,
    log10 as log,
    logspace,
    where,
    zeros,
)
from numpy.typing import NDArray
from pyimpspec.exceptions import ZHITError
from pyimpspec.progress import Progress


_WINDOW_FUNCTIONS: Dict[str, Callable] = {}


def _initialize_window_functions():
    from scipy.signal import windows as scipy_windows

    global _WINDOW_FUNCTIONS

    name: str
    for name in dir(scipy_windows):
        if name.startswith("_"):
            continue
        elif not callable(getattr(scipy_windows, name)):
            continue

        func: Callable = getattr(scipy_windows, name)
        sig: Signature = signature(func)

        if not ("M" in sig.parameters and "sym" in sig.parameters):
            continue
        elif len(sig.parameters) > 2:
            continue

        _WINDOW_FUNCTIONS[name] = func


def _generate_weights(
    log_f: NDArray[float64],
    window: str,
    center: float,
    width: float,
) -> NDArray[float64]:
    from scipy.interpolate import Akima1DInterpolator

    if window not in _WINDOW_FUNCTIONS:
        raise ZHITError(
            f"Unsupported window function: '{window}'! Valid values include:\n- "
            + "\n- ".join(sorted(_WINDOW_FUNCTIONS.keys()))
        )

    weights: NDArray[float64] = zeros(log_f.shape, dtype=float64)
    min_log_f: float = center - width / 2
    max_log_f: float = center + width / 2
    num_points: int = 10 * int(ceil(max_log_f) - floor(min_log_f)) + 1

    x: List[float] = [
        _
        for _ in log(
            logspace(
                floor(min_log_f),
                ceil(max_log_f),
                num=num_points,
            )
        )
        if min_log_f <= _ <= max_log_f
    ]
    if min_log_f not in x:
        x.insert(0, min_log_f)
    if max_log_f not in x:
        x.append(max_log_f)

    weights_interpolator: Akima1DInterpolator = Akima1DInterpolator(
        x,
        _WINDOW_FUNCTIONS[window](M=len(x)),
    )

    i: int
    lf: float
    for i, lf in enumerate(log_f):
        if not (min_log_f <= lf <= max_log_f):
            continue
        weights[i] = weights_interpolator(lf)

    if not (len(weights) == len(log_f)):
        raise ValueError(f"Expected {len(weights)=} == {len(log_f)=}")

    indices = where(weights < 0.0)[0]
    if indices.size > 0:
        weights[indices] = 0.0

    indices = where(weights > 1.0)[0]
    if indices.size > 0:
        weights[indices] = 1.0

    return weights


def _generate_window_options(
    weights: Optional[NDArray[float64]],
    log_f: NDArray[float64],
    window: str,
    center: float,
    width: float,
    prog: Progress,
) -> Dict[str, NDArray[float64]]:
    prog.set_message("Generating weights")

    if len(_WINDOW_FUNCTIONS) == 0:
        _initialize_window_functions()
    window_options: Dict[str, NDArray[float64]] = {}

    if weights is not None:
        window_options["custom"] = weights
        prog.increment()

    elif window == "auto":
        for window in _WINDOW_FUNCTIONS:
            window_options[window] = _generate_weights(log_f, window, center, width)
            prog.increment()

    else:
        window_options[window] = _generate_weights(log_f, window, center, width)
        prog.increment()

    return window_options
