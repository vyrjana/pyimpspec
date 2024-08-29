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

from typing import Dict
from numpy import float64
from numpy.typing import NDArray
from pyimpspec.exceptions import ZHITError
from pyimpspec.typing import Phases
from pyimpspec.progress import Progress
from .modified_sinc import _smooth_like_savgol as modsinc
from .whittaker_henderson import _smooth as whithend


def _smooth_phase(
    smoothing: str,
    num_points: int,
    polynomial_order: int,
    num_iterations: int,
    ln_omega: NDArray[float64],
    phase: Phases,
) -> Phases:
    if smoothing == "none":
        return phase

    elif smoothing == "savgol":
        from scipy.signal import savgol_filter

        return savgol_filter(
            phase,
            window_length=num_points,
            polyorder=polynomial_order,
        )

    elif smoothing == "lowess":
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
        except ImportError:
            raise ImportError(
                "The optional dependency 'statsmodels' could not be imported! "
                + "Consider installing the dependency if LOWESS smoothing is required."
            )

        return lowess(
            phase,
            ln_omega,
            return_sorted=False,
            frac=num_points / len(phase),
            it=num_iterations,
        )

    elif smoothing == "whithend":
        return whithend(
            phase,
            degree=polynomial_order,
            m=num_points,
        )

    elif smoothing == "modsinc":
        return modsinc(
            phase,
            degree=polynomial_order,
            m=num_points,
            is_MS1=False,
        )

    raise ZHITError(f"Unsupported smoothing: '{smoothing}'!")


def _generate_smoothing_options(
    smoothing: str,
    num_points: int,
    polynomial_order: int,
    num_iterations: int,
    ln_omega: NDArray[float64],
    phase_exp: Phases,
    prog: Progress,
) -> Dict[str, Phases]:
    prog.set_message("Smoothing phase data")

    smoothing_options: Dict[str, Phases] = {}
    for smoothing in (
        [
            "none",
            "lowess",
            "modsinc",
            "savgol",
            "whithend",
        ]
        if smoothing == "auto"
        else [smoothing]
    ):
        smoothing_options[smoothing] = _smooth_phase(
            smoothing,
            num_points,
            polynomial_order,
            num_iterations,
            ln_omega,
            phase_exp,
        )
        prog.increment()

    return smoothing_options
