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
    Callable,
    Dict,
    Tuple,
)
from numpy import (
    array,
    flip,
    float64,
)
from numpy.typing import NDArray
from pyimpspec.exceptions import ZHITError
from pyimpspec.progress import Progress
from pyimpspec.typing import Phases


def _interpolate_phase(
    interpolation: str,
    ln_omega: NDArray[float64],
    phase: Phases,
) -> Callable:
    from scipy.interpolate import (
        Akima1DInterpolator,
        CubicSpline,
        PchipInterpolator,
    )

    ln_omega = flip(ln_omega)
    phase = flip(phase)

    if interpolation == "akima":
        return Akima1DInterpolator(ln_omega, phase, method="akima")
    elif interpolation == "makima":
        return Akima1DInterpolator(ln_omega, phase, method="makima")
    elif interpolation == "cubic":
        return CubicSpline(ln_omega, phase)
    elif interpolation == "pchip":
        return PchipInterpolator(ln_omega, phase)

    raise ZHITError(f"Unsupported interpolation: '{interpolation}'!")


def _generate_interpolation_options(
    interpolation: str,
    ln_omega: NDArray[float64],
    smoothing_options: Dict[str, Phases],
    prog: Progress,
) -> Tuple[Dict[str, Dict[str, Callable]], Dict[str, Dict[str, Phases]]]:
    prog.set_message("Interpolating phase data")

    interpolation_options: Dict[str, Dict[str, Callable]] = {}
    simulated_phase: Dict[str, Dict[str, Phases]] = {}

    phase: Phases
    interpolator: Callable
    for interpolation in (
        ["akima", "makima", "cubic", "pchip"] if interpolation == "auto" else [interpolation]
    ):
        interpolation_options[interpolation] = {}
        simulated_phase[interpolation] = {}

        for smoothing, phase in smoothing_options.items():
            interpolator = _interpolate_phase(
                interpolation,
                ln_omega,
                phase,
            )
            interpolation_options[interpolation][smoothing] = interpolator

            simulated_phase[interpolation][smoothing] = array(
                list(map(interpolator, ln_omega))
            )

            prog.increment()

    return (interpolation_options, simulated_phase)
