# pyimpspec is licensed under the GPLv3 or later (https://www.gnu.org/licenses/gpl-3.0.html).
# Copyright 2025 pyimpspec developers
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

from multiprocessing import get_context
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Tuple,
)
from numpy import (
    array,
    float64,
    isnan,
    pi,
)
from numpy.typing import NDArray
from pyimpspec.typing import Phases
from pyimpspec.progress import Progress


def _reconstruct(args) -> Tuple[NDArray[float64], str, str]:
    ln_omega: NDArray[float64]
    interpolator: Any
    derivator: Any
    smoothing: str
    interpolation: str
    admittance: bool
    (
        ln_omega,
        interpolator,
        derivator,
        smoothing,
        interpolation,
        admittance,
    ) = args

    ln_modulus: List[float] = []
    ln_w_s: float = ln_omega[0]
    gamma: float = -pi / 6

    ln_w_0: float
    for ln_w_0 in ln_omega:
        integral = interpolator.integrate(ln_w_s, ln_w_0)
        derivative = derivator(ln_w_0)
        if isnan(derivative):
            derivative = 0

        if admittance:
            ln_modulus.append(-(-2 / pi * integral - gamma * derivative))
        else:
            ln_modulus.append(2 / pi * integral + gamma * derivative)

    return (array(ln_modulus), smoothing, interpolation)


def _reconstruct_modulus_data(
    interpolation_options: Dict[str, Dict[str, Any]],
    simulated_phase: Dict[str, Dict[str, Phases]],
    ln_omega: NDArray[float64],
    admittance: bool,
    num_procs: int,
    prog: Progress,
) -> List[Tuple[NDArray[float64], Phases, str, str]]:
    prog.set_message("Reconstructing modulus data")

    reconstructions: List[Tuple[NDArray[float64], Phases, str, str]] = []
    args: List[Tuple[NDArray[float64], Any, Any, str, str, bool]] = []

    interpolation: str
    for interpolation in interpolation_options:
        smoothing: str
        interpolator: Any
        for smoothing, interpolator in interpolation_options[interpolation].items():
            args.append(
                (
                    ln_omega,
                    interpolator,
                    interpolator.derivative(1),
                    smoothing,
                    interpolation,
                    admittance,
                )
            )

    def _apply_map(
        function: Callable,
        args: List[Tuple[NDArray[float64], Any, Any, str, str, bool]],
        _map: Callable,
    ):
        ln_modulus: NDArray[float64]
        for ln_modulus, smoothing, interpolation in _map(function, args):
            reconstructions.append(
                (
                    ln_modulus,
                    simulated_phase[interpolation][smoothing],
                    smoothing,
                    interpolation,
                )
            )
            prog.increment()

    if len(args) > 1 and num_procs > 1:
        with get_context(method="spawn").Pool(min((
            num_procs,
            len(args),
        ))) as pool:
            _apply_map(
                function=_reconstruct,
                args=args,
                _map=pool.imap_unordered,
            )
    else:
        _apply_map(
            function=_reconstruct,
            args=args,
            _map=map,
        )

    return reconstructions
