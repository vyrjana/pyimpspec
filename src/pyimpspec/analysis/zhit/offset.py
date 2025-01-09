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

from cmath import rect as _rect
from multiprocessing import Pool
from typing import (
    Dict,
    List,
    Tuple,
)
from numpy import (
    complex128,
    exp,
    float64,
    vectorize,
    where,
)
from numpy.typing import NDArray
from pyimpspec.typing import (
    ComplexImpedances,
    Phases,
)
from pyimpspec.analysis.utility import _calculate_pseudo_chisqr
from pyimpspec.exceptions import ZHITError
from pyimpspec.progress import Progress

rect = vectorize(_rect)


def _offset_residual(
    parameters: "Parameters",  # noqa: F821
    reconstruction: NDArray[float64],
    ln_modulus: NDArray[float64],
    weights: NDArray[float64],
) -> NDArray[float64]:
    offset: float = parameters.valuesdict()["offset"]
    errors: NDArray[float64] = ((reconstruction + offset) - ln_modulus) ** 2

    return weights * errors


def _calculate_modulus_offset(
    ln_modulus_fit: NDArray[float64],
    ln_modulus_exp: NDArray[float64],
    weights: NDArray[float64],
) -> float:
    from lmfit import (
        Parameters,
        minimize,
    )
    from lmfit.minimizer import MinimizerResult

    if weights.shape != ln_modulus_exp.shape:
        raise ZHITError(
            f"Expected the weights array to have the following shape: {ln_modulus_exp.shape}"
        )
    if where(weights > 0.0)[0].size == 0:
        raise ZHITError(
            "No data points have a weight greater than zero with the current "
            "settings!"
        )
    if where(weights < 0.0)[0].size > 0:
        raise ZHITError("Weights must be non-negative values!")

    parameters: Parameters = Parameters()
    parameters.add("offset", 0.0)

    fit: MinimizerResult = minimize(
        _offset_residual,
        parameters,
        args=(
            ln_modulus_fit,
            ln_modulus_exp,
            weights,
        ),
    )

    return fit.params.valuesdict()["offset"]


def _adjust_offset(args) -> Tuple[float, NDArray[complex128], str, str, str]:
    (
        ln_modulus,
        phase,
        ln_modulus_exp,
        weights,
        X_exp,
        admittance,
        smoothing,
        interpolation,
        window,
    ) = args

    offset: float = _calculate_modulus_offset(ln_modulus, ln_modulus_exp, weights)
    X_fit: NDArray[complex128] = rect(exp(ln_modulus + offset), phase)

    return (
        _calculate_pseudo_chisqr(
            Z_exp=X_exp ** (-1 if admittance else 1),
            Z_fit=X_fit ** (-1 if admittance else 1),
        ),
        X_fit,
        smoothing,
        interpolation,
        window,
    )


def _adjust_modulus_offset(
    reconstructions: List[Tuple[NDArray[float64], Phases, str, str]],
    window_options: Dict[str, NDArray[float64]],
    ln_modulus_exp: NDArray[float64],
    X_exp: NDArray[complex128],
    admittance: bool,
    num_procs: int,
    prog: Progress,
) -> List[Tuple[float, NDArray[complex128], str, str, str]]:
    prog.set_message("Adjusting modulus offset")

    results: List[Tuple[float, ComplexImpedances, str, str, str]] = []

    args = []
    window: str
    weights: NDArray[float64]
    for window, weights in window_options.items():
        ln_modulus: NDArray[float64]
        phase: Phases
        smoothing: str
        interpolation: str
        for ln_modulus, phase, smoothing, interpolation in reconstructions:
            args.append(
                (
                    ln_modulus,
                    phase,
                    ln_modulus_exp,
                    weights,
                    X_exp,
                    admittance,
                    smoothing,
                    interpolation,
                    window,
                )
            )

    if len(args) > 1 and num_procs > 1:
        with Pool(num_procs) as pool:
            for res in pool.imap_unordered(_adjust_offset, args):
                results.append(res)
                prog.increment()

    else:
        for res in map(_adjust_offset, args):
            results.append(res)
            prog.increment()

    return sorted(results, key=lambda _: _[0])
