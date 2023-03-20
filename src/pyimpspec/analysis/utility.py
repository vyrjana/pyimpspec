# pyimpspec is licensed under the GPLv3 or later (https://www.gnu.org/licenses/gpl-3.0.html).
# Copyright 2023 pyimpspec developers
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

from multiprocessing import cpu_count
from os import environ
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Set,
)
from numpy import (
    __config__ as numpy_config,
    array,
    ceil,
    empty,
    float64,
    floor,
    int64,
    integer,
    isinf,
    issubdtype,
    log10 as log,
    logspace,
    ndarray,
    sum as array_sum,
)
from numpy.typing import NDArray
from pyimpspec.typing import (
    ComplexImpedances,
    ComplexResidual,
    ComplexResiduals,
    Frequencies,
    Frequency,
)


def _interpolate(
    experimental: Frequencies,
    num_per_decade: int,
) -> Frequencies:
    if not isinstance(experimental, ndarray):
        experimental = array(experimental, dtype=Frequency)
    assert (
        issubdtype(type(num_per_decade), integer) and num_per_decade > 0
    ), num_per_decade
    min_f: float64 = min(experimental)
    max_f: float64 = max(experimental)
    assert 0.0 < min_f < max_f
    assert not isinf(max_f)
    log_min_f: int64 = int64(floor(log(min_f)))
    log_max_f: int64 = int64(ceil(log(max_f)))
    f: float64
    freq: List[float64] = [
        f
        for f in logspace(
            log_min_f, log_max_f, num=(log_max_f - log_min_f) * num_per_decade + 1
        )
        if f >= min_f and f <= max_f
    ]
    if min_f not in freq:
        freq.append(min_f)
    if max_f not in freq:
        freq.append(max_f)
    return array(list(sorted(freq, reverse=True)), dtype=Frequency)


def _calculate_residuals(
    Z_exp: ComplexImpedances,
    Z_fit: ComplexImpedances,
) -> ComplexResiduals:
    residuals: ComplexResiduals = empty(Z_exp.shape, dtype=ComplexResidual)
    residuals.real = (Z_exp.real - Z_fit.real) / abs(Z_exp)
    residuals.imag = (Z_exp.imag - Z_fit.imag) / abs(Z_exp)
    return residuals


def _boukamp_weight(Z_exp: ComplexImpedances) -> NDArray[float64]:
    assert isinstance(Z_exp, ndarray), Z_exp
    # See eq. 13 in Boukamp (1995)
    return (Z_exp.real**2 + Z_exp.imag**2) ** -1  # type: ignore


def _calculate_pseudo_chisqr(
    Z_exp: ComplexImpedances,
    Z_fit: ComplexImpedances,
    weight: Optional[NDArray[float64]] = None,
) -> float:
    assert isinstance(Z_exp, ndarray), Z_exp
    assert isinstance(Z_fit, ndarray), Z_fit
    assert isinstance(weight, ndarray) or weight is None, weight
    if weight is None:
        weight = _boukamp_weight(Z_exp)
    # See eq. 14 in Boukamp (1995)
    return float(
        array_sum(
            weight * ((Z_exp.real - Z_fit.real) ** 2 + (Z_exp.imag - Z_fit.imag) ** 2)
        )
    )


NUM_PROCS_OVERRIDE: int = -1


def _set_default_num_procs(num_procs: int):
    """
    Override the default number of parallel process that pyimpspec should use.
    Setting the value to less than one disables any previous override.

    Parameters
    ----------
    num_procs: int
        If the value is greater than zero, then the value is used as the number of processes to use.
        Otherwise, any previous override is disabled.
    """
    assert issubdtype(type(num_procs), integer), num_procs
    global NUM_PROCS_OVERRIDE
    NUM_PROCS_OVERRIDE = num_procs


def _get_default_num_procs() -> int:
    """
    Get the default number of parallel processes that pyimpspec would try to use.
    NumPy may be using libraries that multithreaded, which can lead to poor performance or system responsiveness when combined with pyimpspec's use of multiple processes.
    This function attempts to return a reasonable number of processes depending on the detected libraries (and relevant environment variables):

    - OpenBLAS (``OPENBLAS_NUM_THREADS``)
    - MKL (``MKL_NUM_THREADS``)

    If none the libraries listed above are detected because some other library is used, then the value returned by ``multiprocessing.cpu_count()`` is used.

    Returns
    -------
    int
    """
    if NUM_PROCS_OVERRIDE > 0:
        return NUM_PROCS_OVERRIDE
    num_cores: int = cpu_count()
    multithreaded: Dict[str, str] = {
        "openblas": "OPENBLAS_NUM_THREADS",
        "mkl": "MKL_NUM_THREADS",
    }
    libraries: Set[str] = set()
    key: str
    for key in [_ for _ in map(str.lower, dir(numpy_config)) if "_info" in _]:
        obj: Any = getattr(numpy_config, key)
        if not isinstance(obj, dict):
            continue
        elif "libraries" not in obj:
            continue
        elif len(obj["libraries"]) == 0:
            continue
        libraries.update(set(obj["libraries"]))
    name: str
    for name in libraries:
        lib: str
        env: str
        for lib, env in multithreaded.items():
            if lib in name:
                num_threads: int = int(environ.get(env, -1))
                if num_threads < 0:
                    # Assume that the library will use as many threads as there
                    # are cores available to the system.
                    return 1
                elif num_threads == 1:
                    return num_cores
                else:
                    num_procs: int = num_cores // num_threads
                    return num_procs if num_procs > 1 else 1
    return num_cores
