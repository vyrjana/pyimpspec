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

from multiprocessing import cpu_count
from os import environ
from numpy import (
    __config__ as numpy_config,
    float64,
    int64,
    integer,
    isclose,
    isinf,
    log10 as log,
    logspace,
    sum as array_sum,
)
from numpy.typing import NDArray
from pyimpspec.typing import (
    ComplexImpedances,
    ComplexResiduals,
    Frequencies,
    Frequency,
)
from pyimpspec.typing.helpers import (
    Any,
    Dict,
    List,
    Optional,
    Set,
    _cast_to_floating_array,
    _is_complex_array,
    _is_floating_array,
    _is_integer,
)


def _interpolate(
    experimental: Frequencies,
    num_per_decade: integer,
) -> Frequencies:
    if not _is_floating_array(experimental):
        experimental = _cast_to_floating_array(experimental)

    if len(experimental) < 2:
        raise ValueError(f"Expected an array with at least two values instead of {experimental=}")

    if not _is_integer(num_per_decade):
        raise TypeError(f"Expected an integer instead of {num_per_decade=}")
    elif num_per_decade <= 0:
        raise ValueError(
            f"Expected an integer greater than zero instead of {num_per_decade=}"
        )

    min_f: float64 = min(experimental)
    max_f: float64 = max(experimental)
    if not (0.0 < min_f < max_f):
        raise ValueError(
            f"Expected 0.0 < min_f < max_f instead of {min_f=} and {max_f=}"
        )
    elif isinf(max_f):
        raise ValueError(f"Expected max_f < inf instead of {max_f=}")

    log_min_f: int64 = log(min_f)
    log_max_f: int64 = log(max_f)
    num_decades: int = int(round(log_max_f - log_min_f))

    f: Frequencies = logspace(
        log_max_f,
        log_min_f,
        num=num_decades * num_per_decade + 1,
        dtype=Frequency,
    )

    assert isclose(f, min_f).any(), f
    assert isclose(f, max_f).any(), f

    return f


def _calculate_residuals(
    Z_exp: ComplexImpedances,
    Z_fit: ComplexImpedances,
) -> ComplexResiduals:
    # Eqs. 15 and 16 from Schönleber et al., 2014.
    # DOI:10.1016/j.electacta.2014.01.034
    return (Z_exp - Z_fit) / abs(Z_exp)


def _boukamp_weight(Z_exp: ComplexImpedances) -> NDArray[float64]:
    if not _is_complex_array(Z_exp):
        raise TypeError(f"Expected an array of complex values instead of {Z_exp=}")

    # Eq. 13 in Boukamp, 1995.
    # DOI:10.1149/1.2044210
    return (Z_exp.real**2 + Z_exp.imag**2) ** -1  # type: ignore


def _calculate_pseudo_chisqr(
    Z_exp: ComplexImpedances,
    Z_fit: ComplexImpedances,
    weight: Optional[NDArray[float64]] = None,
) -> float:
    if not _is_complex_array(Z_exp):
        raise TypeError(f"Expected an array of complex values instead of {Z_exp=}")

    if not _is_complex_array(Z_fit):
        raise TypeError(f"Expected an array of complex values instead of {Z_fit=}")

    if not (_is_floating_array(weight) or weight is None):
        raise TypeError(f"Expected None or an array of floats instead of {weight=}")

    if weight is None:
        weight = _boukamp_weight(Z_exp)

    # Eq. 14 in Boukamp, 1995.
    # DOI:10.1149/1.2044210
    return float(
        array_sum(
            weight * ((Z_exp.real - Z_fit.real) ** 2 + (Z_exp.imag - Z_fit.imag) ** 2)
        )
    )


NUM_PROCS_OVERRIDE: int = -1


def set_default_num_procs(num_procs: int):
    """
    Override the default number of parallel process that pyimpspec should use.
    Setting the value to less than one disables any previous override.

    Parameters
    ----------
    num_procs: int
        If the value is greater than zero, then the value is used as the number of processes to use.
        Otherwise, any previous override is disabled.
    """
    if not _is_integer(num_procs):
        raise TypeError(f"Expected an integer instead of {num_procs=}")

    global NUM_PROCS_OVERRIDE
    NUM_PROCS_OVERRIDE = num_procs


def get_default_num_procs() -> int:
    """
    Get the default number of parallel processes that pyimpspec would try to use.
    NumPy may be using libraries that are multithreaded, which can lead to poor performance or system responsiveness when combined with pyimpspec's use of multiple processes.
    This function attempts to return a reasonable number of processes depending on the detected libraries (and relevant environment variables):

    - OpenBLAS (``OPENBLAS_NUM_THREADS``)
    - MKL (``MKL_NUM_THREADS``)

    If none of the libraries listed above are detected because some other library is used, then the value returned by ``multiprocessing.cpu_count()`` is used.

    Returns
    -------
    int
    """
    if NUM_PROCS_OVERRIDE > 0:
        return NUM_PROCS_OVERRIDE
    num_cores: int = cpu_count()

    multithreaded: Dict[str, List[str]] = {
        "openblas": ["OPENBLAS_NUM_THREADS", "GOTO_NUM_THREADS", "OMP_NUM_THREADS"],
        "mkl": ["MKL_NUM_THREADS"],
    }
    libraries: Set[str] = set()

    if hasattr(numpy_config, "CONFIG"):
        key: str
        obj: Any
        for key, obj in numpy_config.CONFIG.items():
            if not isinstance(obj, dict):
                continue

            blas_config: dict = obj.get("blas", {})
            if not blas_config or not blas_config.get("found", False):
                continue

            lib: str
            for lib in multithreaded:
                if lib in blas_config.get("name", ""):
                    libraries.add(lib)

    name: str
    for name in libraries:
        envs: List[str]
        for lib, envs in multithreaded.items():
            if lib in name:
                num_threads: int = -1
                for env in envs:
                    try:
                        num_threads = int(environ.get(env, ""))
                    except ValueError:
                        continue
                    else:
                        break

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
