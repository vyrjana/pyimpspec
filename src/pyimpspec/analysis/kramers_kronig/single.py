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

from pyimpspec.typing.helpers import (
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    _is_boolean,
    _is_integer,
)
from pyimpspec.data import DataSet
from .result import KramersKronigResult
from .algorithms import (
    suggest_num_RC,
    suggest_representation,
)
from .exploratory import evaluate_log_F_ext


def perform_kramers_kronig_test(
    data: DataSet,
    test: str = "real",
    num_RC: int = 0,
    add_capacitance: bool = True,
    add_inductance: bool = True,
    admittance: Optional[bool] = None,
    min_log_F_ext: float = -1.0,
    max_log_F_ext: float = 1.0,
    log_F_ext: float = 0.0,
    num_F_ext_evaluations: int = 20,
    rapid_F_ext_evaluations: bool = True,
    cnls_method: str = "leastsq",
    max_nfev: int = 0,
    timeout: int = 60,
    num_procs: int = -1,
    **kwargs,
) -> KramersKronigResult:
    """
    Performs linear Kramers-Kronig tests, attempts to automatically find a suitable extension of the time constant range, optionally suggests the appropriate immittance representation to test, and automatically suggests the optimal number of RC elements to use.
    The results can be used to check the validity of an impedance spectrum before performing equivalent circuit fitting.
    This function acts as a wrapper for |evaluate_log_F_ext|, |suggest_num_RC_limits|, |suggest_num_RC|, and |suggest_representation|.

    References:

    - `B.A. Boukamp, 1995, J. Electrochem. Soc., 142, 1885-1894 <https://doi.org/10.1149/1.2044210>`_
    - `M. Schönleber, D. Klotz, and E. Ivers-Tiffée, 2014, Electrochim. Acta, 131, 20-27 <https://doi.org/10.1016/j.electacta.2014.01.034>`_
    - `C. Plank, T. Rüther, and M.A. Danzer, 2022, 2022 International Workshop on Impedance Spectroscopy (IWIS), 1-6 <https://doi.org/10.1109/IWIS57888.2022.9975131>`_
    - `V. Yrjänä and J. Bobacka, 2024, Electrochim. Acta, 504, 144951 <https://doi.org/10.1016/j.electacta.2024.144951>`_

    Parameters
    ----------
    data: |DataSet|
        The data set to be tested.

    test: str, optional
        Supported values include "complex", "imaginary", "real", "complex-inv", "imaginary-inv", "real-inv", and "cnls".
        The first three correspond to the complex, imaginary, real tests, respectively, described by Boukamp (1995).
        These three implementations use least squares fitting (see `numpy.linalg.lstsq <https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html>`_).
        The implementations ending with "-inv" use matrix inversion, which was the default in pyimpspec prior to version 5.0.0.
        The "cnls" implementation uses complex non-linear least squares fitting.

    num_RC: int, optional
        The number of RC elements to use.
        A value greater than or equal to one results in the specific number of RC elements being tested.
        Otherwise, the number of RC elements is determined automatically.

    add_capacitance: bool, optional
        Add an additional capacitance in series (or in parallel if ``admittance=True``) with the rest of the circuit.

    add_inductance: bool, optional
        Add an additional inductance in series (or in parallel if ``admittance=True``) with the rest of the circuit.

    admittance: Optional[bool], optional
        If True, then perform the test(s) using the admittance data (:math:`Y = \\frac{1}{Z}`) instead of the impedance data (:math:`Z`).
        Each representation uses a different equivalent circuit model: Fig. 1 for impedance and Fig. 13 for admittance (Boukamp, 1995).
        Operating on the admittance data may be necessary in some cases such as when there is a negative differential resistance.
        If set to None, then both representations are used and the |suggest_representation| is used to pick one.

    min_log_F_ext: float, optional
        The lower limit for |log F_ext|, which extends or contracts the range of time constants.

    max_log_F_ext: float, optional
        The upper limit for |log F_ext|, which extends or contracts the range of time constants.

    log_F_ext: float, optional
        If ``num_F_ext_evaluations == 0``, then ``log_F_ext`` is used directly as the value for |log F_ext|.

    num_F_ext_evaluations: int, optional
        The maximum number of evaluations to perform when trying to automatically estimate the optimal |log F_ext|.
        Values greater than zero cause an approach based on splitting the range of logarithmic extensions into evenly spaced parts, estimating where the minimum is, and evaluating additional points near that minimum.
        Values less than zero cause an approach based on using the differential evolution algorithm to find the minimum.
        A value of zero causes ``log_F_ext`` to be used directly as |log F_ext|.

    rapid_F_ext_evaluations: bool, optional
        If possible, minimize the number of time constants that are tested when evaluating extensions in order to perform the optimization faster.

    cnls_method: str, optional
        The iterative method used to perform the fitting.
        Only relevant when performing "cnls" tests.

    max_nfev: int, optional
        The maximum number of function evaluations.
        If less than one, then no limit is imposed.
        Only relevant when performing "cnls" tests.

    timeout: int, optional
        The maximum amount of time in seconds to spend performing tests.
        Only relevant when performing "cnls" tests.

    num_procs: int, optional
        The number of parallel processes to use when performing tests.
        Only relevant when performing "cnls" tests.

    **kwargs
        Additional keyword arguments are passed on to the algorithms that are used when automatically determining an optimal number of RC elements.

    Returns
    -------
    |KramersKronigResult|

        A single linear Kramers-Kronig test result representing the suggested extension of the range of time constants, the suggested number of RC elements (i.e., time constants), and the suggested representation of the immittance spectrum to test.
    """
    if not _is_integer(num_RC):
        raise TypeError(f"Expected an integer instead of {num_RC=}")

    if not (_is_boolean(admittance) or admittance is None):
        raise TypeError(f"Expected a boolean or None instead of {admittance=}")

    options: List[bool] = [False, True] if admittance is None else [admittance]
    results: List[Union[KramersKronigResult, Tuple[KramersKronigResult, Dict[int, float], int, int]]] = []
    err: Optional[Exception] = None

    for admittance in options:
        try:
            log_F_ext_evaluations: List[Tuple[float, List[KramersKronigResult], float]]
            log_F_ext_evaluations = evaluate_log_F_ext(
                data=data,
                test=test,
                num_RCs=[num_RC] if num_RC > 0 else None,
                add_capacitance=add_capacitance,
                add_inductance=add_inductance,
                admittance=admittance,
                min_log_F_ext=min_log_F_ext,
                max_log_F_ext=max_log_F_ext,
                log_F_ext=log_F_ext,
                num_F_ext_evaluations=num_F_ext_evaluations,
                rapid_F_ext_evaluations=rapid_F_ext_evaluations,
                cnls_method=cnls_method,
                max_nfev=max_nfev,
                timeout=timeout,
                num_procs=num_procs,
            )
            tests: List[KramersKronigResult] = log_F_ext_evaluations[0][1]
        except ValueError as e:
            err = e
            continue

        if num_RC > 0:
            if not (tests[0].num_RC <= num_RC <= tests[-1].num_RC):
                raise ValueError(
                    f"Expected the specified number of RC elements to be {tests[0].num_RC} <= {num_RC=} <= {tests[-1].num_RC}"
                )

            results.append(tests[0])
        else:
            results.append(suggest_num_RC(tests, **kwargs))

    if len(results) == 0:
        if err is not None:
            raise err
        else:
            raise ValueError(f"Expected to have at least one item in {results=}")

    if num_RC > 0:
        if not all(map(lambda t: isinstance(t, KramersKronigResult), results)):
            raise TypeError(f"Expected only KramersKronigResult instances instead of {results=}")

        if len(results) == 1:
            return results[0]
        else:
            return min(results, key=lambda t: t.pseudo_chisqr)

    if not all(map(lambda t: isinstance(t, tuple), results)):
        raise TypeError(f"Expected only tuples instead of {results=}")
    elif not all(map(lambda t: len(t) == 4, results)):
        raise ValueError(f"Expected tuples with four items instead of {results=}")

    if len(results) == 1:
        return results[0][0]
    else:
        return suggest_representation(results)[0]
