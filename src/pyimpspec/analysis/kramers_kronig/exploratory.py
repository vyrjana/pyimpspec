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

from collections import namedtuple
from functools import partial
from multiprocessing import Pool
from multiprocessing.context import TimeoutError as MPTimeoutError
from warnings import (
    catch_warnings,
    filterwarnings,
)
from numpy import (
    argmin,
    argwhere,
    array,
    ceil,
    diff,
    float64,
    int64,
    isclose,
    isnan,
    linspace,
    log10 as log,
    mean,
    zeros,
)
from numpy.typing import NDArray
from pyimpspec.circuit.base import Element
from pyimpspec.circuit.circuit import Circuit
from pyimpspec.data import DataSet
from pyimpspec.exceptions import KramersKronigError
from pyimpspec.progress import Progress
from pyimpspec.typing import (
    ComplexImpedances,
    Frequencies,
)
from pyimpspec.typing.helpers import (
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    _is_boolean,
    _is_floating,
    _is_integer,
    _is_integer_list,
)
from pyimpspec.analysis.utility import (
    _calculate_pseudo_chisqr,
    _calculate_residuals,
    get_default_num_procs,
)
from .result import KramersKronigResult
from .cnls import _test_wrapper as _cnls_test
from .matrix_inversion import _test_wrapper as _inversion_test
from .least_squares import _test_wrapper as _leastsq_test
from .utility import (
    _boukamp_weight,
)
from .algorithms import (
    suggest_num_RC,
    suggest_representation,
)
from .algorithms.utility.logistic import (
    _logistic_derivative,
    _logistic_function,
)
from .algorithms.utility.cubic import (
    _cubic_function,
    _fit_cubic_function,
)
from .algorithms.utility.pseudo_chi_squared import (
    _approximate_transition_and_end_point,
    _calculate_intercept_of_lines,
    _fit_intersecting_lines,
    _intersecting_lines_function,
)


_DEBUG = bool(0)


_KKFits = namedtuple(
    "_KKFits",
    [
        "log_F_ext",
        "num_RCs",
        "circuits",
        "pseudo_chisqrs",
    ],
)


def _use_matrix_inversion(
    test: str,
    f: Frequencies,
    Z_exp: ComplexImpedances,
    weight: NDArray[float64],
    num_RCs: List[int],
    add_capacitance: bool,
    admittance: bool,
    log_F_ext: float,
    prog: Optional[Progress],
) -> _KKFits:
    supported_tests: List[str] = [
        "complex",
        "real",
        "imaginary",
    ]
    if test not in supported_tests:
        raise ValueError(f"{test=} is not among the valid values {supported_tests=}")

    args = (
        (
            test,
            f,
            Z_exp,
            weight,
            num_RC,
            add_capacitance,
            admittance,
            log_F_ext,
        )
        for num_RC in num_RCs
    )
    if prog:
        prog.increment()
        prog.set_message("Performing tests")

    fits: List[Tuple[int, Circuit]] = []

    res: Tuple[int, Circuit]
    for res in map(_inversion_test, args):
        fits.append(res)
        if prog:
            prog.increment()

    fits.sort(key=lambda f: f[0])
    weight = _boukamp_weight(Z_exp, admittance=False)
    pseudo_chisqrs: List[float] = [
        _calculate_pseudo_chisqr(
            Z_exp,
            circuit.get_impedances(f),
            weight,
        )
        for (num_RC, circuit) in fits
    ]

    return _KKFits(
        log_F_ext=log_F_ext,
        num_RCs=[f[0] for f in fits],
        circuits=[f[1] for f in fits],
        pseudo_chisqrs=pseudo_chisqrs,
    )


def _use_cnls(
    f: Frequencies,
    Z_exp: ComplexImpedances,
    weight: NDArray[float64],
    automatically_limit_num_RC: bool,
    num_RCs: List[int],
    add_capacitance: bool,
    add_inductance: bool,
    admittance: bool,
    log_F_ext: float,
    method: str,
    max_nfev: int,
    num_procs: int,
    timeout: int,
    prog: Optional[Progress],
) -> _KKFits:
    def calculate_log_sum_abs_tau_var(circuit: Circuit) -> float64:
        key: str = "C" if admittance else "R"
        total: float = 0.0

        element: Element
        for element in circuit.get_elements():
            parameters: Dict[str, float] = element.get_values()
            if "tau" in parameters and parameters[key] != 0.0:
                total += abs(parameters["tau"] / parameters[key])

        return log(total)

    args = (
        (
            f,
            Z_exp,
            weight,
            num_RC,
            add_capacitance,
            add_inductance,
            admittance,
            log_F_ext,
            method,
            max_nfev,
        )
        for num_RC in num_RCs
    )
    if prog is not None:
        prog.increment()
        prog.set_message("Performing tests")

    fits: List[Tuple[int, Circuit]] = []
    with Pool(num_procs) as pool:
        threshold: Optional[float] = None
        log_sum_abs_tau_var: Dict[int, float] = {}
        max_count: int = 5

        iterator = pool.imap(_cnls_test, args, 1)
        while True:
            try:
                res: Tuple[int, Circuit] = iterator.next(timeout=timeout)
            except MPTimeoutError:
                if len(fits) == 0:
                    raise KramersKronigError(
                        "Timed out before finishing the fitting process! Try increasing the timeout setting."
                    )
                break
            except StopIteration:
                break

            fits.append(res)

            if prog is not None:
                prog.increment()

            if not automatically_limit_num_RC:
                continue

            log_sum_abs_tau_var[fits[-1][0]] = calculate_log_sum_abs_tau_var(res[1])
            if (threshold is None) and (len(fits) >= 5):
                threshold = min(list(log_sum_abs_tau_var.values()))

            if (len(fits) < 10) or (len(fits) % 5 == 0):
                continue

            count: int = sum(
                (1 for f in fits[-max_count:] if log_sum_abs_tau_var[f[0]] < threshold)
            )
            if count >= max_count:
                break

    while True:
        try:
            next(args)
        except StopIteration:
            break

        if prog is not None:
            prog.increment()

    fits.sort(key=lambda f: f[0])
    weight = _boukamp_weight(Z_exp, admittance=False)
    pseudo_chisqrs: List[float] = [
        _calculate_pseudo_chisqr(
            Z_exp,
            circuit.get_impedances(f),
            weight,
        )
        for (num_RC, circuit) in fits
    ]

    return _KKFits(
        log_F_ext=log_F_ext,
        num_RCs=[f[0] for f in fits],
        circuits=[f[1] for f in fits],
        pseudo_chisqrs=pseudo_chisqrs,
    )


def _use_least_squares_fitting(
    test: str,
    f: Frequencies,
    Z_exp: ComplexImpedances,
    weight: NDArray[float64],
    num_RCs: List[int],
    add_capacitance: bool,
    add_inductance: bool,
    admittance: bool,
    log_F_ext: float,
    prog: Optional[Progress],
) -> _KKFits:
    supported_tests: List[str] = [
        "complex",
        "real",
        "imaginary",
    ]
    if test not in supported_tests:
        raise ValueError(f"{test=} is not among the valid values {supported_tests=}")

    args = (
        (
            test,
            f,
            Z_exp,
            weight,
            num_RC,
            add_capacitance,
            add_inductance,
            admittance,
            log_F_ext,
        )
        for num_RC in num_RCs
    )
    if prog:
        prog.increment()
        prog.set_message("Performing tests")

    fits: List[Tuple[int, Circuit]] = []

    res: Tuple[int, Circuit]
    for res in map(_leastsq_test, args):
        fits.append(res)
        if prog:
            prog.increment()

    fits.sort(key=lambda f: f[0])
    weight = _boukamp_weight(Z_exp, admittance=False)
    pseudo_chisqrs: List[float] = [
        _calculate_pseudo_chisqr(
            Z_exp,
            circuit.get_impedances(f),
            weight,
        )
        for (num_RC, circuit) in fits
    ]

    return _KKFits(
        log_F_ext=log_F_ext,
        num_RCs=[f[0] for f in fits],
        circuits=[f[1] for f in fits],
        pseudo_chisqrs=pseudo_chisqrs,
    )


def _perform_tests(
    test: str,
    f: Frequencies,
    Z_exp: ComplexImpedances,
    weight: NDArray[float64],
    automatically_limit_num_RC: bool,
    num_RCs: List[int],
    add_capacitance: bool,
    add_inductance: bool,
    admittance: bool,
    log_F_ext: float,
    cnls_method: str,
    max_nfev: int,
    num_procs: int,
    timeout: int,
    prog: Progress,
    **kwargs,
) -> _KKFits:
    fits: _KKFits
    if test == "cnls":
        fits = _use_cnls(
            f,
            Z_exp,
            weight,
            automatically_limit_num_RC,
            num_RCs,
            add_capacitance,
            add_inductance,
            admittance,
            log_F_ext,
            cnls_method,
            max_nfev,
            num_procs,
            timeout,
            prog,
        )
    elif test.endswith("-inv"):
        fits = _use_matrix_inversion(
            test.replace("-inv", ""),
            f,
            Z_exp,
            weight,
            num_RCs,
            add_capacitance,
            admittance,
            log_F_ext,
            prog,
        )
    else:
        fits = _use_least_squares_fitting(
            test,
            f,
            Z_exp,
            weight,
            num_RCs,
            add_capacitance,
            add_inductance,
            admittance,
            log_F_ext,
            prog,
        )

    return fits


def _wrapper(args: tuple) -> _KKFits:
    (
        log_F_ext,
        kwargs,
    ) = args
    kwargs["log_F_ext"] = log_F_ext
    fits: _KKFits = _perform_tests(**kwargs)

    return fits


def _calculate_statistic(
    fits: _KKFits,
    f: Frequencies,
    test: str,
    target_num_RC: int,
) -> float:
    x: NDArray[float64] = array([num_RC for num_RC in fits.num_RCs], dtype=float64)
    y: NDArray[float64] = log(fits.pseudo_chisqrs)

    if target_num_RC > 0:
        i: int = argmin(abs(x - target_num_RC))

        return mean(y[i:i + 2])

    # Unable to estimate the target num_RC at
    # log(fits.log_F_ext) = 0 for some reason.
    intercept_x: int
    max_x: int
    p: Tuple[float, float, float, float]
    intercept_x, max_x, p = _approximate_transition_and_end_point(x, y)

    min_x: float64 = min(x)
    if isnan(p).any():
        y = zeros(3, dtype=float64)
        for i, _x in enumerate((min_x, intercept_x, max_x)):
            y[i] = y[argmin(abs(x - _x)).flatten()]
    else:
        x = array([min_x, intercept_x, max_x], dtype=float64)
        y = _intersecting_lines_function(x, *p)

    x /= len(f)
    y -= y[2]
    if y[0] != 0.0:
        y /= y[0]

    return x[1] ** 2 + abs(y[1])


def _log_F_ext_residual(
    params: "Parameters",  # noqa: F821
    kwargs: dict,
    cache: List[Tuple[_KKFits, float]],
    prog: Optional[Progress],
) -> float:
    log_F_ext: float = params.valuesdict()["log_F_ext"]
    fits: _KKFits = _wrapper(
        (
            log_F_ext,
            kwargs,
        )
    )
    statistic: float = _calculate_statistic(
        fits,
        f=kwargs["f"],
        test=kwargs["test"],
        target_num_RC=kwargs["target_num_RC"],
    )

    cache.append((fits, statistic))
    if prog is not None:
        prog.increment()

    return statistic


def _evaluate_log_F_ext_using_lmfit(
    min_log_F_ext: float,
    max_log_F_ext: float,
    num_F_ext_evaluations: int,
    rapid_F_ext_evaluations: bool,
    wrapper_kwargs: dict,
    prog: Progress,
    method: str = "differential_evolution",
) -> List[Tuple[_KKFits, float]]:
    from lmfit import minimize, Parameters

    if not (min_log_F_ext <= 0.0 < max_log_F_ext):
        raise ValueError(f"Expected {min_log_F_ext=} <= 0.0 < {max_log_F_ext=}")

    max_nfev: Optional[int] = (
        num_F_ext_evaluations if num_F_ext_evaluations > 0 else None
    )
    if not max_nfev:
        prog.increment()

    # Perform entirely using least squares fitting
    parameters = Parameters()
    parameters.add(
        "log_F_ext",
        value=min((0.1, max_log_F_ext / 2)),
        min=min_log_F_ext,
        max=max_log_F_ext,
    )
    if not max_nfev:
        prog.increment()

    evaluations: List[Tuple[_KKFits, float]] = []

    baseline_result: _KKFits = _wrapper((0.0, wrapper_kwargs))
    target_num_RC: int = _estimate_target_num_RC(baseline_result)
    wrapper_kwargs["target_num_RC"] = target_num_RC

    num_RCs: List[int] = wrapper_kwargs["num_RCs"][:]
    if rapid_F_ext_evaluations and target_num_RC > 0:
        wrapper_kwargs["num_RCs"] = [
            num_RC
            for num_RC in num_RCs
            if num_RC <= min((max(num_RCs), target_num_RC + 5))
        ]

    evaluations.append(
        (
            baseline_result,
            _calculate_statistic(
                baseline_result,
                f=wrapper_kwargs["f"],
                test=wrapper_kwargs["test"],
                target_num_RC=wrapper_kwargs["target_num_RC"],
            ),
        )
    )

    if not max_nfev:
        prog.increment()

    minimize(
        _log_F_ext_residual,
        parameters,
        # Many of the other methods tend to get stuck on local minima,
        # but the following work quite well
        # - "differential_evolution"
        # - "powell"
        # - "slsqp"
        # - "bfgs"
        method=method,
        args=(
            wrapper_kwargs,
            evaluations,
            prog if max_nfev else None,
        ),
        max_nfev=max_nfev,
    )
    if not max_nfev:
        prog.increment()

    if rapid_F_ext_evaluations and set(num_RCs) != set(wrapper_kwargs["num_RCs"]):
        wrapper_kwargs["num_RCs"] = num_RCs
        evaluations = sorted(evaluations, key=lambda e: e[1])
        fits, statistic = evaluations.pop(0)
        fits = _wrapper((fits.log_F_ext, wrapper_kwargs))
        evaluations.insert(0, (fits, statistic))

    if _DEBUG:  # TODO: User-configurable setting?
        _debug_plot_statistic(evaluations)

    return evaluations


def _debug_plot_statistic(evaluations: List[Tuple[_KKFits, float]]):
    x = []
    y = []
    for kk, stat in sorted(evaluations, key=lambda e: e[0].log_F_ext):
        x.append(kk.log_F_ext)
        y.append(stat)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.scatter(x, y)

    ax.set_xlabel(r"$\log{F_{\rm ext}}$")
    ax.set_ylabel(r"statistic")

    plt.show()


def _fit_cubic_and_interpolate(
    x: NDArray[float64],
    y: NDArray[float64],
) -> Tuple[NDArray[float64], NDArray[float64]]:
    j: int = argmin(y)
    delta: int = 2
    if not (delta % 2 == 0):
        raise ValueError(f"Expected {delta % 2=} == 0")

    i: int = j - delta
    k: int = j + delta
    if i < 0:
        i = 0
        k = min((i + 2 * delta, len(x) - 1))
    elif k > len(x) - 1:
        k = len(x) - 1
        i = max((0, k - 2 * delta))

    if (k - i) == (2 * delta) and (k - i) > 3:
        x = x[i:k + 1]
        y = y[i:k + 1]

    p: Tuple[float, float, float, float] = _fit_cubic_function(x, y)

    x_interp: NDArray[float64] = linspace(
        min(x),
        max(x),
        num=int(round((max(x) - min(x)) * 100)),
    )

    return (
        x_interp,
        _cubic_function(x_interp, *p),
    )


def _pick_minimum(
    x: List[float],
    y: List[float],
    x_interp: NDArray[float64],
    y_interp: NDArray[float64],
) -> float:
    from scipy.signal import argrelmin

    candidates: List[Tuple[float, float]] = []

    i: int = argmin(y)
    candidates.append((x[i], y[i]))

    if len(y_interp) > 0:
        i = argmin(y_interp)
        candidates.append((x_interp[i], y_interp[i]))

        indices: NDArray[int64] = argrelmin(y_interp)[0]
        if len(indices) > 0:
            minima: NDArray[float64] = y_interp[indices]
            i = indices[argmin(minima)]
            candidates.append((x_interp[i], y_interp[i]))

    return min(candidates, key=lambda xy: xy[1])[0]


def _fit_logistic_function(
    x: NDArray[float64],
    y: NDArray[float64],
    p0: Optional[Tuple[float, ...]] = None,
    bounds: Optional[Tuple[NDArray[float64], NDArray[float64]]] = None,
) -> Tuple[float, ...]:
    from scipy.optimize import (
        OptimizeWarning,
        curve_fit,
    )

    kwargs = {}
    if p0 is None:
        p0 = [
            -1.0,
            2.0,
            min((10.0, (max(x) - min(x)) / 2.0)),
            1.0,
        ]
    kwargs["p0"] = p0
    if bounds is not None:
        kwargs["bounds"] = bounds

    with catch_warnings():
        filterwarnings("ignore", category=OptimizeWarning)
        filterwarnings("ignore", category=RuntimeWarning)
        p: Tuple[float, ...] = curve_fit(
            _logistic_function,
            x,
            y,
            **kwargs,
        )[0]

    return p


def _estimate_target_num_RC(baseline_result: _KKFits) -> int:
    # The slope of the first few points at 'log_F_ext == 0.0'
    # should provide a decent estimate for the num_RC to target when
    # extrapolated to the lowest log(X²ps) value obtained when the
    # tests were performed at 'log_F_ext == 0.0'.
    from scipy.stats import linregress
    from pandas import Series

    def backup_approach(x: NDArray[float64], y: NDArray[float64]):
        try:
            p = _fit_intersecting_lines(x, y)
        except (ValueError, ZeroDivisionError):
            return -2

        try:
            return int(ceil(_calculate_intercept_of_lines(p[0], p[1], 0.0, min(y))))
        except (ValueError, ZeroDivisionError):
            return -3

    if not (diff(baseline_result.num_RCs) == 1).all():
        return -1

    x: NDArray[float64] = array(baseline_result.num_RCs, dtype=float64)
    y: NDArray[float64] = baseline_result.pseudo_chisqrs[:]
    y = y[: argmin(y) + 1]
    if len(y) < len(x):
        y = y + [min(y)] * (len(baseline_result.pseudo_chisqrs) - len(y))
    y = log(y)

    # Some test implementations may have a very significant drop in
    # X²ps at high num_RC when the number of points per decade is low.
    i: int = len(y) - 1
    while (y[i - 1] - y[i]) > 1.0:
        i -= 1

    x = x[: i + 1]
    y = y[: i + 1]

    std = Series(y).rolling(3, center=True, min_periods=1).std()
    rel_std = (std - min(std)) / (max(std) - min(std))

    try:
        p: Tuple[float, ...] = _fit_logistic_function(x, rel_std)
    except (RuntimeError, ValueError):
        return backup_approach(x, y)

    if _DEBUG:
        import matplotlib.pyplot as plt

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        ax1.scatter(x, y, edgecolor="black", facecolor="none", marker="o")
        ax2.scatter(x, rel_std, color="red", marker="+")

        smooth_x = linspace(min(x), max(x), num=int(ceil(max(x) - min(x))) * 100)
        ax2.plot(
            smooth_x,
            _logistic_function(smooth_x, *p),
            color="red",
            linestyle="-",
        )

        smooth_y = _logistic_derivative(smooth_x, *p)
        ax2.plot(
            smooth_x,
            (smooth_y - min(smooth_y)) / (max(smooth_y) - min(smooth_y)),
            color="blue",
            linestyle="--",
        )

    i = argmin(abs(x - int(ceil(p[2]))))

    if len(x[:i]) < 5:
        slope = _logistic_derivative(p[2], *p)
        intercept = _logistic_function(p[2], *p) - slope * p[2]
        i = argmin(abs(x - (min(rel_std) - intercept) / slope))

        if _DEBUG:
            m = min(argwhere(smooth_y <= max(rel_std)).flatten())
            n = max(argwhere(smooth_y >= min(rel_std)).flatten())
            ax2.plot(
                smooth_x[m:n + 1],
                smooth_y[m:n + 1],
                color="green",
                linestyle=":",
            )

    if len(x[:i]) < 4:
        if _DEBUG:
            plt.close()

        return backup_approach(x, y)

    regression = linregress(x[:i], y[:i])
    slope: float = regression.slope
    intercept: float = regression.intercept
    if slope >= 0.0:
        if _DEBUG:
            plt.close()

        return backup_approach(x, y)

    if _DEBUG:
        smooth_x = linspace(min(x), max(x), num=int(ceil(max(x) - min(x))) * 100)
        smooth_y = slope * smooth_x + intercept
        m = min(argwhere(smooth_y <= max(y)).flatten())
        n = max(argwhere(smooth_y >= min(y)).flatten())
        ax1.plot(
            smooth_x[m:n + 1],
            smooth_y[m:n + 1],
            color="black",
            linestyle=":",
        )
        ax1.axvline(x[i] - 0.5, color="black", linestyle="--")

    max_x: int
    _, max_x, _ = _approximate_transition_and_end_point(x, y)
    for xy in sorted(zip(x, y), key=lambda xy: xy[1]):
        if xy[0] <= max_x:
            break

    target_num_RC: int = (
        int(
            ceil(
                _calculate_intercept_of_lines(
                    slope,
                    intercept,
                    0.0,
                    xy[1],
                )
            )
        )
        + 1
    )

    if _DEBUG:
        print(f"{x[i]=}, {target_num_RC=}")
        ax1.set_xlim(0, min((p[2] + 20, max(x))))
        plt.show()

    return target_num_RC


def _evaluate_log_F_ext_using_custom_approach(
    min_log_F_ext: float,
    max_log_F_ext: float,
    num_F_ext_evaluations: int,
    rapid_F_ext_evaluations: bool,
    wrapper_kwargs: dict,
    prog: Progress,
    _map: Callable,
) -> List[Tuple[_KKFits, float]]:
    if not (min_log_F_ext <= 0.0 < max_log_F_ext):
        raise ValueError(f"Expected {min_log_F_ext=} <= 0.0 < {max_log_F_ext=}")

    baseline_result: _KKFits = _wrapper((0.0, wrapper_kwargs))
    target_num_RC: int = _estimate_target_num_RC(baseline_result)
    wrapper_kwargs["target_num_RC"] = target_num_RC

    num_RCs: List[int] = wrapper_kwargs["num_RCs"][:]
    if rapid_F_ext_evaluations and target_num_RC > 0:
        wrapper_kwargs["num_RCs"] = [
            num_RC
            for num_RC in num_RCs
            if num_RC <= min((max(num_RCs), target_num_RC + 5))
        ]

    prog.increment()

    # Find the approximate location of the minimum
    stage_1_num_F_ext_evaluations: int = int(ceil(num_F_ext_evaluations / 2)) + 1
    stage_1: NDArray[float64] = linspace(
        min_log_F_ext,
        max_log_F_ext,
        num=stage_1_num_F_ext_evaluations,
    )

    stage_1_results: List[_KKFits] = []
    log_F_ext: float64
    for res in _map(
        _wrapper,
        (
            (log_F_ext, wrapper_kwargs)
            for log_F_ext in stage_1
            if not isclose(log_F_ext, 0.0)
        ),
    ):
        stage_1_results.append(res)
        prog.increment()

    step_size: float64 = abs(stage_1[1] - stage_1[0])
    stage_1_results.append(baseline_result)
    stage_1_results.sort(key=lambda kk: kk.log_F_ext)

    x: List[float] = []
    y: List[float] = []
    for fits in sorted(stage_1_results, key=lambda res: res.log_F_ext):
        x.append(fits.log_F_ext)
        y.append(
            _calculate_statistic(
                fits,
                f=wrapper_kwargs["f"],
                test=wrapper_kwargs["test"],
                target_num_RC=wrapper_kwargs["target_num_RC"],
            )
        )

    x_interp_1: NDArray[float64]
    y_interp_1: NDArray[float64]
    x_interp_1, y_interp_1 = _fit_cubic_and_interpolate(x, y)

    stage_1_minimum: float64
    stage_1_minimum = _pick_minimum(x, y, x_interp_1, y_interp_1)

    # Refine the range near the approximated minimum
    min_log_F_ext = max(
        (
            min_log_F_ext,
            stage_1_minimum - step_size / 2,
        )
    )
    max_log_F_ext = min(
        (
            max_log_F_ext,
            min_log_F_ext + step_size,
        )
    )

    stage_2_num_F_ext_evaluations: int = num_F_ext_evaluations - len(stage_1_results)

    x.clear()
    for log_F_ext in linspace(
        min_log_F_ext,
        max_log_F_ext,
        num=max((3, stage_2_num_F_ext_evaluations + 1)),
    )[1:-1]:
        if not isclose(log_F_ext, stage_1, atol=1e-4).any():
            x.append(log_F_ext)

    stage_2: NDArray[float64] = array(x)
    stage_2_results: List[_KKFits] = []
    for res in _map(
        _wrapper,
        ((log_F_ext, wrapper_kwargs) for log_F_ext in stage_2),
    ):
        stage_2_results.append(res)
        prog.increment()

    intermediate_results: List[_KKFits] = stage_1_results + stage_2_results

    x.clear()
    y.clear()
    for fits in sorted(intermediate_results, key=lambda res: res.log_F_ext):
        x.append(fits.log_F_ext)
        y.append(
            _calculate_statistic(
                fits,
                f=wrapper_kwargs["f"],
                test=wrapper_kwargs["test"],
                target_num_RC=wrapper_kwargs["target_num_RC"],
            ),
        )

    x_interp_2: NDArray[float64]
    y_interp_2: NDArray[float64]
    x_interp_2, y_interp_2 = _fit_cubic_and_interpolate(x, y)

    stage_2_minimum: float64
    stage_2_minimum = _pick_minimum(x, y, x_interp_2, y_interp_2)

    if not (
        isclose(stage_2_minimum, stage_1, atol=1e-4).any()
        or isclose(stage_2_minimum, stage_2, atol=1e-4).any()
    ):
        intermediate_results.append(_wrapper((stage_2_minimum, wrapper_kwargs)))

    evaluations: List[Tuple[_KKFits, float]] = []
    for fits in intermediate_results:
        evaluations.append(
            (
                fits,
                _calculate_statistic(
                    fits,
                    f=wrapper_kwargs["f"],
                    test=wrapper_kwargs["test"],
                    target_num_RC=wrapper_kwargs["target_num_RC"],
                ),
            )
        )

    if rapid_F_ext_evaluations and set(num_RCs) != set(wrapper_kwargs["num_RCs"]):
        wrapper_kwargs["num_RCs"] = num_RCs
        evaluations = sorted(evaluations, key=lambda e: e[1])
        fits, statistic = evaluations.pop(0)
        fits = _wrapper((fits.log_F_ext, wrapper_kwargs))
        evaluations.insert(0, (fits, statistic))

    if _DEBUG:  # TODO: User-configurable setting?
        _debug_plot_statistic(evaluations)

    return evaluations


def evaluate_log_F_ext(
    data: DataSet,
    test: str = "real",
    num_RCs: Optional[List[int]] = None,
    add_capacitance: bool = True,
    add_inductance: bool = True,
    admittance: bool = False,
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
) -> List[Tuple[float, List[KramersKronigResult], float]]:
    """
    Evaluates extensions (or contractions) of the range of time constants in order to find an optimum.
    Linear Kramers-Kronig tests are performed at various ranges of time constants.
    The limits of the default range are defined by the reciprocals of the maximum and minimum excitation frequencies.
    The lower and upper limits of the extended (or contracted) ranges are defined as :math:`\\tau_{\\rm min} = 1/(F_{\\rm ext}\\omega_{\\rm max})` and :math:`\\tau_{\\rm max} = F_{\\rm ext}/\\omega_{\\rm min}`, respectively.

    References:

    - `V. Yrjänä and J. Bobacka, 2024, Electrochim. Acta, 504, 144951 <https://doi.org/10.1016/j.electacta.2024.144951>`_
    - `B.A. Boukamp, 1995, J. Electrochem. Soc., 142, 1885-1894 <https://doi.org/10.1149/1.2044210>`_

    Parameters
    ----------
    data: |DataSet|
        The data set to be tested.

    test: str, optional
        See |perform_kramers_kronig_test| for details.

    num_RCs: Optional[List[int]], optional
        See |perform_exploratory_kramers_kronig_tests| for details.

    add_capacitance: bool, optional
        See |perform_kramers_kronig_test| for details.

    add_inductance: bool, optional
        See |perform_kramers_kronig_test| for details.

    admittance: bool, optional
        Perform the linear Kramers-Kronig test on the admittance representation instead of the impedance representation.

    min_log_F_ext: float, optional
        See |perform_kramers_kronig_test| for details.

    max_log_F_ext: float, optional
        See |perform_kramers_kronig_test| for details.

    log_F_ext: float, optional
        See |perform_kramers_kronig_test| for details.

    num_F_ext_evaluations: int, optional
        See |perform_kramers_kronig_test| for details.

    rapid_F_ext_evaluations: bool, optional
        See |perform_kramers_kronig_test| for details.

    cnls_method: str, optional
        See |perform_kramers_kronig_test| for details.

    max_nfev: int, optional
        See |perform_kramers_kronig_test| for details.

    timeout: int, optional
        See |perform_kramers_kronig_test| for details.

    num_procs: int, optional
        See |perform_kramers_kronig_test| for details.

    **kwargs

    Returns
    -------
    List[Tuple[float, List[|KramersKronigResult|], float]]

        A list of tuples containing:

        - The extension in terms of decades beyond the default range.
        - A list of linear Kramers-Kronig test results performed with a different number of RC elements (i.e., time constants).
        - The statistic indicating the quality of the extension (the smaller the better).

        The list of tuples is sorted from best to worst. The list of |KramersKronigResult| instances within each tuple is sorted from lowest to highest number of RC elements, and the optimal number of RC elements still needs to be determined.
    """
    if not isinstance(test, str):
        raise TypeError(f"Expected a string instead of {test=}")

    if not _is_boolean(add_capacitance):
        raise TypeError(f"Expected a boolean instead of {add_capacitance=}")

    if not _is_boolean(add_inductance):
        raise TypeError(f"Expected a boolean instead of {add_inductance=}")
    elif not add_inductance and test.endswith("-inv"):
        raise ValueError(
            "The tests implemented using matrix inversion must include the series/parallel inductance"
        )

    if not _is_boolean(admittance):
        raise TypeError(f"Expected a boolean instead of {admittance=}")

    if num_RCs is None:
        num_RCs = []
    elif not _is_integer_list(num_RCs):
        raise TypeError(f"Expected None or a list of integers instead of {num_RCs=}")

    if not _is_floating(min_log_F_ext):
        raise TypeError(f"Expected a float instead of {max_log_F_ext=}")
    elif min_log_F_ext > 0.0:
        raise ValueError(f"Expected {min_log_F_ext=} <= 0.0")

    if not _is_floating(max_log_F_ext):
        raise TypeError(f"Expected a float instead of {max_log_F_ext=}")
    elif max_log_F_ext <= 0.0:
        raise ValueError(f"Expected {max_log_F_ext=} > 0.0")

    if not _is_floating(log_F_ext):
        raise TypeError(f"Expected a float instead of {log_F_ext=}")

    if not _is_integer(num_F_ext_evaluations):
        raise TypeError(f"Expected an integer instead of {num_F_ext_evaluations=}")

    if not _is_boolean(rapid_F_ext_evaluations):
        raise TypeError(f"Expected a boolean instead of {rapid_F_ext_evaluations=}")

    if not isinstance(cnls_method, str):
        raise TypeError(f"Expected a string instead of {cnls_method=}")

    if not _is_integer(max_nfev):
        raise TypeError(f"Expected an integer instead of {max_nfev=}")

    if not _is_integer(timeout):
        raise TypeError(f"Expected an integer instead of {timeout=}")

    if not _is_integer(num_procs):
        raise TypeError(f"Expected an integer instead of {num_procs=}")
    elif num_procs < 1:
        num_procs = max((get_default_num_procs() - abs(num_procs), 1))

    f: Frequencies = data.get_frequencies()
    Z_exp: ComplexImpedances = data.get_impedances()

    automatically_limit_num_RC: bool = len(num_RCs) == 0
    num_points: int = len(f)
    max_num_RC: int = 2 * num_points - 5
    if test.endswith("-inv"):
        max_num_RC = min((num_points + 10, max_num_RC))

    if len(num_RCs) > 0 and max(num_RCs) > max_num_RC:
        raise KramersKronigError(
            f"The maximum value of num_RCs must be less than or equal to {max_num_RC}"
        )

    if automatically_limit_num_RC:
        num_RCs = list(range(2, max_num_RC + 1))
    else:
        num_RCs = sorted(num_RCs)

    if not all(map(lambda n: 2 <= n <= max_num_RC, num_RCs)):
        raise KramersKronigError(
            f"Expected all values in num_RCs to be withing the range [2,{max_num_RC}] instead of {num_RCs}"
        )

    num_steps: int = 2  # Calculating weight and preparing arguments

    estimate_log_F_ext: bool = num_F_ext_evaluations != 0
    if estimate_log_F_ext:
        if not automatically_limit_num_RC:
            raise ValueError(
                "Expected the range of RC elements to be determined automatically when evaluating extensions of the range of time constants (i.e., expected 'num_RCs == []' when 'num_F_ext_evaluations != 0')"
            )
        elif abs(num_F_ext_evaluations) < 10:
            raise ValueError(
                f"Expected at least 10 evaluations instead of {abs(num_F_ext_evaluations)=} for the optimization of log F_ext"
            )

        if num_F_ext_evaluations < 0:
            num_steps += abs(num_F_ext_evaluations) + 2
        else:
            num_steps += abs(num_F_ext_evaluations)
    else:
        num_steps += len(num_RCs)

    prog: Progress
    with Progress(
        "Preparing arguments",
        total=num_steps + 1,
        N=(5 if test == "cnls" else 10),
    ) as prog:
        weight: NDArray[float64] = _boukamp_weight(Z_exp, admittance=admittance)
        prog.increment()

        wrapper_kwargs = dict(
            test=test,
            f=f,
            Z_exp=Z_exp,
            weight=weight,
            automatically_limit_num_RC=automatically_limit_num_RC,
            num_RCs=num_RCs,
            add_capacitance=add_capacitance,
            add_inductance=add_inductance,
            admittance=admittance,
            cnls_method=cnls_method,
            max_nfev=max_nfev,
            num_procs=num_procs,
            timeout=timeout,
            prog=prog if not estimate_log_F_ext else None,
        )

        evaluations: List[Tuple[_KKFits, float]]
        if not estimate_log_F_ext:
            fits: _KKFits = _perform_tests(
                log_F_ext=log_F_ext,
                **wrapper_kwargs,
            )
            evaluations = [(fits, 0.0)]
        else:
            prog.set_message("Evaluating time constant ranges")
            evaluation_kwargs: dict = dict(
                min_log_F_ext=min_log_F_ext,
                max_log_F_ext=max_log_F_ext,
                num_F_ext_evaluations=abs(num_F_ext_evaluations),
                rapid_F_ext_evaluations=rapid_F_ext_evaluations,
                wrapper_kwargs=wrapper_kwargs,
                prog=prog,
            )
            if num_F_ext_evaluations <= 0:
                evaluations = _evaluate_log_F_ext_using_lmfit(**evaluation_kwargs)
            elif num_procs > 1:
                # TODO: Figure out why this causes a RuntimeError related to
                # the matplotlib window. Tends to happen when using the CLI and
                # several windows have been shown. The same doesn't happen when,
                # e.g., performing multiple fits in a row via the CLI.
                # EDIT: Seems to be related to using an interactive
                # matplotlib backend such as TkAgg. Probably need to figure out
                # how to handle using a combination of Agg for generating
                # figures and then also supporting displaying them using, e.g.,
                # TkAgg.
                from matplotlib import get_backend

                with Pool(num_procs) as pool:
                    evaluations = _evaluate_log_F_ext_using_custom_approach(
                        _map=pool.map if get_backend().lower() == "agg" else map,
                        **evaluation_kwargs,
                    )
            else:
                evaluations = _evaluate_log_F_ext_using_custom_approach(
                    _map=map,
                    **evaluation_kwargs,
                )

    f: Frequencies = data.get_frequencies()
    Z_exp: ComplexImpedances = data.get_impedances()
    result_sets: List[Tuple[float, List[KramersKronigResult]]] = []

    if not isinstance(test, str):
        raise TypeError(f"Expected a string instead of {test=}")

    fits: _KKFits
    statistic: float
    for fits, statistic in sorted(evaluations, key=lambda e: e[1]):
        results: List[KramersKronigResult] = []
        for circuit, pseudo_chisqr in zip(
            fits.circuits,
            fits.pseudo_chisqrs,
        ):
            Z_fit: ComplexImpedances = circuit.get_impedances(f)
            results.append(
                KramersKronigResult(
                    circuit=circuit,
                    pseudo_chisqr=pseudo_chisqr,
                    frequencies=f,
                    impedances=Z_fit,
                    # Residuals calculated according to eqs. 15 and 16
                    # in Schönleber et al. (2014)
                    residuals=_calculate_residuals(
                        Z_exp=Z_exp,
                        Z_fit=Z_fit,
                    ),
                    test=test,
                )
            )

        result_sets.append((fits.log_F_ext, results, statistic))

    return result_sets


def _evaluate_representations(
    representations: List[bool],
    data: DataSet,
    test: str = "real",
    num_RCs: Optional[List[int]] = None,
    add_capacitance: bool = True,
    add_inductance: bool = True,
    min_log_F_ext: float = -1.0,
    max_log_F_ext: float = 1.0,
    log_F_ext: float = 0.0,
    num_F_ext_evaluations: int = 20,
    rapid_F_ext_evaluations: bool = True,
    cnls_method: str = "leastsq",
    max_nfev: int = 0,
    timeout: int = 60,
    num_procs: int = -1,
) -> List[Tuple[float, List[KramersKronigResult], float]]:
    evaluations: List[Tuple[float, List[KramersKronigResult], float]] = []

    admittance: bool
    for admittance in representations:
        evaluations.append(
            evaluate_log_F_ext(
                data=data,
                test=test,
                num_RCs=num_RCs,
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
            )[0]
        )

    return evaluations


def perform_exploratory_kramers_kronig_tests(
    data: DataSet,
    test: str = "real",
    num_RCs: Optional[List[int]] = None,
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
) -> Tuple[
    List[KramersKronigResult],
    Tuple[KramersKronigResult, Dict[int, float], int, int],
]:
    """
    Similar to |perform_kramers_kronig_test| but returns some intermediate results rather than only the final |KramersKronigResult|.
    This function acts as a wrapper for |evaluate_log_F_ext|, |suggest_num_RC|, and |suggest_representation|.

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
        See |perform_kramers_kronig_test| for details.

    num_RCs: Optional[List[int]], optional
        A list of integers of the various numbers of RC elements to test.
        If a None value is provided (i.e., the default), then the range of values to test is determined automatically.

    add_capacitance: bool, optional
        See |perform_kramers_kronig_test| for details.

    add_inductance: bool, optional
        See |perform_kramers_kronig_test| for details.

    admittance: bool, optional
        See |perform_kramers_kronig_test| for details.

    min_log_F_ext: float, optional
        See |perform_kramers_kronig_test| for details.

    max_log_F_ext: float, optional
        See |perform_kramers_kronig_test| for details.

    log_F_ext: float, optional
        See |perform_kramers_kronig_test| for details.

    num_F_ext_evaluations: int, optional
        See |perform_kramers_kronig_test| for details.

    rapid_F_ext_evaluations: bool, optional
        See |perform_kramers_kronig_test| for details.

    cnls_method: str, optional
        See |perform_kramers_kronig_test| for details.

    max_nfev: int, optional
        See |perform_kramers_kronig_test| for details.

    timeout: int, optional
        See |perform_kramers_kronig_test| for details.

    num_procs: int, optional
        See |perform_kramers_kronig_test| for details.

    **kwargs
        See |perform_kramers_kronig_test| for details.

    Returns
    -------
    Tuple[List[|KramersKronigResult|], Tuple[|KramersKronigResult|, Dict[int, float], int, int]]

        A tuple containing a list of |KramersKronigResult| and the corresponding result of |suggest_num_RC| for the suggested extension of the range of time constants and the suggested representation of the immittance spectrum to test.
    """
    if not (_is_boolean(admittance) or admittance is None):
        raise TypeError(f"Expected a boolean or None instead of {admittance=}")

    evaluations: List[Tuple[float, List[KramersKronigResult], float]]
    evaluations = _evaluate_representations(
        representations=[False, True] if admittance is None else [admittance],
        data=data,
        test=test,
        num_RCs=num_RCs,
        add_capacitance=add_capacitance,
        add_inductance=add_inductance,
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
    assert len(evaluations) > 0, evaluations

    suggestions: List[Tuple[KramersKronigResult, Dict[int, float], int, int]] = []

    tests: List[KramersKronigResult] = []
    for _, tests, *_ in evaluations:
        suggestions.append(suggest_num_RC(tests, **kwargs))

    if not (len(suggestions) == len(evaluations) >= 1):
        raise ValueError(f"Expected {len(suggestions)=} == {len(evaluations)=} >= 1")

    suggestion = suggest_representation(suggestions)

    for _, tests, *_ in evaluations:
        if tests[0].admittance == suggestion[0].admittance:
            break
    else:
        raise ValueError(f"Expected {tests[0].admittance=} == {suggestion[0].admittance=}")

    return (tests, suggestion)
