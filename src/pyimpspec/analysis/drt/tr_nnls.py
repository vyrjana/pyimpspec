# pyimpspec is licensed under the GPLv3 or later (https://www.gnu.org/licenses/gpl-3.0.html).
# Copyright 2022 pyimpspec developers
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

# This module uses Tikhonov regularization and non-negative least squares
# DRT-python-code commit: 9663ed8b331f521a9fcdb0b58fb2b34693df938c

from typing import (
    List,
)
from numpy import (
    array,
    float64,
    floating,
    identity,
    integer,
    int64,
    issubdtype,
    log as ln,
    logspace,
    ndarray,
    pi,
    sqrt,
    sum as array_sum,
    zeros,
)
from scipy.optimize import nnls
from pyimpspec.data import DataSet
from pyimpspec.analysis.fitting import _calculate_residuals
from pyimpspec.analysis.drt.result import (
    DRTResult,
    _calculate_chisqr,
)
import pyimpspec.progress as progress


_MODES: List[str] = ["real", "imaginary"]


def _generate_tikhonov_matrix(A: ndarray, I: ndarray, lambda_value: float) -> ndarray:
    return (A.T @ A) + lambda_value * I


def _solve(A: ndarray, b: ndarray) -> ndarray:
    return nnls(A, b)[0]


def _suggest_lambda(
    lambda_values: ndarray,
    solution_norms: ndarray,
) -> float:
    a: ndarray = zeros(lambda_values.size - 1, dtype=float64)
    for i in range(0, lambda_values.size - 1):
        a[i] = solution_norms[i] - solution_norms[i + 1]
    b: ndarray = zeros(a.size - 1, dtype=float64)
    for i in range(0, b.size - 1):
        b[i] = a[i] - a[i + 1]
    c: float
    for i, c in reversed(list(enumerate(b))):
        if c < 0.0:
            return lambda_values[(i + 1) if i < lambda_values.size - 2 else i]
    return lambda_values[-1]


def _calculate_drt_tr_nnls(
    data: DataSet,
    mode: str,
    lambda_value: float,
    num_procs: int,
) -> DRTResult:
    assert (
        hasattr(data, "get_frequency")
        and callable(data.get_frequency)
        and hasattr(data, "get_impedance")
        and callable(data.get_impedance)
    ), data
    assert (
        type(mode) is str and mode in _MODES
    ), f"Valid mode values: {', '.join(_MODES)}"
    assert issubdtype(type(lambda_value), floating), lambda_value
    assert issubdtype(type(num_procs), integer), num_procs
    # This is simply to avoid locking up the GUI thread in DearEIS.
    progress_message: str = "Preparing matrices"
    num_matrices: int = 4
    progress.update_every_N_percent(0, total=num_matrices, message=progress_message)
    i: int
    f: ndarray = data.get_frequency()
    Z: ndarray = data.get_impedance()
    omega: ndarray = 2 * pi * f
    tau: ndarray = 1 / f
    ln_tau: ndarray = ln(tau)
    delta_ln_tau: ndarray = zeros(omega.size, dtype=float64)
    for i in range(1, omega.size - 1):
        delta_ln_tau[i] = 0.5 * (ln_tau[i + 1] - ln_tau[i - 1])
    delta_ln_tau[0] = 0.5 * (ln_tau[1] - ln_tau[0])
    delta_ln_tau[-1] = 0.5 * (ln_tau[-1] - ln_tau[-2])
    progress.update_every_N_percent(1, total=num_matrices, message=progress_message)
    # Subtract the high-frequency resistance, which will be added back later,
    # and then normalize the impedances.
    R_inf: float = Z[0].real
    Z -= R_inf
    R_pol: float = Z[-1].real - Z[0].real
    Z_norm: ndarray = array(
        list(map(lambda _: complex(_.real / R_pol, _.imag / R_pol), Z))
    )
    # Added back so that the residuals and chi-square calculations will be correct.
    Z += R_inf
    # Prepare matrices and vectors
    I: ndarray = identity(omega.size, dtype=int64)
    progress.update_every_N_percent(2, total=num_matrices, message=progress_message)
    A: ndarray = zeros(
        (
            omega.size,
            omega.size,
        ),
        dtype=float64,
    )
    product: ndarray
    for i in range(0, omega.size):
        product = omega[i] * tau
        A[i, :] = (
            (product if mode == "imaginary" else 1) * delta_ln_tau / (1 + product**2)
        )
    progress.update_every_N_percent(3, total=num_matrices, message=progress_message)
    b: ndarray = A.T @ (Z_norm.real if mode == "real" else -Z_norm.imag)
    progress.update_every_N_percent(4, total=num_matrices, message=progress_message)
    progress_message = "Calculating DRT"
    progress.update_every_N_percent(0, message=progress_message)
    A_tikh: ndarray
    g_tau: ndarray
    if lambda_value <= 0.0:
        # Determine suitable regularization parameter if one hasn't been provided.
        min_log_lambda: int = -15
        max_log_lambda: int = 0
        lambda_values = logspace(
            min_log_lambda,
            max_log_lambda,
            # If the number of points is too high, then there is a risk of suggesting
            # a lambda value that is unnecessarily high.
            num=(max_log_lambda - min_log_lambda) * 1 + 1,
        )
        A_tikhs: List[ndarray] = []
        g_taus: List[ndarray] = []
        # residuals: ndarray = zeros(lambda_values.size, dtype=float64)
        solution_norms: ndarray = zeros(lambda_values.size, dtype=float64)
        for i, lambda_value in enumerate(lambda_values):
            A_tikhs.append(_generate_tikhonov_matrix(A, I, lambda_value))
            g_taus.append(_solve(A_tikhs[-1], b))
            # residuals[i] = sqrt(array_sum(((A.T @ A) @ g_taus[-1] - b) ** 2))
            solution_norms[i] = sqrt(array_sum(g_taus[-1] ** 2))
            progress.update_every_N_percent(
                i + 1,
                total=len(lambda_values),
                message=progress_message,
            )
        lambda_value = _suggest_lambda(lambda_values, solution_norms)
        A_tikh = A_tikhs[-1]
        g_tau = g_taus[-1]
        for i, lv in enumerate(lambda_values):
            if lv == lambda_value:
                A_tikh = A_tikhs[i]
                g_tau = g_taus[i]
                break
    else:
        A_tikh = _generate_tikhonov_matrix(A, I, lambda_value)
        g_tau = _solve(A_tikh, b)
        progress.update_every_N_percent(1, message=progress_message)
    R_pol_synthetic: float = array_sum(g_tau * delta_ln_tau)  # Should be (close to) 1.0
    # W: ndarray = A_tikh @ g_tau
    # res: float = sqrt(array_sum((W - b) ** 2))
    # lhs: float = sqrt(array_sum(W**2))
    Z_re_im: ndarray = zeros(omega.size, dtype=float64)
    for i in range(0, omega.size):
        product = omega[i] * tau
        Z_re_im[i] = array_sum(
            delta_ln_tau
            * ((product if mode == "imaginary" else 1) * g_tau / (1 + product**2))
        )
    Z_re_im = R_pol * Z_re_im
    Z_fit: ndarray
    if mode == "real":
        Z_fit = array(list(map(lambda _: complex(*_), zip(Z_re_im + R_inf, Z.imag))))
    else:
        Z_fit = array(list(map(lambda _: complex(*_), zip(Z.real, -Z_re_im))))
    gamma: ndarray = g_tau * R_pol
    return DRTResult(
        "TR-NNLS",
        tau,
        gamma,
        f,
        Z_fit,
        *_calculate_residuals(Z, Z_fit),
        # "tr-rbf" method with credible_intervals
        array([]),  # Mean
        array([]),  # Lower bound
        array([]),  # Upper bound
        # "bht" method
        array([]),  # Imaginary gamma
        {},
        # Stats
        _calculate_chisqr(Z, Z_fit),
        lambda_value,
    )
