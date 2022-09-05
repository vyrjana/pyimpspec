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

# This module uses Tikhonov regularization and either radial basis function or piecewise linear discretization
# pyDRTtools commit: 65ea54d9332a0c6594de852f0242a88e20ec4427

from multiprocessing import (
    Pool,
    cpu_count,
)
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
)
from numpy import (
    abs as array_abs,
    arccos,
    arctan2,
    argmin,
    array,
    concatenate,
    cos,
    cumsum,
    diff,
    divide,
    exp,
    eye,
    finfo,
    floating,
    inf,
    integer,
    issubdtype,
    log as ln,
    log10 as log,
    logspace,
    mean,
    min as array_min,
    ndarray,
    ones,
    pi,
    quantile,
    real,
    sin,
    sqrt,
    square,
    std,
    sum as array_sum,
    where,
    zeros,
)
from numpy.linalg import cholesky
from numpy.matlib import repmat
from numpy.random import randn
from scipy import integrate
from scipy.linalg import (
    inv as invert,
    solve as solve_linalg,
    toeplitz,
)
from scipy.optimize import (
    fsolve,
)
from cvxopt import (
    matrix,
    solvers as cvxopt_solvers,
)
from pyimpspec.data import DataSet
from pyimpspec.analysis.fitting import _calculate_residuals
from .result import (
    DRTError,
    DRTResult,
    _calculate_chisqr,
)
from .tr_nnls import _suggest_lambda
import pyimpspec.progress as progress


RBF_TYPES: List[str] = [
    "c0-matern",
    "c2-matern",
    "c4-matern",
    "c6-matern",
    "cauchy",
    "gaussian",
    "inverse-quadratic",
    "inverse-quadric",
    "piecewise-linear",
]

RBF_SHAPES: List[str] = ["fwhm", "factor"]

MODES: List[str] = ["complex", "real", "imaginary"]


def _generate_truncated_multivariate_gaussians(
    F: ndarray,  # m * d dimensions
    g: ndarray,  # m * 1 dimensions
    M: ndarray,  # d * d dimensions, symmetric and definite positive
    mu_r: ndarray,  # d * 1 dimensions
    initial_X: ndarray,  # d * 1 dimensions
    cov: bool = True,  # True -> M is the covariance and the mean is mu_r, False -> M is a precision matrix (log-density == -1/2 X'*M*X + r'*X)
    L: int = 1,  # Number of samples
) -> ndarray:
    """
    Algorithm described in http://arxiv.org/abs/1208.4118

    F: ndarray
        m * d dimensions

    g: ndarray
        m * 1 dimensions

    M: ndarray
        d * d dimensions, symmetric and definite positive

    mu_r: ndarray
        d * 1 dimensions

    initial_X: ndarray
        d * 1 dimensions

    cov: bool = True
        True -> M is the covariance and the mean is mu_r
        False -> M is a precision matrix (log-density == -1/2 X'*M*X + r'*X)

    L: int = 1
        Number of samples

    Returns an array (d * L dimensions) where each column is a sample
    """
    assert (
        g.shape[0] == F.shape[0]
    ), f"Constraint dimensions do not match: {g.shape[0]} != {F.shape[0]}"
    R: ndarray = cholesky(M)  # .T?
    R = R.T  # change the lower matrix to upper matrix
    mu: ndarray
    if cov is True:  # Using M as a covariance matrix
        mu = mu_r
        g = g + F @ mu
        F = F @ R.T
        initial_X = solve_linalg(R.T, (initial_X - mu))
    else:  # Using M as a precision matrix
        mu = solve_linalg(R, solve_linalg(R.T, mu_r))
        g = g + F @ mu
        F = solve_linalg(R, F)
        initial_X = R @ (initial_X - mu)
    assert (F @ initial_X + g).any() >= 0, "Inconsistent initial condition!"
    # Dimension of mean vector; each sample must be of this dimension
    d: int = initial_X.shape[0]
    bounce_count: int = 0  # TODO: Never really used, only incremented
    near_zero: float = 1e-12
    # Squared Euclidean norm of constraint matrix columns
    F2: ndarray = array_sum(square(F), axis=1)
    Ft: ndarray = F.T
    last_X: ndarray = initial_X
    Xs: ndarray = zeros(
        (
            d,
            L,
        )
    )
    Xs[:, 0] = initial_X
    progress_message: str = "Calculating credible intervals"
    progress.update_every_N_percent(0, message=progress_message)
    # Generate samples
    i: int = 2
    while i <= L:
        if i % 100 == 0:
            progress.update_every_N_percent(i + 1, total=L, message=progress_message)
        stop: bool = False
        j: int = -1
        # Generate initial velocity from normal distribution
        V0: ndarray = randn(d)
        X: ndarray = last_X
        T: float = pi / 2
        tt: float = 0.0
        while True:
            a: ndarray = real(V0)
            b: ndarray = X
            fa: ndarray = F @ a
            fb: ndarray = F @ b
            U: ndarray = sqrt(square(fa) + square(fb))
            # Has to be arctan2 not arctan
            phi: ndarray = arctan2(-fa, fb)
            # Find the locations where the constraints were hit
            pn: ndarray = array(array_abs(divide(g, U)) <= 1)
            if pn.any():
                inds: ndarray = where(pn)[0]
                phn: ndarray = phi[pn]
                t1: ndarray = array_abs(
                    -1.0 * phn + arccos(divide(-1.0 * g[pn], U[pn]))
                )
                # If there was a previous reflection (j > -1) and there is a potential
                # reflection at the sample plane, then make sure that a new reflection
                # at j is not found because of numerical error
                if j > -1:
                    if pn[j] == 1:
                        temp: ndarray = cumsum(pn)
                        indj = temp[j] - 1
                        tt1 = t1[indj]
                        if (
                            array_abs(tt1) < near_zero
                            or array_abs(tt1 - pi) < near_zero
                        ):
                            t1[indj] = inf
                mt: float = array_min(t1)
                m_ind = argmin(t1)
                j = inds[m_ind]
            else:
                mt = T
            # Update travel time
            tt = tt + mt
            if tt >= T:
                mt = mt - (tt - T)
                stop = True
            # Update position and velocity
            X = a * sin(mt) + b * cos(mt)
            V = a * cos(mt) - b * sin(mt)
            if stop:
                break
            # Update new velocity
            qj = F[j, :] @ V / F2[j]
            V0 = V - 2 * qj * Ft[:, j]
            bounce_count += 1
        if (F @ X + g).all() > 0:
            Xs[:, i - 1] = X
            last_X = X
            i = i + 1
    if cov:
        Xs = R.T @ Xs + repmat(mu.reshape(mu.shape[0], 1), 1, L)
    else:
        Xs = solve_linalg(R, Xs) + repmat(mu.reshape(mu.shape[0], 1), 1, L)
    return Xs


def _calculate_credible_intervals(
    num_RL: int,
    num_samples: int,
    mu: ndarray,
    Sigma_inv: ndarray,
    x: ndarray,
    tau_fine: ndarray,
    tau: ndarray,
    epsilon: float,
    rbf_type: str,
) -> Tuple[ndarray, ndarray, ndarray]:
    # Calculation of credible interval according to Bayesian statistics
    mu = mu[num_RL:]
    Sigma_inv = Sigma_inv[num_RL:, num_RL:]
    # Cholesky transform instead of direct inverse
    L_Sigma_inv: ndarray = cholesky(Sigma_inv)
    L_Sigma_agm: ndarray = invert(L_Sigma_inv)
    Sigma: ndarray = L_Sigma_agm.T @ L_Sigma_agm
    # Using generate_tmg from HMC_exact.py to sample the truncated Gaussian distribution
    Xs: ndarray = _generate_truncated_multivariate_gaussians(
        eye(x.shape[0]),
        finfo(float).eps * ones(mu.shape[0]),
        Sigma,
        mu,
        x,
        True,
        num_samples,
    )
    lower_bound: ndarray
    upper_bound: ndarray
    # map array to gamma
    _, lower_bound = _x_to_gamma(
        quantile(Xs[:, 501:], 0.005, axis=1),
        tau_fine,
        tau,
        epsilon,
        rbf_type,
    )
    _, upper_bound = _x_to_gamma(
        quantile(Xs[:, 501:], 0.995, axis=1),
        tau_fine,
        tau,
        epsilon,
        rbf_type,
    )
    _, mean_gamma = _x_to_gamma(
        mean(Xs[:, 501:], axis=1),
        tau_fine,
        tau,
        epsilon,
        rbf_type,
    )
    return (
        mean_gamma,
        lower_bound,
        upper_bound,
    )


def _rbf_epsilon_functions(func: Callable) -> Callable:
    switch: Dict[str, Callable] = {
        "gaussian": lambda x: exp(-((x) ** 2)) - 0.5,
        "c0-matern": lambda x: exp(-abs(x)) - 0.5,
        "c2-matern": lambda x: exp(-abs(x)) * (1 + abs(x)) - 0.5,
        "c4-matern": lambda x: 1 / 3 * exp(-abs(x)) * (3 + 3 * abs(x) + abs(x) ** 2)
        - 0.5,
        "c6-matern": lambda x: 1
        / 15
        * exp(-abs(x))
        * (15 + 15 * abs(x) + 6 * abs(x) ** 2 + abs(x) ** 3)
        - 0.5,
        "inverse-quadratic": lambda x: 1 / (1 + (x) ** 2) - 0.5,
        "inverse-quadric": lambda x: 1 / sqrt(1 + (x) ** 2) - 0.5,
        "cauchy": lambda x: 1 / (1 + abs(x)) - 0.5,
        "piecewise-linear": lambda x: 0.0,
    }
    assert set(RBF_TYPES) == set(switch.keys())

    def wrapper(*args, **kwargs):
        kwargs["rbf_functions"] = switch
        return func(*args, **kwargs)

    return wrapper


@_rbf_epsilon_functions
def _compute_epsilon(
    f: ndarray,
    rbf_shape: str,
    shape_coeff: float,
    rbf_type: str,
    rbf_functions: Dict[str, Callable],
) -> float:
    if rbf_type == "piecewise-linear":
        return 0.0
    elif rbf_shape == "fwhm":
        FWHM_coeff: ndarray = 2 * fsolve(rbf_functions[rbf_type], 1)
        delta: float = mean(diff(ln(1 / f.reshape(f.shape[0]))))
        return (shape_coeff * FWHM_coeff / delta)[0]
    # "factor"
    return shape_coeff


def _rbf_A_matrix_functions(func: Callable) -> Callable:
    switch: Dict[str, Callable] = {
        "c0-matern": lambda x, epsilon: exp(-abs(epsilon * x)),
        "c2-matern": lambda x, epsilon: exp(-abs(epsilon * x)) * (1 + abs(epsilon * x)),
        "c4-matern": lambda x, epsilon: 1
        / 3
        * exp(-abs(epsilon * x))
        * (3 + 3 * abs(epsilon * x) + abs(epsilon * x) ** 2),
        "c6-matern": lambda x, epsilon: 1
        / 15
        * exp(-abs(epsilon * x))
        * (
            15
            + 15 * abs(epsilon * x)
            + 6 * abs(epsilon * x) ** 2
            + abs(epsilon * x) ** 3
        ),
        "cauchy": lambda x, epsilon: 1 / (1 + abs(epsilon * x)),
        "gaussian": lambda x, epsilon: exp(-((epsilon * x) ** 2)),
        "inverse-quadratic": lambda x, epsilon: 1 / (1 + (epsilon * x) ** 2),
        "inverse-quadric": lambda x, epsilon: 1 / sqrt(1 + (epsilon * x) ** 2),
        "piecewise-linear": lambda x, epsilon: 0.0,
    }
    assert set(RBF_TYPES) == set(switch.keys())

    def wrapper(*args, **kwargs):
        kwargs["rbf_functions"] = switch
        return func(*args, **kwargs)

    return wrapper


@_rbf_A_matrix_functions
def _A_matrix_element(
    f: float,
    tau: float,
    epsilon: float,
    real: bool,
    rbf_type: str,
    rbf_functions: Dict[str, Callable],
) -> float:
    alpha: float = 2 * pi * f * tau
    rbf_func: Callable = rbf_functions[rbf_type]
    integrand: Callable
    if real is True:
        integrand = (
            lambda x: 1.0 / (1.0 + (alpha**2) * exp(2.0 * x)) * rbf_func(x, epsilon)
        )
    else:
        integrand = (
            lambda x: alpha
            / (1.0 / exp(x) + (alpha**2) * exp(x))
            * rbf_func(x, epsilon)
        )
    return integrate.quad(integrand, -50, 50, epsabs=1e-9, epsrel=1e-9)[0]


def _assemble_A_matrix(args) -> ndarray:
    f: ndarray
    tau: ndarray
    epsilon: float
    real: bool
    rbf_type: str
    (
        f,
        tau,
        epsilon,
        real,
        rbf_type,
    ) = args
    w: ndarray = 2 * pi * f
    num_freqs: int = f.shape[0]
    num_taus: int = tau.shape[0]
    A: ndarray
    i: int
    j: int
    # Check if the frequencies are ln-spaced
    if (
        num_freqs == num_taus
        and (std(diff(ln(1 / f))) / mean(diff(ln(1 / f)))) < 0.01
        and rbf_type != "piecewise-linear"
    ):
        # Use the Toeplitz trick
        C: ndarray = zeros(num_freqs)
        for i in range(0, num_freqs):
            C[i] = _A_matrix_element(f[i], tau[0], epsilon, real, rbf_type)
        R: ndarray = zeros(num_taus)
        for j in range(0, num_taus):
            R[j] = _A_matrix_element(f[0], tau[j], epsilon, real, rbf_type)
        A = toeplitz(C, R)
    else:
        # Use brute force
        A = zeros(
            (
                num_freqs,
                num_taus,
            )
        )
        for i in range(0, num_freqs):
            for j in range(0, num_taus):
                if rbf_type == "piecewise-linear":
                    if real is True:
                        A[i, j] = (
                            0.5
                            / (1 + (w[i] * tau[j]) ** 2)
                            * ln(
                                (tau[j] if j == num_taus - 1 else tau[j + 1])
                                / (tau[j] if j == 0 else tau[j - 1])
                            )
                        )
                    else:
                        A[i, j] = (
                            # -0.5
                            0.5
                            * (w[i] * tau[j])
                            / (1 + (w[i] * tau[j]) ** 2)
                            * ln(
                                (tau[j] if j == num_taus - 1 else tau[j + 1])
                                / (tau[j] if j == 0 else tau[j - 1])
                            )
                        )
                else:
                    A[i, j] = _A_matrix_element(f[i], tau[j], epsilon, real, rbf_type)
    return (1 if real is True else -1) * A


def _inner_product_rbf(
    f_i: float,
    f_j: float,
    epsilon: float,
    derivative_order: int,
    rbf_type: str,
) -> float:
    a: float = epsilon * ln(f_i / f_j)
    if rbf_type == "c0-matern":
        if derivative_order == 1:
            return epsilon * (1 - abs(a)) * exp(-abs(a))
        elif derivative_order == 2:
            return epsilon**3 * (1 + abs(a)) * exp(-abs(a))
    elif rbf_type == "c2-matern":
        if derivative_order == 1:
            return epsilon / 6 * (3 + 3 * abs(a) - abs(a) ** 3) * exp(-abs(a))
        elif derivative_order == 2:
            return (
                epsilon**3
                / 6
                * (3 + 3 * abs(a) - 6 * abs(a) ** 2 + abs(a) ** 3)
                * exp(-abs(a))
            )
    elif rbf_type == "c4-matern":
        if derivative_order == 1:
            return (
                epsilon
                / 30
                * (
                    105
                    + 105 * abs(a)
                    + 30 * abs(a) ** 2
                    - 5 * abs(a) ** 3
                    - 5 * abs(a) ** 4
                    - abs(a) ** 5
                )
                * exp(-abs(a))
            )
        elif derivative_order == 2:
            return (
                epsilon**3
                / 30
                * (45 + 45 * abs(a) - 15 * abs(a) ** 3 - 5 * abs(a) ** 4 + abs(a) ** 5)
                * exp(-abs(a))
            )
    elif rbf_type == "c6-matern":
        if derivative_order == 1:
            return (
                epsilon
                / 140
                * (
                    10395
                    + 10395 * abs(a)
                    + 3780 * abs(a) ** 2
                    + 315 * abs(a) ** 3
                    - 210 * abs(a) ** 4
                    - 84 * abs(a) ** 5
                    - 14 * abs(a) ** 6
                    - abs(a) ** 7
                )
                * exp(-abs(a))
            )
        elif derivative_order == 2:
            return (
                epsilon**3
                / 140
                * (
                    2835
                    + 2835 * abs(a)
                    + 630 * abs(a) ** 2
                    - 315 * abs(a) ** 3
                    - 210 * abs(a) ** 4
                    - 42 * abs(a) ** 5
                    + abs(a) ** 7
                )
                * exp(-abs(a))
            )
    elif rbf_type == "cauchy":
        if a == 0:
            if derivative_order == 1:
                return 2 / 3 * epsilon
            elif derivative_order == 2:
                return 8 / 5 * epsilon**3
        else:
            numerator: float
            denominator: float
            if derivative_order == 1:
                numerator = abs(a) * (2 + abs(a)) * (
                    4 + 3 * abs(a) * (2 + abs(a))
                ) - 2 * (1 + abs(a)) ** 2 * (4 + abs(a) * (2 + abs(a))) * ln(1 + abs(a))
                denominator = abs(a) ** 3 * (1 + abs(a)) * (2 + abs(a)) ** 3
                return 4 * epsilon * numerator / denominator
            elif derivative_order == 2:
                numerator = abs(a) * (2 + abs(a)) * (
                    -96
                    + abs(a)
                    * (2 + abs(a))
                    * (-30 + abs(a) * (2 + abs(a)))
                    * (4 + abs(a) * (2 + abs(a)))
                ) + 12 * (1 + abs(a)) ^ 2 * (
                    16 + abs(a) * (2 + abs(a)) * (12 + abs(a) * (2 + abs(a)))
                ) * ln(
                    1 + abs(a)
                )
                denominator = abs(a) ^ 5 * (1 + abs(a)) * (2 + abs(a)) ** 5
                return 8 * epsilon ^ 3 * numerator / denominator
    elif rbf_type == "gaussian":
        if derivative_order == 1:
            return -epsilon * (-1 + a**2) * exp(-(a**2 / 2)) * sqrt(pi / 2)
        elif derivative_order == 2:
            return (
                epsilon**3
                * (3 - 6 * a**2 + a**4)
                * exp(-(a**2 / 2))
                * sqrt(pi / 2)
            )
    elif rbf_type == "inverse-quadratic":
        if derivative_order == 1:
            return 4 * epsilon * (4 - 3 * a**2) * pi / ((4 + a**2) ** 3)
        elif derivative_order == 2:
            return (
                48
                * (16 + 5 * a**2 * (-8 + a**2))
                * pi
                * epsilon**3
                / ((4 + a**2) ** 5)
            )
    elif rbf_type == "inverse-quadric":
        y_i: float = -ln(f_i)
        y_j: float = -ln(f_j)
        rbf_i: Callable = lambda y: 1 / sqrt(1 + (epsilon * (y - y_i)) ** 2)
        rbf_j: Callable = lambda y: 1 / sqrt(1 + (epsilon * (y - y_j)) ** 2)
        delta: float
        sqr_drbf_dy: Callable
        if derivative_order == 1:
            delta = 1e-8
            sqr_drbf_dy = (
                lambda y: 1
                / (2 * delta)
                * (rbf_i(y + delta) - rbf_i(y - delta))
                * 1
                / (2 * delta)
                * (rbf_j(y + delta) - rbf_j(y - delta))
            )
        elif derivative_order == 2:
            delta = 1e-4
            sqr_drbf_dy = (
                lambda y: 1
                / (delta ^ 2)
                * (rbf_i(y + delta) - 2 * rbf_i(y) + rbf_i(y - delta))
                * 1
                / (delta ^ 2)
                * (rbf_j(y + delta) - 2 * rbf_j(y) + rbf_j(y - delta))
            )
        return integrate.quad(sqr_drbf_dy, -50, 50, epsabs=1e-9, epsrel=1e-9)[0]
    assert rbf_type not in RBF_TYPES, f"Unsupported RBF type: {rbf_type}"
    return -1.0  # Just to satisfy mypy


def _assemble_M_matrix(
    tau: ndarray,
    epsilon: float,
    derivative_order: int,
    rbf_type: str,
) -> ndarray:
    f: ndarray = 1 / tau
    num_freqs: int = f.shape[0]
    num_taus: int = tau.shape[0]
    M: ndarray
    i: int
    j: int
    # Check if the collocation points are sufficiently ln-spaced
    if (
        std(diff(ln(tau))) / mean(diff(ln(tau)))
    ) < 0.01 and rbf_type != "piecewise-linear":
        # Apply the Toeplitz trick to compute the M matrix
        C: ndarray = zeros(num_taus)
        for i in range(0, num_taus):
            C[i] = _inner_product_rbf(
                f[0],
                f[i],
                epsilon,
                derivative_order,
                rbf_type,
            )  # TODO: Maybe use tau instead of freq (pyDRTtools comment)
        R: ndarray = zeros(num_taus)
        for j in range(0, num_taus):
            R[j] = _inner_product_rbf(
                f[j],
                f[0],
                epsilon,
                derivative_order,
                rbf_type,
            )
        M = toeplitz(C, R)
    elif rbf_type == "piecewise-linear":
        if derivative_order == 1:
            M = zeros(
                (
                    num_freqs - 1,
                    num_freqs,
                )
            )
            for i in range(0, num_freqs - 1):
                delta_loc: float = ln((1 / f[i + 1]) / (1 / f[i]))
                M[i, i] = -1 / delta_loc
                M[i, i + 1] = 1 / delta_loc
        elif derivative_order == 2:
            M = zeros(
                (
                    num_taus - 2,
                    num_taus,
                )
            )
            for i in range(0, num_taus - 2):
                delta_loc = ln(tau[i + 1] / tau[i])

                if i == 0 or i == num_taus - 3:
                    M[i, i] = 2.0 / (delta_loc**2)
                    M[i, i + 1] = -4.0 / (delta_loc**2)
                    M[i, i + 2] = 2.0 / (delta_loc**2)
                else:
                    M[i, i] = 1.0 / (delta_loc**2)
                    M[i, i + 1] = -2.0 / (delta_loc**2)
                    M[i, i + 2] = 1.0 / (delta_loc**2)
        M = M.T @ M
    else:
        # Brute force
        M = zeros(
            (
                num_taus,
                num_taus,
            )
        )
        for i in range(0, num_taus):
            for j in range(0, num_taus):
                M[i, j] = _inner_product_rbf(
                    f[i],
                    f[j],
                    epsilon,
                    derivative_order,
                    rbf_type,
                )  # TODO: Maybe use tau instead of freq? See previous pyDRTtools comment.
    return M


def _quad_format(
    A: ndarray,
    b: ndarray,
    M: ndarray,
    lambda_value: float,
) -> Tuple[ndarray, ndarray]:
    H: ndarray = 2 * (A.T @ A + lambda_value * M)
    H = (H.T + H) / 2
    c: ndarray = -2 * b.T @ A
    return (
        H,
        c,
    )


def _quad_format_combined(
    A_re: ndarray,
    A_im: ndarray,
    b_re: ndarray,
    b_im: ndarray,
    M: ndarray,
    lambda_value: float,
) -> Tuple[ndarray, ndarray]:
    H: ndarray = 2 * ((A_re.T @ A_re + A_im.T @ A_im) + lambda_value * M)
    H = (H.T + H) / 2
    c: ndarray = -2 * (b_im.T @ A_im + b_re.T @ A_re)
    return (
        H,
        c,
    )


def _solve_qp(
    H: ndarray,
    c: ndarray,
    G: Optional[ndarray] = None,
    h: Optional[ndarray] = None,
    A: Optional[ndarray] = None,
    b: Optional[ndarray] = None,
) -> ndarray:
    # cvxopt
    args: List[matrix] = [matrix(H), matrix(c)]
    if G is not None:
        assert h is not None
        args.extend([matrix(G), matrix(h)])
    if A is not None:
        assert b is not None
        args.extend([matrix(A), matrix(b)])
    cvxopt_solvers.options["abstol"] = 1e-15
    cvxopt_solvers.options["reltol"] = 1e-15
    solution: dict = cvxopt_solvers.qp(*args)
    if "optimal" not in solution["status"]:
        raise DRTError("Failed to find optimal solution!")
    return array(solution["x"]).reshape((H.shape[1],))


def _rbf_gamma_functions(func: Callable) -> Callable:
    switch: Dict[str, Callable] = {
        "c0-matern": lambda x, epsilon: exp(-abs(epsilon * x)),
        "c2-matern": lambda x, epsilon: exp(-abs(epsilon * x)) * (1 + abs(epsilon * x)),
        "c4-matern": lambda x, epsilon: 1
        / 3
        * exp(-abs(epsilon * x))
        * (3 + 3 * abs(epsilon * x) + abs(epsilon * x) ** 2),
        "c6-matern": lambda x, epsilon: 1
        / 15
        * exp(-abs(epsilon * x))
        * (
            15
            + 15 * abs(epsilon * x)
            + 6 * abs(epsilon * x) ** 2
            + abs(epsilon * x) ** 3
        ),
        "cauchy": lambda x, epsilon: 1 / (1 + abs(epsilon * x)),
        "gaussian": lambda x, epsilon: exp(-((epsilon * x) ** 2)),
        "inverse-quadratic": lambda x, epsilon: 1 / (1 + (epsilon * x) ** 2),
        "inverse-quadric": lambda x, epsilon: 1 / sqrt(1 + (epsilon * x) ** 2),
        "piecewise-linear": lambda x, epsilon: 0.0,
    }
    assert set(RBF_TYPES) == set(switch.keys())

    def wrapper(*args, **kwargs):
        kwargs["rbf_functions"] = switch
        return func(*args, **kwargs)

    return wrapper


@_rbf_gamma_functions
def _x_to_gamma(
    x: ndarray,
    tau_fine: ndarray,
    tau: ndarray,
    epsilon: float,
    rbf_type: str,
    rbf_functions: Dict[str, Callable],
) -> Tuple[ndarray, ndarray]:
    # TODO: double check this to see if the function is correct (pyDRTtools comment)
    if rbf_type == "piecewise-linear":
        return (
            tau,
            x,
        )
    num_taus: int = tau.shape[0]
    num_fine_taus: int = tau_fine.shape[0]
    B: ndarray = zeros(
        (
            num_fine_taus,
            num_taus,
        )
    )
    rbf: Callable = rbf_functions[rbf_type]
    i: int
    j: int
    for i in range(0, num_fine_taus):
        for j in range(0, num_taus):
            delta_ln_tau = ln(tau_fine[i]) - ln(tau[j])
            B[i, j] = rbf(delta_ln_tau, epsilon)
    return (
        tau_fine,
        B @ x,
    )


def _prepare_complex_matrices(
    A_re: ndarray,
    A_im: ndarray,
    b_re: ndarray,
    b_im: ndarray,
    M: ndarray,
    lambda_value: float,
    f: ndarray,
    num_freqs: int,
    num_taus: int,
    inductance: bool,
) -> Tuple[ndarray, ndarray, ndarray, ndarray, ndarray, int]:
    num_RL: int = 2 if inductance is True else 1
    tmp: ndarray  # Used for temporary binding of matrices
    tmp = A_re
    A_re = zeros(
        (
            num_freqs,
            num_taus + num_RL,
        )
    )
    A_re[:, num_RL:] = tmp
    A_re[:, 1 if inductance is True else 0] = 1

    tmp = A_im
    A_im = zeros(
        (
            num_freqs,
            num_taus + num_RL,
        )
    )
    A_im[:, num_RL:] = tmp
    if inductance is True:
        A_im[:, 0] = 2 * pi * f

    tmp = M
    M = zeros(
        (
            num_taus + num_RL,
            num_taus + num_RL,
        )
    )
    M[num_RL:, num_RL:] = tmp

    H, c = _quad_format_combined(
        A_re,
        A_im,
        b_re,
        b_im,
        M,
        lambda_value,
    )
    return (
        A_re,
        A_im,
        M,
        H,
        c,
        num_RL,
    )


def _prepare_real_matrices(
    A_re: ndarray,
    A_im: ndarray,
    b_re: ndarray,
    b_im: ndarray,
    M: ndarray,
    lambda_value: float,
    num_freqs: int,
    num_taus: int,
) -> Tuple[ndarray, ndarray, ndarray, ndarray, ndarray, int]:
    num_RL: int = 1
    tmp: ndarray  # Used for temporary binding of matrices
    tmp = A_re
    A_re = zeros(
        (
            num_freqs,
            num_taus + num_RL,
        )
    )
    A_re[:, num_RL:] = tmp
    A_re[:, 0] = 1

    tmp = A_im
    A_im = zeros(
        (
            num_freqs,
            num_taus + num_RL,
        )
    )
    A_im[:, num_RL:] = tmp

    tmp = M
    M = zeros(
        (
            num_taus + num_RL,
            num_taus + num_RL,
        )
    )
    M[num_RL:, num_RL:] = tmp

    H, c = _quad_format(
        A_re,
        b_re,
        M,
        lambda_value,
    )
    return (
        A_re,
        A_im,
        M,
        H,
        c,
        num_RL,
    )


def _prepare_imaginary_matrices(
    A_re: ndarray,
    A_im: ndarray,
    b_re: ndarray,
    b_im: ndarray,
    M: ndarray,
    lambda_value: float,
    f: ndarray,
    num_freqs: int,
    num_taus: int,
    inductance: bool,
) -> Tuple[ndarray, ndarray, ndarray, ndarray, ndarray, int]:
    num_RL: int = 1 if inductance is True else 0
    tmp: ndarray  # Used for temporary binding of matrices
    tmp = A_re
    A_re = zeros(
        (
            num_freqs,
            num_taus + num_RL,
        )
    )
    A_re[:, num_RL:] = tmp

    tmp = A_im
    A_im = zeros(
        (
            num_freqs,
            num_taus + num_RL,
        )
    )
    A_im[:, num_RL:] = tmp
    if inductance is True:
        A_im[:, 0] = 2 * pi * f
    tmp = M
    M = zeros(
        (
            num_taus + num_RL,
            num_taus + num_RL,
        )
    )
    M[num_RL:, num_RL:] = tmp

    H, c = _quad_format(
        A_im,
        b_im,
        M,
        lambda_value,
    )
    return (
        A_re,
        A_im,
        M,
        H,
        c,
        num_RL,
    )


def _lambda_process(
    args,
) -> Optional[
    Tuple[
        float,
        float,
        ndarray,
        ndarray,
        ndarray,
        ndarray,
        ndarray,
        ndarray,
        ndarray,
        int,
    ]
]:
    A_re: ndarray
    A_im: ndarray
    Z: ndarray
    M: ndarray
    lambda_value: float
    f: ndarray
    tau: ndarray
    tau_fine: ndarray
    epsilon: float
    mode: str
    rbf_type: str
    inductance: bool
    maximum_symmetry: float
    (
        A_re,
        A_im,
        Z,
        M,
        lambda_value,
        f,
        tau,
        tau_fine,
        epsilon,
        mode,
        rbf_type,
        inductance,
        maximum_symmetry,
    ) = args
    num_freqs: int = f.size
    num_taus: int = tau.size
    num_RL: int  # The number of R and/or L elements in series
    H: ndarray
    c: ndarray
    if mode == "complex":
        A_re, A_im, M, H, c, num_RL = _prepare_complex_matrices(
            A_re,
            A_im,
            Z.real,
            Z.imag,
            M,
            lambda_value,
            f,
            num_freqs,
            num_taus,
            inductance,
        )
    elif mode == "real":
        A_re, A_im, M, H, c, num_RL = _prepare_real_matrices(
            A_re,
            A_im,
            Z.real,
            Z.imag,
            M,
            lambda_value,
            num_freqs,
            num_taus,
        )
    elif mode == "imaginary":
        A_re, A_im, M, H, c, num_RL = _prepare_imaginary_matrices(
            A_re,
            A_im,
            Z.real,
            Z.imag,
            M,
            lambda_value,
            f,
            num_freqs,
            num_taus,
            inductance,
        )
    try:
        x: ndarray = _solve_qp(H, c)
    except Exception:
        return None
    gamma: ndarray
    _, gamma = _x_to_gamma(x[num_RL:], tau_fine, tau, epsilon, rbf_type)
    min_gamma: float = abs(min(gamma))
    max_gamma: float = abs(max(gamma))
    score: float = 1.0 - ((max_gamma - min_gamma) / max(min_gamma, max_gamma))
    if score > maximum_symmetry:
        return None
    Z_fit: ndarray = array(
        list(map(lambda _: complex(_[0], _[1]), zip(A_re @ x, A_im @ x)))
    )
    return (
        lambda_value,
        sqrt(array_sum(gamma**2)),
        x,
        Z_fit,
        A_re,
        A_im,
        M,
        H,
        c,
        num_RL,
    )


def _calculate_drt_tr_rbf(
    data: DataSet,
    mode: str = "complex",
    lambda_value: float = 1e-3,
    rbf_type: str = "gaussian",
    derivative_order: int = 1,
    rbf_shape: str = "fwhm",
    shape_coeff: float = 0.5,
    inductance: bool = False,
    credible_intervals: bool = False,
    num_samples: int = 2000,
    maximum_symmetry: float = 0.3,
    num_procs: int = -1,
) -> DRTResult:
    assert type(mode) is str and mode in MODES, f"Valid mode values: {', '.join(MODES)}"
    assert issubdtype(type(lambda_value), floating), lambda_value
    assert (
        type(rbf_type) is str and rbf_type in RBF_TYPES
    ), f"Valid rbf_type values: {', '.join(RBF_TYPES)}"
    assert (
        issubdtype(type(derivative_order), integer) and 1 <= derivative_order <= 2
    ), "Valid derivative_order values: 1, 2"
    assert (
        type(rbf_shape) is str and rbf_shape in RBF_SHAPES
    ), f"Valid rbf_shape values: {', '.join(RBF_SHAPES)}"
    assert (
        issubdtype(
            type(shape_coeff),
            floating,
        )
        and shape_coeff > 0.0
    ), shape_coeff
    assert type(inductance) is bool, inductance
    assert type(credible_intervals) is bool, credible_intervals
    if credible_intervals is True:
        assert num_samples >= 1000, f"{num_samples} is not enough samples!"
    assert issubdtype(type(num_samples), integer) and num_samples > 0, num_samples
    assert (
        issubdtype(
            type(maximum_symmetry),
            floating,
        )
        and 0.0 <= maximum_symmetry <= 1.0
    ), maximum_symmetry
    assert issubdtype(type(num_procs), integer), num_procs
    if num_procs < 1:
        num_procs = cpu_count()
    min_log_lambda: float = -15.0
    max_log_lambda: float = 0.0
    lambda_values: ndarray = (
        logspace(
            min_log_lambda,
            max_log_lambda,
            num=round((max_log_lambda - min_log_lambda)) * 1 + 1,
        )
        if lambda_value <= 0.0
        else array([lambda_value])
    )
    # TODO: Figure out if f and Z need to be altered depending on the value
    # of the 'inductance' argument!
    f: ndarray = data.get_frequency()
    Z: ndarray = data.get_impedance()
    tau: ndarray = 1 / f
    tau_fine: ndarray = logspace(
        log(tau.min()) - 0.5, log(tau.max()) + 0.5, 10 * f.shape[0]
    )
    num_freqs: int = f.size
    epsilon: float = _compute_epsilon(f, rbf_shape, shape_coeff, rbf_type)
    progress_message: str = "Preparing matrices"
    progress.update_every_N_percent(0, total=3, message=progress_message)
    A_re: ndarray
    A_im: ndarray
    with Pool(2 if num_procs > 1 else 1) as pool:
        for i, res in enumerate(
            pool.imap(
                _assemble_A_matrix,
                [
                    (
                        f,
                        tau,
                        epsilon,
                        True,
                        rbf_type,
                    ),
                    (
                        f,
                        tau,
                        epsilon,
                        False,
                        rbf_type,
                    ),
                ],
            )
        ):
            progress.update_every_N_percent(i + 1, total=3, message=progress_message)
            if i == 0:
                A_re = res
            else:
                A_im = res
    M: ndarray = _assemble_M_matrix(tau, epsilon, derivative_order, rbf_type)
    progress.update_every_N_percent(1, message=progress_message)
    with Pool(num_procs) as pool:
        args = (
            (
                A_re,
                A_im,
                Z,
                M,
                lambda_value,
                f,
                tau,
                tau_fine,
                epsilon,
                mode,
                rbf_type,
                inductance,
                maximum_symmetry if lambda_values.size > 1 else 1.0,
            )
            for lambda_value in lambda_values
        )
        results: List[
            Tuple[
                float,
                float,
                ndarray,
                ndarray,
                ndarray,
                ndarray,
                ndarray,
                ndarray,
                ndarray,
                int,
            ]
        ] = []
        progress_message = "Calculating DRT"
        progress.update_every_N_percent(0, message=progress_message)
        i: int
        for i, res in enumerate(pool.imap_unordered(_lambda_process, args)):
            progress.update_every_N_percent(
                i + 1,
                total=len(lambda_values),
                message=progress_message,
            )
            if res is not None:
                results.append(res)
    if len(results) == 0:
        raise DRTError("Failed to perform calculations! Try tweaking the settings.")
    if len(results) > 1:
        results.sort(key=lambda _: _[0])
        lambda_value = _suggest_lambda(
            array(list(map(lambda _: _[0], results))),
            array(list(map(lambda _: _[1], results))),
        )
        results = list(filter(lambda _: _[0] == lambda_value, results))
    lambda_value, _, x, Z_fit, A_re, A_im, M, H, c, num_RL = results[0]
    sigma_re_im: float
    if mode == "complex":
        sigma_re_im = std(concatenate([Z_fit.real - Z.real, Z_fit.imag - Z.imag]))
    elif mode == "real":
        sigma_re_im = std(Z_fit.real - Z.real)
    elif mode == "imaginary":
        sigma_re_im = std(Z_fit.imag - Z.imag)
    inv_V: ndarray = 1 / sigma_re_im**2 * eye(num_freqs)
    Sigma_inv: ndarray
    mu_numerator: ndarray
    if mode == "complex":
        Sigma_inv = (
            (A_re.T @ inv_V @ A_re)
            + (A_im.T @ inv_V @ A_im)
            + (lambda_value / sigma_re_im**2) * M
        )
        mu_numerator = A_re.T @ inv_V @ Z.real + A_im.T @ inv_V @ Z.imag
    elif mode == "real":
        Sigma_inv = (A_re.T @ inv_V @ A_re) + (lambda_value / sigma_re_im**2) * M
        mu_numerator = A_re.T @ inv_V @ Z.real
    elif mode == "imaginary":
        Sigma_inv = (A_im.T @ inv_V @ A_im) + (lambda_value / sigma_re_im**2) * M
        mu_numerator = A_im.T @ inv_V @ Z.imag
    Sigma_inv = (Sigma_inv + Sigma_inv.T) / 2
    L_Sigma_inv: ndarray = cholesky(Sigma_inv)
    mu: ndarray = solve_linalg(L_Sigma_inv.T, solve_linalg(L_Sigma_inv, mu_numerator))
    # TODO: Why were L and R defined only to not be used?
    L: float
    R: float
    if num_RL == 0:
        L, R = 0.0, 0.0
    elif num_RL == 1:
        if mode == "imaginary":
            L, R = x[0], 0.0
        else:
            L, R = 0.0, x[0]
    elif num_RL == 2:
        L, R = x[0:2]
    x = x[num_RL:]
    return DRTResult(
        "TR-RBF",
        *_x_to_gamma(x, tau_fine, tau, epsilon, rbf_type),
        f,
        Z_fit,
        *_calculate_residuals(Z, Z_fit),
        # "tr-rbf" method with credible_intervals
        *(
            _calculate_credible_intervals(
                num_RL,
                num_samples,
                mu,
                Sigma_inv,
                x,
                tau_fine,
                tau,
                epsilon,
                rbf_type,
            )
            if credible_intervals is True
            else (
                array([]),  # Mean
                array([]),  # Lower bound
                array([]),  # Upper bound
            )
        ),
        # "bht" method
        array([]),  # Imaginary gamma
        {},
        # Stats
        _calculate_chisqr(Z, Z_fit),
        lambda_value,
    )
