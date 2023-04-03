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

# This module uses Tikhonov regularization and either radial basis function or piecewise linear discretization
# 10.1016/j.electacta.2015.09.097
# - 10.1016/j.electacta.2015.03.123
# - 10.1016/j.electacta.2017.07.050
# Based on code from https://github.com/ciuccislab/pyDRTtools.
# pyDRTtools commit: 65ea54d9332a0c6594de852f0242a88e20ec4427

from dataclasses import dataclass
from multiprocessing import Pool
from multiprocessing.context import TimeoutError as MPTimeoutError
from time import sleep
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
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
    float64,
    floating,
    inf,
    int64,
    integer,
    issubdtype,
    log as ln,
    log10 as log,
    logspace,
    mean,
    min as array_min,
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
from numpy.linalg import (
    cholesky,
    norm,
)
from numpy.matlib import repmat
from numpy.random import randn
from numpy.typing import NDArray
from pyimpspec.data import DataSet
from pyimpspec.analysis.utility import (
    _calculate_residuals,
    _calculate_pseudo_chisqr,
    _get_default_num_procs,
)
from pyimpspec.exceptions import DRTError
from .result import DRTResult
from pyimpspec.progress import Progress
from pyimpspec.typing import (
    ComplexImpedance,
    ComplexImpedances,
    Frequencies,
    Indices,
    Gamma,
    Gammas,
    TimeConstant,
    TimeConstants,
)
from .utility import _l_curve_corner_search

_SOLVER_IMPORTED: bool = False


@dataclass(frozen=True)
class TRRBFResult(DRTResult):
    """
    An object representing the results of calculating the distribution of relaxation times in a data set using Tikhonov regularization and radial basis function (or piecewise linear) discretization (TR-RBF).

    Parameters
    ----------
    time_constants: |TimeConstants|
        The time constants.

    gammas: |Gammas|
        The gamma values.

    frequencies: |Frequencies|
        The frequencies of the impedance spectrum.

    impedances: |ComplexImpedances|
        The impedance produced by the model.

    residuals: |ComplexResiduals|
        The residuals of the impedances of the model and the data set.

    mean_gammas: |Gammas|
        The mean gamma values of the Bayesian credible intervals.

    lower_bounds: |Gammas|
        The lower bound gamma values of the Bayesian credible intervals.

    upper_bounds: |Gammas|
        The upper bound gamma values of the Bayesian credible intervals.

    pseudo_chisqr: float
        The pseudo chi-squared value, |pseudo chi-squared|, of the modeled impedance (eq. 14 in Boukamp, 1995).

    lambda_value: float
        The lambda value that was ultimately used.
    """

    gammas: Gammas
    mean_gammas: Gammas
    lower_bounds: Gammas
    upper_bounds: Gammas
    lambda_value: float

    def get_label(self) -> str:
        return "TR-RBF"

    def get_drt_credible_intervals_data(
        self,
    ) -> Tuple[TimeConstants, Gammas, Gammas, Gammas]:
        """
        Get the data necessary to plot the Bayesian credible intervals for this DRT result: the time constants, the mean gamma values, the lower bound gamma values, and the upper bound gamma values.

        Returns
        -------
        Tuple[|TimeConstants|, |Gammas|, |Gammas|, |Gammas|]
        """
        if not self.mean_gammas.any():
            return (
                array([], dtype=TimeConstant),
                array([], dtype=Gamma),
                array([], dtype=Gamma),
                array([], dtype=Gamma),
            )
        return (
            self.time_constants,
            self.mean_gammas,
            self.lower_bounds,
            self.upper_bounds,
        )

    def get_gammas(self) -> Gammas:
        """
        Get the gamma values.

        Returns
        -------
        |Gammas|
        """
        return self.gammas

    def get_drt_data(self) -> Tuple[TimeConstants, Gammas]:
        """
        Get the data necessary to plot this DRTResult as a DRT plot: the time constants and the corresponding gamma values.

        Returns
        -------
        Tuple[|TimeConstants|, |Gammas|]
        """
        return (
            self.time_constants,  # type: ignore
            self.gammas,  # type: ignore
        )

    def get_peaks(self, threshold: float = 0.0) -> Tuple[TimeConstants, Gammas]:
        """
        Get the time constants (in seconds) and gamma (in ohms) of peaks with magnitudes greater than the threshold.
        The threshold and the magnitudes are all relative to the magnitude of the highest peak.

        Parameters
        ----------
        threshold: float, optional
            The minimum peak height threshold (relative to the height of the tallest peak) for a peak to be included.

        Returns
        -------
        Tuple[|TimeConstants|, |Gammas|]
        """
        indices: Indices = self._get_peak_indices(
            threshold,
            self.gammas,  # type: ignore
        )
        return (
            self.time_constants[indices],  # type: ignore
            self.gammas[indices],  # type: ignore
        )

    def to_peaks_dataframe(
        self,
        threshold: float = 0.0,
        columns: Optional[List[str]] = None,
    ) -> "DataFrame":  # noqa: F821
        from pandas import DataFrame

        if columns is None:
            columns = [
                "tau (s)",
                "gamma (ohm)",
            ]
        assert isinstance(columns, list), columns
        assert len(columns) == 2
        indices: Indices = self._get_peak_indices(
            threshold,
            self.gammas,  # type: ignore
        )
        return DataFrame.from_dict(
            {
                columns[0]: self.time_constants[indices],  # type: ignore
                columns[1]: self.gammas[indices],  # type: ignore
            }
        )

    def to_statistics_dataframe(
        self,
    ) -> "DataFrame":  # noqa: F821
        from pandas import DataFrame

        statistics: Dict[str, Union[int, float, str]] = {
            "Log pseudo chi-squared": log(self.pseudo_chisqr),
            "Lambda": self.lambda_value,
        }
        return DataFrame.from_dict(
            {
                "Label": list(statistics.keys()),
                "Value": list(statistics.values()),
            }
        )


_RBF_TYPES: List[str] = [
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

_RBF_SHAPES: List[str] = ["fwhm", "factor"]

_MODES: List[str] = ["complex", "real", "imaginary"]


def _generate_truncated_multivariate_gaussians(
    F: NDArray[float64],  # m * d dimensions
    g: NDArray[float64],  # m * 1 dimensions
    M: NDArray[float64],  # d * d dimensions, symmetric and definite positive
    mu_r: NDArray[float64],  # d * 1 dimensions
    initial_X: NDArray[float64],  # d * 1 dimensions
    cov: bool = True,  # True -> M is the covariance and the mean is mu_r, False -> M is a precision matrix (log-density == -1/2 X'*M*X + r'*X)
    L: int = 1,  # Number of samples
) -> NDArray[float64]:
    """
    Algorithm described in http://arxiv.org/abs/1208.4118

    F: NDArray[float64]
        m * d dimensions

    g: NDArray[float64]
        m * 1 dimensions

    M: NDArray[float64]
        d * d dimensions, symmetric and definite positive

    mu_r: NDArray[float64]
        d * 1 dimensions

    initial_X: NDArray[float64]
        d * 1 dimensions

    cov: bool = True
        True -> M is the covariance and the mean is mu_r
        False -> M is a precision matrix (log-density == -1/2 X'*M*X + r'*X)

    L: int = 1
        Number of samples

    Returns an array (d * L dimensions) where each column is a sample
    """
    from scipy.linalg import solve as solve_linalg

    assert (
        g.shape[0] == F.shape[0]
    ), f"Constraint dimensions do not match: {g.shape[0]} != {F.shape[0]}"
    R: NDArray[float64] = cholesky(M)  # .T?
    R = R.T  # change the lower matrix to upper matrix
    mu: NDArray[float64]
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
    near_zero: float = 1e-12
    # Squared Euclidean norm of constraint matrix columns
    F2: NDArray[float64] = array_sum(square(F), axis=1)
    Ft: NDArray[float64] = F.T
    last_X: NDArray[float64] = initial_X
    Xs: NDArray[float64] = zeros(
        (
            d,
            L,
        ),
        dtype=float64,
    )
    Xs[:, 0] = initial_X
    # Generate samples
    i: int = 2
    while i <= L:
        stop: bool = False
        j: int = -1
        # Generate initial velocity from normal distribution
        V0: NDArray[float64] = randn(d)
        X: NDArray[float64] = last_X
        T: float = pi / 2
        tt: float = 0.0
        while True:
            a: NDArray[float64] = real(V0)
            b: NDArray[float64] = X
            fa: NDArray[float64] = F @ a
            fb: NDArray[float64] = F @ b
            U: NDArray[float64] = sqrt(square(fa) + square(fb))
            # Has to be arctan2 not arctan
            phi: NDArray[float64] = arctan2(-fa, fb)
            # Find the locations where the constraints were hit
            pn: NDArray[float64] = array(array_abs(divide(g, U)) <= 1)
            if pn.any():
                inds: NDArray[int64] = where(pn)[0]
                phn: NDArray[float64] = phi[pn]
                t1: NDArray[float64] = array_abs(
                    -1.0 * phn + arccos(divide(-1.0 * g[pn], U[pn]))
                )
                # If there was a previous reflection (j > -1) and there is a potential
                # reflection at the sample plane, then make sure that a new reflection
                # at j is not found because of numerical error
                if j > -1:
                    if pn[j] == 1:
                        temp: NDArray[float64] = cumsum(pn)
                        indj = temp[j] - 1
                        tt1 = t1[indj]
                        if (
                            array_abs(tt1) < near_zero
                            or array_abs(tt1 - pi) < near_zero
                        ):
                            t1[indj] = inf
                mt: float64 = array_min(t1)
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
        if (F @ X + g).all() > 0:
            Xs[:, i - 1] = X
            last_X = X
            i = i + 1
    if cov:
        Xs = R.T @ Xs + repmat(mu.reshape(mu.shape[0], 1), 1, L)
    else:
        Xs = solve_linalg(R, Xs) + repmat(mu.reshape(mu.shape[0], 1), 1, L)
    return Xs


def _calculate_credible_intervals(args) -> Tuple[Gammas, Gammas, Gammas]:
    from scipy.linalg import inv as invert

    num_RL: int
    num_samples: int
    mu: NDArray[float64]
    Sigma_inv: NDArray[float64]
    x: NDArray[float64]
    tau_fine: TimeConstants
    tau: TimeConstants
    epsilon: float
    rbf_type: str
    (
        num_RL,
        num_samples,
        mu,
        Sigma_inv,
        x,
        tau_fine,
        tau,
        epsilon,
        rbf_type,
    ) = args
    # Calculation of credible interval according to Bayesian statistics
    mu = mu[num_RL:]
    Sigma_inv = Sigma_inv[num_RL:, num_RL:]
    # Cholesky transform instead of direct inverse
    L_Sigma_inv: NDArray[float64] = cholesky(Sigma_inv)
    L_Sigma_agm: NDArray[float64] = invert(L_Sigma_inv)
    Sigma: NDArray[float64] = L_Sigma_agm.T @ L_Sigma_agm
    # Using generate_tmg from HMC_exact.py to sample the truncated Gaussian distribution
    Xs: NDArray[float64] = _generate_truncated_multivariate_gaussians(
        eye(x.shape[0], dtype=float64),
        finfo(float).eps * ones(mu.shape[0], dtype=float64),
        Sigma,
        mu,
        x,
        True,
        num_samples,
    )
    lower_bound: Gammas
    upper_bound: Gammas
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
    assert set(_RBF_TYPES) == set(switch.keys())

    def wrapper(*args, **kwargs):
        kwargs["rbf_functions"] = switch
        return func(*args, **kwargs)

    return wrapper


@_rbf_epsilon_functions
def _compute_epsilon(
    f: Frequencies,
    rbf_shape: str,
    shape_coeff: float,
    rbf_type: str,
    rbf_functions: Dict[str, Callable],
) -> float:
    if rbf_type == "piecewise-linear":
        return 0.0
    elif rbf_shape == "fwhm":
        from scipy.optimize import fsolve

        FWHM_coeff: NDArray[float64] = 2 * fsolve(rbf_functions[rbf_type], 1)
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
    assert set(_RBF_TYPES) == set(switch.keys())

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
    from scipy.integrate import quad

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
    return quad(integrand, -50, 50, epsabs=1e-9, epsrel=1e-9)[0]


def _assemble_A_matrix(args) -> NDArray[float64]:
    f: Frequencies
    tau: TimeConstants
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
    w: NDArray[float64] = 2 * pi * f
    num_freqs: int = f.shape[0]
    num_taus: int = tau.shape[0]
    A: NDArray[float64]
    i: int
    j: int
    # Check if the frequencies are ln-spaced
    if (
        num_freqs == num_taus
        and (std(diff(ln(1 / f))) / mean(diff(ln(1 / f)))) < 0.01
        and rbf_type != "piecewise-linear"
    ):
        # Use the Toeplitz trick
        from scipy.linalg import toeplitz

        C: NDArray[float64] = zeros(num_freqs, dtype=float64)
        for i in range(0, num_freqs):
            C[i] = _A_matrix_element(f[i], tau[0], epsilon, real, rbf_type)
        R: NDArray[float64] = zeros(num_taus, dtype=float64)
        for j in range(0, num_taus):
            R[j] = _A_matrix_element(f[0], tau[j], epsilon, real, rbf_type)
        A = toeplitz(C, R)
    else:
        # Use brute force
        A = zeros(
            (
                num_freqs,
                num_taus,
            ),
            dtype=float64,
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
                ) + 12 * (1 + abs(a)) ** 2 * (
                    16 + abs(a) * (2 + abs(a)) * (12 + abs(a) * (2 + abs(a)))
                ) * ln(
                    1 + abs(a)
                )
                denominator = abs(a) ** 5 * (1 + abs(a)) * (2 + abs(a)) ** 5
                return 8 * epsilon**3 * numerator / denominator
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
        from scipy.integrate import quad

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
                / (delta**2)
                * (rbf_i(y + delta) - 2 * rbf_i(y) + rbf_i(y - delta))
                * 1
                / (delta**2)
                * (rbf_j(y + delta) - 2 * rbf_j(y) + rbf_j(y - delta))
            )
        return quad(sqr_drbf_dy, -50, 50, epsabs=1e-9, epsrel=1e-9)[0]
    assert rbf_type not in _RBF_TYPES, f"Unsupported RBF type: {rbf_type}"
    return -1.0  # Just to satisfy mypy


def _assemble_M_matrix(
    tau: TimeConstants,
    epsilon: float,
    derivative_order: int,
    rbf_type: str,
) -> NDArray[float64]:
    f: Frequencies = 1 / tau
    num_freqs: int = f.shape[0]
    num_taus: int = tau.shape[0]
    M: NDArray[float64]
    i: int
    j: int
    # Check if the collocation points are sufficiently ln-spaced
    if (
        std(diff(ln(tau))) / mean(diff(ln(tau)))
    ) < 0.01 and rbf_type != "piecewise-linear":
        # Apply the Toeplitz trick to compute the M matrix
        from scipy.linalg import toeplitz

        C: NDArray[float64] = zeros(num_taus, dtype=float64)
        for i in range(0, num_taus):
            C[i] = _inner_product_rbf(
                f[0],
                f[i],
                epsilon,
                derivative_order,
                rbf_type,
            )  # TODO: Maybe use tau instead of freq (pyDRTtools comment)
        R: NDArray[float64] = zeros(num_taus, dtype=float64)
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
                ),
                dtype=float64,
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
                ),
                dtype=float64,
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
            ),
            dtype=float64,
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
    A: NDArray[float64],
    b: NDArray[float64],
    M: NDArray[float64],
    lambda_value: float,
) -> Tuple[NDArray[float64], NDArray[float64]]:
    H: NDArray[float64] = 2 * (A.T @ A + lambda_value * M)
    H = (H.T + H) / 2
    c: NDArray[float64] = -2 * b.T @ A
    return (
        H,
        c,
    )


def _quad_format_combined(
    A_re: NDArray[float64],
    A_im: NDArray[float64],
    b_re: NDArray[float64],
    b_im: NDArray[float64],
    M: NDArray[float64],
    lambda_value: float,
) -> Tuple[NDArray[float64], NDArray[float64]]:
    H: NDArray[float64] = 2 * ((A_re.T @ A_re + A_im.T @ A_im) + lambda_value * M)
    H = (H.T + H) / 2
    c: NDArray[float64] = -2 * (b_im.T @ A_im + b_re.T @ A_re)
    return (
        H,
        c,
    )


def _solve_qp_cvxpy(
    H: NDArray[float64],
    c: NDArray[float64],
) -> NDArray[float64]:
    from cvxpy import (
        Minimize,
        Problem,
        Variable,
        quad_form,
    )

    N_out: int = c.shape[0]
    x: Variable = Variable(shape=N_out, value=ones(N_out, dtype=float64))
    l: NDArray[float64] = zeros(N_out, dtype=float64)
    prob: Problem = Problem(Minimize((1 / 2) * quad_form(x, H) + c @ x), [x >= l])
    prob.solve(
        # verbose=True,
        eps_abs=1e-10,
        eps_rel=1e-10,
        sigma=1.00e-08,
        max_iter=200000,
        eps_prim_inf=1e-5,
        eps_dual_inf=1e-5,
    )
    return x.value


def _solve_qp_cvxopt(
    H: NDArray[float64],
    c: NDArray[float64],
    G: Optional[NDArray[float64]] = None,
    h: Optional[NDArray[float64]] = None,
    A: Optional[NDArray[float64]] = None,
    b: Optional[NDArray[float64]] = None,
) -> NDArray[float64]:
    try:
        from kvxopt import (
            matrix,
            solvers,
        )
    except ImportError:
        from cvxopt import (
            matrix,
            solvers,
        )
    args: List[matrix] = [matrix(H), matrix(c)]
    if G is not None:
        assert h is not None
        args.extend([matrix(G), matrix(h)])
    if A is not None:
        assert b is not None
        args.extend([matrix(A), matrix(b)])
    solvers.options["abstol"] = 1e-15
    solvers.options["reltol"] = 1e-15
    solution: dict = solvers.qp(*args)
    if "optimal" not in solution["status"]:
        raise DRTError("Failed to find optimal solution!")
    return array(solution["x"]).reshape((H.shape[1],))


def _solve_qp(
    H: NDArray[float64],
    c: NDArray[float64],
    G: Optional[NDArray[float64]] = None,
    h: Optional[NDArray[float64]] = None,
    A: Optional[NDArray[float64]] = None,
    b: Optional[NDArray[float64]] = None,
) -> NDArray[float64]:
    try:
        return _solve_qp_cvxpy(H, c)
    except Exception:
        return _solve_qp_cvxopt(H, c, G, h, A, b)


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
    assert set(_RBF_TYPES) == set(switch.keys())

    def wrapper(*args, **kwargs):
        kwargs["rbf_functions"] = switch
        return func(*args, **kwargs)

    return wrapper


@_rbf_gamma_functions
def _x_to_gamma(
    x: NDArray[float64],
    tau_fine: TimeConstants,
    tau: TimeConstants,
    epsilon: float,
    rbf_type: str,
    rbf_functions: Dict[str, Callable],
) -> Tuple[TimeConstants, Gammas]:
    # TODO: double check this to see if the function is correct (pyDRTtools comment)
    if rbf_type == "piecewise-linear":
        return (
            tau,
            x,
        )
    num_taus: int = tau.shape[0]
    num_fine_taus: int = tau_fine.shape[0]
    B: NDArray[float64] = zeros(
        (
            num_fine_taus,
            num_taus,
        ),
        dtype=float64,
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
    A_re: NDArray[float64],
    A_im: NDArray[float64],
    b_re: NDArray[float64],
    b_im: NDArray[float64],
    M: NDArray[float64],
    lambda_value: float,
    f: Frequencies,
    num_freqs: int,
    num_taus: int,
    inductance: bool,
) -> Tuple[
    NDArray[float64],
    NDArray[float64],
    NDArray[float64],
    NDArray[float64],
    NDArray[float64],
    int,
]:
    num_RL: int = 2 if inductance is True else 1
    tmp: NDArray[float64]  # Used for temporary binding of matrices
    tmp = A_re
    A_re = zeros(
        (
            num_freqs,
            num_taus + num_RL,
        ),
        dtype=float64,
    )
    A_re[:, num_RL:] = tmp
    A_re[:, 1 if inductance is True else 0] = 1

    tmp = A_im
    A_im = zeros(
        (
            num_freqs,
            num_taus + num_RL,
        ),
        dtype=float64,
    )
    A_im[:, num_RL:] = tmp
    if inductance is True:
        A_im[:, 0] = 2 * pi * f

    tmp = M
    M = zeros(
        (
            num_taus + num_RL,
            num_taus + num_RL,
        ),
        dtype=float64,
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
    A_re: NDArray[float64],
    A_im: NDArray[float64],
    b_re: NDArray[float64],
    b_im: NDArray[float64],
    M: NDArray[float64],
    lambda_value: float,
    num_freqs: int,
    num_taus: int,
) -> Tuple[
    NDArray[float64],
    NDArray[float64],
    NDArray[float64],
    NDArray[float64],
    NDArray[float64],
    int,
]:
    num_RL: int = 1
    tmp: NDArray[float64]  # Used for temporary binding of matrices
    tmp = A_re
    A_re = zeros(
        (
            num_freqs,
            num_taus + num_RL,
        ),
        dtype=float64,
    )
    A_re[:, num_RL:] = tmp
    A_re[:, 0] = 1

    tmp = A_im
    A_im = zeros(
        (
            num_freqs,
            num_taus + num_RL,
        ),
        dtype=float64,
    )
    A_im[:, num_RL:] = tmp

    tmp = M
    M = zeros(
        (
            num_taus + num_RL,
            num_taus + num_RL,
        ),
        dtype=float64,
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
    A_re: NDArray[float64],
    A_im: NDArray[float64],
    b_re: NDArray[float64],
    b_im: NDArray[float64],
    M: NDArray[float64],
    lambda_value: float,
    f: Frequencies,
    num_freqs: int,
    num_taus: int,
    inductance: bool,
) -> Tuple[
    NDArray[float64],
    NDArray[float64],
    NDArray[float64],
    NDArray[float64],
    NDArray[float64],
    int,
]:
    num_RL: int = 1 if inductance is True else 0
    tmp: NDArray[float64]  # Used for temporary binding of matrices
    tmp = A_re
    A_re = zeros(
        (
            num_freqs,
            num_taus + num_RL,
        ),
        dtype=float64,
    )
    A_re[:, num_RL:] = tmp

    tmp = A_im
    A_im = zeros(
        (
            num_freqs,
            num_taus + num_RL,
        ),
        dtype=float64,
    )
    A_im[:, num_RL:] = tmp
    if inductance is True:
        A_im[:, 0] = 2 * pi * f
    tmp = M
    M = zeros(
        (
            num_taus + num_RL,
            num_taus + num_RL,
        ),
        dtype=float64,
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
        NDArray[float64],
        ComplexImpedances,
        NDArray[float64],
        NDArray[float64],
        NDArray[float64],
        NDArray[float64],
        NDArray[float64],
        int,
    ]
]:
    A_re: NDArray[float64]
    A_im: NDArray[float64]
    Z: ComplexImpedances
    M: NDArray[float64]
    lambda_value: float
    f: Frequencies
    tau: TimeConstants
    tau_fine: TimeConstants
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
    H: NDArray[float64]
    c: NDArray[float64]
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
        x: NDArray[float64] = _solve_qp(H, c)
    except Exception:
        return None
    gamma: Gammas
    _, gamma = _x_to_gamma(x[num_RL:], tau_fine, tau, epsilon, rbf_type)
    min_gamma: float = abs(min(gamma))
    max_gamma: float = abs(max(gamma))
    score: float = 1.0 - ((max_gamma - min_gamma) / max(min_gamma, max_gamma))
    if score > maximum_symmetry:
        return None
    Z_fit: ComplexImpedances = array(
        list(map(lambda _: complex(_[0], _[1]), zip(A_re @ x, A_im @ x))),
        dtype=ComplexImpedance,
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


def _suggest_lambda(
    lambda_values: NDArray[float64],
    solution_norms: NDArray[float64],
) -> float:
    a: NDArray[float64] = zeros(lambda_values.size - 1, dtype=float64)
    for i in range(0, lambda_values.size - 1):
        a[i] = solution_norms[i] - solution_norms[i + 1]
    b: NDArray[float64] = zeros(a.size - 1, dtype=float64)
    for i in range(0, b.size - 1):
        b[i] = a[i] - a[i + 1]
    c: float
    for i, c in reversed(list(enumerate(b))):
        if c < 0.0:
            return lambda_values[(i + 1) if i < lambda_values.size - 2 else i]
    return lambda_values[-1]


def _attempt_importing_solver():
    try:
        import cvxpy
    except ImportError:
        try:
            import kvxopt
        except ImportError:
            import cvxopt


def _l_curve_P(
    lambda_value: float,
    A_re: NDArray[float64],
    A_im: NDArray[float64],
    Z_exp: ComplexImpedances,
    M: NDArray[float64],
    f: Frequencies,
    tau: TimeConstants,
    tau_fine: TimeConstants,
    epsilon: float,
    mode: str,
    rbf_type: str,
    inductance: bool,
    maximum_symmetry: float,
) -> Tuple[float64, float64]:
    result: Optional[
        Tuple[
            float,
            float,
            NDArray[float64],
            ComplexImpedances,
            NDArray[float64],
            NDArray[float64],
            NDArray[float64],
            NDArray[float64],
            NDArray[float64],
            int,
        ]
    ] = _lambda_process(
        (
            A_re,
            A_im,
            Z_exp,
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
        )
    )
    if result is None:
        return (inf, inf)
    assert result is not None
    return (
        log(norm(result[3] - Z_exp)**2),
        log(result[1] ** 2),
    )


def calculate_drt_tr_rbf(
    data: DataSet,
    mode: str = "complex",
    lambda_value: float = -1.0,
    rbf_type: str = "gaussian",
    derivative_order: int = 1,
    rbf_shape: str = "fwhm",
    shape_coeff: float = 0.5,
    inductance: bool = False,
    credible_intervals: bool = False,
    num_samples: int = 10000,
    maximum_symmetry: float = 0.3,
    timeout: int = 60,
    num_procs: int = 0,
    **kwargs,
) -> TRRBFResult:
    """
    Calculates the distribution of relaxation times (DRT) for a given data set using Tikhonov regularization and radial basis (or piecewise linear) discretization (TR-RBF method).

    References:

    - Wan, T. H., Saccoccio, M., Chen, C., and Ciucci, F., 2015, Electrochim. Acta, 184, 483-499 (https://doi.org/10.1016/j.electacta.2015.09.097)
    - Ciucci, F. and Chen, C., 2015, Electrochim. Acta, 167, 439-454 (https://doi.org/10.1016/j.electacta.2015.03.123)
    - Effat, M. B. and Ciucci, F., 2017, Electrochim. Acta, 247, 1117-1129 (https://doi.org/10.1016/j.electacta.2017.07.050)

    Parameters
    ----------
    data: DataSet
        The data set to use in the calculations.

    mode: str, optional
        Which parts of the data are to be included in the calculations.
        Valid values include:

        - "complex"
        - "real"
        - "imaginary"

    lambda_value: float, optional
        The Tikhonov regularization parameter.
        If the value is equal to or less than zero, then an attempt will be made to automatically find a suitable value.
        If the value is between -1.5 and 0.0, then a custom approach is used.
        If the value is less than -1.5, then the L-curve corner search algorithm (DOI:10.1088/2633-1357/abad0d) is used.

    rbf_type: str, optional
        The type of function to use for discretization.
        Valid values include:

        - "gaussian"
        - "c0-matern"
        - "c2-matern"
        - "c4-matern"
        - "c6-matern"
        - "inverse-quadratic"
        - "inverse-quadric"
        - "cauchy"
        - "piecewise-linear"

    derivative_order: int, optional
        The order of the derivative used during discretization.

    rbf_shape: str, optional
        The shape control of the radial basis functions.
        Valid values include:

        - "fwhm": full width half maximum
        - "factor": `shape_coeff` is used directly

    shape_coeff: float, optional
        The full width at half maximum (FWHM) coefficient affecting the chosen shape type.

    inductance: bool, optional
        If true, then an inductive element is included in the calculations.

    credible_intervals: bool, optional
        If true, then the credible intervals are also calculated for the DRT results according to Bayesian statistics.

    num_samples: int, optional
        The number of samples drawn when calculating the Bayesian credible intervals.
        A greater number provides better accuracy but requires more time.

    maximum_symmetry: float, optional
        A maximum limit (between 0.0 and 1.0) for the relative vertical symmetry of the DRT.
        A high degree of symmetry is common for results where the gamma value oscillates wildly (e.g., due to a small regularization parameter).
        The TR-RBF method only uses this limit when the regularization parameter (lambda) is not provided.

    timeout: int, optional
        The number of seconds to wait for the calculation of credible intervals to complete.

    num_procs: int, optional
        The maximum number of parallel processes to use.
        A value less than 1 results in an attempt to figure out a suitable value based on, e.g., the number of cores detected.
        Additionally, a negative value can be used to reduce the number of processes by that much (e.g., to leave one core for a GUI thread).

    Returns
    -------
    TRRBFResult
    """
    from scipy.linalg import solve as solve_linalg

    global _SOLVER_IMPORTED
    if _SOLVER_IMPORTED is False:
        try:
            _attempt_importing_solver()
        except ImportError:
            raise DRTError("Failed to import any of the supported convex optimizers!")
        else:
            _SOLVER_IMPORTED = True

    assert hasattr(data, "get_frequencies") and callable(data.get_frequencies)
    assert hasattr(data, "get_impedances") and callable(data.get_impedances)
    assert isinstance(mode, str), mode
    if mode not in _MODES:
        raise DRTError("Valid mode values: '" + "', '".join(_MODES))
    assert issubdtype(type(lambda_value), floating), lambda_value
    assert type(rbf_type) is str, rbf_type
    if rbf_type not in _RBF_TYPES:
        raise DRTError("Valid rbf_type values: '" + "', '".join(_RBF_TYPES))
    assert issubdtype(type(derivative_order), integer), derivative_order
    if not (1 <= derivative_order <= 2):
        raise DRTError("Valid derivative_order values: 1, 2")
    assert type(rbf_shape) is str, rbf_shape
    if rbf_shape not in _RBF_SHAPES:
        raise DRTError("Valid rbf_shape values: '" + "', '".join(_RBF_SHAPES))
    assert issubdtype(type(shape_coeff), floating), shape_coeff
    if shape_coeff <= 0.0:
        raise DRTError("The shape coefficient must be greater than 0.0!")
    assert isinstance(inductance, bool), inductance
    assert isinstance(credible_intervals, bool), credible_intervals
    assert issubdtype(type(num_samples), integer), num_samples
    if credible_intervals is True and num_samples < 1000:
        raise DRTError(f"{num_samples} is not enough samples!")
    assert issubdtype(type(maximum_symmetry), floating), maximum_symmetry
    if not (0.0 <= maximum_symmetry <= 1.0):
        raise DRTError("The maximum symmetry must be between 0.0 and 1.0 (inclusive)!")
    assert issubdtype(type(timeout), integer), timeout
    if credible_intervals is True and timeout < 1:
        raise DRTError("The timeout must be greater than 0!")
    assert issubdtype(type(num_procs), integer), num_procs
    if num_procs < 1:
        num_procs = _get_default_num_procs() - abs(num_procs)
        if num_procs < 1:
            num_procs = 1
    # TODO: Switch over to using the new cross-validation-based method(s)?
    min_log_lambda: float = -7.0
    max_log_lambda: float = 0.0
    lambda_values: NDArray[float64] = (
        logspace(
            min_log_lambda,
            max_log_lambda,
            num=round(max_log_lambda - min_log_lambda) + 1,
        )
        if -1.5 <= lambda_value <= 0.0
        else array([lambda_value])
    )
    # TODO: Figure out if f and Z need to be altered depending on the value
    # of the 'inductance' argument!
    f: Frequencies = data.get_frequencies()
    Z_exp: ComplexImpedances = data.get_impedances()
    tau: TimeConstants = 1 / f
    tau_fine: TimeConstants = logspace(
        log(tau.min()) - 0.5, log(tau.max()) + 0.5, 10 * f.shape[0]
    )
    num_freqs: int = f.size
    epsilon: float = _compute_epsilon(f, rbf_shape, shape_coeff, rbf_type)
    num_steps: int = 0
    num_steps += 3  # A_re, A_im, and M matrices
    num_steps += len(lambda_values)
    if credible_intervals is True:
        num_steps += timeout
    prog: Progress
    with Progress("Preparing matrices", total=num_steps + 1) as prog:
        A_re: NDArray[float64]
        A_im: NDArray[float64]
        i: int
        args = [
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
        ]
        if num_procs > 1:
            with Pool(2) as pool:
                for i, res in enumerate(pool.imap(_assemble_A_matrix, args)):
                    if i == 0:
                        A_re = res
                    else:
                        A_im = res
                    prog.increment()
        else:
            A_re = _assemble_A_matrix(args[0])
            prog.increment()
            A_im = _assemble_A_matrix(args[1])
            prog.increment()
        M: NDArray[float64] = _assemble_M_matrix(
            tau,
            epsilon,
            derivative_order,
            rbf_type,
        )
        prog.increment()
        if len(lambda_values) == 1 and lambda_values[0] < -1.5:
            lambda_values[0] = _l_curve_corner_search(
                lambda _: _l_curve_P(
                    _,
                    A_re,
                    A_im,
                    Z_exp,
                    M,
                    f,
                    tau,
                    tau_fine,
                    epsilon,
                    mode,
                    rbf_type,
                    inductance,
                    maximum_symmetry,
                ),
                minimum=1e-10,
                maximum=1,
            )
        args = (
            (
                A_re,
                A_im,
                Z_exp,
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
        prog.set_message("Calculating DRT")
        results: List[
            Tuple[
                float,
                float,
                NDArray[float64],
                ComplexImpedances,
                NDArray[float64],
                NDArray[float64],
                NDArray[float64],
                NDArray[float64],
                NDArray[float64],
                int,
            ]
        ] = []
        if len(lambda_values) > 1 and num_procs > 1:
            with Pool(num_procs) as pool:
                for res in pool.imap_unordered(_lambda_process, args):
                    if res is not None:
                        results.append(res)
                    prog.increment()
        else:
            for res in map(_lambda_process, args):
                if res is not None:
                    results.append(res)
                prog.increment()
    if len(results) == 0:
        raise DRTError("Failed to perform calculations! Try tweaking the settings.")
    if len(results) > 1:
        results.sort(key=lambda _: _[0])
        lambda_value = _suggest_lambda(
            array(list(map(lambda _: _[0], results)), dtype=float64),
            array(list(map(lambda _: _[1], results)), dtype=float64),
        )
        results = list(filter(lambda _: _[0] == lambda_value, results))
    lambda_value, _, x, Z_fit, A_re, A_im, M, H, c, num_RL = results[0]
    sigma_re_im: float
    if mode == "complex":
        sigma_re_im = std(
            concatenate([Z_fit.real - Z_exp.real, Z_fit.imag - Z_exp.imag])
        )
    elif mode == "real":
        sigma_re_im = std(Z_fit.real - Z_exp.real)
    elif mode == "imaginary":
        sigma_re_im = std(Z_fit.imag - Z_exp.imag)
    inv_V: NDArray[float64] = 1 / sigma_re_im**2 * eye(num_freqs)
    Sigma_inv: NDArray[float64]
    mu_numerator: NDArray[float64]
    if mode == "complex":
        Sigma_inv = (
            (A_re.T @ inv_V @ A_re)
            + (A_im.T @ inv_V @ A_im)
            + (lambda_value / sigma_re_im**2) * M
        )
        mu_numerator = A_re.T @ inv_V @ Z_exp.real + A_im.T @ inv_V @ Z_exp.imag
    elif mode == "real":
        Sigma_inv = (A_re.T @ inv_V @ A_re) + (lambda_value / sigma_re_im**2) * M
        mu_numerator = A_re.T @ inv_V @ Z_exp.real
    elif mode == "imaginary":
        Sigma_inv = (A_im.T @ inv_V @ A_im) + (lambda_value / sigma_re_im**2) * M
        mu_numerator = A_im.T @ inv_V @ Z_exp.imag
    Sigma_inv = (Sigma_inv + Sigma_inv.T) / 2
    L_Sigma_inv: NDArray[float64] = cholesky(Sigma_inv)
    mu: NDArray[float64] = solve_linalg(
        L_Sigma_inv.T, solve_linalg(L_Sigma_inv, mu_numerator)
    )
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
    time_constants: TimeConstants
    time_constants, gamma = _x_to_gamma(x, tau_fine, tau, epsilon, rbf_type)
    if credible_intervals is True:
        prog.set_message("Calculating credible intervals")
        args = (
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
        pool = Pool(1)
        async_result = pool.map_async(_calculate_credible_intervals, (args,))
        while timeout > 0:
            if async_result.ready():
                prog.increment(step=timeout)
                break
            sleep(1.0)
            prog.increment()
            timeout -= 1
        try:
            (mean_gamma, lower_gamma, upper_gamma,) = async_result.get(
                timeout=2
            )[0]
        except MPTimeoutError:
            pool.close()
            raise DRTError(
                "Timed out while calculating credible intervals! Adjust the timeout limit and try again."
            )
        pool.close()
    else:
        mean_gamma, lower_gamma, upper_gamma = (
            array([]),  # Mean
            array([]),  # Lower bound
            array([]),  # Upper bound
        )
    return TRRBFResult(
        time_constants=time_constants,
        gammas=gamma,
        frequencies=f,
        impedances=Z_fit,
        residuals=_calculate_residuals(Z_exp, Z_fit),
        mean_gammas=mean_gamma,
        lower_bounds=lower_gamma,
        upper_bounds=upper_gamma,
        pseudo_chisqr=_calculate_pseudo_chisqr(Z_exp, Z_fit),
        lambda_value=lambda_value,
    )
