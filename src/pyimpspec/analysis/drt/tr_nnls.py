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

# This module uses Tikhonov regularization and non-negative least squares
# 10.1149/1945-7111/abf508
# Based on code from https://github.com/akulikovsky/DRT-python-code.
# DRT-python-code commit: 9663ed8b331f521a9fcdb0b58fb2b34693df938c

from dataclasses import dataclass
from numpy import (
    array,
    float64,
    fromiter,
    identity,
    int64,
    log as ln,
    log10 as log,
    logspace,
    pi,
    polyfit,
    sqrt,
    sum as array_sum,
    zeros,
)
from numpy.linalg import norm
from numpy.typing import NDArray
from pyimpspec.data import DataSet
from pyimpspec.analysis.utility import (
    _calculate_residuals,
    _calculate_pseudo_chisqr,
)
from pyimpspec.analysis.drt.result import DRTResult
from pyimpspec.progress import Progress
from pyimpspec.typing import (
    ComplexImpedance,
    ComplexImpedances,
    Frequencies,
    Gammas,
    Indices,
    TimeConstants,
)
from pyimpspec.typing.helpers import (
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    _is_floating,
    _is_integer,
)
from .utility import _l_curve_corner_search


@dataclass(frozen=True)
class TRNNLSResult(DRTResult):
    """
    An object representing the results of calculating the distribution of relaxation times in a data set using Tikhonov regularization and non-negative least squares fitting (TR-NNLS).

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
        The residuals of the real and imaginary parts of the model and the data set.

    pseudo_chisqr: float
        The pseudo chi-squared value, |pseudo chi-squared|, of the modeled impedance (eq. 14 in Boukamp, 1995).

    lambda_value: float
        The lambda value that was used.
    """

    gammas: Gammas
    lambda_value: float

    def get_label(self) -> str:
        return "TR-NNLS"

    def get_gammas(self) -> Gammas:
        """
        Get the gamma values.

        Returns
        -------
        |Gammas|
        """
        return self.gammas

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
        elif not isinstance(columns, list):
            raise TypeError(f"Expected a list of strings instead of {columns=}")
        elif len(columns) != 2:
            raise ValueError(f"Expected a list with 2 items instead of {len(columns)=}")
        elif not all(map(lambda s: isinstance(s, str), columns)):
            raise TypeError(f"Expected a list of strings instead of {columns=}")
        elif len(set(columns)) != 2:
            raise ValueError(
                f"Expected a list of 2 unique strings instead of {columns=}"
            )

        time_constants: TimeConstants
        gammas: Gammas
        time_constants, gammas = self.get_peaks(threshold=threshold)

        return DataFrame.from_dict(
            {
                columns[0]: time_constants,  # type: ignore
                columns[1]: gammas,  # type: ignore
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


_MODES: List[str] = ["real", "imaginary"]


def _calculate_delta_ln_tau(tau: TimeConstants) -> NDArray[float64]:
    ln_tau: NDArray[float64] = ln(tau)
    delta_ln_tau: NDArray[float64] = zeros(tau.size, dtype=float64)

    i: int
    for i in range(1, tau.size - 1):
        delta_ln_tau[i] = 0.5 * (ln_tau[i + 1] - ln_tau[i - 1])

    delta_ln_tau[0] = 0.5 * (ln_tau[1] - ln_tau[0])
    delta_ln_tau[-1] = 0.5 * (ln_tau[-1] - ln_tau[-2])

    return delta_ln_tau


def _normalize_impedance(
    Z: ComplexImpedances,
) -> Tuple[ComplexImpedances, float, float]:
    R_inf: float = Z[0].real  # High-frequency resistance
    Z_norm: ComplexImpedances = Z - R_inf

    R_pol: float = Z_norm[-1].real - Z_norm[0].real
    Z_norm /= R_pol

    return (
        Z_norm,
        R_inf,
        R_pol,
    )


def _generate_A_matrix(
    omega: NDArray[float64],
    tau: TimeConstants,
    delta_ln_tau: NDArray[float64],
    is_imaginary: bool,
) -> NDArray[float64]:
    A: NDArray[float64] = zeros(
        (
            omega.size,
            omega.size,
        ),
        dtype=float64,
    )

    product: NDArray[float64]
    for i in range(0, omega.size):
        product = omega[i] * tau
        A[i, :] = (product if is_imaginary else 1) * delta_ln_tau / (1 + product**2)

    return A


def _generate_b_vector(
    A: NDArray[float64],
    Z_norm: ComplexImpedances,
    is_imaginary: bool,
) -> NDArray[float64]:
    return A.T @ (-Z_norm.imag if is_imaginary else Z_norm.real)


def _generate_tikhonov_matrix(
    A: NDArray[float64],
    I: NDArray[float64],
    lambda_value: float,
) -> NDArray[float64]:
    return (A.T @ A) + lambda_value * I


def _solve(
    A: NDArray[float64],
    b: NDArray[float64],
    maxiter: Optional[int],
) -> NDArray[float64]:
    from scipy.optimize import nnls

    return nnls(A, b, maxiter=maxiter)[0]


def _generate_lambda_values(
    log_minimum: int,
    log_maximum: int,
    num_per_decade: int,
) -> NDArray[float64]:
    return logspace(
        log_minimum,
        log_maximum,
        num=(log_maximum - log_minimum) * num_per_decade + 1,
    )


def _test_lambda_values(
    A: NDArray[float64],
    b: NDArray[float64],
    I: NDArray[float64],
    lambda_values: NDArray[float64],
    maxiter: Optional[int],
) -> Tuple[NDArray[float64], NDArray[float64]]:
    prog: Progress
    with Progress(
        "Testing different lambda values",
        total=len(lambda_values) + 1,
    ) as prog:
        solution_norms: NDArray[float64] = zeros(lambda_values.size, dtype=float64)

        i: int
        for i, lambda_value in enumerate(lambda_values):
            A_tikh: NDArray[float64] = _generate_tikhonov_matrix(A, I, lambda_value)
            g_tau: NDArray[float64] = _solve(A_tikh, b, maxiter)
            solution_norms[i] = sqrt(array_sum(g_tau**2))
            prog.increment()

    return (
        lambda_values,
        solution_norms,
    )


def _reduce_points_by_radius(
    x: NDArray[float64],
    y: NDArray[float64],
    r: float,
) -> Tuple[NDArray[float64], NDArray[float64]]:
    raw_indices: List[int] = [0]

    i: int = 0
    while i < x.size - 1:
        i += 1
        j = raw_indices[-1]
        if ((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2) ** (1 / 2) < r:
            continue
        raw_indices.append(i)

    indices: Indices = array(raw_indices, dtype=int64)

    return (
        x[indices],
        y[indices],
    )


def _suggest_lambda(
    lambda_values: NDArray[float64],
    solution_norms: NDArray[float64],
) -> float:
    lambda_values, solution_norms = _reduce_points_by_radius(
        lambda_values,
        solution_norms,
        r=2e-2,
    )

    n: int = 5
    m1: float
    c1: float
    (m1, c1) = polyfit(lambda_values[:n], solution_norms[:n], deg=1)

    m2: float
    c2: float
    (m2, c2) = polyfit(lambda_values[-n:], solution_norms[-n:], deg=1)

    return (c2 - c1) / (m1 - m2)


def _generate_model_impedance(
    omega: NDArray[float64],
    tau: TimeConstants,
    delta_ln_tau: NDArray[float64],
    A_tikh: NDArray[float64],
    g_tau: NDArray[float64],
    Z: ComplexImpedances,
    R_inf: float,
    R_pol: float,
    is_imaginary: bool,
) -> ComplexImpedances:
    Z_re_im: NDArray[float64] = zeros(omega.size, dtype=float64)

    i: int
    for i in range(0, omega.size):
        product = omega[i] * tau
        Z_re_im[i] = array_sum(
            delta_ln_tau
            * ((product if is_imaginary else 1) * g_tau / (1 + product**2))
        )

    Z_re_im = R_pol * Z_re_im

    if is_imaginary:
        return fromiter(
            map(lambda _: complex(*_), zip(Z.real, -Z_re_im)),
            dtype=ComplexImpedance,
            count=len(Z),
        )

    return fromiter(
        map(lambda _: complex(*_), zip(Z_re_im + R_inf, Z.imag)),
        dtype=ComplexImpedance,
        count=len(Z),
    )


def _l_curve_P(
    lambda_value: float,
    A: NDArray[float64],
    b: NDArray[float64],
    I: NDArray[float64],
    maxiter: Optional[int],
) -> Tuple[float64, float64]:
    A_tikh: NDArray[float64] = _generate_tikhonov_matrix(A, I, lambda_value)
    g_tau: NDArray[float64] = _solve(A_tikh, b, maxiter)

    return (
        log(norm(A_tikh @ g_tau - b) ** 2),
        log(norm(g_tau) ** 2),
    )


def calculate_drt_tr_nnls(
    data: DataSet,
    mode: str = "real",
    lambda_value: float = -1.0,
    max_iter: int = -1,
    **kwargs,
) -> TRNNLSResult:
    """
    Calculates the distribution of relaxation times (DRT) for a given data set using Tikhonov regularization and non-negative least squares fitting (TR-NNLS method).

    References:

    - `Kulikovsky, A., 2021, J. Electrochem. Soc., 168, 044512 <https://doi.org/10.1149/1945-7111/abf508>`_

    Parameters
    ----------
    data: DataSet
        The data set to use in the calculations.

    mode: str, optional
        Which parts of the data are to be included in the calculations.
        Valid values include:
        - "real"
        - "imaginary"

    lambda_value: float, optional
        The Tikhonov regularization parameter.
        If the value is equal to or less than zero, then an attempt will be made to automatically find a suitable value.
        If the value is between -1.5 and 0.0, then a custom approach is used.
        If the value is less than -1.5, then the L-curve corner search algorithm (DOI:10.1088/2633-1357/abad0d) is used.

    max_iter: int, optional
        The maximum number of iterations.
        If set to less than one, then the default of `scipy.optimize.nnls`_ is used.

    Returns
    -------
    |TRNNLSResult|
    """
    if not isinstance(mode, str):
        raise TypeError(f"Expected a string instead of {mode=}")
    elif mode not in _MODES:
        raise ValueError("Valid mode values: '" + "', '".join(_MODES))

    if not _is_floating(lambda_value):
        raise TypeError(f"Expected a float instead of {lambda_value=}")

    if not _is_integer(max_iter):
        raise TypeError("Expected an integer instead of {max_iter=}")

    maxiter: Optional[int] = max_iter if max_iter > 0 else None

    prog: Progress
    with Progress("Preparing matrices", total=6) as prog:
        is_imaginary: bool = mode == "imaginary"
        f: Frequencies = data.get_frequencies()
        if len(f) < 1:
            raise ValueError(
                f"There are no unmasked data points in the '{data.get_label()}' data set parsed from '{data.get_path()}'"
            )

        Z_exp: ComplexImpedances = data.get_impedances()
        omega: NDArray[float64] = 2 * pi * f
        tau: TimeConstants = 1 / omega
        delta_ln_tau: NDArray[float64] = _calculate_delta_ln_tau(tau)
        prog.increment()

        Z_norm: ComplexImpedances
        R_inf: float
        R_pol: float
        Z_norm, R_inf, R_pol = _normalize_impedance(Z_exp)
        # Prepare matrices and vectors
        I: NDArray[float64] = identity(omega.size, dtype=int64)
        A: NDArray[float64] = _generate_A_matrix(omega, tau, delta_ln_tau, is_imaginary)
        prog.increment()

        b: NDArray[float64] = _generate_b_vector(A, Z_norm, is_imaginary)
        prog.increment()

        A_tikh: NDArray[float64]
        g_tau: NDArray[float64]
        # Try to determine a suitable regularization parameter if one hasn't
        # been provided.
        if lambda_value < -1.5:
            lambda_value = _l_curve_corner_search(
                lambda _: _l_curve_P(_, A, b, I, maxiter),
                minimum=1e-10,
                maximum=1,
            )

        elif lambda_value <= 0.0:
            lambda_value = _suggest_lambda(
                *_test_lambda_values(
                    A,
                    b,
                    I,
                    _generate_lambda_values(
                        log_minimum=-7,
                        log_maximum=0,
                        num_per_decade=10,
                    ),
                    maxiter,
                ),
            )

        prog.set_message("Calculating DRT")
        A_tikh = _generate_tikhonov_matrix(A, I, lambda_value)
        prog.increment()

        g_tau = _solve(A_tikh, b, maxiter)
        prog.increment()

        # R_pol_synthetic: float = array_sum(g_tau * delta_ln_tau)  # Should be (close to) 1.0
        Z_fit: ComplexImpedances = _generate_model_impedance(
            omega,
            tau,
            delta_ln_tau,
            A_tikh,
            g_tau,
            Z_exp,
            R_inf,
            R_pol,
            is_imaginary,
        )
        gamma: Gammas = g_tau * R_pol

    return TRNNLSResult(
        time_constants=tau,
        gammas=gamma,
        frequencies=f,
        impedances=Z_fit,
        residuals=_calculate_residuals(Z_exp, Z_fit),
        pseudo_chisqr=_calculate_pseudo_chisqr(Z_exp, Z_fit),
        lambda_value=lambda_value,
    )
