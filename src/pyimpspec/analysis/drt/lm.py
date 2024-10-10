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

# This module implements the Loewner method
# 10.1016/j.laa/2007.03.008
# 10.2139/ssrn.4217752
# 10.3390/batteries9020132
# 10.1016/j.jpowsour.2023.233575

# Based on source code from
# https://github.com/projectsEECandDRI/DRT-from-Loewner-framework
# (commit 19a40ea7d53d6d62ea5d7eeb63584818f2e36a7b)
# with some modifications.

from dataclasses import dataclass
from numpy import (
    argwhere,
    array,
    complex128,
    concatenate,
    flip,
    float64,
    full,
    int64,
    log10 as log,
    nan,
    pi,
    sqrt,
    zeros,
)
from numpy.linalg import (
    solve,
    svd,
    matrix_rank,
)
from numpy.typing import NDArray
from pyimpspec.data import DataSet
from pyimpspec.analysis.kramers_kronig.single import (
    KramersKronigResult,
    perform_kramers_kronig_test,
)
from pyimpspec.analysis.utility import (
    _calculate_pseudo_chisqr,
    _calculate_residuals,
    get_default_num_procs,
)
from .result import DRTResult
from .peak_analysis import DRTPeaks
from pyimpspec.progress import Progress
from pyimpspec.typing import (
    ComplexImpedance,
    ComplexImpedances,
    Frequencies,
    Gamma,
    Gammas,
    Indices,
    TimeConstant,
    TimeConstants,
)
from pyimpspec.typing.helpers import (
    Dict,
    List,
    NDArray,
    Optional,
    Tuple,
    Union,
    float64,
    _is_integer,
)


_DEBUG: bool = bool(0)

_MODEL_ORDER_METHODS: List[str] = [
    "matrix_rank",
    "pseudo_chisqr",
]


@dataclass(frozen=True)
class LMResult(DRTResult):
    """
    An object representing the results of calculating the distribution of relaxation times in a data set using the Loewner method.

    Parameters
    ----------
    time_constants: |TimeConstants|
        The time constants.

    gammas: Gammas
        All gamma values. Positive and negative values correspond to resistive-capacitive and resistive-inductive peaks, respectively.

    frequencies: |Frequencies|
        The frequencies of the impedance spectrum.

    impedances: |ComplexImpedances|
        The impedance produced by the model.

    residuals: |ComplexResiduals|
        The residuals of the real parts of the model and the data set.

    pseudo_chisqr: float
        The pseudo chi-squared value, |pseudo chi-squared|, of the modeled impedance (eq. 14 in Boukamp, 1995).

    singular_values: NDArray[float64]
        The singular values as a function of the model order :math:`k \\in [1..n]` where :math:`n` is the closest value divisible by 2 that is less than the total number of frequencies in the |DataSet| that was analyzed.
    """

    gammas: Gammas
    singular_values: NDArray[float64]

    @property
    def _resistive_capacitive_time_constants(self) -> NDArray[float64]:
        indices: NDArray[int64] = argwhere(self.gammas >= 0.0).flatten()

        return self.time_constants[indices]

    @property
    def _resistive_capacitive_gammas(self) -> NDArray[float64]:
        indices: NDArray[int64] = argwhere(self.gammas >= 0.0).flatten()

        return self.gammas[indices]

    @property
    def _resistive_inductive_time_constants(self) -> NDArray[float64]:
        indices: NDArray[int64] = argwhere(self.gammas < 0.0).flatten()

        return self.time_constants[indices]

    @property
    def _resistive_inductive_gammas(self) -> NDArray[float64]:
        indices: NDArray[int64] = argwhere(self.gammas < 0.0).flatten()

        return abs(self.gammas[indices])

    def get_label(self) -> str:
        return f"LM (k={len(self.time_constants)})"

    def get_gammas(self) -> Tuple[Gammas, Gammas]:
        """
        Get the gamma values for the resistive-capacitive and resistive-inductive peaks.

        Returns
        -------
        Tuple[|Gammas|, |Gammas|]
        """
        return (
            self._resistive_capacitive_gammas,
            self._resistive_inductive_gammas,
        )

    def get_drt_data(self) -> Tuple[TimeConstants, Gammas, TimeConstants, Gammas]:
        """
        Get the data necessary to plot this DRT result as a DRT plot: the time constants and corresponding gamma values for the resistive-capacitive and the resistive-inductive peaks.

        Returns
        -------
        Tuple[|TimeConstants|, |Gammas|, |TimeConstants|, |Gammas|]
        """
        return (
            self._resistive_capacitive_time_constants,
            self._resistive_capacitive_gammas,
            self._resistive_inductive_time_constants,
            self._resistive_inductive_gammas,
        )

    def to_peaks_dataframe(
        self,
        threshold: float = 0.0,
        columns: Optional[List[str]] = None,
    ) -> "DataFrame":  # noqa: F821
        from pandas import DataFrame

        if columns is None:
            columns = [
                "tau, RC (s)",
                "gamma, RC (ohm)",
                "tau, RL (s)",
                "gamma, RL (ohm)",
            ]
        elif not isinstance(columns, list):
            raise TypeError(f"Expected a list of strings instead of {columns=}")
        elif len(columns) != 4:
            raise ValueError(f"Expected a list with 4 items instead of {len(columns)=}")
        elif not all(map(lambda s: isinstance(s, str), columns)):
            raise TypeError(f"Expected a list of strings instead of {columns=}")
        elif len(set(columns)) != 4:
            raise ValueError(
                f"Expected a list of 4 unique strings instead of {columns=}"
            )

        def get_peak_indices(threshold, gammas):
            if gammas.size > 0:
                max_gamma = max(gammas)
            else:
                max_gamma = 1.0

            return argwhere((gammas / max_gamma) > threshold).flatten()

        def pad(
            tau: TimeConstants,
            gamma: Gammas,
            width: int,
        ) -> Tuple[TimeConstants, Gammas]:
            tmp_tau: TimeConstants = full(width, nan, dtype=TimeConstant)
            tmp_tau[: tau.size] = tau
            tmp_gamma: Gammas = full(width, nan, dtype=Gamma)
            tmp_gamma[: gamma.size] = gamma
            return (
                tmp_tau,
                tmp_gamma,
            )

        indices_RC: Indices = get_peak_indices(
            threshold,
            self._resistive_capacitive_gammas,
        )
        indices_RL: Indices = get_peak_indices(
            threshold,
            self._resistive_inductive_gammas,
        )

        width: int = max(
            (
                indices_RC.size,
                indices_RL.size,
            )
        )


        tau_RC: TimeConstants
        gamma_RC: Gammas
        tau_RC, gamma_RC = pad(
            tau=self._resistive_capacitive_time_constants[indices_RC],
            gamma=self._resistive_capacitive_gammas[indices_RC],
            width=width,
        )

        tau_RL: TimeConstants
        gamma_RL: Gammas
        tau_RL, gamma_RL = pad(
            tau=self._resistive_inductive_time_constants[indices_RL],
            gamma=self._resistive_inductive_gammas[indices_RL],
            width=width,
        )

        return DataFrame.from_dict(
            {
                columns[0]: tau_RC,
                columns[1]: gamma_RC,
                columns[2]: tau_RL,
                columns[3]: gamma_RL,
            }
        )

    def to_statistics_dataframe(
        self,
    ) -> "DataFrame":  # noqa: F821
        from pandas import DataFrame

        statistics: Dict[str, Union[int, float, str]] = {
            "Log pseudo chi-squared": log(self.pseudo_chisqr),
            "Model order (k)": len(self.time_constants),
        }

        return DataFrame.from_dict(
            {
                "Label": list(statistics.keys()),
                "Value": list(statistics.values()),
            }
        )

    def get_peaks(
        self,
        threshold: float = 0.0,
    ) -> Tuple[TimeConstants, Gammas, TimeConstants, Gammas]:
        """
        Get the time constants (in seconds) of peaks with magnitudes greater than the threshold.
        The threshold and the magnitudes are all relative to the magnitude of the highest peak.

        Parameters
        ----------
        threshold: float, optional
            The minimum peak height threshold (relative to the height of the tallest peak) for a peak to be included.

        Returns
        -------
        Tuple[|TimeConstants|, |Gammas|, |TimeConstants|, |Gammas|]
        """
        def get_peak_indices(threshold: float, gamma: Gammas) -> Indices:
            if len(gamma) == 0:
                return array([], dtype=int64)

            max_g: float = max(gamma)
            if max_g == 0.0:
                return array([], dtype=int64)

            indices: Indices = array(list(range(len(gamma))), dtype=int64)

            return array(
                list(
                    filter(
                        lambda i: gamma[i] / max_g > threshold and gamma[i] > 0.0, indices
                    )
                ),
                dtype=int64,
            )

        indices_RC: Indices = get_peak_indices(threshold, self._resistive_capacitive_gammas)
        indices_RL: Indices = get_peak_indices(threshold, self._resistive_inductive_gammas)

        return (
            self._resistive_capacitive_time_constants[indices_RC],
            self._resistive_capacitive_gammas[indices_RC],
            self._resistive_inductive_time_constants[indices_RL],
            self._resistive_inductive_gammas[indices_RL],
        )

    def analyze_peaks(
        self,
        num_peaks: int = 0,
        peak_positions: Optional[Union[List[float], NDArray[float64]]] = None,
        disallow_skew: bool = False,
    ) -> Tuple[DRTPeaks, DRTPeaks]:
        """
        Analyze the peaks present in a distribution of relaxation times using skew normal distributions.
        
        **Peak analysis of the DRT obtained using the Loewner method is not supported.**

        Parameters
        ----------
        num_peaks: int, optional
            If greater than zero, then analyze only that number of peaks (sorted from highest to lowest gamma values).

        peak_positions: Optional[Union[List[float], NDArray[float64]]], optional
            Analyze only the peaks at the provided positions.

        disallow_skew: bool, optional
            If true, then normal distributions are used instead of skew normal distributions.

        Returns
        -------
        Tuple[|DRTPeaks|, |DRTPeaks|]
            Neither |DRTPeaks| instance actually contains any results corresponding to peaks.
        """
        return (
            DRTPeaks(
                self._resistive_capacitive_time_constants,
                peaks=[],
                suffix="RC",
            ),
            DRTPeaks(
                self._resistive_inductive_time_constants,
                peaks=[],
                suffix="RL",
            ),
        )


def _interleave_complex_conjugate(values: NDArray[complex128]) -> NDArray[complex128]:
    dst: NDArray[complex128] = zeros(len(values) * 2, dtype=complex128)
    dst[0::2] += values.T
    dst[1::2] += values.conj().T

    return dst


def _generate_loewner_matrices(
    s: NDArray[complex128],
    Z: ComplexImpedances,
) -> Tuple[
        NDArray[float64],
        NDArray[float64],
        NDArray[float64],
        NDArray[float64],
]:
    from scipy.linalg import block_diag

    # Establish the number of points to use
    n: int = len(s)
    while n % 2 != 0:
        n -= 1

    # Generate the vectors of values for the two disjointed sets
    la: NDArray[complex128] = _interleave_complex_conjugate(s[0::2][:n // 2])
    W: NDArray[complex128] = _interleave_complex_conjugate(Z[0::2][:n // 2])
    mu: NDArray[complex128] = _interleave_complex_conjugate(s[1::2][:n // 2])
    V: NDArray[complex128] = _interleave_complex_conjugate(Z[1::2][:n // 2])

    # Generate the Loewner and shifted Loewner matrices
    L: NDArray[complex128] = zeros((n, n), dtype=complex128)
    Ls: NDArray[complex128] = zeros((n, n), dtype=complex128)

    i: int
    for i in range(0, len(mu)):
        j: int
        for j in range(0, len(la)):
            L[i][j] = (V[i] - W[j]) / (mu[i] - la[j])
            Ls[i][j] = (mu[i] * V[i] - la[j] * W[j]) / (mu[i] - la[j])

    # Transform the complex-valued matrices into real-valued matrices
    block: NDArray[complex128] = array(
        [
            [1, 1j],
            [1, -1j]
        ]
    ) / sqrt(2)
    P: NDArray[complex128] = block_diag(*[block for _ in range(0, n//2)])
    Pstar: NDArray[complex128] = P.conj().T

    L = Pstar @ L @ P
    Ls = Pstar @ Ls @ P
    W = (W.T @ P).T
    V = Pstar @ V

    return (
        L.real,
        Ls.real,
        V.real,
        W.real,
    )

def _generate_reduced_order_model(
    L: NDArray[float64],
    Ls: NDArray[float64],
    V: NDArray[float64],
    W: NDArray[float64],
    k: int,
) -> Tuple[
    NDArray[float64],
    NDArray[float64],
    NDArray[float64],
    NDArray[float64],
    NDArray[float64],
]:
    singular_values: NDArray[float64]
    _, singular_values, _ = svd(concatenate((L, Ls), 1))

    Y_L: NDArray[float64]
    X_L: NDArray[float64]
    Y_L, _, X_L = svd(L)

    Yk: NDArray[float64] = Y_L[:, :k]
    Xk: NDArray[float64] = X_L[:, :k]

    Ek = -Yk.T @ L @ Xk
    Ak = -Yk.T @ Ls @ Xk
    Bk = Yk.T @ V
    Ck = W.T @ Xk

    return (
        Ek,
        Ak,
        Bk,
        Ck,
        singular_values,
    )


def _calculate_model_impedance(
    s: NDArray[complex128],
    Ek: NDArray[float64],
    Ak: NDArray[float64],
    Bk: NDArray[float64],
    Ck: NDArray[float64],
) -> ComplexImpedances:
    Z_fit: ComplexImpedances = zeros(s.shape, dtype=ComplexImpedance)

    i: int
    for i in range(0, len(s)):
        A: NDArray[complex128] = s[i] * Ek - Ak
        B: NDArray[complex128] = solve(A, Bk)
        Z_fit[i] = Ck @ B

    return Z_fit



def _extract_peaks(
    Ek: NDArray[float64],
    Ak: NDArray[float64],
    Bk: NDArray[float64],
    Ck: NDArray[float64],
) -> Tuple[TimeConstants, Gammas]:
    from scipy.linalg import eig

    eigenvalues: NDArray[complex128]
    eigenvectors: NDArray[complex128]
    eigenvalues, eigenvectors = eig(Ak, Ek)

    Bt: NDArray[complex128] = solve(eigenvectors, solve(Ek, Bk))
    Ct: NDArray[complex128] = Ck @ eigenvectors
    residues: NDArray[complex128] = Bt * Ct.T

    time_constants: TimeConstants = abs(-1 / eigenvalues)
    gammas: Gammas = (-residues / eigenvalues).real
    
    return (
        time_constants,
        gammas,
    )


def calculate_drt_lm(
    data: DataSet,
    model_order: int = 0,
    model_order_method: str = "matrix_rank",
    num_procs: int = -1,
    **kwargs,
) -> LMResult:
    """
    Calculates the distribution of relaxation times (DRT) using the Loewner method (LM).

    References:

    - `Mayo, A.J. and Antoulas, A.C., 2007, Linear Algebra Its Appl. 425, 634–662 <https://doi.org/10.1016/j.laa.2007.03.008>`_,
    - `Sorrentino, A., Patel, B., Gosea, I.V., Antoulas, A.C., and Vidaković-Koch, T., 2022, SSRN <http://doi.org/10.2139/ssrn.4217752>`_
    - `Rüther, T., Gosea, I.V., Jahn, L., Antoulas, A.C., and Danzer, M.A., 2023, Batteries, 9, 132 <https://doi.org/10.3390/batteries9020132>`_
    - `Sorrentino, A., Patel, B., Gosea, I.V., Antoulas, A.C., and Vidaković-Koch, T., 2023, J. Power Sources, 585, 223575 <https://doi.org/10.1016/j.jpowsour.2023.233575>`_

    Parameters
    ----------
    data: DataSet
        The data set to use in the calculations.

    model_order: int, optional
        The order of the model (k).

    model_order_method: str, optional
        How to automatically pick the order of the model if the model order is not specified:
        
        - "matrix_rank"
        - "pseudo_chisqr"

    num_procs: int, optional
        The maximum number of parallel processes to use.
        A value less than 1 results in an attempt to figure out a suitable value based on, e.g., the number of cores detected.
        Additionally, a negative value can be used to reduce the number of processes by that much (e.g., to leave one core for a GUI thread).

    Returns
    -------
    LMResult
    """
    f: Frequencies = data.get_frequencies()
    n: int = len(f)
    while n % 2 != 0:
        n -= 1

    if n <= 2:
        raise ValueError(f"Expected more frequencies (i.e., {len(f)=} > 2)")

    if not _is_integer(model_order):
        raise TypeError(f"Expected an integer instead of {model_order=}")
    elif not (model_order <= n):
        raise ValueError(f"Expected {model_order=} <= {n=}")

    if not isinstance(model_order_method, str):
        raise TypeError(f"Expected a string instead of {model_order_method=}")
    elif model_order_method not in _MODEL_ORDER_METHODS:
        raise ValueError(f"Unsupported {model_order_method=} encountered instead of one of the following:\n- " + "\n- ".join(_MODEL_ORDER_METHODS))

    if not _is_integer(num_procs):
        raise TypeError(f"Expected an integer instead of {num_procs=}")
    elif num_procs < 1:
        num_procs = max((get_default_num_procs() - abs(num_procs), 1))

    prog: Progress
    with Progress("Preparing matrices", total=3) as prog:
        s: NDArray[complex128] = flip(1j * 2 * pi * f)
        Z_exp: ComplexImpedances = flip(data.get_impedances())
        
        L: NDArray[float64]
        Ls: NDArray[float64]
        V: NDArray[float64]
        W: NDArray[float64]
        L, Ls, V, W = _generate_loewner_matrices(s, Z_exp)
        prog.increment()
        prog.set_message("Generating reduced model")

        Z_exp = flip(Z_exp)

        reduced_model_has_been_generated: bool = False
        Ek: NDArray[float64]
        Ak: NDArray[float64]
        Bk: NDArray[float64]
        Ck: NDArray[float64]
        Z_fit: ComplexImpedances
        pseudo_chisqr: float

        # If the model order has not been specified, then pick one
        k: int = model_order
        if k < 1 and model_order_method == "matrix_rank":
            k = matrix_rank(concatenate((L, Ls), 1))
        elif k < 1 and model_order_method == "pseudo_chisqr":
            test: KramersKronigResult = perform_kramers_kronig_test(
                data,
                test=kwargs.get("test", "complex"),
                num_RC=kwargs.get("num_RC", 0),
                add_capacitance=kwargs.get("add_capacitance", True),
                add_inductance=kwargs.get("add_inductance", True),
                admittance=kwargs.get("admittance", None),
                min_log_F_ext=kwargs.get("min_log_F_ext", -1.0),
                max_log_F_ext=kwargs.get("max_log_F_ext", 1.0),
                log_F_ext=kwargs.get("log_F_ext", 0.0),
                num_F_ext_evaluations=kwargs.get("num_F_ext_evaluations", 20),
                rapid_F_ext_evaluations=kwargs.get("rapid_F_ext_evaluations", True),
                cnls_method=kwargs.get("cnls_method", "leastsq"),
                max_nfev=kwargs.get("max_nfev", 0),
                timeout=kwargs.get("timeout", 60),
                num_procs=kwargs.get("num_procs", -1),
            )

            k_values: List[int] = []
            pseudo_chisqr_values: List[float] = []

            for k in range(1, n+1):
                k_values.append(k)
                Ek, Ak, Bk, Ck, singular_values = _generate_reduced_order_model(
                    L,
                    Ls,
                    V,
                    W,
                    k=k,
                )
                reduced_model_has_been_generated = True

                Z_fit = _calculate_model_impedance(
                    s,
                    Ek,
                    Ak,
                    Bk,
                    Ck,
                )

                Z_fit = flip(Z_fit)
                pseudo_chisqr = _calculate_pseudo_chisqr(Z_exp=Z_exp, Z_fit=Z_fit)
                pseudo_chisqr_values.append(pseudo_chisqr)
                if pseudo_chisqr <= test.pseudo_chisqr:
                    break

            if _DEBUG and reduced_model_has_been_generated:
                import matplotlib.pyplot as plt
                from pandas import Series

                fig, ax = plt.subplots()

                ax.set_xlim(0, n + 1)
                ax.axhline(log(test.pseudo_chisqr), color="black", linestyle=":")
                ax.scatter(k_values, log(pseudo_chisqr_values), color="blue", marker="o")

                ax2 = ax.twinx()
                std = Series(log(pseudo_chisqr_values)).rolling(3, center=True, min_periods=1).std()
                ax2.scatter(k_values, std, color="red", marker="+")

                plt.show()
        elif k < 1:
            raise NotImplementedError(f"Unsupported {model_order_method=}")

        if not reduced_model_has_been_generated:
            Ek, Ak, Bk, Ck, singular_values = _generate_reduced_order_model(
                L,
                Ls,
                V,
                W,
                k=k,
            )

            Z_fit = _calculate_model_impedance(
                s,
                Ek,
                Ak,
                Bk,
                Ck,
            )

            Z_fit = flip(Z_fit)
            pseudo_chisqr = _calculate_pseudo_chisqr(Z_exp=Z_exp, Z_fit=Z_fit)

            reduced_model_has_been_generated = True

        assert reduced_model_has_been_generated

        prog.increment()
        prog.set_message("Calculating time constants")

        time_constants: TimeConstants
        gammas: Gammas
        time_constants, gammas = _extract_peaks(Ek, Ak, Bk, Ck)

    return LMResult(
        time_constants=time_constants,
        gammas=gammas,
        frequencies=f,
        impedances=Z_fit,
        residuals=_calculate_residuals(Z_exp, Z_fit),
        pseudo_chisqr=pseudo_chisqr,
        singular_values=singular_values,
    )
