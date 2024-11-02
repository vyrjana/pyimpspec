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

# This module implements the Bayesian Hilbert transform method
# - 10.1016/j.electacta.2020.136864
# Based on code from https://github.com/ciuccislab/pyDRTtools.
# pyDRTtools commit: 3694b9b4cef9b29d623bef7300280810ec351d46

from contextlib import redirect_stdout
from dataclasses import dataclass
from multiprocessing import Pool
from os import devnull
from sys import version_info as _python_version_info
from numpy import (
    array,
    diag,
    empty_like,
    exp,
    eye,
    float64,
    full,
    log as ln,
    log10 as log,
    logical_and,
    logspace,
    nan,
    pi,
    sqrt,
    squeeze,
    sum as array_sum,
    zeros,
)
from numpy.linalg import (
    cholesky,
    inv,
    norm,
    solve as solve_linalg,
)
from numpy.random import rand
from numpy.typing import NDArray
from pyimpspec.data import DataSet
from pyimpspec.analysis.utility import (
    _calculate_residuals,
    _calculate_pseudo_chisqr,
    get_default_num_procs,
)
from pyimpspec.exceptions import DRTError
from .result import DRTResult
from .peak_analysis import (
    DRTPeaks,
    _analyze_peaks,
)
from .tr_rbf import (
    _RBF_SHAPES,
    _RBF_TYPES,
    _compute_epsilon,
    _x_to_gamma,
)
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
    Callable,
    Dict,
    IO,
    List,
    NDArray,
    Optional,
    Tuple,
    Union,
    float64,
    _is_integer,
    _is_floating,
)


@dataclass(frozen=True)
class BHTResult(DRTResult):
    """
    An object representing the results of calculating the distribution of relaxation times in a data set using the Bayesian Hilbert transfrom (BHT) method.

    Parameters
    ----------
    time_constants: |TimeConstants|
        The time constants.

    real_gammas: |Gammas|
        The gamma values calculated based on the real part of the impedance spectrum.

    imaginary_gammas: |Gammas|
        The gamma values calculated based on the imaginary part of the impedance spectrum.

    frequencies: |Frequencies|
        The frequencies of the impedance spectrum.

    impedances: |ComplexImpedances|
        The impedance produced by the model.

    residuals: |ComplexResiduals|
        The residuals of the real parts of the model and the data set.

    pseudo_chisqr: float
        The pseudo chi-squared value, |pseudo chi-squared|, of the modeled impedance (eq. 14 in Boukamp, 1995).

    scores: Dict[str, complex]
        The scores calculated by the BHT method.
    """

    real_gammas: Gammas
    imaginary_gammas: Gammas
    scores: Dict[str, complex]

    def get_label(self) -> str:
        return "BHT"

    def get_gammas(self) -> Tuple[Gammas, Gammas]:
        """
        Get the gamma values calculated based on the real or imaginary parts of the impedance spectrum.

        Returns
        -------
        Tuple[|Gammas|, |Gammas|]
        """
        return (self.real_gammas, self.imaginary_gammas)

    def get_drt_data(self) -> Tuple[TimeConstants, Gammas, Gammas]:
        """
        Get the data necessary to plot this DRT result as a DRT plot: the time constants and the corresponding real and imaginary gamma values.

        Returns
        -------
        Tuple[|TimeConstants|, |Gammas|, |Gammas|]
        """
        return (
            self.time_constants,
            self.real_gammas,
            self.imaginary_gammas,
        )

    def to_peaks_dataframe(
        self,
        threshold: float = 0.0,
        columns: Optional[List[str]] = None,
    ) -> "DataFrame":  # noqa: F821
        from pandas import DataFrame

        if columns is None:
            columns = [
                "tau, real (s)",
                "gamma, real (ohm)",
                "tau, imag. (s)",
                "gamma, imag. (ohm)",
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

        indices_re: Indices = self._get_peak_indices(threshold, self.real_gammas)
        indices_im: Indices = self._get_peak_indices(threshold, self.imaginary_gammas)
        width: int = max(indices_re.size, indices_im.size)

        tau_re: TimeConstants
        gamma_re: Gammas
        tau_re, gamma_re = pad(
            self.time_constants[indices_re],
            self.real_gammas[indices_re],
            width,
        )

        tau_im: TimeConstants
        gamma_im: Gammas
        tau_im, gamma_im = pad(
            self.time_constants[indices_im],
            self.imaginary_gammas[indices_im],
            width,
        )

        return DataFrame.from_dict(
            {
                columns[0]: tau_re,
                columns[1]: gamma_re,
                columns[2]: tau_im,
                columns[3]: gamma_im,
            }
        )

    def to_statistics_dataframe(
        self,
    ) -> "DataFrame":  # noqa: F821
        from pandas import DataFrame

        statistics: Dict[str, Union[int, float, str]] = {
            "Log pseudo chi-squared": log(self.pseudo_chisqr),
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
        Get the time constants (in seconds) and gammas (in ohms) of peaks with magnitudes greater than the threshold.
        The threshold and the magnitudes are all relative to the magnitude of the highest peak.

        Parameters
        ----------
        threshold: float, optional
            The minimum peak height threshold (relative to the height of the tallest peak) for a peak to be included.

        Returns
        -------
        Tuple[|TimeConstants|, |Gammas|, |TimeConstants|, |Gammas|]
        """
        indices_re: Indices = self._get_peak_indices(threshold, self.real_gammas)
        indices_im: Indices = self._get_peak_indices(threshold, self.imaginary_gammas)

        return (
            self.time_constants[indices_re],
            self.real_gammas[indices_re],
            self.time_constants[indices_im],
            self.imaginary_gammas[indices_im],
        )

    def get_scores(self) -> Dict[str, complex]:
        """
        Get the scores for the data set.
        The scores are represented as complex values where the real and imaginary parts have magnitudes ranging from 0.0 to 1.0.
        A consistent impedance spectrum should score high.

        Returns
        -------
        Dict[str, complex]
        """
        return self.scores

    def to_scores_dataframe(
        self,
        columns: Optional[List[str]] = None,
        rows: Optional[List[str]] = None,
    ) -> "DataFrame":  # noqa: F821
        """
        Get the scores for the data set as a `pandas.DataFrame`_ object that can be used to generate, e.g., a Markdown table.

        Parameters
        ----------
        columns: Optional[List[str]], optional
            The labels for the column headers.

        rows: Optional[List[str]], optional
            The labels for the rows.

        Returns
        -------
        `pandas.DataFrame`_
        """
        from pandas import DataFrame

        if columns is None:
            columns = [
                "Score",
                "Real (%)",
                "Imag. (%)",
            ]
        elif not isinstance(columns, list):
            raise TypeError(f"Expected a list of strings instead of {columns=}")
        elif len(columns) != 3:
            raise ValueError(f"Expected a list with 3 items instead of {len(columns)=}")
        elif not all(map(lambda s: isinstance(s, str), columns)):
            raise TypeError(f"Expected a list of strings instead of {columns=}")
        elif len(set(columns)) != 3:
            raise ValueError(
                f"Expected a list of 3 unique strings instead of {columns=}"
            )

        if rows is None:
            rows = [
                "Mean",
                "Residuals, 1 sigma",
                "Residuals, 2 sigma",
                "Residuals, 3 sigma",
                "Hellinger distance",
                "Jensen-Shannon distance",
            ]
        elif not isinstance(rows, list):
            raise TypeError(f"Expected a list of strings instead of {rows=}")
        elif len(rows) != 6:
            raise ValueError(f"Expected a list with 6 items instead of {len(rows)=}")
        elif not all(map(lambda s: isinstance(s, str), rows)):
            raise TypeError(f"Expected a list of strings instead of {rows=}")
        elif len(set(rows)) != 6:
            raise ValueError(f"Expected a list of 6 unique strings instead of {rows=}")

        return DataFrame.from_dict(
            {
                columns[0]: rows,
                columns[1]: [
                    self.scores["mean"].real * 100,
                    self.scores["residuals_1sigma"].real * 100,
                    self.scores["residuals_2sigma"].real * 100,
                    self.scores["residuals_3sigma"].real * 100,
                    self.scores["hellinger_distance"].real * 100,
                    self.scores["jensen_shannon_distance"].real * 100,
                ],
                columns[2]: [
                    self.scores["mean"].imag * 100,
                    self.scores["residuals_1sigma"].imag * 100,
                    self.scores["residuals_2sigma"].imag * 100,
                    self.scores["residuals_3sigma"].imag * 100,
                    self.scores["hellinger_distance"].imag * 100,
                    self.scores["jensen_shannon_distance"].imag * 100,
                ],
            }
        )

    def analyze_peaks(
        self,
        num_peaks: int = 0,
        peak_positions: Optional[Union[List[float], NDArray[float64]]] = None,
        disallow_skew: bool = False,
    ) -> Tuple[DRTPeaks, DRTPeaks]:
        """
        Analyze the peaks present in a distribution of relaxation times using skew normal distributions.

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
            The first and second |DRTPeaks| instance corresponds to the DRT generated based on the real or imaginary part, respectively, of the impedance spectrum.
        """
        peaks_real: DRTPeaks = _analyze_peaks(
            self.time_constants,
            self.real_gammas,
            num_peaks=num_peaks,
            peak_positions=peak_positions,
            disallow_skew=disallow_skew,
            suffix="real",
        )
        peaks_imag: DRTPeaks = _analyze_peaks(
            self.time_constants,
            self.imaginary_gammas,
            num_peaks=num_peaks,
            peak_positions=peak_positions,
            disallow_skew=disallow_skew,
            suffix="imag.",
        )
        object.__setattr__(self, "_peak_analysis", (peaks_real, peaks_imag))

        return (peaks_real, peaks_imag)


def _compute_res_scores(
    res: NDArray[float64],
    band: NDArray[float64],
) -> NDArray[float64]:
    # Count the points fallen inside the 1, 2, and 3 sigma credible bands
    count: NDArray[float64] = zeros(3, dtype=float64)

    i: int
    for i in range(3):
        count[i] = array_sum(logical_and(res < (i + 1) * band, res > -(i + 1) * band))

    return count / len(res)


def _compute_SHD(
    mu_P: NDArray[float64],
    Sigma_P: NDArray[float64],
    mu_Q: NDArray[float64],
    Sigma_Q: NDArray[float64],
) -> NDArray[float64]:
    # Squared Hellinger distance
    sigma_P: NDArray[float64] = sqrt(diag(Sigma_P))
    sigma_Q: NDArray[float64] = sqrt(diag(Sigma_Q))

    sum_cov: NDArray[float64] = sigma_P**2 + sigma_Q**2
    prod_cov: NDArray[float64] = sigma_P * sigma_Q

    return 1.0 - sqrt(2.0 * prod_cov / sum_cov) * exp(
        -0.25 * (mu_P - mu_Q) ** 2 / sum_cov
    )


def _compute_JSD(
    mu_P: NDArray[float64],
    Sigma_P: NDArray[float64],
    mu_Q: NDArray[float64],
    Sigma_Q: NDArray[float64],
    num_samples: int,
) -> NDArray[float64]:
    # Jensen-Shannon distance (JSD)
    from scipy.stats import multivariate_normal

    JSD: NDArray[float64] = empty_like(mu_P, dtype=float64)

    i: int
    for i in range(mu_P.size):
        RV_p = multivariate_normal(mean=mu_P[i], cov=Sigma_P[i, i])
        RV_q = multivariate_normal(mean=mu_Q[i], cov=Sigma_Q[i, i])

        x: NDArray[float64] = RV_p.rvs(num_samples)
        p_x: NDArray[float64] = RV_p.pdf(x)
        q_x: NDArray[float64] = RV_q.pdf(x)
        m_x: NDArray[float64] = (p_x + q_x) / 2.0

        y: NDArray[float64] = RV_q.rvs(num_samples)
        p_y: NDArray[float64] = RV_p.pdf(y)
        q_y: NDArray[float64] = RV_q.pdf(y)
        m_y: NDArray[float64] = (p_y + q_y) / 2.0

        dKL_pm: NDArray[float64] = ln(p_x / m_x).mean()
        dKL_qm: NDArray[float64] = ln(q_y / m_y).mean()
        JSD[i] = 0.5 * (dKL_pm + dKL_qm)

    return JSD


def _NMLL_fct(
    theta: NDArray[float64],
    Z: ComplexImpedances,
    A: NDArray[float64],
    L: NDArray[float64],
    num_freqs: int,
    num_taus: int,
) -> NDArray[float64]:
    sigma_n: NDArray[float64]
    sigma_beta: NDArray[float64]
    sigma_lambda: NDArray[float64]
    sigma_n, sigma_beta, sigma_lambda = theta

    W: NDArray[float64] = (
        1 / (sigma_beta**2) * eye(num_taus + 1, dtype=float64)
        + 1 / (sigma_lambda**2) * L.T @ L
    )

    # W = 0.5 * (W.T + W)
    K_agm: NDArray[float64] = 1 / (sigma_n**2) * (A.T @ A) + W

    # K_agm = 0.5 * (K_agm.T + K_agm)
    L_W: NDArray[float64] = cholesky(W)
    L_agm: NDArray[float64] = cholesky(K_agm)

    # Compute mu_x
    u: NDArray[float64] = solve_linalg(L_agm.T, solve_linalg(L_agm, A.T @ Z))
    mu_x: NDArray[float64] = 1 / (sigma_n**2) * u

    # Compute loss
    E_mu_x: NDArray[float64] = 0.5 / (sigma_n**2) * norm(A @ mu_x - Z) ** 2 + 0.5 * (
        mu_x.T @ (W @ mu_x)
    )

    val_1: NDArray[float64] = array_sum(ln(diag(L_W)))
    val_2: NDArray[float64] = -array_sum(ln(diag(L_agm)))
    val_3: NDArray[float64] = -num_freqs / 2.0 * ln(sigma_n**2)
    val_4: NDArray[float64] = -E_mu_x
    val_5: float = -num_freqs / 2 * ln(2 * pi)

    return -(val_1 + val_2 + val_3 + val_4 + val_5)


def _compute_A_re(w: NDArray[float64], tau: NDArray[float64]) -> NDArray[float64]:
    num_freqs: int = w.shape[0]
    num_taus: int = tau.shape[0]

    A_re: NDArray[float64] = zeros(
        (
            num_freqs,
            num_taus + 1,
        ),
        dtype=float64,
    )
    A_re[:, 0] = 1.0

    i: int
    j: int
    for i in range(0, num_freqs):
        for j in range(0, num_taus):
            A_re[i, j + 1] = (
                0.5
                / (1 + (w[i] * tau[j]) ** 2)
                * ln(
                    (tau[j] if j == num_taus - 1 else tau[j + 1])
                    / (tau[j] if j == 0 else tau[j - 1])
                )
            )

    return A_re


def _compute_A_H_re(w: NDArray[float64], tau: NDArray[float64]) -> NDArray[float64]:
    return _compute_A_re(w, tau)[:, 1:]


def _compute_A_im(w: NDArray[float64], tau: NDArray[float64]) -> NDArray[float64]:
    num_freqs: int = w.shape[0]
    num_taus: int = tau.shape[0]

    A_im: NDArray[float64] = zeros(
        (
            num_freqs,
            num_taus + 1,
        ),
        dtype=float64,
    )
    A_im[:, 0] = w

    i: int
    j: int
    for i in range(0, num_freqs):
        for j in range(0, num_taus):
            A_im[i, j + 1] = (
                -0.5
                * (w[i] * tau[j])
                / (1 + (w[i] * tau[j]) ** 2)
                * ln(
                    (tau[j] if j == num_taus - 1 else tau[j + 1])
                    / (tau[j] if j == 0 else tau[j - 1])
                )
            )

    return A_im


def _compute_A_H_im(w: NDArray[float64], tau: NDArray[float64]) -> NDArray[float64]:
    return _compute_A_im(w, tau)[:, 1:]


def _compute_L(tau: NDArray[float64], derivative_order: int) -> NDArray[float64]:
    num_taus: int = tau.shape[0]
    L: NDArray[float64] = zeros(
        (
            num_taus - 2,
            num_taus + 1,
        ),
        dtype=float64,
    )

    i: int
    delta_loc: float
    if derivative_order == 1:
        for i in range(0, num_taus - 2):
            delta_loc = ln(tau[i + 1] / tau[i])
            factors: NDArray[float64] = array([1.0, -2.0, 1.0], dtype=float64)

            if i == 0 or i == num_taus - 3:
                factors *= 2

            L[i, i + 1] = factors[0] / (delta_loc**2)
            L[i, i + 2] = factors[1] / (delta_loc**2)
            L[i, i + 3] = factors[2] / (delta_loc**2)

    elif derivative_order == 2:
        for i in range(0, num_taus - 2):
            delta_loc = ln(tau[i + 1] / tau[i])

            if i == 0:
                L[i, i + 1] = -3.0 / (2 * delta_loc)
                L[i, i + 2] = 4.0 / (2 * delta_loc)
                L[i, i + 3] = -1.0 / (2 * delta_loc)

            elif i == num_taus - 2:
                L[i, i] = 1.0 / (2 * delta_loc)
                L[i, i + 1] = -4.0 / (2 * delta_loc)
                L[i, i + 2] = 3.0 / (2 * delta_loc)

            else:
                L[i, i] = 1.0 / (2 * delta_loc)
                L[i, i + 2] = -1.0 / (2 * delta_loc)
    else:
        raise NotImplementedError(f"Unsupported {derivative_order=}")

    return L


def _compute_mu_score(
    mu_Z_DRT: NDArray[float64],
    mu_Z_H: NDArray[float64],
) -> float:
    return float(1.0 - (norm(mu_Z_DRT - mu_Z_H) / (norm(mu_Z_DRT) + norm(mu_Z_H))))


def _compute_real_residual_scores(
    mu_R_inf: float,
    cov_R_inf: NDArray[float64],
    sigma_n_im: NDArray[float64],
    mu_Z_H_re: NDArray[float64],
    Z_exp: ComplexImpedances,
    Sigma_Z_H_re: NDArray[float64],
) -> NDArray[float64]:
    res_re: NDArray[float64] = mu_R_inf + mu_Z_H_re - Z_exp.real
    band_re: NDArray[float64] = sqrt(cov_R_inf + diag(Sigma_Z_H_re) + sigma_n_im**2)

    return _compute_res_scores(res_re, band_re)


def _compute_imaginary_residual_scores(
    mu_L_0: float,
    cov_L_0: NDArray[float64],
    sigma_n_re: NDArray[float64],
    mu_Z_H_im: NDArray[float64],
    omega: NDArray[float64],
    Z_exp: ComplexImpedances,
    Sigma_Z_H_im: NDArray[float64],
) -> NDArray[float64]:
    res_im: NDArray[float64] = omega * mu_L_0 + mu_Z_H_im - Z_exp.imag
    band_im: NDArray[float64] = sqrt(
        (omega**2) * cov_L_0 + diag(Sigma_Z_H_im) + sigma_n_re**2
    )

    return _compute_res_scores(res_im, band_im)


def _compute_HD_score(
    mu_Z_DRT,
    Sigma_Z_DRT,
    mu_Z_H,
    Sigma_Z_H,
) -> float:
    # Squared Hellinger distance (SHD)
    # which is bounded between 0 and 1
    # we are going to score w.r.t. the Hellinger distance (HD)
    # the score uses 1 to mean good (this means close)
    # and 0 means bad (far away) => that's the opposite of the distance
    return 1.0 - sqrt(_compute_SHD(mu_Z_DRT, Sigma_Z_DRT, mu_Z_H, Sigma_Z_H)).mean()


def _compute_JSD_score(
    mu_Z_DRT,
    Sigma_Z_DRT,
    mu_Z_H,
    Sigma_Z_H,
    num_samples: int,
) -> float:
    # Compute the Jensen-Shannon distance (JSD). The JSD is a symmetrized
    # relative entropy (discrepancy), so highest value means more entropy.
    # We are going to reverse that by taking (ln(2)-JSD)/ln(2), which means
    # higher value less relative entropy (discrepancy).
    return (
        ln(2)
        - _compute_JSD(
            mu_Z_DRT,
            Sigma_Z_DRT,
            mu_Z_H,
            Sigma_Z_H,
            num_samples,
        ).mean()
    ) / ln(2)


def _calculate_scores(
    theta_0: NDArray[float64],
    f: Frequencies,
    Z_exp: ComplexImpedances,
    out_dict_real: dict,
    out_dict_imag: dict,
    num_samples: int,
) -> dict:
    prog: Progress
    with Progress("Calculating scores", total=12) as prog:
        # scores
        # s_mu - distance between means:
        mu_Z_DRT_re: NDArray[float64] = out_dict_real["mu_Z_DRT"]
        mu_Z_H_re: NDArray[float64] = out_dict_imag["mu_Z_H"]
        s_mu_re: float = _compute_mu_score(mu_Z_DRT_re, mu_Z_H_re)
        prog.increment()

        mu_Z_DRT_im: NDArray[float64] = out_dict_imag["mu_Z_DRT"]
        mu_Z_H_im: NDArray[float64] = out_dict_real["mu_Z_H"]
        s_mu_im: float = _compute_mu_score(mu_Z_DRT_im, mu_Z_H_im)
        prog.increment()

        # s_JSD - Jensen-Shannon Distance:
        # we need the means (above) and covariances (below)
        # for the computation of the JSD
        Sigma_Z_DRT_re: NDArray[float64] = out_dict_real["Sigma_Z_DRT"]
        Sigma_Z_DRT_im: NDArray[float64] = out_dict_imag["Sigma_Z_DRT"]
        Sigma_Z_H_re: NDArray[float64] = out_dict_imag["Sigma_Z_H"]
        Sigma_Z_H_im: NDArray[float64] = out_dict_real["Sigma_Z_H"]

        # s_res - residual score:
        # real part
        s_res_re: NDArray[float64] = _compute_real_residual_scores(
            out_dict_real["mu_gamma"][0],
            diag(out_dict_real["Sigma_gamma"])[0],
            out_dict_imag["theta"][0],
            mu_Z_H_re,
            Z_exp,
            Sigma_Z_H_re,
        )
        prog.increment()

        # imaginary part
        s_res_im: NDArray[float64] = _compute_imaginary_residual_scores(
            out_dict_imag["mu_gamma"][0],
            diag(out_dict_imag["Sigma_gamma"])[0],
            out_dict_real["theta"][0],
            mu_Z_H_im,
            2 * pi * f,
            Z_exp,
            Sigma_Z_H_im,
        )
        prog.increment()

        s_HD_re: float = _compute_HD_score(
            mu_Z_DRT_re,
            Sigma_Z_DRT_re,
            mu_Z_H_re,
            Sigma_Z_H_re,
        )
        prog.increment()

        s_HD_im: float = _compute_HD_score(
            mu_Z_DRT_im,
            Sigma_Z_DRT_im,
            mu_Z_H_im,
            Sigma_Z_H_im,
        )
        prog.increment()

        s_JSD_re: float = _compute_JSD_score(
            mu_Z_DRT_re,
            Sigma_Z_DRT_re,
            mu_Z_H_re,
            Sigma_Z_H_re,
            num_samples,
        )
        prog.increment()

        s_JSD_im: float = _compute_JSD_score(
            mu_Z_DRT_im,
            Sigma_Z_DRT_im,
            mu_Z_H_im,
            Sigma_Z_H_im,
            num_samples,
        )

    return {
        "hellinger_distance": complex(s_HD_re, s_HD_im),
        "jensen_shannon_distance": complex(s_JSD_re, s_JSD_im),
        "mean": complex(s_mu_re, s_mu_im),
        "residuals_1sigma": complex(s_res_re[0], s_res_im[0]),
        "residuals_2sigma": complex(s_res_re[1], s_res_im[1]),
        "residuals_3sigma": complex(s_res_re[2], s_res_im[2]),
    }


def _single_hilbert_transform_estimate(
    theta_0: NDArray[float64],
    Z_exp: NDArray[float64],
    A: NDArray[float64],
    A_H: NDArray[float64],
    L: NDArray[float64],
    num_freqs: int,
    num_taus: int,
) -> dict:
    import warnings
    from scipy.optimize import (
        OptimizeResult,
        OptimizeWarning,
        minimize,
    )

    fp: IO
    with open(devnull, "w") as fp:
        with redirect_stdout(fp):
            kw: dict
            if _python_version_info.major == 3 and _python_version_info.minor < 11:
                kw = {}
            else:
                kw = {"category": OptimizeWarning}

            with warnings.catch_warnings(**kw):
                warnings.simplefilter("ignore")
                res: OptimizeResult = minimize(
                    _NMLL_fct,
                    squeeze(theta_0),
                    args=(Z_exp, A, L, num_freqs, num_taus),
                    options={"gtol": 1e-8, "disp": True},
                )

    sigma_n: float
    sigma_beta: float
    sigma_lambda: float
    sigma_n, sigma_beta, sigma_lambda = res.x

    # Compute the probability density functions of data regression
    # $K_agm = A.T A +\lambda L.T L$
    W: NDArray[float64] = (
        1 / (sigma_beta**2) * eye(num_taus + 1, dtype=float64)
        + 1 / (sigma_lambda**2) * L.T @ L
    )
    K_agm: NDArray[float64] = 1 / (sigma_n**2) * (A.T @ A) + W

    # Cholesky factorization
    L_agm: NDArray[float64] = cholesky(K_agm)
    inv_L_agm: NDArray[float64] = inv(L_agm)
    inv_K_agm: NDArray[float64] = inv_L_agm.T @ inv_L_agm

    # Compute the gamma ~ N(mu_gamma, Sigma_gamma)
    Sigma_gamma: NDArray[float64] = inv_K_agm
    # .real is also in the original pyDRTTools code
    mu_gamma: NDArray[float64] = 1 / (sigma_n**2) * (Sigma_gamma @ A.T) @ Z_exp.real

    # Compute, from gamma, the Z ~ N(mu_Z, Sigma_Z)
    mu_Z: NDArray[float64] = A @ mu_gamma
    Sigma_Z: NDArray[float64] = A @ (Sigma_gamma @ A.T) + sigma_n**2 * eye(
        num_freqs, dtype=float64
    )

    # Compute, from gamma, the Z_DRT ~ N(mu_Z_DRT, Sigma_Z_DRT)
    A_DRT: NDArray[float64] = A[:, 1:]
    mu_gamma_DRT: NDArray[float64] = mu_gamma[1:]
    Sigma_gamma_DRT: NDArray[float64] = Sigma_gamma[1:, 1:]
    mu_Z_DRT: NDArray[float64] = A_DRT @ mu_gamma_DRT
    Sigma_Z_DRT: NDArray[float64] = A_DRT @ (Sigma_gamma_DRT @ A_DRT.T)

    # Compute, from gamma, the Z_H_conj ~ N(mu_Z_H_conj, Sigma_Z_H_conj)
    mu_Z_H: NDArray[float64] = A_H @ mu_gamma[1:]
    Sigma_Z_H: NDArray[float64] = A_H @ (Sigma_gamma[1:, 1:] @ A_H.T)

    return {
        "mu_gamma": mu_gamma,
        "Sigma_gamma": Sigma_gamma,
        "mu_Z": mu_Z,
        "Sigma_Z": Sigma_Z,
        "mu_Z_DRT": mu_Z_DRT,
        "Sigma_Z_DRT": Sigma_Z_DRT,
        "mu_Z_H": mu_Z_H,
        "Sigma_Z_H": Sigma_Z_H,
        "theta": res.x,
    }


def _calculate_symmetry_score(
    result: Tuple[float, NDArray[float64], dict, dict],
    tau_fine: NDArray[float64],
    tau: NDArray[float64],
    epsilon: float,
    rbf_type: str,
) -> float:
    # Calculate gamma for each result and check if it is asymmetrical
    pseudo_chisqr: float
    theta_0: NDArray[float64]
    data_real: dict
    data_imag: dict
    pseudo_chisqr, theta_0, data_real, data_imag = result

    _, gamma = _x_to_gamma(
        data_real["mu_gamma"][1:],
        tau_fine,
        tau,
        epsilon,
        rbf_type,
    )

    min_gamma: float = abs(min(gamma))
    max_gamma: float = abs(max(gamma))
    score: float = 1.0 - ((max_gamma - min_gamma) / max(min_gamma, max_gamma))

    _, gamma = _x_to_gamma(
        data_imag["mu_gamma"][1:],
        tau_fine,
        tau,
        epsilon,
        rbf_type,
    )

    min_gamma = abs(min(gamma))
    max_gamma = abs(max(gamma))
    score += 1.0 - ((max_gamma - min_gamma) / max(min_gamma, max_gamma))

    return score / 2.0


def _hilbert_transform_process(
    args: tuple,
) -> Union[Optional[Tuple[float, NDArray[float64], dict, dict]], Exception]:
    theta_0: NDArray[float64]
    w: NDArray[float64]
    Z: ComplexImpedances
    A_re: NDArray[float64]
    A_im: NDArray[float64]
    A_H_re: NDArray[float64]
    A_H_im: NDArray[float64]
    L: NDArray[float64]
    num_freqs: int
    num_taus: int
    maximum_symmetry: float
    tau_fine: NDArray[float64]
    tau: NDArray[float64]
    epsilon: float
    rbf_type: str
    (
        theta_0,
        w,
        Z,
        A_re,
        A_im,
        A_H_re,
        A_H_im,
        L,
        num_freqs,
        num_taus,
        maximum_symmetry,
        tau_fine,
        tau,
        epsilon,
        rbf_type,
    ) = args
    try:
        data_real: dict = _single_hilbert_transform_estimate(
            theta_0,
            Z.real,
            A_re,
            A_H_im,
            L,
            num_freqs,
            num_taus,
        )

        theta_0 = data_real["theta"]
        data_imag: dict = _single_hilbert_transform_estimate(
            theta_0,
            Z.imag,
            A_im,
            A_H_re,
            L,
            num_freqs,
            num_taus,
        )

        mu_R_inf: float = data_real["mu_gamma"][0]
        mu_Z_H_re: NDArray[float64] = data_imag["mu_Z_H"]
        mu_L_0: float = data_imag["mu_gamma"][0]
        mu_Z_H_im: NDArray[float64] = data_real["mu_Z_H"]

        Z_fit: ComplexImpedances = array(
            list(
                map(
                    lambda _: complex(*_),
                    zip(
                        mu_R_inf + mu_Z_H_re,
                        w * mu_L_0 + mu_Z_H_im,
                    ),
                )
            ),
            dtype=ComplexImpedance,
        )
        pseudo_chisqr: float = _calculate_pseudo_chisqr(Z_exp=Z, Z_fit=Z_fit)

        result: Tuple[float, NDArray[float64], dict, dict] = (
            pseudo_chisqr,
            theta_0,
            data_real,
            data_imag,
        )
        if (
            maximum_symmetry > 0.0 and _calculate_symmetry_score(
                result,
                tau_fine,
                tau,
                epsilon,
                rbf_type,
            )
            > maximum_symmetry
        ):
            # The result is most likely poor (lots of strong oscillation).
            return None

        return (
            pseudo_chisqr,
            theta_0,
            data_real,
            data_imag,
        )
    except Exception as err:
        return err


def _perform_attempts(
    w: NDArray[float64],
    Z: ComplexImpedances,
    A_re: NDArray[float64],
    A_im: NDArray[float64],
    A_H_re: NDArray[float64],
    A_H_im: NDArray[float64],
    num_freqs: int,
    num_taus: int,
    maximum_symmetry: float,
    tau_fine: NDArray[float64],
    tau: NDArray[float64],
    epsilon: float,
    rbf_type: str,
    derivative_order: int,
    num_attempts: int,
    num_procs: int,
) -> Tuple[float, NDArray[float64], dict, dict]:
    L: NDArray[float64] = _compute_L(tau, derivative_order)
    theta_0_generator: Callable = lambda: 10 ** (6 * rand(3, 1) - 3)

    args = (
        (
            theta_0_generator(),
            w,
            Z,
            A_re,
            A_im,
            A_H_re,
            A_H_im,
            L,
            num_freqs,
            num_taus,
            maximum_symmetry,
            tau_fine,
            tau,
            epsilon,
            rbf_type,
        )
        for _ in range(0, num_attempts)
    )

    results: List[Tuple[float, NDArray[float64], dict, dict]] = []
    errors: List[Exception] = []

    prog: Progress
    with Progress("Calculating Hilbert transforms", total=num_attempts + 1) as prog:
        if num_procs > 1:
            with Pool(num_procs) as pool:
                for res in pool.imap_unordered(
                    _hilbert_transform_process,
                    args,
                ):
                    prog.increment()
                    if isinstance(res, Exception):
                        errors.append(res)
                    elif res is not None:
                        results.append(res)

        else:
            for res in map(_hilbert_transform_process, args):
                prog.increment()
                if isinstance(res, Exception):
                    errors.append(res)
                elif res is not None:
                    results.append(res)

    if len(results) == 0:
        if len(errors) > 0:
            raise errors.pop(0)
        else:
            raise DRTError("Failed to perform calculations! Try tweaking the settings.")

    results.sort(key=lambda _: _[0])

    return results[0]


def _calculate_model_impedance(
    w: NDArray[float64],
    data_real: dict,
    data_imag: dict,
) -> ComplexImpedances:
    # Real part
    mu_R_inf: float = data_real["mu_gamma"][0]
    mu_Z_H_im: NDArray[float64] = data_real["mu_Z_H"]

    # Imaginary part
    mu_L_0: float = data_imag["mu_gamma"][0]
    mu_Z_H_re: NDArray[float64] = data_imag["mu_Z_H"]

    # Means and bounds
    mu_Z_H_re_agm: NDArray[float64] = mu_R_inf + mu_Z_H_re
    mu_Z_H_im_agm: NDArray[float64] = w * mu_L_0 + mu_Z_H_im

    return array(
        list(
            map(
                lambda _: complex(*_),
                zip(
                    mu_Z_H_re_agm,
                    mu_Z_H_im_agm,
                ),
            )
        ),
        dtype=ComplexImpedance,
    )


def calculate_drt_bht(
    data: DataSet,
    rbf_type: str = "gaussian",
    derivative_order: int = 1,
    rbf_shape: str = "fwhm",
    shape_coeff: float = 0.5,
    num_samples: int = 2000,
    num_attempts: int = 10,
    maximum_symmetry: float = 0.5,
    num_procs: int = -1,
    **kwargs,
) -> BHTResult:
    """
    Calculates the distribution of relaxation times (DRT) using the Bayesian Hilbert transform (BHT) method.

    References:

    - `Liu, J., Wan, T. H., and Ciucci, F., 2020, Electrochim. Acta, 357, 136864 <https://doi.org/10.1016/j.electacta.2020.136864>`_

    Parameters
    ----------
    data: DataSet
        The data set to use in the calculations.

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
        The number of samples drawn when calculating Jensen-Shannon distance.
        A greater number provides better accuracy but requires more time.

    num_attempts: int, optional
        The minimum number of attempts to make when trying to find suitable random initial values.
        A greater number should provide better results at the expense of time.

    maximum_symmetry: float, optional
        A maximum limit (between 0.0 and 1.0) for the relative vertical symmetry of the DRT.
        A high degree of symmetry is common for results where the gamma value oscillates wildly.
        A low value for the limit should improve the results but may cause the BHT method to take longer to finish.

    num_procs: int, optional
        The maximum number of parallel processes to use.
        A value less than 1 results in an attempt to figure out a suitable value based on, e.g., the number of cores detected.
        Additionally, a negative value can be used to reduce the number of processes by that much (e.g., to leave one core for a GUI thread).

    Returns
    -------
    |BHTResult|
    """
    if not isinstance(rbf_type, str):
        raise TypeError(f"Expected a string instead of {rbf_type=}")
    elif rbf_type not in _RBF_TYPES:
        raise ValueError("Valid rbf_type values: '" + "', '".join(_RBF_TYPES))

    if not _is_integer(derivative_order):
        raise TypeError(f"Expected an integer instead of {derivative_order=}")
    elif not (1 <= derivative_order <= 2):
        raise ValueError("Valid derivative_order values: 1, 2")

    if not isinstance(rbf_shape, str):
        raise TypeError(f"Expected a string instead of {rbf_shape=}")
    elif rbf_shape not in _RBF_SHAPES:
        raise ValueError("Valid rbf_shape values: '" + "', '".join(_RBF_SHAPES))

    if not _is_floating(shape_coeff):
        raise TypeError(f"Expected a float instead of {shape_coeff=}")
    elif shape_coeff <= 0.0:
        raise ValueError("The shape coefficient must be greater than 0.0")

    if not _is_integer(num_samples):
        raise TypeError(f"Expected an integer instead of {num_samples=}")
    elif num_samples < 1:
        raise ValueError("The number of samples must be greater than 0")

    if not _is_integer(num_attempts):
        raise TypeError(f"Expected an integer instead of {num_attempts=}")
    elif num_attempts < 1:
        raise ValueError("The number of attempts must be greater than 0")

    if not _is_floating(maximum_symmetry):
        raise TypeError(f"Expected a float instead of {maximum_symmetry=}")
    elif not (0.0 <= maximum_symmetry <= 1.0):
        raise ValueError("The maximum symmetry must be in the range [0.0, 1.0]")

    if not _is_integer(num_procs):
        raise TypeError(f"Expected an integer instead of {num_procs=}")
    elif num_procs < 1:
        num_procs = max((get_default_num_procs() - abs(num_procs), 1))

    prog: Progress
    with Progress("Preparing matrices", total=5) as prog:
        f: Frequencies = data.get_frequencies()
        if len(f) < 1:
            raise ValueError(
                f"There are no unmasked data points in the '{data.get_label()}' data set parsed from '{data.get_path()}'"
            )

        tau: NDArray[float64] = 1 / f
        tau_fine: NDArray[float64] = logspace(
            log(tau.min()) - 0.5,
            log(tau.max()) + 0.5,
            10 * f.shape[0],
        )

        w: NDArray[float64] = 2 * pi * f
        epsilon: float = _compute_epsilon(
            f,
            rbf_shape,
            shape_coeff,
            rbf_type,
        )

        A_re: NDArray[float64] = _compute_A_re(w, tau)
        prog.increment()

        A_im: NDArray[float64] = _compute_A_im(w, tau)
        prog.increment()

        A_H_re: NDArray[float64] = _compute_A_H_re(w, tau)
        prog.increment()

        A_H_im: NDArray[float64] = _compute_A_H_im(w, tau)
        prog.increment()

        Z_exp: ComplexImpedances = data.get_impedances()
        theta_0: NDArray[float64]
        data_real: dict
        data_imag: dict
        pseudo_chisqr, theta_0, data_real, data_imag = _perform_attempts(
            w,
            Z_exp,
            A_re,
            A_im,
            A_H_re,
            A_H_im,
            f.shape[0],
            tau.shape[0],
            maximum_symmetry,
            tau_fine,
            tau,
            epsilon,
            rbf_type,
            derivative_order,
            num_attempts,
            num_procs,
        )

        # Scores seem to be fine based on comparison with the ZARC example used in the article
        scores: dict = _calculate_scores(
            theta_0,
            f,
            Z_exp,
            data_real,
            data_imag,
            num_samples,
        )
        prog.set_message("Calculating model impedance")
        Z_fit: ComplexImpedances = _calculate_model_impedance(w, data_real, data_imag)

    time_constants: TimeConstants
    time_constants, real_gammas = _x_to_gamma(
        data_real["mu_gamma"][1:],
        tau_fine,
        tau,
        epsilon,
        rbf_type,
    )

    _, imaginary_gammas = _x_to_gamma(
        data_imag["mu_gamma"][1:],
        tau_fine,
        tau,
        epsilon,
        rbf_type,
    )

    return BHTResult(
        time_constants=time_constants,
        real_gammas=real_gammas,
        imaginary_gammas=imaginary_gammas,
        frequencies=f,
        impedances=Z_fit,
        residuals=_calculate_residuals(Z_exp, Z_fit),
        pseudo_chisqr=pseudo_chisqr,
        scores=scores,
    )
