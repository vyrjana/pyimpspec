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

# This module implements the Bayesian Hilbert transform method
# pyDRTtools commit: 65ea54d9332a0c6594de852f0242a88e20ec4427

from contextlib import redirect_stdout
from multiprocessing import (
    Pool,
    cpu_count,
)
from os import devnull
from typing import (
    Callable,
    IO,
    List,
    Optional,
    Tuple,
)
from numpy import (
    array,
    diag,
    empty_like,
    exp,
    eye,
    floating,
    integer,
    issubdtype,
    log as ln,
    log10 as log,
    logical_and,
    logspace,
    ndarray,
    pi,
    sqrt,
    squeeze,
    sum as array_sum,
    zeros,
)
from numpy.linalg import (
    cholesky,
    inv as invert,
    norm,
    solve as solve_linalg,
)
from numpy.random import (
    rand,
)
from scipy.optimize import (
    OptimizeResult,
    minimize,
)
from scipy.stats import multivariate_normal
from pyimpspec.data import DataSet
from pyimpspec.analysis.fitting import _calculate_residuals
from .result import (
    DRTError,
    DRTResult,
    _calculate_chisqr,
)
from .tr_rbf import (
    _x_to_gamma,
    _compute_epsilon,
    RBF_TYPES,
    RBF_SHAPES,
)
import pyimpspec.progress as progress


def _compute_res_scores(res: ndarray, band: ndarray) -> ndarray:
    # Count the points fallen inside the 1, 2, and 3 sigma credible bands
    count: ndarray = zeros(3)
    i: int
    for i in range(3):
        count[i] = array_sum(logical_and(res < (i + 1) * band, res > -(i + 1) * band))
    return count / len(res)


def _compute_SHD(
    mu_P: ndarray,
    Sigma_P: ndarray,
    mu_Q: ndarray,
    Sigma_Q: ndarray,
) -> ndarray:
    # Squared Hellinger distance
    sigma_P: ndarray = sqrt(diag(Sigma_P))
    sigma_Q: ndarray = sqrt(diag(Sigma_Q))
    sum_cov: ndarray = sigma_P**2 + sigma_Q**2
    prod_cov: ndarray = sigma_P * sigma_Q
    return 1.0 - sqrt(2.0 * prod_cov / sum_cov) * exp(
        -0.25 * (mu_P - mu_Q) ** 2 / sum_cov
    )


def _compute_JSD(
    mu_P: ndarray,
    Sigma_P: ndarray,
    mu_Q: ndarray,
    Sigma_Q: ndarray,
    num_samples: int,
) -> ndarray:
    # Jensen-Shannon distance (JSD)
    JSD: ndarray = empty_like(mu_P)
    i: int
    for i in range(mu_P.size):
        RV_p = multivariate_normal(mean=mu_P[i], cov=Sigma_P[i, i])
        RV_q = multivariate_normal(mean=mu_Q[i], cov=Sigma_Q[i, i])
        x: ndarray = RV_p.rvs(num_samples)
        p_x: ndarray = RV_p.pdf(x)
        q_x: ndarray = RV_q.pdf(x)
        m_x: ndarray = (p_x + q_x) / 2.0
        y: ndarray = RV_q.rvs(num_samples)
        p_y: ndarray = RV_p.pdf(y)
        q_y: ndarray = RV_q.pdf(y)
        m_y: ndarray = (p_y + q_y) / 2.0
        dKL_pm: ndarray = ln(p_x / m_x).mean()
        dKL_qm: ndarray = ln(q_y / m_y).mean()
        JSD[i] = 0.5 * (dKL_pm + dKL_qm)
    return JSD


def _NMLL_fct(
    theta: ndarray,
    Z: ndarray,
    A: ndarray,
    L: ndarray,
    num_freqs: int,
    num_taus: int,
) -> ndarray:
    sigma_n: ndarray
    sigma_beta: ndarray
    sigma_lambda: ndarray
    sigma_n, sigma_beta, sigma_lambda = theta
    W: ndarray = (
        1 / (sigma_beta**2) * eye(num_taus + 1) + 1 / (sigma_lambda**2) * L.T @ L
    )
    # W = 0.5 * (W.T + W)
    K_agm: ndarray = 1 / (sigma_n**2) * (A.T @ A) + W
    # K_agm = 0.5 * (K_agm.T + K_agm)
    L_W: ndarray = cholesky(W)
    L_agm: ndarray = cholesky(K_agm)
    # Compute mu_x
    u: ndarray = solve_linalg(L_agm.T, solve_linalg(L_agm, A.T @ Z))
    mu_x: ndarray = 1 / (sigma_n**2) * u
    # Compute loss
    E_mu_x: ndarray = 0.5 / (sigma_n**2) * norm(A @ mu_x - Z) ** 2 + 0.5 * (
        mu_x.T @ (W @ mu_x)
    )
    val_1: ndarray = array_sum(ln(diag(L_W)))
    val_2: ndarray = -array_sum(ln(diag(L_agm)))
    val_3: ndarray = -num_freqs / 2.0 * ln(sigma_n**2)
    val_4: ndarray = -E_mu_x
    val_5: ndarray = -num_freqs / 2 * ln(2 * pi)
    return -(val_1 + val_2 + val_3 + val_4 + val_5)


def _compute_A_re(w: ndarray, tau: ndarray) -> ndarray:
    num_freqs: int = w.shape[0]
    num_taus: int = tau.shape[0]
    A_re: ndarray = zeros(
        (
            num_freqs,
            num_taus + 1,
        )
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


def _compute_A_H_re(w: ndarray, tau: ndarray) -> ndarray:
    return _compute_A_re(w, tau)[:, 1:]


def _compute_A_im(w: ndarray, tau: ndarray) -> ndarray:
    num_freqs: int = w.shape[0]
    num_taus: int = tau.shape[0]
    A_im: ndarray = zeros(
        (
            num_freqs,
            num_taus + 1,
        )
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


def _compute_A_H_im(w: ndarray, tau: ndarray) -> ndarray:
    return _compute_A_im(w, tau)[:, 1:]


def _compute_L(tau: ndarray, derivative_order: int) -> ndarray:
    num_taus: int = tau.shape[0]
    L: ndarray = zeros(
        (
            num_taus - 2,
            num_taus + 1,
        )
    )
    i: int
    delta_loc: float
    if derivative_order == 1:
        for i in range(0, num_taus - 2):
            delta_loc = ln(tau[i + 1] / tau[i])
            factors: ndarray = array([1.0, -2.0, 1.0])
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
        raise Exception(f"Unsupported derivative order: {derivative_order}")
    return L


def _calculate_scores(
    theta_0: ndarray,
    f: ndarray,
    Z_exp: ndarray,
    out_dict_real: dict,
    out_dict_imag: dict,
    num_samples: int,
) -> dict:
    progress_message: str = "Calculating scores"
    num_scores: int = 12
    progress.update_every_N_percent(0, total=num_scores, message=progress_message)
    # scores
    # s_mu - distance between means:
    mu_Z_DRT_re: ndarray = out_dict_real["mu_Z_DRT"]
    mu_Z_H_re: ndarray = out_dict_imag["mu_Z_H"]
    discrepancy_re = norm(mu_Z_DRT_re - mu_Z_H_re) / (
        norm(mu_Z_DRT_re) + norm(mu_Z_H_re)
    )
    s_mu_re: float = 1.0 - discrepancy_re  # type: ignore
    progress.update_every_N_percent(1, total=num_scores, message=progress_message)
    mu_Z_DRT_im: ndarray = out_dict_imag["mu_Z_DRT"]
    mu_Z_H_im: ndarray = out_dict_real["mu_Z_H"]
    discrepancy_im = norm(mu_Z_DRT_im - mu_Z_H_im) / (
        norm(mu_Z_DRT_im) + norm(mu_Z_H_im)
    )
    s_mu_im: float = 1.0 - discrepancy_im  # type: ignore
    progress.update_every_N_percent(2, total=num_scores, message=progress_message)
    # s_JSD - Jensen-Shannon Distance:
    # we need the means (above) and covariances (below)
    # for the computation of the JSD
    Sigma_Z_DRT_re: ndarray = out_dict_real["Sigma_Z_DRT"]
    Sigma_Z_DRT_im: ndarray = out_dict_imag["Sigma_Z_DRT"]
    Sigma_Z_H_re: ndarray = out_dict_imag["Sigma_Z_H"]
    Sigma_Z_H_im: ndarray = out_dict_real["Sigma_Z_H"]
    # s_res - residual score:
    # real part
    # retrieve distribution of R_inf
    mu_R_inf: float = out_dict_real["mu_gamma"][0]
    cov_R_inf: ndarray = diag(out_dict_real["Sigma_gamma"])[0]
    # we will also need omega an estimate of the error
    sigma_n_im: ndarray = out_dict_imag["theta"][0]
    # R_inf+Z_H_re-Z_exp has:
    # mean:
    res_re: ndarray = mu_R_inf + mu_Z_H_re - Z_exp.real
    # std:
    band_re: ndarray = sqrt(cov_R_inf + diag(Sigma_Z_H_re) + sigma_n_im**2)
    s_res_re: ndarray = _compute_res_scores(res_re, band_re)
    progress.update_every_N_percent(5, total=num_scores, message=progress_message)
    # imaginary part
    # retrieve distribution of L_0
    mu_L_0: float = out_dict_imag["mu_gamma"][0]
    cov_L_0: ndarray = diag(out_dict_imag["Sigma_gamma"])[0]
    # we will also need omega
    omega_vec: ndarray = 2.0 * pi * f
    # and an estimate of the error
    sigma_n_re: ndarray = out_dict_real["theta"][0]
    # R_inf+Z_H_re-Z_exp has:
    # mean:
    res_im: ndarray = omega_vec * mu_L_0 + mu_Z_H_im - Z_exp.imag
    # std:
    band_im: ndarray = sqrt(
        (omega_vec**2) * cov_L_0 + diag(Sigma_Z_H_im) + sigma_n_re**2
    )
    s_res_im: ndarray = _compute_res_scores(res_im, band_im)
    progress.update_every_N_percent(8, total=num_scores, message=progress_message)
    # Squared Hellinger distance (SHD)
    # which is bounded between 0 and 1
    # we are going to score w.r.t. the Hellinger distance (HD)
    # the score uses 1 to mean good (this means close)
    # and 0 means bad (far away) => that's the opposite of the distance
    SHD_re: ndarray = _compute_SHD(mu_Z_DRT_re, Sigma_Z_DRT_re, mu_Z_H_re, Sigma_Z_H_re)
    s_HD_re: float = 1.0 - sqrt(SHD_re).mean()
    progress.update_every_N_percent(9, total=num_scores, message=progress_message)
    SHD_im: ndarray = _compute_SHD(mu_Z_DRT_im, Sigma_Z_DRT_im, mu_Z_H_im, Sigma_Z_H_im)
    s_HD_im: float = 1.0 - sqrt(SHD_im).mean()
    progress.update_every_N_percent(10, total=num_scores, message=progress_message)
    # compute the Jensen-Shannon distance (JSD)
    # the JSD is a symmetrized relative entropy (discrepancy), so highest value means more entropy
    # we are going to reverse that by taking (ln(2)-JSD)/ln(2)
    # which means higher value less relative entropy (discrepancy)
    JSD_re: ndarray = _compute_JSD(
        mu_Z_DRT_re,
        Sigma_Z_DRT_re,
        mu_Z_H_re,
        Sigma_Z_H_re,
        num_samples,
    )
    s_JSD_re: float = (ln(2) - JSD_re.mean()) / ln(2)
    progress.update_every_N_percent(11, total=num_scores, message=progress_message)
    JSD_im: ndarray = _compute_JSD(
        mu_Z_DRT_im,
        Sigma_Z_DRT_im,
        mu_Z_H_im,
        Sigma_Z_H_im,
        num_samples,
    )
    s_JSD_im: float = (ln(2) - JSD_im.mean()) / ln(2)
    progress.update_every_N_percent(12, total=num_scores, message=progress_message)
    return {
        "hellinger_distance": complex(s_HD_re, s_HD_im),
        "jensen_shannon_distance": complex(s_JSD_re, s_JSD_im),
        "mean": complex(s_mu_re, s_mu_im),
        "residuals_1sigma": complex(s_res_re[0], s_res_im[0]),
        "residuals_2sigma": complex(s_res_re[1], s_res_im[1]),
        "residuals_3sigma": complex(s_res_re[2], s_res_im[2]),
    }


def _single_hilbert_transform_estimate(
    theta_0: ndarray,
    Z_exp: ndarray,
    A: ndarray,
    A_H: ndarray,
    L: ndarray,
    num_freqs: int,
    num_taus: int,
):
    fp: IO
    with open(devnull, "w") as fp:
        with redirect_stdout(fp):
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
    W: ndarray = (
        1 / (sigma_beta**2) * eye(num_taus + 1) + 1 / (sigma_lambda**2) * L.T @ L
    )
    K_agm: ndarray = 1 / (sigma_n**2) * (A.T @ A) + W
    # Cholesky factorization
    L_agm: ndarray = cholesky(K_agm)
    inv_L_agm: ndarray = invert(L_agm)
    inv_K_agm: ndarray = inv_L_agm.T @ inv_L_agm
    # Compute the gamma ~ N(mu_gamma, Sigma_gamma)
    Sigma_gamma: ndarray = inv_K_agm
    mu_gamma: ndarray = 1 / (sigma_n**2) * (Sigma_gamma @ A.T) @ Z_exp.real
    # Compute, from gamma, the Z ~ N(mu_Z, Sigma_Z)
    mu_Z: ndarray = A @ mu_gamma
    Sigma_Z: ndarray = A @ (Sigma_gamma @ A.T) + sigma_n**2 * eye(num_freqs)
    # Compute, from gamma, the Z_DRT ~ N(mu_Z_DRT, Sigma_Z_DRT)
    A_DRT: ndarray = A[:, 1:]
    mu_gamma_DRT: ndarray = mu_gamma[1:]
    Sigma_gamma_DRT: ndarray = Sigma_gamma[1:, 1:]
    mu_Z_DRT: ndarray = A_DRT @ mu_gamma_DRT
    Sigma_Z_DRT: ndarray = A_DRT @ (Sigma_gamma_DRT @ A_DRT.T)
    # Compute, from gamma, the Z_H_conj ~ N(mu_Z_H_conj, Sigma_Z_H_conj)
    mu_Z_H: ndarray = A_H @ mu_gamma[1:]
    Sigma_Z_H: ndarray = A_H @ (Sigma_gamma[1:, 1:] @ A_H.T)
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
    result: Tuple[float, ndarray, dict, dict],
    tau_fine: ndarray,
    tau: ndarray,
    epsilon: float,
    rbf_type: str,
) -> float:
    # Calculate gamma for each result and check if it is asymmetrical
    chisqr: float
    theta_0: ndarray
    data_real: dict
    data_imag: dict
    chisqr, theta_0, data_real, data_imag = result
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
) -> Optional[Tuple[float, ndarray, dict, dict]]:
    theta_0: ndarray
    w: ndarray
    Z: ndarray
    A_re: ndarray
    A_im: ndarray
    A_H_re: ndarray
    A_H_im: ndarray
    L: ndarray
    num_freqs: int
    num_taus: int
    maximum_symmetry: float
    tau_fine: ndarray
    tau: ndarray
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
        mu_Z_H_re: ndarray = data_imag["mu_Z_H"]
        mu_L_0: float = data_imag["mu_gamma"][0]
        mu_Z_H_im: ndarray = data_real["mu_Z_H"]
        Z_fit: ndarray = array(
            list(
                map(
                    lambda _: complex(*_),
                    zip(
                        mu_R_inf + mu_Z_H_re,
                        w * mu_L_0 + mu_Z_H_im,
                    ),
                )
            )
        )
        z: complex = array_sum((Z_fit - Z) ** 2 / Z)
        chisqr: float = sqrt(z.real**2 + z.imag**2)
        result: Tuple[float, ndarray, dict, dict] = (
            chisqr,
            theta_0,
            data_real,
            data_imag,
        )
        if (
            _calculate_symmetry_score(
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
            chisqr,
            theta_0,
            data_real,
            data_imag,
        )
    except Exception:
        return None


def _calculate_drt_bht(
    data: DataSet,
    rbf_type: str,
    derivative_order: int,
    rbf_shape: str,
    shape_coeff: float,
    num_samples: int,
    num_attempts: int,
    maximum_symmetry: float,
    num_procs: int,
) -> DRTResult:
    assert (
        hasattr(data, "get_frequency")
        and callable(data.get_frequency)
        and hasattr(data, "get_impedance")
        and callable(data.get_impedance)
    ), data
    assert type(rbf_type) is str and rbf_type in RBF_TYPES, rbf_type
    assert (
        issubdtype(type(derivative_order), integer) and 1 <= derivative_order <= 2
    ), derivative_order
    assert type(rbf_shape) is str and rbf_shape in RBF_SHAPES, rbf_shape
    assert (
        issubdtype(
            type(shape_coeff),
            floating,
        )
        and shape_coeff > 0.0
    ), shape_coeff
    assert issubdtype(type(num_samples), integer) and num_samples > 0, num_samples
    assert issubdtype(type(num_attempts), integer) and num_attempts >= 1, num_attempts
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
    f: ndarray = data.get_frequency()
    Z: ndarray = data.get_impedance()
    tau: ndarray = 1 / f
    tau_fine: ndarray = logspace(
        log(tau.min()) - 0.5,
        log(tau.max()) + 0.5,
        10 * f.shape[0],
    )
    w: ndarray = 2 * pi * f
    num_freqs: int = f.shape[0]
    num_taus: int = tau.shape[0]
    epsilon: float = _compute_epsilon(
        f,
        rbf_shape,
        shape_coeff,
        rbf_type,
    )
    A_re: ndarray = _compute_A_re(w, tau)
    A_im: ndarray = _compute_A_im(w, tau)
    A_H_re: ndarray = _compute_A_H_re(w, tau)
    A_H_im: ndarray = _compute_A_H_im(w, tau)
    L: ndarray = _compute_L(tau, derivative_order)
    theta_0_generator: Callable = lambda: 10 ** (6 * rand(3, 1) - 3)
    with Pool(num_procs) as pool:
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
        results: List[Tuple[float, ndarray, dict, dict]] = []
        progress_message: str = "Calculating Hilbert transforms"
        progress.update_every_N_percent(0, total=num_attempts, message=progress_message)
        for i, res in enumerate(
            pool.imap_unordered(
                _hilbert_transform_process,
                args,
            )
        ):
            progress.update_every_N_percent(
                i + 1,
                total=num_attempts,
                message=progress_message,
            )
            if res is not None:
                results.append(res)
    if len(results) == 0:
        raise DRTError("Failed to perform calculations! Try tweaking the settings.")
    results.sort(key=lambda _: _[0])
    theta_0: ndarray
    data_real: dict
    data_imag: dict
    _, theta_0, data_real, data_imag = results[0]
    # Scores seem to be fine based on comparison with the ZARC example used in the article
    scores: dict = _calculate_scores(
        theta_0,
        f,
        Z,
        data_real,
        data_imag,
        num_samples,
    )
    # Real part
    mu_Z_re: ndarray = data_real["mu_Z"]
    cov_Z_r: ndarray = diag(data_real["Sigma_Z"])
    mu_R_inf: float = data_real["mu_gamma"][0]
    cov_R_inf: ndarray = diag(data_real["Sigma_gamma"])[0]
    mu_Z_DRT_re: ndarray = data_real["mu_Z_DRT"]
    cov_Z_DRT_re: ndarray = diag(data_real["Sigma_Z_DRT"])
    mu_Z_H_im: ndarray = data_real["mu_Z_H"]
    cov_Z_H_im: ndarray = diag(data_real["Sigma_Z_H"])
    sigma_n_re = data_real["theta"][0]
    # Imaginary part
    mu_Z_im: ndarray = data_imag["mu_Z"]
    cov_Z_im: ndarray = diag(data_imag["Sigma_Z"])
    mu_L_0: float = data_imag["mu_gamma"][0]
    cov_L_0: ndarray = diag(data_imag["Sigma_gamma"])[0]
    mu_Z_DRT_im: ndarray = data_imag["mu_Z_DRT"]
    cov_Z_DRT_im: ndarray = diag(data_imag["Sigma_Z_DRT"])
    mu_Z_H_re: ndarray = data_imag["mu_Z_H"]
    cov_Z_H_re: ndarray = diag(data_imag["Sigma_Z_H"])
    sigma_n_im: float = data_imag["theta"][0]
    # Means and bounds
    mu_Z_H_re_agm: ndarray = mu_R_inf + mu_Z_H_re
    mu_Z_H_im_agm: ndarray = w * mu_L_0 + mu_Z_H_im
    # band_agm_re = sqrt(cov_R_inf + cov_Z_H_re + sigma_n_im**2)
    # band_agm_im = sqrt((w**2) * cov_L_0 + cov_Z_H_im + sigma_n_re**2)
    Z_fit: ndarray = array(
        list(
            map(
                lambda _: complex(*_),
                zip(
                    mu_Z_H_re_agm,
                    mu_Z_H_im_agm,
                ),
            )
        )
    )
    return DRTResult(
        "BHT",
        *_x_to_gamma(
            data_real["mu_gamma"][1:],
            tau_fine,
            tau,
            epsilon,
            rbf_type,
        ),
        f,
        Z_fit,
        *_calculate_residuals(Z, Z_fit),
        # "tr-rbf" method with credible_intervals
        array([]),  # Mean
        array([]),  # Lower bound
        array([]),  # Upper bound
        # "bht" method
        _x_to_gamma(
            data_imag["mu_gamma"][1:],
            tau_fine,
            tau,
            epsilon,
            rbf_type,
        )[1],
        scores,
        # Stats
        _calculate_chisqr(Z, Z_fit),
        -1.0,
    )
