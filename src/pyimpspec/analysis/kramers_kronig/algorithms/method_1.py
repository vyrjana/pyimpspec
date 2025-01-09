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

from numpy import (
    array,
    ceil,
    float64,
    inf,
    isclose,
    isnan,
    log10 as log,
    nan,
)
from numpy.typing import NDArray
from pyimpspec.analysis.kramers_kronig.result import KramersKronigResult
from pyimpspec.circuit.circuit import Circuit
from pyimpspec.typing.helpers import (
    Dict,
    List,
    Tuple,
    _is_floating,
)
from .utility.mu import _calculate_mu
from .utility.common import (
    _generate_pseudo_chisqr_offsets,
    _truncate_circuits,
)
from .utility.logistic import (
    _logistic_derivative,
    _logistic_function,
)


def _calculate_score(
    circuit: Circuit,
    pseudo_chisqr: float,
    mu_criterion: float = 0.85,
    beta: float = 0.75,
) -> float:
    r"""
    Calculate a score based on the |mu| and |pseudo chi-squared| values of the fitted circuit, and the provided |mu crit|:

    :math:`S = \frac{-log{\chi^2_{\rm ps}}}{\left(\mu_{\rm crit} - \mu\right)^\beta}`

    If a circuit has :math:`\mu \geq \mu_{\rm crit}` or if |mu| is |numpy.nan|, then |numpy.nan| is returned.
    This score is not a part of the original algorithm proposed by Schönleber et al. (2014), but rather a measure for avoiding premature terminations that can occur in the original algorithm if, e.g., |mu| fluctuates too much at low numbers of RC elements.
    The equation above rewards circuits that have a good fit (i.e., a low |pseudo chi-squared| value) and penalizes large differences between |mu crit| and |mu|.
    The value of |beta| is determined heuristically and 0.75 seems to work well.

    References:

    - `V. Yrjänä and J. Bobacka, 2024, Electrochim. Acta, 504, 144951 <https://doi.org/10.1016/j.electacta.2024.144951>`_
    - `M. Schönleber, D. Klotz, and E. Ivers-Tiffée, 2014, Electrochim. Acta, 131, 20-27 <https://doi.org/10.1016/j.electacta.2014.01.034>`_

    Parameters
    ----------
    circuit: Circuit
        The circuit that was fitted as part of the Kramers-Kronig test.

    pseudo_chisqr: float
        The pseudo chi-squared corresponding to the fitted circuit.

    mu_criterion: float, optional
        The |mu crit| to apply.

    beta: float, optional
        The exponent used in the denominator.

    Returns
    -------
    float
    """
    mu: float = _calculate_mu(circuit)

    if isnan(mu) or mu >= mu_criterion:
        return nan

    return -log(pseudo_chisqr) / ((mu_criterion - mu) ** beta)


def _fit_logistic_function(tests: List[KramersKronigResult]) -> Tuple[float, ...]:
    from scipy.optimize import curve_fit

    x: NDArray[float64] = array([t.num_RC for t in tests], dtype=float64)
    y: NDArray[float64] = array(
        [
            mu if not isnan(mu) else 1.0
            for mu in (_calculate_mu(t.circuit) for t in tests)
        ]
    )
    a: float = max(y) - min(y)
    d: float = max(y) - a

    p: Tuple[float, ...] = curve_fit(
        lambda x, b, c: _logistic_function(x, a, b, c, d),
        x,
        y,
        p0=(1.0, 10),
        bounds=(
            [-inf, min(x) + 1],
            [inf, max(x) - 1],
        ),
    )[0]

    return tuple((a, *p, d))


def suggest(
    tests: List[KramersKronigResult],
    lower_limit: int = 0,
    upper_limit: int = 0,
    mu_criterion: float = 0.85,
    beta: float = 0.75,
    relative_scores: bool = True,
) -> Dict[int, float]:
    """
    The value |mu| describes the ratio of the total mass of negative resistances to the total mass of positive resistances:

    :math:`\\mu = 1 - \\frac{\\Sigma_{R_k < 0} |R_k|}{\\Sigma_{R_k \\geq 0} |R_k|}`

    |mu| ranges from 0.0 to 1.0 and these extremes represent overfitting and underfitting, respectively.
    Overfitting manifests as an oscillating fitted impedance spectrum, which is made possible by a mix of positive and negative resistances.
    The number of RC elements is incremented until the corresponding |mu| drops below the threshold |mu crit|.

    The first modification is to adapt the equation above for use with validation of immittance data in the admittance representation:

    :math:`\\mu = 1 - \\frac{\\Sigma_{C_k < 0} |C_k|}{\\Sigma_{C_k \\geq 0} |C_k|}`

    The denominator can be less than one, so the calculated values are clamped to only range from 0.0 to 1.0.

    The second modification is that the iteration is done in reverse (i.e., by decrementing the number of RC elements instead of incrementing it) since there can be significant fluctuation of |mu| at low numbers of RC elements, which can cause the iterative process to stop too early.

    The third modification is to calculate an additional score, :math:`S`, as follows:

    :math:`S = \\frac{-\\log{\\chi^2_{\\rm ps}}}{{\\left(\\mu_{\\rm crit} - \\mu\\right)}^{\\beta}}`

    The exponent |beta| is determined heuristically and a value of 0.75 seems to work well.
    Only |mu| values less than |mu crit| are considered when calculating :math:`S`.
    The use of this score helps to deal with the fluctuation that affects the use of |mu| directly.
    If |beta| is set to zero, then the second and third modification are skipped.

    If |mu| is negative, then an alternative approach is used whereby a logistic function is fitted to a plot of |mu| versus the number of RC elements.
    The intercept (rounded up) of the slope at the midpoint of that function and a line at the highest point of the function is used to pick the optimal number of RC elements.

    The returned dictionary maps the number of RC elements to a score ranging from 0.0 to 1.0 with the latter representing the highest-ranking candidate.

    References:

    - `M. Schönleber, D. Klotz, and E. Ivers-Tiffée, 2014, Electrochim. Acta, 131, 20-27 <https://doi.org/10.1016/j.electacta.2014.01.034>`_
    - `V. Yrjänä and J. Bobacka, 2024, Electrochim. Acta, 504, 144951 <https://doi.org/10.1016/j.electacta.2024.144951>`_

    Parameters
    ----------
    tests: List[|KramersKronigResult|]
        The test results to evaluate.

    lower_limit: int, optional
        The lower limit to enforce for the number of RC elements.
        If this value is less than one, then no limit is enforced.
        If both the lower and upper limit are greater than zero, then the lower limit must have a smaller value than the upper limit.

    upper_limit: int, optional
        The upper limit to enforce for the number of RC elements.
        If this value is less than one, then no limit is enforced.
        If both the lower and upper limit are greater than zero, then the upper limit must have a greater value than the lower limit.

    mu_criterion: float, optional
        The |mu crit| to apply.
        Schönleber et al. (2014) recommended 0.85 based on their experiences.
        If a value less than zero is provided, then the alternative approach based on fitting a logistic function to pick the optimal number of RC elements is used.

    beta: float, optional
        Exponent used to tweak the influence of the proximity of |mu| to |mu crit| when calculating scores.
        If set to zero, then the iteration direction is not reversed and the score :math:`S` is not calculated.

    relative_scores: bool, optional
        Return relative scores ranging from 0.0 to 1.0 (from worst to best) rather than the raw values.

    Returns
    -------
    Dict[int, float]

        A dictionary mapping the number of RC elements to its corresponding score.
    """
    if not _is_floating(mu_criterion):
        raise TypeError(f"Expected a float instead of {mu_criterion=}")
    elif isclose(mu_criterion, 0.0):
        raise ValueError(f"Expected {mu_criterion=} != 0.0")
    elif not ((0.0 < mu_criterion < 1.0) or (-1.0 <= mu_criterion < 0.0)):
        raise ValueError(
            f"Expected (0.0 < {mu_criterion=} < 1.0) or (-1.0 < {mu_criterion=} < 0.0)"
        )

    if not _is_floating(beta):
        raise TypeError(f"Expected a float instead of {beta=}")
    elif beta < 0.0:
        raise ValueError(
            f"Expected a value greater than or equal to zero instead of {beta=}"
        )

    circuits: Dict[int, Circuit] = {t.num_RC: t.circuit for t in tests}
    circuits = _truncate_circuits(circuits, lower_limit, upper_limit)
    scores: Dict[int, float]

    # Original approach
    if isclose(beta, 0.0):
        if not relative_scores:
            scores = {
                num_RC: (mu if not isnan(mu) else 1.0)
                for num_RC, mu in {
                    num_RC: _calculate_mu(circuit)
                    for num_RC, circuit in circuits.items()
                }.items()
            }

            if any(map(isnan, scores.values())):
                raise ValueError(
                    f"Expected values that are not NaN instead of {scores=}"
                )

            return {
                num_RC: (value if not isnan(value) else 1.0)
                for num_RC, value in scores.items()
            }

        scores = {
            num_RC: (mu if not (isnan(mu) or mu >= mu_criterion) else nan)
            for num_RC, mu in {
                num_RC: _calculate_mu(circuit) for num_RC, circuit in circuits.items()
            }.items()
        }
        if all(map(isnan, scores.values())):
            # None of the mu values are below the chosen threshold so pick the
            # highest number of RC elements.
            return {
                num_RC: (1.0 if num_RC == max(circuits.keys()) else 0.0)
                for num_RC in circuits
            }

        # Pick the first instance where the mu value drops below the chosen
        # threshold.
        lowest_num_RC: int = min(
            (num_RC for num_RC, value in scores.items() if not isnan(value))
        )
        return {
            num_RC: (1.0 if num_RC == lowest_num_RC else 0.0)
            for num_RC in scores.keys()
        }

    pseudo_chisqrs: Dict[int, float] = {
        t.num_RC: t.pseudo_chisqr for t in tests if t.num_RC in circuits
    }

    # Modified approach 1
    if mu_criterion < 0.0:
        p: Tuple[float, ...] = _fit_logistic_function(tests)
        slope: float = _logistic_derivative(p[2], *p)
        intercept: float = _logistic_function(p[2], *p) - slope * p[2]
        if not relative_scores:
            return {
                num_RC: _logistic_function(num_RC, *p) for num_RC in circuits.keys()
            }

        offset_factor: float = 1e-6
        offsets: Dict[int, float] = _generate_pseudo_chisqr_offsets(
            pseudo_chisqrs, factor=offset_factor
        )

        target_num_RC: float = ceil((intercept - (p[0] + p[3])) / (0.0 - slope))
        scores = {
            num_RC: abs(target_num_RC - num_RC) + (offset_factor - offsets[num_RC])
            for num_RC in circuits.keys()
        }
        min_score: float = min(scores.values())
        max_score: float = max(scores.values()) - min_score

        scores = {
            num_RC: 1.0 - (value - min_score) / max_score
            for num_RC, value in scores.items()
        }

        return scores

    # Modified approach 2
    scores = {
        num_RC: _calculate_score(
            circuit,
            pseudo_chisqrs[num_RC],
            mu_criterion=mu_criterion,
            beta=beta,
        )
        for num_RC, circuit in circuits.items()
    }
    if all(map(isnan, scores.values())):
        # None of the mu values are below the chosen threshold so pick the
        # highest number of RC elements.
        return {
            num_RC: (1.0 if num_RC == max(circuits.keys()) else 0.0)
            for num_RC in circuits
        }

    # Iterative in reverse and pick the highest number of RC elements before
    # the mu values rise above the chosen threshold.
    num_RCs: List[int] = sorted(circuits.keys())
    i: int = len(num_RCs)
    while i > 0:
        i -= 1
        score: float = scores[num_RCs[i]]
        if isnan(score):
            i += 1
            break

    if i >= len(num_RCs):
        i = len(num_RCs) - 1

    non_nan_scores: List[float] = [
        scores[num_RC] for num_RC in num_RCs[i:] if not isnan(scores[num_RC])
    ]
    if len(non_nan_scores) == 0:
        return {num_RC: 0.0 for num_RC in scores.keys()}

    if not relative_scores:
        return {
            num_RC: (
                (scores[num_RC] if not isnan(scores[num_RC]) else 0.0)
                if j >= i
                else 0.0
            )
            for j, num_RC in enumerate(num_RCs)
        }

    min_score = min(non_nan_scores)
    max_score = max(non_nan_scores) - min_score
    if max_score == 0.0:
        min_score = 0.0
        max_score = max(non_nan_scores)

    return {
        num_RC: float(((scores[num_RC] - min_score) / max_score) if j >= i else 0.0)
        for j, num_RC in enumerate(num_RCs)
    }
