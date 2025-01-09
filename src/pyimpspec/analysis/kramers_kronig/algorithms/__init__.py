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
    diff,
    exp,
    float64,
    isclose,
    log10 as log,
    mean,
)
from numpy.typing import NDArray
from pyimpspec.analysis.kramers_kronig.result import KramersKronigResult
from pyimpspec.circuit.circuit import Circuit
from pyimpspec.typing import Frequencies
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
from .method_1 import suggest as suggest_num_RC_method_1
from .method_2 import suggest as suggest_num_RC_method_2
from .method_3 import suggest as suggest_num_RC_method_3
from .method_4 import suggest as suggest_num_RC_method_4
from .method_5 import suggest as suggest_num_RC_method_5
from .method_6 import suggest as suggest_num_RC_method_6
from .representation import suggest as suggest_representation
from .utility.pseudo_chi_squared import _approximate_transition_and_end_point
from .utility.common import subdivide_frequencies
from .utility.osculating_circle import calculate_curvatures


def suggest_num_RC_limits(
    tests: List[KramersKronigResult],
    lower_limit: int = 0,
    upper_limit: int = 0,
    limit_delta: int = 0,
    threshold: float = 4.0,
) -> Tuple[int, int]:
    """
    Suggest lower and upper limits for the range of number of time constants where the optimal number of time constants should be looked for.
    The lower limit is the point at which incrementing the number of time constants no longer significantly improves the fit.
    The upper limit is suggested based on the mean distances between sign changes of the curvatures and when those drop below a threshold.

    References:

    - `V. Yrjänä and J. Bobacka, 2024, Electrochim. Acta, 504, 144951 <https://doi.org/10.1016/j.electacta.2024.144951>`_

    Parameters
    ----------
    tests: List[|KramersKronigResult|]
        The test results to evaluate.

    lower_limit: int, optional
        The lower limit to enforce if this value is greater than zero.
        If this value is set to zero, then the lower limit is automatically estimated.
        If this value is less than zero, then no lower limit is enforced.

    upper_limit: int, optional
        The upper limit to enforce if this value is greater than zero.
        If this value is set to zero, then the upper limit is automatically estimated.
        If this value is less than zero, then no upper limit is enforced.

    limit_delta: int, optional
        Alternative way of defining the upper limit as lower limit + delta.
        Only used if the value is greater than zero.

    threshold: float, optional
        The threshold for the mean distance between curvature sign changes.
        This value is used when estimating the upper limit.

    Returns
    -------
    Tuple[int, int]

        The suggested lower and upper limits for the number of RC elements where the optimal number of RC elements is likely to exist. Values outside these limits are likely to result in under- or overfitting.
    """
    manually_defined_lower_limit: bool = lower_limit > 0
    manually_defined_upper_limit: bool = upper_limit > 0

    if manually_defined_lower_limit and manually_defined_upper_limit:
        if limit_delta > 0:
            return (
                max((lower_limit, tests[0].num_RC)),
                min((lower_limit + limit_delta, upper_limit, tests[-1].num_RC)),
            )

        return (
            max((lower_limit, tests[0].num_RC)),
            min((upper_limit, tests[-1].num_RC)),
        )

    f: Frequencies = tests[0].get_frequencies()
    x: NDArray[float64]
    if tests[0].test == "complex":
        x = array([t.num_RC for t in tests], dtype=float64)
    else:
        x = array([t.num_RC for t in tests if t.num_RC <= len(f)], dtype=float64)

    min_x: int = int(min(x))
    max_x: int = int(max(x))

    y: NDArray[float64] = log([t.pseudo_chisqr for t in tests if t.num_RC <= max_x])
    possibly_single_resistor_or_capacitor: bool = (
        min_x == 2 and (diff(x) == 1).all() and (y[:5] < -2).all()
    )

    if not manually_defined_lower_limit:
        if possibly_single_resistor_or_capacitor:
            lower_limit = min_x
        else:
            lower_limit, max_x, _ = _approximate_transition_and_end_point(x, y)

        lower_limit = max((min_x, lower_limit))
        if manually_defined_upper_limit:
            lower_limit = max((min_x, lower_limit - 1))

    if not manually_defined_upper_limit:
        if possibly_single_resistor_or_capacitor:
            upper_limit = min((max_x, len(f)))
        else:
            if lower_limit >= max_x:
                lower_limit = max_x - 1
                upper_limit = max_x
            else:
                mean_distances: Dict[int, float] = suggest_num_RC_method_5(
                    tests,
                    lower_limit=lower_limit,
                    upper_limit=0,
                    relative_scores=False,
                )

                for upper_limit, value in reversed(mean_distances.items()):
                    if upper_limit <= max_x and value >= threshold:
                        upper_limit = min(
                            (
                                max_x,
                                max((lower_limit + (limit_delta or 1), upper_limit)),
                            )
                        )
                        break
                else:
                    upper_limit = min((max_x, lower_limit + (limit_delta or 1)))

        if upper_limit <= lower_limit:
            upper_limit = min((max_x, lower_limit + 1))

    if upper_limit <= lower_limit:
        if upper_limit >= max_x:
            lower_limit = max((min_x, upper_limit - (limit_delta or 1)))
        else:
            upper_limit = min((max_x, lower_limit + (limit_delta or 1)))

    if upper_limit <= lower_limit:
        raise ValueError(f"Expected {lower_limit=} < {upper_limit=}")

    if (
        not manually_defined_lower_limit
        and lower_limit > min_x
        and min(tests, key=lambda t: t.num_RC).num_RC < lower_limit
    ):
        best_fit_below_lower_limit: KramersKronigResult = min(
            [t for t in tests if t.num_RC < lower_limit],
            key=lambda t: t.pseudo_chisqr,
        )
        lower_limit_fit: KramersKronigResult = [
            t for t in tests if t.num_RC == lower_limit
        ][0]
        if best_fit_below_lower_limit.pseudo_chisqr < lower_limit_fit.pseudo_chisqr:
            lower_limit = best_fit_below_lower_limit.num_RC

    if limit_delta > 0:
        upper_limit = min((lower_limit + limit_delta, max_x))

    return (lower_limit, upper_limit)


def _choose_methods(
    tests: List[KramersKronigResult],
    lower_limit: int,
    upper_limit: int,
    limit_delta: int,
    methods: List[int],
    **kwargs,
) -> Tuple[Dict[int, Callable], int, int]:
    lower_limit, upper_limit = suggest_num_RC_limits(
        tests,
        lower_limit,
        upper_limit,
        limit_delta,
    )
    if lower_limit >= upper_limit:
        raise ValueError(f"Expected {lower_limit=} < {upper_limit=}")

    algorithms: Dict[int, Callable] = {
        1: lambda: suggest_num_RC_method_1(
            tests,
            lower_limit=lower_limit,
            upper_limit=upper_limit,
            mu_criterion=kwargs.get("mu_criterion", 0.85),
            beta=kwargs.get("beta", 0.75),
        ),
        2: lambda: suggest_num_RC_method_2(
            tests,
            lower_limit=lower_limit,
            upper_limit=upper_limit,
        ),
        3: lambda: suggest_num_RC_method_3(
            tests,
            lower_limit=lower_limit,
            upper_limit=upper_limit,
        ),
        4: lambda: suggest_num_RC_method_4(
            tests,
            lower_limit=lower_limit,
            upper_limit=upper_limit,
        ),
        5: lambda: suggest_num_RC_method_5(
            tests,
            lower_limit=lower_limit,
            upper_limit=upper_limit,
        ),
        6: lambda: suggest_num_RC_method_6(
            tests,
            lower_limit=lower_limit,
            upper_limit=upper_limit,
        ),
    }

    selection: Dict[int, Callable] = {
        k: v for k, v in algorithms.items() if k in methods
    }
    if len(selection) == 0:
        if len(methods) == 0:
            raise ValueError(f"Unsupported suggestion method(s): {methods=}")
        selection = algorithms

    return (selection, lower_limit, upper_limit)


def _suggest_using_mean(
    tests: List[KramersKronigResult],
    methods: Dict[int, Callable],
) -> Tuple[KramersKronigResult, Dict[int, float]]:
    """
    Take the highest-ranking test result from each method and then pick the
    test result closest to the mean number of RC elements.
    """
    total_scores: Dict[int, float] = {t.num_RC: 0.0 for t in tests}
    suggestions: List[int] = []

    m: int
    func: Callable
    for m, func in methods.items():
        d: Dict[int, float] = func()
        if not isinstance(d, dict):
            raise TypeError(f"Expected a dictionary instead of {d=}")
        elif len(d) == 0:
            continue

        if not all(map(lambda k: _is_integer(k), d.keys())):
            raise TypeError(f"Expected all keys to be integers instead of {d.keys()=}")
        elif not all(map(lambda k: k > 0, d.keys())):
            raise ValueError(
                f"Expected all keys to be integers greater than zero instead of {d.keys()=}"
            )
        elif not all(map(lambda v: _is_floating(v), d.values())):
            raise TypeError(
                f"Expected all values to be floats instead of {d.values()=}"
            )
        elif not all(map(lambda v: 0.0 <= v <= 1.0, d.values())):
            raise ValueError(
                f"Expected all values to be floats in the range [0.0, 1.0] instead of {d.values()=}"
            )

        suggestions.append(sorted(d.items(), key=lambda kv: kv[1], reverse=True)[0][0])

    i: int
    for i in range(min(suggestions), max(suggestions) + 1):
        total_scores[i] += 1.0 * suggestions.count(i)

    num_RC: int = int(round(mean(suggestions)))

    return (
        sorted(
            tests,
            key=lambda t: (abs(t.num_RC - num_RC), t.pseudo_chisqr),
        )[0],
        total_scores,
    )


def _suggest_using_ranking(
    tests: List[KramersKronigResult],
    methods: Dict[int, Callable],
) -> Tuple[KramersKronigResult, Dict[int, float]]:
    """
    Assign scores to the test results after ranking them according to different
    approaches. Pick the overall highest-scoring test result.
    """
    total_scores: Dict[int, float] = {t.num_RC: 0.0 for t in tests}
    a: float = 1.0
    b: float = 1.0

    m: int
    func: Callable
    for m, func in methods.items():
        d: Dict[int, float] = func()
        if not isinstance(d, dict):
            raise TypeError(f"Expected a dictionary instead of {d=}")
        elif len(d) == 0:
            continue

        if not all(map(lambda k: _is_integer(k), d.keys())):
            raise TypeError(f"Expected all keys to be integers instead of {d.keys()=}")
        elif not all(map(lambda k: k > 0, d.keys())):
            raise ValueError(
                f"Expected all keys to be integers greater than zero instead of {d.keys()=}"
            )
        elif not all(map(lambda v: _is_floating(v), d.values())):
            raise TypeError(
                f"Expected all values to be floats instead of {d.values()=}"
            )
        elif not all(map(lambda v: 0.0 <= v <= 1.0, d.values())):
            raise ValueError(
                f"Expected all values to be floats in the range [0.0, 1.0] instead of {d.values()=}"
            )

        i: int
        num_RC: int
        score: float
        for i, (num_RC, score) in enumerate(
            sorted(d.items(), key=lambda kv: kv[1], reverse=True)
        ):
            total_scores[num_RC] += a * exp(-b * i)

    return (
        sorted(
            tests,
            key=lambda t: (total_scores.get(t.num_RC, 0.0), -log(t.pseudo_chisqr)),
            reverse=True,
        )[0],
        total_scores,
    )


def _suggest_using_sum(
    tests: List[KramersKronigResult],
    methods: Dict[int, Callable],
) -> Tuple[KramersKronigResult, Dict[int, float]]:
    """
    Each method returns relative scores (0.0 to 1.0) that are added together.
    Overlapping suggestions end up with higher total scores and the highest-
    scoring result is chosen.
    """
    total_scores: Dict[int, float] = {t.num_RC: 0.0 for t in tests}

    m: int
    func: Callable
    for m, func in methods.items():
        d: Dict[int, float] = func()
        if not isinstance(d, dict):
            raise TypeError(f"Expected a dictionary instead of {d=}")
        elif len(d) == 0:
            continue

        if not all(map(lambda k: _is_integer(k), d.keys())):
            raise TypeError(f"Expected all keys to be integers instead of {d.keys()=}")
        elif not all(map(lambda k: k > 0, d.keys())):
            raise ValueError(
                f"Expected all keys to be integers greater than zero instead of {d.keys()=}"
            )
        elif not all(map(lambda v: _is_floating(v), d.values())):
            raise TypeError(
                f"Expected all values to be floats instead of {d.values()=}"
            )
        elif not all(map(lambda v: 0.0 <= v <= 1.0, d.values())):
            raise ValueError(
                f"Expected all values to be floats in the range [0.0, 1.0] instead of {d.values()=}"
            )

        i: int
        num_RC: int
        score: float
        for i, (num_RC, score) in enumerate(
            sorted(d.items(), key=lambda kv: kv[1], reverse=True)
        ):
            total_scores[num_RC] += score

    return (
        sorted(
            tests,
            key=lambda t: (total_scores.get(t.num_RC, 0.0), -log(t.pseudo_chisqr)),
            reverse=True,
        )[0],
        total_scores,
    )


def _suggest_using_default(
    tests: List[KramersKronigResult],
    lower_limit: int,
    upper_limit: int,
    limit_delta: int,
    **kwargs,
) -> Tuple[KramersKronigResult, Dict[int, float], int, int]:
    lower_limit, upper_limit = suggest_num_RC_limits(
        tests,
        lower_limit,
        upper_limit,
        limit_delta,
    )
    if lower_limit >= upper_limit:
        raise ValueError(f"Expected {lower_limit=} < {upper_limit=}")

    f: Frequencies = tests[0].get_frequencies()
    subdivided_frequencies: Frequencies
    if "subdivided_frequencies" in kwargs:
        subdivided_frequencies = kwargs["subdivided_frequencies"]
    else:
        subdivided_frequencies = subdivide_frequencies(f)

    curvatures: Dict[int, NDArray[float64]]
    if "curvatures" in kwargs:
        curvatures = kwargs["curvatures"]
    else:
        circuits: Dict[int, Circuit] = {t.num_RC: t.circuit for t in tests}
        curvatures = {
            num_RC: calculate_curvatures(
                circuit.get_impedances(subdivided_frequencies)
            )
            for num_RC, circuit in circuits.items()
        }

    num_sign_changes: Dict[int, float] = suggest_num_RC_method_4(
        tests=tests,
        lower_limit=min(tests, key=lambda t: t.num_RC).num_RC,
        upper_limit=max(tests, key=lambda t: t.num_RC).num_RC,
        offset_factor=0.0,
        relative_scores=False,
        subdivided_frequencies=subdivided_frequencies,
        curvatures=curvatures,
    )
    norms: Dict[int, float] = suggest_num_RC_method_3(
        tests,
        lower_limit=lower_limit,
        upper_limit=upper_limit,
        relative_scores=False,
        subdivided_frequencies=subdivided_frequencies,
        curvatures=curvatures,
    )
    mean_distances: Dict[int, float] = suggest_num_RC_method_5(
        tests,
        lower_limit=lower_limit,
        upper_limit=upper_limit,
        relative_scores=False,
        subdivided_frequencies=subdivided_frequencies,
        curvatures=curvatures,
    )

    tests = [t for t in tests if lower_limit <= t.num_RC <= upper_limit]
    log_pseudo_chisqrs: Dict[int, float] = {
        t.num_RC: log(t.pseudo_chisqr) for t in tests
    }

    # Try to whittle down the num_RC to suggest based on minimizing the number
    # of sign changes among the curvatures. If that doesn't provide a single
    # option, then take the options and maximize the mean distance between
    # sign changes among the curvatures. If that still doesn't provide a single
    # option, then take the remaining options and pick the option with the best
    # fit.
    modified_scores: Dict[int, float] = {
        num_RC: num_sign_changes[num_RC] for num_RC in mean_distances.keys()
    }
    offset_factor: float = 0.1
    top_candidates: List[int] = []

    offsets: Dict[int, float]
    invert: bool
    for offsets, invert in (
        ({}, False),
        (norms, False),
        (mean_distances, True),
        (log_pseudo_chisqrs, False),
    ):
        if len(offsets) > 0:
            min_value: float = min(offsets.values())
            max_value: float = max(offsets.values()) - min_value

            if max_value > 0.0:
                num_RC: int
                value: float
                for num_RC, value in offsets.items():
                    if num_RC not in modified_scores:
                        continue

                    value = (value - min_value) / max_value
                    if invert:
                        value = 1.0 - value

                    modified_scores[num_RC] += offset_factor * value

        top_candidates.clear()

        min_score: float = min(modified_scores.values())

        num_RC: int
        score: float
        for num_RC, score in modified_scores.items():
            if isclose(score, min_score):
                top_candidates.append(num_RC)

        if len(top_candidates) == 1:
            break

    if len(top_candidates) > 1:
        # If by some chance there are still two or more options,
        # then pick the lowest num_RC among those options.
        top_candidates.sort()
        for num_RC in top_candidates[1:]:
            modified_scores[num_RC] += 1e-4
    elif len(top_candidates) == 0:
        raise NotImplementedError()

    relative_scores: Dict[int, float]
    min_score: float = min(modified_scores.values())
    max_score = max(modified_scores.values()) - min_score
    if max_score > 0.0:
        relative_scores = {
            num_RC: 1.0 - (score - min_score) / max_score
            for num_RC, score in modified_scores.items()
        }
    else:
        relative_scores = {num_RC: 1.0 for num_RC in modified_scores.keys()}

    suggested_test: KramersKronigResult = sorted(
        tests,
        key=lambda t: relative_scores.get(t.num_RC, 0.0),
        reverse=True,
    )[0]

    # In some cases there may be a lower num_RC that offers a better fit.
    # E.g., there may be a hump near the lower limit of the num_RC range
    # where the optimal num_RC should exist.
    suggested_log_pseudo_chisqr: float64 = log(suggested_test.pseudo_chisqr)

    log_pseudo_chisqr: float
    for num_RC, log_pseudo_chisqr in sorted(
        log_pseudo_chisqrs.items(),
        key=lambda kv: kv[1],
    ):
        if (
            num_RC < suggested_test.num_RC
            and log_pseudo_chisqr < suggested_log_pseudo_chisqr
            and num_sign_changes[num_RC] <= num_sign_changes[suggested_test.num_RC]
        ):
            suggested_test = [t for t in tests if t.num_RC == num_RC][0]
            break

    return (suggested_test, relative_scores, lower_limit, upper_limit)


def suggest_num_RC(
    tests: List[KramersKronigResult],
    lower_limit: int = 0,
    upper_limit: int = 0,
    limit_delta: int = 0,
    methods: Optional[List[int]] = None,
    use_mean: bool = False,
    use_ranking: bool = False,
    use_sum: bool = False,
    **kwargs,
) -> Tuple[KramersKronigResult, Dict[int, float], int, int]:
    """
    Suggest the optimal number of RC elements to use as part of the linear Kramers-Kronig test by applying one or more of the following methods:

    - 1: |mu|-criterion (Schönleber et al., 2014). With optional modifications that are enabled by default (Yrjänä and Bobacka, 2024).
    - 2: The norm of the fitted variables (Plank et al., 2022).
    - 3: The norm of the curvatures across the fitted impedance spectrum (Plank et al., 2022). With optional modifications that are enabled by default (Yrjänä and Bobacka, 2024).
    - 4: The number of sign changes across the curvatures of the fitted impedance spectrum (Plank et al., 2022). With optional modifications that are enabled by default (Yrjänä and Bobacka, 2024).
    - 5: The mean distance between sign changes across the curvatures of the fitted impedance spectrum (Yrjänä and Bobacka, 2024).
    - 6: The apex of a |log sum abs tau R| (or |log sum abs tau C|) versus the number of RC elements (Yrjänä and Bobacka, 2024).

    If multiple methods are used, then one of several approaches can be used to determine which number of RC elements to suggest:

    - Each method suggests a number of RC elements and the mean is chosen.
    - Each method ranks the different numbers of RC elements, exponentially decreasing points are assigned based on rank, the points assigned by each method are summed up, and the highest-scoring number of RC elements is chosen.
    - Each method returns a relative score from 0.0 to 1.0 (worst to best), the relative scores are added up, and the highest-scoring number of RC elements is chosen.

    If no methods are chosen, then the default approach is used:

    - Use method 4 to obtain an initial list of candidates.
    - Use method 3 to reduce the list of candidates.
    - Use method 5 to reduce the list of candidates, if necessary.
    - Use |pseudo chi-squared| to reduce the list of candidates, if necessary.
    - Try to find a lower number of RC elements with a lower |pseudo chi-squared| and an equal number or fewer sign changes among the curvatures.

    If the lower and/or upper limit is not specified, then |suggest_num_RC_limits| is used to estimate the limit(s).

    References:

    - `M. Schönleber, D. Klotz, and E. Ivers-Tiffée, 2014, Electrochim. Acta, 131, 20-27 <https://doi.org/10.1016/j.electacta.2014.01.034>`_
    - `C. Plank, T. Rüther, and M.A. Danzer, 2022, 2022 International Workshop on Impedance Spectroscopy (IWIS), 1-6 <https://doi.org/10.1109/IWIS57888.2022.9975131>`_
    - `V. Yrjänä and J. Bobacka, 2024, Electrochim. Acta, 504, 144951 <https://doi.org/10.1016/j.electacta.2024.144951>`_

    Parameters
    ----------
    tests: List[|KramersKronigResult|]
        The test results to evaluate.

    lower_limit: int, optional
        See |suggest_num_RC_limits| for details.

    upper_limit: int, optional
        See |suggest_num_RC_limits| for details.

    limit_delta: int, optional
        See |suggest_num_RC_limits| for details.

    methods: Optional[List[int]], optional
        A list of integers corresponding to the supported methods.

    use_mean: bool, optional
        If true, then the mean value of the number of RC elements suggested by each of the selected methods is chosen.

    use_ranking: bool, optional
        If true, then each selected method ranks the numbers of RC elements, a score is assigned based on ranking, and the highest-scoring number of RC elements is chosen.

    use_sum: bool, optional
        If true, then the scores returned by each of the selected methods are summed up and the highest-scoring number of RC elements is chosen.

    **kwargs
        Keyword arguments are passed on to the underlying methods.

    Returns
    -------
    Tuple[|KramersKronigResult|, Dict[int, float], int, int]

        A tuple containing:

        - The |KramersKronigResult| corresponding to the suggested number of RC elements.
        - A dictionary that maps the number of RC elements to their corresponding scores.
        - The lower limit for the number of RC elements to consider.
        - The upper limit for the number of RC elements to consider.
    """
    if not isinstance(tests, list):
        raise TypeError(f"Expected a list instead of {tests=}")
    tests = sorted(tests, key=lambda t: t.num_RC)

    if not _is_integer(lower_limit):
        raise TypeError(f"Expected an integer instead of {lower_limit=}")
    elif not (lower_limit >= 0):
        raise ValueError(
            f"Expected an integer greater than or equal to zero instead of {lower_limit=}"
        )

    if not _is_integer(upper_limit):
        raise TypeError(f"Expected an integer instead of {upper_limit=}")
    elif upper_limit < 0:
        upper_limit = tests[-1].num_RC + upper_limit

    if not _is_integer(limit_delta):
        raise TypeError(f"Expected an integer instead of {limit_delta=}")

    if methods is None:
        methods = []
    elif not _is_integer_list(methods):
        raise TypeError(f"Expected None or a list of integers instead of {methods=}")

    if not _is_boolean(use_mean):
        raise TypeError(f"Expected a boolean instead of {use_mean=}")
    elif not _is_boolean(use_ranking):
        raise TypeError(f"Expected a boolean instead of {use_ranking=}")
    elif not _is_boolean(use_sum):
        raise TypeError(f"Expected a boolean instead of {use_sum=}")
    elif sum((use_sum, use_mean, use_ranking)) > 1:
        raise ValueError(
            f"Only one way of combining suggestions can be active at a time instead of {use_mean=}, {use_ranking=}, and {use_sum=}"
        )

    selection: Dict[int, Callable]
    if (any((use_mean, use_ranking, use_sum)) and len(methods) > 0) or len(
        methods
    ) == 1:
        selection, lower_limit, upper_limit = _choose_methods(
            tests,
            lower_limit,
            upper_limit,
            limit_delta,
            methods,
            **kwargs,
        )

        if use_sum:
            suggested_test, total_scores = _suggest_using_sum(tests, selection)
        elif use_mean:
            suggested_test, total_scores = _suggest_using_mean(tests, selection)
        elif use_ranking:
            suggested_test, total_scores = _suggest_using_ranking(tests, selection)
        elif len(methods) == 1:
            suggested_test, total_scores = _suggest_using_sum(tests, selection)
        else:
            raise NotImplementedError()

    elif len(methods) == 0:
        suggested_test, total_scores, lower_limit, upper_limit = _suggest_using_default(
            tests,
            lower_limit,
            upper_limit,
            limit_delta,
            **kwargs,
        )
    elif len(methods) > 1:
        raise ValueError(
            "Multiple methods for suggesting the optimal number of RC elements have been chosen, but the manner in which the scores are combined has not been chosen!"
        )
    else:
        raise NotImplementedError()

    return (
        suggested_test,
        total_scores,
        lower_limit,
        upper_limit,
    )
