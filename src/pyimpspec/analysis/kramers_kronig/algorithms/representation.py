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
    arctan2,
    argmax,
    argmin,
    complex128,
    cos,
    degrees,
    float64,
    inf,
    isnan,
    linspace,
    log10 as log,
    pi,
    sign,
    sin,
    tan,
)
from pyimpspec.analysis.kramers_kronig.result import KramersKronigResult
from pyimpspec.typing import Frequencies
from pyimpspec.typing.helpers import (
    Dict,
    List,
    Tuple,
    NDArray,
    _is_boolean,
    _is_integer,
    _is_floating,
)
from .utility.osculating_circle import (
    _get_osculating_circle,
    _find_x_axis_intersections,
)
from .utility.pseudo_chi_squared import _calculate_intercept_of_lines


_DEBUG = bool(0)


def _extrapolate_spectrum(X_slope: NDArray[complex128], admittance: bool) -> float64:
    numerator: float64 = (X_slope[0] - X_slope[1]).imag
    denominator: float64 = (X_slope[0] - X_slope[1]).real

    slope: float64
    if denominator == 0.0:
        slope = inf if numerator > 0 else -inf
    else:
        slope = numerator / denominator

    slope *= 1 if admittance else -1

    return slope


def _does_extrapolated_slope_tend_to_zero(
    X_slope: NDArray[complex128],
    high_frequency: bool,
) -> bool:
    if high_frequency and abs(X_slope[0].imag) < abs(X_slope[1].imag):
        return True
    elif not high_frequency and abs(X_slope[-1].imag) < abs(X_slope[-2].imag):
        return True
    elif abs(X_slope[-1].imag) == abs(X_slope[-2].imag) == 0.0:
        return True

    return False


def _pick_intersection(
    intersections: List[Tuple[float, float]],
    x_center: float,
    y_center: float,
    kappa: float,
    X_circle: NDArray[complex128],
    admittance: bool,
) -> Tuple[float, float, float, float, float, float, float]:
    x_start: float = X_circle[1].real
    y_start: float = X_circle[1].imag * (1 if admittance else -1)

    start_angle: float = arctan2(y_start - y_center, x_start - x_center)
    start_angle += 2 * pi
    start_angle %= 2 * pi

    rotation_direction: int = int(sign(kappa)) * (-1 if admittance else 1)

    candidates: List[Tuple[float, float, float, float]] = []
    x_end: float
    y_end: float
    for x_end, y_end in intersections:
        end_angle: float = arctan2(y_end - y_center, x_end - x_center)
        end_angle += 2 * pi
        end_angle %= 2 * pi

        angle_difference: float = end_angle - start_angle
        angle_difference *= rotation_direction
        if angle_difference < 0.0:
            angle_difference += 2 * pi

        candidates.append((x_end, y_end, end_angle, angle_difference))

        if _DEBUG:
            print(
                f"{degrees(start_angle)=:.3f}, {degrees(end_angle)=:.3f}, {degrees(angle_difference)=:.3f}"
            )

    candidates.sort(key=lambda t: t[-1])
    x_end, y_end, end_angle, angle_difference = candidates[0]

    return (
        start_angle,
        x_start,
        y_start,
        end_angle,
        x_end,
        y_end,
        angle_difference,
    )


def _imaginary_extremes_tend_toward_zero(
    representations: List[Tuple[KramersKronigResult, Dict[int, float], int, int]]
) -> List[float]:
    results: List[float] = []

    i: int
    test: KramersKronigResult
    for i, (test, *_) in enumerate(representations):
        if _DEBUG:
            from pyimpspec.plot.mpl import show, plot_nyquist
            from matplotlib.pyplot import Circle

            figure, axes = plot_nyquist(
                test,
                colors={"impedance": "black"},
                admittance=test.admittance,
                legend=False,
            )
            figure, axes = plot_nyquist(
                test,
                colors={"impedance": "red"},
                admittance=test.admittance,
                line=True,
                legend=False,
                figure=figure,
                axes=axes,
            )
            ax = axes[0]
            x_lim = ax.get_xlim()
            y_lim = ax.get_ylim()
            ax.axhline(0.0, color="black", alpha=0.5)

        score: float = 0.0

        f: Frequencies = test.get_frequencies()

        j: int
        f_circle: NDArray[float64]
        for j, f_circle in enumerate((f[:3][::-1], f[-3:])):
            if _DEBUG:
                color: str = "blue" if j == 0 else "green"
                print()

            X_circle: NDArray[complex128] = test.circuit.get_impedances(f_circle) ** (
                -1 if test.admittance else 1
            )

            x_center: float
            y_center: float
            kappa: float
            x_center, y_center, kappa = _get_osculating_circle(
                test.circuit,
                f_circle,
                test.admittance,
            )

            rotation_direction: int = int(sign(kappa))
            if test.admittance:
                rotation_direction *= -1
            else:
                y_center *= -1

            f_slope: List[float] = (
                [
                    max(f),
                    max(f) * 0.9999,
                ]
                if j == 0
                else [
                    min(f) * 1.0001,
                    min(f),
                ]
            )
            X_slope: NDArray[complex128] = test.circuit.get_impedances(
                f_slope,
            ) ** (-1 if test.admittance else 1)
            slope: float64 = _extrapolate_spectrum(X_slope, test.admittance)

            extrapolated_slope_tends_to_zero: bool = (
                _does_extrapolated_slope_tend_to_zero(
                    X_slope,
                    high_frequency=j == 0,
                )
            )
            if extrapolated_slope_tends_to_zero:
                score += 0.5

            if _DEBUG:
                print(
                    f"{test.admittance=}, {test.num_RC=}, {x_center=}, {y_center=}, {kappa=}, {slope=}"
                )
                offset: float = (
                    X_slope[0].imag * (1 if test.admittance else -1)
                    - slope * X_slope[0].real
                )
                x_intercept: float = _calculate_intercept_of_lines(
                    slope,
                    offset,
                    0.0,
                    0.0,
                )
                if extrapolated_slope_tends_to_zero:
                    ax.plot(
                        [X_slope[j].real, x_intercept],
                        [slope * _x + offset for _x in [X_slope[j].real, x_intercept]],
                        color=color,
                        linestyle=":",
                    )
                    ax.scatter(
                        x_intercept,
                        slope * x_intercept + offset,
                        edgecolor=color,
                        facecolor="none",
                        marker="o",
                    )
                else:
                    ax.axline(
                        (
                            X_slope[j].real,
                            X_slope[j].imag * (1 if test.admittance else -1),
                        ),
                        slope=slope,
                        color=color,
                        linestyle=":",
                    )

            if kappa == 0.0 or isnan(x_center) or isnan(y_center):
                # Straight line
                continue

            radius: float = abs(1 / kappa)
            intersections: List[Tuple[float, float]]
            intersections = _find_x_axis_intersections(x_center, y_center, radius)
            if len(intersections) > 0:
                start_angle: float
                x_start: float
                y_start: float
                end_angle: float
                x_end: float
                y_end: float
                angle_difference: float
                (
                    start_angle,
                    x_start,
                    y_start,
                    end_angle,
                    x_end,
                    y_end,
                    angle_difference,
                ) = _pick_intersection(
                    intersections,
                    x_center,
                    y_center,
                    kappa,
                    X_circle,
                    test.admittance,
                )

                score += 0.5 * (1.0 - angle_difference / (2 * pi))

                if _DEBUG:
                    ax.axline(
                        (x_center, y_center),
                        slope=tan(start_angle),
                        color="magenta",
                        linestyle=":",
                    )
                    ax.scatter(
                        x_start,
                        y_start,
                        facecolor="none",
                        edgecolor="magenta",
                        marker="o",
                        zorder=20,
                    )

                    ax.axline(
                        (x_center, y_center),
                        slope=tan(end_angle),
                        color="magenta",
                        linestyle="--",
                    )
                    ax.scatter(
                        x_end,
                        y_end,
                        edgecolor="none",
                        facecolor="magenta",
                        marker="o",
                        zorder=20,
                    )

                    angles = linspace(
                        start_angle,
                        start_angle + angle_difference * rotation_direction,
                        num=360,
                    )
                    radius_factor = 0.9
                    x = radius * radius_factor * cos(angles) + x_center
                    y = radius * radius_factor * sin(angles) + y_center
                    ax.plot(x, y, color="magenta", linestyle=":")

            if _DEBUG:
                ax.add_patch(
                    Circle(
                        (x_center, y_center),
                        radius,
                        edgecolor=color,
                        facecolor="none",
                        linestyle="--",
                    )
                )
                ax.scatter(x_center, y_center, marker="+", color=color)

                print(f"{intersections=}")
                for x, y in intersections:
                    ax.scatter(x, y, color=color, marker="x")

        results.append(score)

        if _DEBUG:
            print(f"{results[i]=}, {representations[i][0].admittance=}")
            ax.set_xlim(*x_lim)
            ax.set_ylim(*y_lim)
            show()

    return results


def _suggest_representation(
    representations: List[Tuple[KramersKronigResult, Dict[int, float], int, int]]
) -> Tuple[KramersKronigResult, Dict[int, float], int, int]:
    if not (len(representations) == 2):
        raise ValueError(
            f"Expected a list with two tuples instead of {representations=}"
        )

    representations = sorted(representations, key=lambda t: t[0].pseudo_chisqr)

    log_pseudo_chisqr: NDArray[float64] = log(
        [t[0].pseudo_chisqr for t in representations]
    )
    if abs(log_pseudo_chisqr[1] - log_pseudo_chisqr[0]) > 0.5:
        if _DEBUG:
            i: int
            for i, (test, *_) in enumerate(representations):
                print(f"{test.admittance=}, {test.num_RC=}, {log_pseudo_chisqr[i]=}")

        return representations[0]

    scores: Dict[int, float] = {i: 0 for i in range(0, len(representations))}

    if _DEBUG:
        admittances = {i: representations[i][0].admittance for i in scores.keys()}

    num_RCs: List[float] = [t[0].num_RC + t[0].pseudo_chisqr for t in representations]
    i = argmin(num_RCs)
    scores[i] += 1
    if _DEBUG:
        print(f"{scores=}, {admittances=}, {num_RCs=}")

    lower_limits: List[float] = [t[2] + t[0].pseudo_chisqr for t in representations]
    i = argmin(lower_limits)
    scores[i] += 1
    if _DEBUG:
        print(f"{scores=}, {admittances=}, {lower_limits=}")

    imaginary_extremes_tend_to_zero: List[float] = _imaginary_extremes_tend_toward_zero(
        representations
    )
    i = argmax(imaginary_extremes_tend_to_zero)
    scores[i] += 1
    if _DEBUG:
        print(f"{scores=}, {admittances=}, {imaginary_extremes_tend_to_zero=}")

    score: float
    for i, score in sorted(
        scores.items(),
        key=lambda kv: (
            max(scores.values()) - kv[1],
            representations[kv[0]][0].pseudo_chisqr,
        ),
    ):
        return representations[i]

    return representations[0]


def suggest(
    suggestions: List[Tuple[KramersKronigResult, Dict[int, float], int, int]]
) -> Tuple[KramersKronigResult, Dict[int, float], int, int]:
    """
    Suggest the most appropriate representation (impedance or admittance) of the immittance spectrum that was tested.
    If the difference between |pseudo chi-squared| values is greater than 0.5 decades, then the representation that provides the best fit is picked.
    Otherwise, the representations are scored according to various criteria:

    - One point to whichever has the lowest number of RC elements.
    - One point to whichever has the lowest lower limit for the number of RC elements.
    - One point to whichever comes closest to having the imaginary part of each frequency extreme reach zero.

    The tuple in the input list that corresponds to the representation with the highest score is returned.

    References:

    - `V. Yrjänä and J. Bobacka, 2024, Electrochim. Acta, 504, 144951 <https://doi.org/10.1016/j.electacta.2024.144951>`_

    Parameters
    ----------
    suggestions: List[Tuple[|KramersKronigResult|, Dict[int, float], int, int]]
        A list obtained by processing List[|KramersKronigResult|] for different representations with |suggest_num_RC| and collecting the return values.

    Returns
    -------
    Tuple[|KramersKronigResult|, Dict[int, float], int, int]

        A tuple containing:

        - The |KramersKronigResult| corresponding to the suggested number of RC elements and representation.
        - A dictionary that maps the number of RC elements to their corresponding scores for the suggested representation.
        - The lower limit for the number of RC elements to consider for the suggested representation.
        - The upper limit for the number of RC elements to consider for the suggested representation.
    """
    if not isinstance(suggestions, list):
        raise TypeError(f"Expected a list instead of {suggestions=}")
    elif len(suggestions) < 1:
        raise ValueError(f"Expected at least on item in {suggestions=}")
    elif not all(map(lambda t: isinstance(t, tuple) and len(t) == 4, suggestions)):
        raise TypeError(
            f"Expected a list of tuples with four items instead of {suggestions=}"
        )

    for suggestion in suggestions:
        test: KramersKronigResult
        scores: Dict[int, float]
        lower_limit: int
        upper_limit: int
        test, scores, lower_limit, upper_limit = suggestion

        if not _is_boolean(test.admittance):
            raise TypeError(f"Expected a boolean instead of {test.admittance=}")
        elif not isinstance(scores, dict):
            raise TypeError(
                f"Expected the second item in each tuple to be a dictionary instead of {suggestion=}"
            )
        elif not all(map(lambda k: _is_integer(k), scores.keys())):
            raise TypeError(
                f"Expected only integer keys in the second item of the tuple instead of {scores=}"
            )
        elif not all(map(lambda v: _is_floating(v), scores.values())):
            raise TypeError(
                f"Expected only float values in the second item of the tuple instead of {scores=}"
            )
        elif not _is_integer(lower_limit):
            raise TypeError(
                f"Expected the third item in each tuple to be an integer instead of {suggestion=}"
            )
        elif not _is_integer(upper_limit):
            raise TypeError(
                f"Expected the fourth item in each tuple to be an integer instead of {suggestion=}"
            )

    if len(suggestions) == 1:
        return suggestions[0]

    return _suggest_representation(suggestions)
