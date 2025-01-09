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
    argwhere,
    array,
    float64,
    int64,
    linspace,
    log10 as log,
    zeros,
)
from numpy.typing import NDArray
from pyimpspec.data import DataSet
from pyimpspec.analysis import KramersKronigResult
from pyimpspec.analysis.kramers_kronig.utility import _format_log_F_ext_for_latex
from pyimpspec.typing import (
    ComplexImpedances,
    Frequencies,
)
from pyimpspec.typing.helpers import (
    Dict,
    List,
    Optional,
    Tuple,
    _is_boolean,
    _is_integer,
    _is_floating,
)
from pyimpspec.plot.colors import (
    COLOR_BLACK,
    COLOR_BLUE,
    COLOR_MAGENTA,
    COLOR_ORANGE,
    COLOR_RED,
    COLOR_TEAL,
)
from .markers import (
    MARKER_CIRCLE,
    MARKER_SQUARE,
)
from .helpers import (
    _color_axis,
    _combine_legends,
    _configure_log_limits,
    _configure_log_scale,
    _format_coord_two_y_axes,
    _validate_figure,
)
from .bode import plot_bode
from .pseudo_chisqr import plot_pseudo_chisqr
from .nyquist import plot_nyquist
from .residuals import plot_residuals
from .real_imaginary import plot_real_imaginary


def plot_log_F_ext(
    evaluations: List[Tuple[float, List[KramersKronigResult], float]],
    projection: str = "3d",
    figure: Optional["Figure"] = None,  # noqa: F821
    title: Optional[str] = None,
    legend: bool = True,
    axes: Optional[List["Axes"]] = None,  # noqa: F821
    adjust_axes: bool = True,
    **kwargs,
) -> Tuple["Figure", List["Axes"]]:  # noqa: F821
    """
    Plot the results of evaluating different values of |log F_ext|.

    Parameters
    ----------
    evaluations: List[Tuple[float, List[KramersKronigResult], float]]
        A list of results obtained with |evaluate_log_F_ext|.

    projection: str, optional
        Either '3d' or '2d' depending on the projection to use.

    figure: Optional[Figure], optional
        The matplotlib.figure.Figure instance to use when plotting the data.

    title: Optional[str], optional
        The title of the figure.
        If no title is provided, then one will be provided.

    legend: bool, optional
        Whether or not to add a legend.

    axes: Optional[List[Axes]], optional
        A list of matplotlib.axes.Axes instances to use when plotting the data.

    adjust_axes: bool, optional
        Whether or not to adjust the axes (label, scale, limits, etc.).

    **kwargs

    Returns
    -------
    Tuple[|Figure|, List[|Axes|]]
    """
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.colors import (
        Colormap,
        Normalize,
    )
    from matplotlib import colormaps
    from matplotlib.cm import ScalarMappable
    from matplotlib.ticker import MaxNLocator
    from matplotlib.collections import PathCollection

    if not isinstance(evaluations, list):
        raise TypeError(f"Expected a list instead of {evaluations=}")
    elif not all(
        map(
            lambda e: isinstance(e, tuple) and len(e) == 3,
            evaluations,
        )
    ):
        raise TypeError(f"Expected tuples of length 3 intead of {evaluations=}")

    if not isinstance(projection, str):
        raise TypeError(f"Expected a string instead of {projection=}")
    elif projection not in ("2d", "3d"):
        raise ValueError(
            f"Expected a string with the value '2d' or '3d' instead of {projection=}"
        )
    three_dimensional: bool = projection == "3d"

    axis: Axes
    if figure is None:
        if three_dimensional:
            figure, axis = plt.subplots(subplot_kw={"projection": "3d"})
        else:
            figure, axis = plt.subplots()
        axes = [axis]
    else:
        assert axes is not None
        axis = axes[0]

    _validate_figure(figure, axes, num_axes=1)

    if title is None:
        title = (
            "Time constant range extensions\n"
            + r"Best $\log{F_{\rm ext}} = "
            + _format_log_F_ext_for_latex(min(evaluations, key=lambda e: e[2])[0])
            + r"$"
        )
    elif not isinstance(title, str):
        raise TypeError(f"Expected a string or None instead of {title=}")

    if title != "":
        figure.suptitle(title)

    count: int = len(evaluations)
    cmap: Colormap = colormaps.get_cmap("rainbow")
    colors: List[Tuple[float, float, float, float]] = [
        cmap(i / count) for i in range(0, count)
    ]
    zorders: List[int] = list(range(count + 1, 0, -1))

    log_F_ext: float
    tests: List[KramersKronigResult]
    statistic: float
    color: Tuple[float, float, float, float]
    zorder: int
    for (log_F_ext, tests, statistic), color, zorder in zip(
        evaluations,
        colors,
        zorders,
    ):
        scatter_kwargs = dict(
            label=f"{log_F_ext:.3g}",
            marker=".",
            color=color,
        )
        x: NDArray[int64] = array([t.num_RC for t in tests], dtype=int64)
        y: NDArray[float64]
        if three_dimensional:
            y = array([log_F_ext] * len(x), dtype=float64)
            z: NDArray[float64] = log([t.pseudo_chisqr for t in tests])
            axis.plot(
                x,
                y,
                z,
                color=COLOR_BLACK,
                alpha=0.25,
            )
            axis.scatter(
                x,
                y,
                z,
                **scatter_kwargs,
            )
        else:
            y = array([t.pseudo_chisqr for t in tests])
            axis.scatter(
                x,
                y,
                zorder=zorder,
                **scatter_kwargs,
            )
            axis.plot(
                x,
                y,
                zorder=zorder,
                color=color,
                alpha=0.25,
            )

    if not _is_boolean(adjust_axes):
        raise TypeError(f"Expected a boolean instead of {adjust_axes=}")
    elif adjust_axes:
        axis.set_xlabel(r"$N_\tau$")
        if three_dimensional:
            axis.set_zlabel(r"$\log{\chi^{2}_{\rm ps}}$")
            axis.set_ylabel(r"$\log{F_{\rm ext}}$")
        else:
            axis.set_ylabel(r"$\chi^{2}_{\rm ps}$")
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))

        if not three_dimensional:
            _configure_log_scale(axis, y=True)
            _configure_log_limits(axis, y=True)

        cbar = figure.colorbar(
            ScalarMappable(norm=Normalize(0, 1), cmap=cmap),
            ax=axis,
            location="bottom",
            fraction=0.05,
            shrink=0.5,
            label="Best to worst (left to right)",
        )
        cbar.set_ticks([])

        if three_dimensional:
            pane_color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 0.0)
            axis.xaxis.set_pane_color(pane_color)
            axis.yaxis.set_pane_color(pane_color)
            axis.zaxis.set_pane_color(pane_color)

    if not _is_boolean(legend):
        raise TypeError(f"Expected a boolean instead of {legend=}")
    elif legend and not three_dimensional:
        handles: List[PathCollection]
        labels: List[str]
        handles, labels = axis.get_legend_handles_labels()
        pairs: List[Tuple[PathCollection, str]] = list(
            map(
                lambda t: (t[0], float(t[1])),
                zip(handles, labels),
            )
        )

        handles = []
        labels = []
        for handle, label in sorted(pairs, key=lambda t: t[1]):
            handles.append(handle)
            labels.append(label)

        axis.legend(handles, labels, title=r"$\log{F_{\rm ext}}$", ncols=2)

    return (
        figure,
        axes,
    )


# Mu-criterion
def _plot_suggestion_method_1(
    tests: List[KramersKronigResult],
    colors: Dict[str, str],
    markers: Dict[str, str],
    legend_kwargs: dict,
    i: int,
    j: int,
    show_excluded: bool,
    figure: "Figure",  # noqa: F821
    axes: List["Axes"],  # noqa: F821
    **kwargs,
):
    from pyimpspec.analysis.kramers_kronig.algorithms.method_1 import (
        _calculate_mu,
        _fit_logistic_function,
        _logistic_derivative,
        _logistic_function,
        suggest,
    )
    from pyimpspec.analysis.kramers_kronig.algorithms.utility.pseudo_chi_squared import (
        _calculate_intercept_of_lines,
    )

    label: str = r"$\mu$"
    axes[0].set_ylabel(label)

    mu_criterion: float = kwargs.get("mu_criterion", 0.85)
    if mu_criterion > 0.0:
        axes[0].axhline(
            mu_criterion,
            color=colors.get("mu_criterion", COLOR_BLACK),
            linestyle=":",
            label=r"$\mu_{\rm crit}$",
        )

    x: List[int] = [t.num_RC for t in tests]
    scores: Dict[int, float] = {t.num_RC: _calculate_mu(t.circuit) for t in tests}
    y: List[float] = [scores[num_RC] for num_RC in x]

    if show_excluded:
        axes[0].scatter(
            [_ for k, _ in enumerate(x) if not (i <= k < j)],
            [_ for k, _ in enumerate(y) if not (i <= k < j)],
            marker="o",
            edgecolor=colors.get("mu", COLOR_BLACK),
            facecolor="none",
            label=label + ", excl.",
        )

    axes[0].scatter(
        x if not show_excluded else x[i:j],
        y if not show_excluded else y[i:j],
        marker="o",
        color=colors.get("mu", COLOR_BLACK),
        label=label + (", incl." if show_excluded else ""),
    )

    beta: float = kwargs.get("beta", 0.75)
    if mu_criterion < 0.0:
        p: Tuple[float, ...] = _fit_logistic_function(tests)
        x = linspace(min(x), max(x), num=int(round(max(x) - min(x))) * 100)
        y = _logistic_function(x, *p)
        axes[0].plot(x, y, color=COLOR_RED)

        slope = _logistic_derivative(p[2], *p)
        intercept = _logistic_function(p[2], *p) - slope * p[2]
        y = slope * x + intercept
        i = min(argwhere(y <= 1.05).flatten())
        j = max(argwhere(y >= -0.05).flatten())
        axes[0].axhline(p[0] + p[3], color=COLOR_RED, linestyle=":")
        axes[0].plot(
            x[i:j + 1],
            y[i:j + 1],
            color=COLOR_RED,
            linestyle=":",
        )

    elif beta > 0.0:
        label = r"$S_{\rm rel}$"
        axes[1].set_ylabel(label)
        scores = suggest(
            tests,
            mu_criterion=kwargs.get("mu_criterion", 0.85),
            beta=beta,
            relative_scores=True,
        )
        y = [scores[num_RC] for num_RC in x]

        if show_excluded:
            axes[1].bar(
                [_ for k, _ in enumerate(x) if not (i <= k < j)],
                [_ for k, _ in enumerate(y) if not (i <= k < j)],
                edgecolor=colors.get("mu_score", COLOR_MAGENTA),
                color="none",
                alpha=0.5,
                label=label + ", excl.",
            )

        axes[1].bar(
            x if not show_excluded else x[i:j],
            y if not show_excluded else y[i:j],
            color=colors.get("mu_score", COLOR_MAGENTA),
            alpha=0.5,
            label=label + (", incl." if show_excluded else ""),
        )

        colors["right_yaxis"] = COLOR_MAGENTA
        axes[1].set_ylim(0.0, None)

    if mu_criterion <= 0.0 or beta <= 0.0:
        axes[1].set_ylabel("")
        axes[1].set_yticks([])

    legend_kwargs.update(
        {
            "facecolor": "white",
            "framealpha": 1.0,
            "loc": "upper right",
        }
    )


# Norm of fitted variables
def _plot_suggestion_method_2(
    tests: List[KramersKronigResult],
    colors: Dict[str, str],
    markers: Dict[str, str],
    legend_kwargs: dict,
    i: int,
    j: int,
    show_excluded: bool,
    figure: "Figure",  # noqa: F821
    axes: List["Axes"],  # noqa: F821
    **kwargs,
):
    from pyimpspec.analysis.kramers_kronig.algorithms.method_2 import (
        suggest,
    )

    axes[0].set_ylabel(r"$||\theta(N_\tau)||/N_\tau$")
    axes[1].set_yticks([])

    x: List[int] = [t.num_RC for t in tests]
    scores: Dict[int, float] = suggest(
        tests,
        relative_scores=False,
    )
    y: List[float] = [scores[num_RC] for num_RC in x]

    if show_excluded:
        axes[0].scatter(
            x,
            y,
            marker="o",
            edgecolor=colors.get("theta", COLOR_BLACK),
            facecolor="none",
            label="Excl.",
        )

    axes[0].scatter(
        x if not show_excluded else x[i:j],
        y if not show_excluded else y[i:j],
        marker="o",
        color=colors.get("theta", COLOR_BLACK),
        label="Incl." if show_excluded else None,
    )

    axes[0].set_yscale("log")


# Norm of curvatures
def _plot_suggestion_method_3(
    tests: List[KramersKronigResult],
    colors: Dict[str, str],
    markers: Dict[str, str],
    legend_kwargs: dict,
    i: int,
    j: int,
    show_excluded: bool,
    figure: "Figure",  # noqa: F821
    axes: List["Axes"],  # noqa: F821
    **kwargs,
):
    from pyimpspec.analysis.kramers_kronig.algorithms.method_3 import (
        suggest,
    )

    axes[0].set_ylabel(r"$||\kappa||$")
    axes[1].set_yticks([])

    x: List[int] = [t.num_RC for t in tests]
    scores: Dict[int, float] = suggest(
        tests,
        subdivision=kwargs.get("subdivision", 4),
        relative_scores=False,
    )
    y: List[float] = [scores[num_RC] for num_RC in x]

    if show_excluded:
        axes[0].scatter(
            x,
            y,
            marker="o",
            edgecolor=colors.get("kappa", COLOR_BLACK),
            facecolor="none",
            label="Excl.",
        )

    axes[0].scatter(
        x if not show_excluded else x[i:j],
        y if not show_excluded else y[i:j],
        marker="o",
        color=colors.get("kappa", COLOR_BLACK),
        label="Incl." if show_excluded else None,
    )

    axes[0].set_yscale("log")


# Num sign changes in curvatures
def _plot_suggestion_method_4(
    tests: List[KramersKronigResult],
    colors: Dict[str, str],
    markers: Dict[str, str],
    legend_kwargs: dict,
    i: int,
    j: int,
    show_excluded: bool,
    figure: "Figure",  # noqa: F821
    axes: List["Axes"],  # noqa: F821
    **kwargs,
):
    from pyimpspec.analysis.kramers_kronig.algorithms.method_4 import (
        suggest,
    )
    from matplotlib.ticker import MaxNLocator

    x: List[int] = [t.num_RC for t in tests]
    scores: Dict[int, float] = suggest(
        tests,
        subdivision=kwargs.get("subdivision", 4),
        offset_factor=0.0,
        relative_scores=False,
    )
    y1: List[float] = [scores[num_RC] for num_RC in x]
    y2: List[float] = [t.pseudo_chisqr for t in tests]

    if show_excluded:
        axes[0].scatter(
            x,
            y1,
            marker="o",
            edgecolor=colors.get("kappa", COLOR_BLACK),
            facecolor="none",
            label=r"$\kappa$, excl.",
        )

        axes[1].scatter(
            x,
            y2,
            marker=".",
            edgecolor=colors.get("distance", COLOR_MAGENTA),
            facecolor="none",
            label=r"$\chi_{\rm ps}^2$, excl.",
        )

    axes[0].scatter(
        x if not show_excluded else x[i:j],
        y1 if not show_excluded else y1[i:j],
        marker="o",
        color=colors.get("kappa", COLOR_BLACK),
        label=r"$\kappa$, incl." if show_excluded else None,
    )

    axes[1].scatter(
        x if not show_excluded else x[i:j],
        y2 if not show_excluded else y2[i:j],
        marker=".",
        color=colors.get("distance", COLOR_MAGENTA),
        label=r"$\chi_{\rm ps}^2$, incl." if show_excluded else None,
    )

    axes[0].yaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0].set_ylabel(r"Num. sgn($\kappa$) changes")
    axes[1].set_ylabel(r"$\chi_{\rm ps}^2$")
    axes[1].set_yscale("log")
    colors["right_yaxis"] = COLOR_MAGENTA


# Average distance between sign changes in curvatures
def _plot_suggestion_method_5(
    tests: List[KramersKronigResult],
    colors: Dict[str, str],
    markers: Dict[str, str],
    legend_kwargs: dict,
    i: int,
    j: int,
    show_excluded: bool,
    figure: "Figure",  # noqa: F821
    axes: List["Axes"],  # noqa: F821
    **kwargs,
):
    from pyimpspec.analysis.kramers_kronig.algorithms.method_5 import (
        suggest,
    )
    from matplotlib.ticker import MaxNLocator

    axes[0].set_ylabel(r"Mean distance between sgn($\kappa$) changes")
    axes[1].set_yticks([])

    x: List[int] = [t.num_RC for t in tests]
    scores: Dict[int, float] = suggest(
        tests,
        relative_scores=False,
    )
    y: List[float] = [scores[num_RC] for num_RC in x]

    if show_excluded:
        axes[0].scatter(
            x,
            y,
            marker="o",
            edgecolor=colors.get("kappa", COLOR_BLACK),
            facecolor="none",
            label="Excl.",
        )

    axes[0].scatter(
        x if not show_excluded else x[i:j],
        y if not show_excluded else y[i:j],
        marker="o",
        color=colors.get("kappa", COLOR_BLACK),
        label="Incl." if show_excluded else None,
    )

    axes[0].yaxis.set_major_locator(MaxNLocator(integer=True))


# Apex of the sum of C (or R) values calculated from fitted R (or C) values
def _plot_suggestion_method_6(
    tests: List[KramersKronigResult],
    colors: Dict[str, str],
    markers: Dict[str, str],
    legend_kwargs: dict,
    i: int,
    j: int,
    show_excluded: bool,
    figure: "Figure",  # noqa: F821
    axes: List["Axes"],  # noqa: F821
    **kwargs,
):
    from pyimpspec.analysis.kramers_kronig.algorithms.method_6 import (
        _calculate_log_sum_abs_tau_var,
        suggest,
    )

    axes[0].set_ylabel(
        r"$\log{\Sigma_{k=1}^{N_\tau} |\tau_k / "
        + (r"C_k" if tests[0].admittance else r"R_k")
        + r"|}$"
    )
    axes[1].set_yticks([])

    x: List[int] = [t.num_RC for t in tests]
    y: List[float] = [_calculate_log_sum_abs_tau_var(t.circuit) for t in tests]

    if show_excluded:
        axes[0].scatter(
            [_ for k, _ in enumerate(x) if not (i <= k < j)],
            [_ for k, _ in enumerate(y) if not (i <= k < j)],
            marker="o",
            edgecolor=colors.get("tau_var", COLOR_BLACK),
            facecolor="none",
            label="Excl.",
        )

    axes[0].scatter(
        x if not show_excluded else x[i:j],
        y if not show_excluded else y[i:j],
        marker="o",
        color=colors.get("tau_var", COLOR_BLACK),
        label="Incl." if show_excluded else None,
    )

    scores: Dict[int, float] = suggest(tests, relative_scores=False)
    y: List[float] = [scores[num_RC] for num_RC in x]

    for k in range(0, len(y)):
        if y[k] < y[0]:
            break
    else:
        while ((y[k] - y[k - 1]) > 0.0) and (k > 1):
            k -= 1
        if k < 2:
            k = len(y) - 1

    axes[0].plot(
        x[:k],
        y[:k],
        color=colors.get("tau_var_fit", COLOR_RED),
        label="Fit",
    )


def _adjust_method_format_coord(
    method: int,
    axes: List["Axes"]  # noqa: F821
):
    x_fmt: str = ""
    y1_fmt: str = ""
    y2_fmt: str = ""
    if method == 1:
        pass
    elif method == 2:
        y2_fmt = "none"
    elif method == 3:
        y2_fmt = "none"
    elif method == 4:
        y2_fmt = "none"
    elif method == 5:
        y2_fmt = "none"
    elif method == 6:
        y2_fmt = "none"
    else:
        raise NotImplementedError()

    axes[1].format_coord = _format_coord_two_y_axes(
        ax2=axes[1],
        ax1=axes[0],
        x_fmt=x_fmt,
        y1_fmt=y1_fmt,
        y2_fmt=y2_fmt,
        x_prefix="x=",
        y1_prefix="y1=" if y2_fmt.strip() != "none" else "y=",
        y2_prefix="y2=",
    )


def plot_num_RC_suggestion_method(
    tests: List[KramersKronigResult],
    method: int,
    lower_limit: int = 0,
    upper_limit: int = 0,
    colors: Optional[Dict[str, str]] = None,
    markers: Optional[Dict[str, str]] = None,
    figure: Optional["Figure"] = None,  # noqa: F821
    legend: bool = True,
    axes: Optional[List["Axes"]] = None,  # noqa: F821
    adjust_axes: bool = True,
    colored_axes: bool = False,
    **kwargs,
) -> Tuple["Figure", List["Axes"]]:  # noqa: F821
    """
    Plot the data used by a specific method to suggest the optimal number of RC elements.

    Parameters
    ----------
    tests: List[KramersKronigResult]
        The test results to plot.

    method: int
        The integer identifier assigned to a method. See |suggest_num_RC| for more information about each method.

    lower_limit: int, optional
        The lower limit of the number of RC elements to consider valid.

    upper_limit: int, optional
        The upper limit of the number of RC elements to consider valid.

    colors: Optional[Dict[str, str]], optional
        The colors of the markers or lines. Valid keys: 'chisqr', 'num_RC', 'real', 'imaginary', 'data_impedance', 'impedance', 'data_magnitude', 'data_phase', 'magnitude', 'phase'.

    markers: Optional[Dict[str, str]], optional
        The markers to use when not plotting lines. Valid keys: 'chisqr', 'num_RC', 'real', 'imaginary', 'data_impedance', 'data_magnitude', 'data_phase'.

    figure: Optional[|Figure|], optional
        The matplotlib.figure.Figure instance to use when plotting the data.

    legend: bool, optional
        Whether or not to add a legend.

    axes: Optional[List[|Axes|]], optional
        A list of matplotlib.axes.Axes instances to use when plotting the data.

    adjust_axes: bool, optional
        Whether or not to adjust the axes (label, scale, limits, etc.).

    colored_axes: bool, optional
        Color the y-axes.

    **kwargs

    Returns
    -------
    Tuple[|Figure|, List[|Axes|]]
    """
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.ticker import MaxNLocator

    if not isinstance(tests, list):
        raise TypeError(f"Expected a list instead of {tests=}")

    if colors is None:
        colors = {}
    elif not isinstance(colors, dict):
        raise TypeError(f"Expected a dictionary or None instead of {colors=}")
    colors["left_yaxis"] = COLOR_BLACK
    colors["right_yaxis"] = COLOR_BLACK

    if markers is None:
        markers = {}
    elif not isinstance(markers, dict):
        raise TypeError(f"Expected a dictionary or None instead of {markers=}")

    axis: Axes
    if figure is None:
        figure, axis = plt.subplots()
        axes = [axis, axis.twinx()]

    _validate_figure(figure, axes, num_axes=2)

    x = [t.num_RC for t in tests]

    i: int = 0
    if (lower_limit > 0) and (lower_limit in x):
        i = x.index(lower_limit)

    j: int = len(x) - 1
    if (upper_limit > 0) and (upper_limit > lower_limit) and (upper_limit in x):
        j = x.index(upper_limit) + 1

    show_excluded: bool = not (i == 0 and j == -1)  # TODO: This is always true?
    legend_kwargs = {}

    options = (
        _plot_suggestion_method_1,
        _plot_suggestion_method_2,
        _plot_suggestion_method_3,
        _plot_suggestion_method_4,
        _plot_suggestion_method_5,
        _plot_suggestion_method_6,
    )

    if not _is_integer(method):
        raise TypeError(f"Expected an integer instead of {method=}")
    elif not (1 <= method <= len(options)):
        raise ValueError(f"Expected a value in the range [1, {len(options)}]")

    options[method - 1](
        tests,
        colors,
        markers,
        legend_kwargs,
        i,
        j,
        show_excluded,
        figure,
        axes,
        **kwargs,
    )

    if not _is_boolean(legend):
        raise TypeError(f"Expected a boolean instead of {legend=}")
    elif legend:
        handles, labels = _combine_legends(axes)
        if len(handles) != len(labels):
            raise ValueError(f"Expected {len(handles)=} == {len(labels)=}")

        k: int
        for k in range(0, len(labels)):
            if r"N_{\tau\rm, opt}" in labels[k]:
                labels.append(labels.pop(k))
                handles.append(handles.pop(k))
                break

        axes[1].legend(handles, labels, **legend_kwargs)

    if not _is_boolean(adjust_axes):
        raise TypeError(f"Expected a boolean instead of {adjust_axes=}")
    elif adjust_axes:
        min_x: int = 0
        max_x: int = min(
            (
                max(x) + 1,
                max(
                    (
                        upper_limit + 5,
                        min(
                            (
                                max(x),
                                len(tests[0].get_frequencies()),
                            )
                        ),
                    )
                ),
            )
        )

        axes[0].set_xlim(min_x, max_x)
        axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
        axes[0].set_xlabel(r"$N_\tau$")
        _adjust_method_format_coord(method, axes[:2])

    if not _is_boolean(colored_axes):
        raise TypeError(f"Expected a boolean instead of {colored_axes=}")
    elif colored_axes and len(x) > 1:
        _color_axis(axes[0], colors["left_yaxis"], right=False)
        _color_axis(axes[1], colors["right_yaxis"], right=True)

    return (
        figure,
        axes,
    )


def _validate_suggestion(suggestion: Tuple[KramersKronigResult, Dict[int, float], int, int]):
    if not isinstance(suggestion, tuple):
        raise TypeError(f"Expected a tuple instead of {suggestion=}")
    elif not isinstance(suggestion[1], dict):
        raise TypeError(f"Expected a dictionary instead of {suggestion[0]=}")
    elif not all(map(lambda k: _is_integer(k), suggestion[1].keys())):
        raise TypeError(f"Expected integer keys instead of {suggestion[1].keys()=}")
    elif not all(map(lambda v: _is_floating(v), suggestion[1].values())):
        raise TypeError(f"Expected float values instead of {suggestion[1].values()=}")
    elif not _is_integer(suggestion[2]):
        raise TypeError(f"Expected an integer instead of {suggestion[2]=}")
    elif not _is_integer(suggestion[3]):
        raise TypeError(f"Expected an integer instead of {suggestion[3]=}")


def plot_num_RC_suggestion(
    suggestion: Tuple[KramersKronigResult, Dict[int, float], int, int],
    colors: Optional[Dict[str, str]] = None,
    markers: Optional[Dict[str, str]] = None,
    figure: Optional["Figure"] = None,  # noqa: F821
    legend: bool = True,
    axes: Optional[List["Axes"]] = None,  # noqa: F821
    adjust_axes: bool = True,
    colored_axes: bool = False,
    **kwargs,
) -> Tuple["Figure", List["Axes"]]:  # noqa: F821
    """
    Plot the scores used to suggest the optimal number of RC elements and highlight that number of RC elements.

    Parameters
    ----------
    suggestion: Tuple[KramersKronigResult, Dict[int, float], int, int]
        The return value of |suggest_num_RC|

    colors: Optional[Dict[str, str]], optional
        The colors of the markers or lines. Valid keys: 'chisqr', 'num_RC', 'real', 'imaginary', 'data_impedance', 'impedance', 'data_magnitude', 'data_phase', 'magnitude', 'phase'.

    markers: Optional[Dict[str, str]], optional
        The markers to use when not plotting lines. Valid keys: 'chisqr', 'num_RC', 'real', 'imaginary', 'data_impedance', 'data_magnitude', 'data_phase'.

    figure: Optional[|Figure|], optional
        The matplotlib.figure.Figure instance to use when plotting the data.

    legend: bool, optional
        Whether or not to add a legend.

    axes: Optional[List[|Axes|]], optional
        A list of matplotlib.axes.Axes instances to use when plotting the data.

    adjust_axes: bool, optional
        Whether or not to adjust the axes (label, scale, limits, etc.).

    colored_axes: bool, optional
        Color the y-axes.

    **kwargs

    Returns
    -------
    Tuple[|Figure|, List[|Axes|]]
    """
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.ticker import MaxNLocator

    _validate_suggestion(suggestion)

    if colors is None:
        colors = {}
    elif not isinstance(colors, dict):
        raise TypeError(f"Expected a dictionary or None instead of {colors=}")
    color_num_RC = colors.get("num_RC", COLOR_BLACK)
    color_suggested = colors.get("suggested", COLOR_MAGENTA)

    if markers is None:
        markers = {}
    elif not isinstance(markers, dict):
        raise TypeError(f"Expected a dictionary or None instead of {markers=}")

    axis: Axes
    if figure is None:
        figure, axis = plt.subplots()
        axes = [axis]

    _validate_figure(figure, axes, num_axes=1)

    x: List[int] = sorted(suggestion[1].keys())
    y: List[float] = [suggestion[1][k] for k in x]
    do_plot_score: bool = bool(max(y) > 0.0)
    if do_plot_score:
        axes[0].bar(x, y, color=color_suggested, alpha=0.25, label="Score")

    test: KramersKronigResult = suggestion[0]
    axes[0].axvline(
        test.num_RC,
        color=color_num_RC,
        linestyle="--",
        label=r"$N_{\tau\rm, opt} = " + str(test.num_RC) + "$",
    )

    if not _is_boolean(legend):
        raise TypeError(f"Expected a boolean instead of {legend=}")
    elif legend:
        axes[0].legend()

    if not _is_boolean(adjust_axes):
        raise TypeError(f"Expected a boolean instead of {adjust_axes=}")
    elif adjust_axes:
        axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
        axes[0].set_xlabel(r"$N_\tau$")

        # In case the user manually chooses a number of RC elements outside of
        # the suggested limits.
        min_x: float
        max_x: float
        min_x, max_x = axes[0].get_xlim()
        if test.num_RC > max_x:
            axes[0].set_xlim(min_x, test.num_RC + 1)

        if do_plot_score:
            axes[0].yaxis.set_major_locator(MaxNLocator(integer=True))
            axes[0].set_ylabel("Score")
        else:
            axes[0].set_yticks([])

    if not _is_boolean(colored_axes):
        raise TypeError(f"Expected a boolean instead of {colored_axes=}")
    elif colored_axes and len(y) > 1 and do_plot_score:
        _color_axis(axes[0], color_suggested, right=True)

    return (figure, axes)


def plot_kramers_kronig_tests(
    tests: List[KramersKronigResult],
    suggestion: Tuple[KramersKronigResult, Dict[int, float], int, int],
    data: DataSet,
    admittance: bool = False,
    estimate_noise: bool = False,
    colors: Optional[Dict[str, str]] = None,
    markers: Optional[Dict[str, str]] = None,
    figure: Optional["Figure"] = None,  # noqa: F821
    title: Optional[str] = None,
    legend: bool = True,
    axes: Optional[List["Axes"]] = None,  # noqa: F821
    adjust_axes: bool = True,
    colored_axes: bool = False,
    limit: float = 0.55,
    moving_average_width: int = 0,
    **kwargs,
) -> Tuple["Figure", List["Axes"]]:  # noqa: F821
    """
    Plot the results of exploratory Kramers-Kronig tests as a Nyquist plot, a Bode plot, a plot of the residuals, and a plot of the |pseudo chi-squared| values and the scores used to suggest the number of RC elements.

    Parameters
    ----------
    tests: List[KramersKronigResult]
        The test results to plot.

    suggestion: Tuple[KramersKronigResult, List[int], int, int]
        The return value of |suggest_num_RC|.

    data: DataSet
        The DataSet instance that was tested.

    admittance: bool, optional
        Whether or not to plot the admittance representation of the data instead of the impedance representation.

    estimate_noise: bool, optional
        Whether or not to plot the estimated standard deviation of the noise based on the pseudo chi-squared value and the number of measured frequencies. The 3-sigma standard deviations are indicated in the plots. The noise is assumed to be Gaussian and the noise in each part (real and imaginary) is assumed to be independent of the other. See `equation 16 in Yrjänä and Bobacka (2024) <https://doi.org/10.1016/j.electacta.2024.144951>`_ for details.

    colors: Optional[Dict[str, str]], optional
        The colors of the markers or lines. Valid keys: 'chisqr', 'num_RC', 'real', 'imaginary', 'data_impedance', 'impedance', 'data_magnitude', 'data_phase', 'magnitude', 'phase'.

    markers: Optional[Dict[str, str]], optional
        The markers to use when not plotting lines. Valid keys: 'chisqr', 'num_RC', 'real', 'imaginary', 'data_impedance', 'data_magnitude', 'data_phase'.

    figure: Optional[|Figure|], optional
        The matplotlib.figure.Figure instance to use when plotting the data.

    title: Optional[str], optional
        The title of the figure.
        If no title is provided, then the label of the DataSet is used instead.

    legend: bool, optional
        Whether or not to add a legend.

    axes: Optional[List[|Axes|]], optional
        A list of matplotlib.axes.Axes instances to use when plotting the data.

    adjust_axes: bool, optional
        Whether or not to adjust the axes (label, scale, limits, etc.).

    colored_axes: bool, optional
        Color the y-axes.

    moving_average_width: int, optional
        The width of the moving average to use when plotting the residuals. Must be an odd integer number greater than or equal to three. Otherwise, the moving averages are not plotted.

    **kwargs

    Returns
    -------
    Tuple[|Figure|, List[|Axes|]]
    """
    # TODO: estimate_noise docstring, reference corresponding equation in the final publication.
    import matplotlib.pyplot as plt

    if not isinstance(tests, list):
        raise TypeError(f"Expected a list instead of {tests=}")

    _validate_suggestion(suggestion)

    if colors is None:
        colors = {}
    elif not isinstance(colors, dict):
        raise TypeError(f"Expected a dictionary or None instead of {colors=}")
    color_chisqr: str = colors.get("chisqr", COLOR_BLACK)
    color_num_RC: str = colors.get("num_RC", COLOR_BLACK)
    color_suggested: str = colors.get("suggested", COLOR_MAGENTA)
    color_real: str = colors.get("real", COLOR_TEAL)
    color_imaginary: str = colors.get("imaginary", COLOR_MAGENTA)
    color_data_impedance: str = colors.get("data_impedance", COLOR_BLACK)
    color_impedance: str = colors.get("impedance", COLOR_RED)
    color_data_magnitude: str = colors.get("data_magnitude", COLOR_BLACK)
    color_data_phase: str = colors.get("data_phase", COLOR_BLACK)
    color_magnitude: str = colors.get("magnitude", COLOR_BLUE)
    color_phase: str = colors.get("phase", COLOR_ORANGE)
    color_data_real: str = colors.get("data_real", COLOR_BLACK)
    color_data_imaginary: str = colors.get("data_imaginary", COLOR_BLACK)

    if markers is None:
        markers = {}
    elif not isinstance(markers, dict):
        raise TypeError(f"Expected a dictionary or None instead of {markers=}")
    marker_data_impedance: str = markers.get("data_impedance", MARKER_CIRCLE)
    marker_data_magnitude: str = markers.get("data_magnitude", MARKER_CIRCLE)
    marker_data_phase: str = markers.get("data_phase", MARKER_SQUARE)
    marker_data_real: str = markers.get("data_real", MARKER_CIRCLE)
    marker_data_imaginary: str = markers.get("data_imaginary", MARKER_SQUARE)

    if figure is None:
        figure, axes = plt.subplots(2, 2)
        assert axes is not None
        axes = [
            # Pseudo chi-squared and score
            axes[0][0],
            axes[0][0].twinx(),
            # Relative residuals
            axes[0][1],
            axes[0][1].twinx(),
            # Nyquist
            axes[1][0],
            # Bode (or real and imaginary if indicating the estimated noise)
            axes[1][1],
            axes[1][1].twinx(),
        ]
    assert axes is not None

    _validate_figure(figure, axes, num_axes=7)

    if title is None:
        title = f"{data.get_label()}\n{suggestion[0].get_label()}"
    elif not isinstance(title, str):
        raise TypeError(f"Expected a string or None instead of {title=}")

    if title != "":
        figure.suptitle(title)

    test: KramersKronigResult
    suggested_num_RCs: List[int]
    lower_limit: int
    upper_limit: int
    test, suggested_num_RCs, lower_limit, upper_limit = suggestion

    pct_noise: float = -1.0
    if estimate_noise:
        pct_noise = test.get_estimated_percent_noise()

    plot_pseudo_chisqr(
        tests,
        lower_limit=lower_limit,
        upper_limit=upper_limit,
        colors={
            "chisqr": color_chisqr,
        },
        legend=False,
        colored_axes=colored_axes,
        figure=figure,
        axes=[axes[0]],
        adjust_axes=adjust_axes,
    )

    plot_num_RC_suggestion(
        suggestion,
        colors={
            "num_RC": color_num_RC,
            "suggested": color_suggested,
        },
        legend=False,
        colored_axes=colored_axes,
        figure=figure,
        axes=[axes[1]],
        adjust_axes=adjust_axes,
    )

    if not _is_boolean(legend):
        raise TypeError(f"Expected a boolean instead of {legend=}")
    elif legend:
        handles, labels = _combine_legends(axes[:2])
        if len(suggestion[1]) < 2:
            if len(handles) != len(labels):
                raise ValueError(f"Expected {len(handles)=} == {len(labels)=}")

            k: int
            for k in range(0, len(labels)):
                if r"\chi^2_{\rm ps" not in labels[k]:
                    continue

                if "excl" in labels[k]:
                    labels[k] = "Excl."
                elif "incl" in labels[k]:
                    labels[k] = "Incl."

        axes[1].legend(handles, labels)

    plot_residuals(
        test,
        admittance=admittance,
        colors={
            "real": color_real,
            "imaginary": color_imaginary,
        },
        legend=legend,
        colored_axes=colored_axes,
        figure=figure,
        axes=[axes[2], axes[3]],
        adjust_axes=adjust_axes,
        limit=limit,
        moving_average_width=moving_average_width,
    )

    plot_nyquist(
        test,
        admittance=admittance,
        colors={
            "impedance": color_impedance,
        },
        markers={
            "impedance": ".",
        },
        line=False,
        label="",
        legend=False,
        colored_axes=colored_axes,
        figure=figure,
        axes=[axes[4]],
        adjust_axes=adjust_axes,
    )

    plot_nyquist(
        data,
        admittance=admittance,
        colors={
            "impedance": color_data_impedance,
        },
        markers={
            "impedance": marker_data_impedance,
        },
        label="Data" if title else None,
        legend=False,
        figure=figure,
        axes=[axes[4]],
        adjust_axes=adjust_axes,
    )

    plot_nyquist(
        test,
        admittance=admittance,
        colors={
            "impedance": color_impedance,
        },
        line=True,
        label="Fit" if title else None,
        legend=legend,
        colored_axes=colored_axes,
        figure=figure,
        axes=[axes[4]],
        adjust_axes=adjust_axes,
    )

    if not _is_boolean(estimate_noise):
        raise TypeError(f"Expected a boolean instead of {estimate_noise=}")
    elif estimate_noise and pct_noise > 0.0:
        handles = []
        labels = []
        for i, linestyle in ((1, ":"), (2, "--"), (3, "-.")):
            for j in (i, -i):
                l = axes[3].axhline(
                    j * pct_noise,
                    color="black",
                    linestyle=linestyle,
                    alpha=0.25,
                    zorder=-5,
                )
                if j > 0:
                    handles.append(l)
                    labels.append(r"$" + str(i) + r"\sigma$")

        if legend:
            existing_legend = axes[3].get_legend()
            axes[3].legend(
                handles,
                labels,
                ncol=3,
                loc="lower center",
                title="Est. equiv. Gaussian noise",
            )
            if existing_legend is not None:
                axes[3].add_artist(existing_legend)

        plot_real_imaginary(
            test,
            admittance=admittance,
            colors={
                "real": color_real,
                "imaginary": color_imaginary,
            },
            markers={
                "real": ".",
                "imaginary": ".",
            },
            line=False,
            label="",
            legend=False,
            colored_axes=colored_axes,
            figure=figure,
            axes=axes[5:7],
            adjust_axes=adjust_axes,
        )

        plot_real_imaginary(
            data,
            admittance=admittance,
            colors={
                "real": color_data_real,
                "imaginary": color_data_imaginary,
            },
            markers={
                "real": marker_data_real,
                "imaginary": marker_data_imaginary,
            },
            label="Data" if title else None,
            legend=False,
            figure=figure,
            axes=axes[5:7],
            adjust_axes=adjust_axes,
        )

        plot_real_imaginary(
            test,
            admittance=admittance,
            colors={
                "real": color_real,
                "imaginary": color_imaginary,
            },
            line=True,
            label="Fit" if title else None,
            legend=legend,
            colored_axes=colored_axes,
            figure=figure,
            axes=axes[5:7],
            adjust_axes=adjust_axes,
        )

        num_per_decade: int = kwargs.get("num_per_decade", 100)
        f: Frequencies = test.get_frequencies(num_per_decade=num_per_decade)
        X_fit: ComplexImpedances = test.get_impedances(
            num_per_decade=num_per_decade
        ) ** (-1 if admittance else 1)
        X_extremes: NDArray[float64] = zeros((2, len(X_fit)), X_fit.dtype)

        sd: NDArray[float64] = pct_noise / 100 * abs(X_fit)
        n_sigma: int = 3

        X_extremes[0][:].real = X_fit.real + sd * n_sigma
        X_extremes[0][:].imag = X_fit.imag + sd * n_sigma
        X_extremes[1][:].real = X_fit.real - sd * n_sigma
        X_extremes[1][:].imag = X_fit.imag - sd * n_sigma

        axes[5].fill_between(
            x=f,
            y2=X_extremes[0].real,
            y1=X_extremes[1].real,
            label=r"Est. $3\sigma$, " + (r"Re($Y$)" if admittance else r"Re($Z$)"),
            color=color_real,
            alpha=0.25,
        )
        axes[6].fill_between(
            x=f,
            y2=X_extremes[0].imag * (1 if admittance else -1),
            y1=X_extremes[1].imag * (1 if admittance else -1),
            label=r"Est. $3\sigma$, " + (r"Im($Y$)" if admittance else r"$-$Im($Z$)"),
            color=color_imaginary,
            alpha=0.25,
        )
        if legend:
            handles, labels = _combine_legends(axes[5:7])
            axes[6].legend(handles, labels)

    else:
        plot_bode(
            test,
            admittance=admittance,
            colors={
                "magnitude": color_magnitude,
                "phase": color_phase,
            },
            markers={
                "magnitude": ".",
                "phase": ".",
            },
            line=False,
            label="",
            colored_axes=colored_axes,
            legend=False,
            figure=figure,
            axes=axes[5:7],
            adjust_axes=adjust_axes,
        )

        plot_bode(
            data,
            admittance=admittance,
            colors={
                "magnitude": color_data_magnitude,
                "phase": color_data_phase,
            },
            markers={
                "magnitude": marker_data_magnitude,
                "phase": marker_data_phase,
            },
            label="Data" if title else None,
            legend=False,
            figure=figure,
            axes=axes[5:7],
            adjust_axes=adjust_axes,
        )

        plot_bode(
            test,
            admittance=admittance,
            colors={
                "magnitude": color_magnitude,
                "phase": color_phase,
            },
            line=True,
            label="Fit" if title else None,
            colored_axes=colored_axes,
            legend=legend,
            figure=figure,
            axes=axes[5:7],
            adjust_axes=adjust_axes,
        )

    if not _is_boolean(adjust_axes):
        raise TypeError(f"Expected a boolean instead of {adjust_axes=}")
    elif adjust_axes:
        axes[1].format_coord = _format_coord_two_y_axes(
            ax1=axes[0],
            ax2=axes[1],
        )

    return (
        figure,
        axes,
    )
