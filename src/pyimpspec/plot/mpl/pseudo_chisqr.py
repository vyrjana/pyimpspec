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

from pyimpspec.analysis import (
    KramersKronigResult,
)
from pyimpspec.typing.helpers import (
    Dict,
    List,
    Optional,
    Tuple,
    _is_boolean,
    _is_integer,
)
from pyimpspec.plot.colors import COLOR_BLACK
from .markers import MARKER_CIRCLE
from .helpers import (
    _color_axis,
    _configure_log_limits,
    _configure_log_scale,
    _initialize_figure,
    _validate_figure,
)


def plot_pseudo_chisqr(
    tests: Optional[List[KramersKronigResult]],
    lower_limit: int = -1,
    upper_limit: int = -1,
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
    Plot the |pseudo chi-squared| values of Kramers-Kronig test results.

    Parameters
    ----------
    tests: Optional[List[KramersKronigResult]]
        The results to plot.

    lower_limit: int, optional
        The lower limit of the number of RC elements to consider valid.

    upper_limit: int, optional
        The upper limit of the number of RC elements to consider valid.

    colors: Optional[Dict[str, str]], optional
        The colors of the markers or lines. Valid keys: 'chisqr'.

    markers: Optional[Dict[str, str]], optional
        The markers to use when not plotting lines. Valid keys: 'chisqr'.

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
    from matplotlib.axes import Axes
    from matplotlib.ticker import MaxNLocator

    if figure is None:
        figure, axes = _initialize_figure(num_rows=1, num_cols=1)
    assert axes is not None

    _validate_figure(figure, axes, num_axes=1)
    axis: Axes = axes[0]

    if not isinstance(tests, list):
        raise TypeError(f"Expected a list instead of {tests=}")

    if colors is None:
        colors = {}
    elif not isinstance(colors, dict):
        raise TypeError(f"Expected a dictionary or None instead of {colors=}")
    color_chisqr: str = colors.get("chisqr", COLOR_BLACK)

    if markers is None:
        markers = {}
    elif not isinstance(markers, dict):
        raise TypeError(f"Expected a dictionary or None instead of {markers=}")
    marker_chisqr: str = markers.get("chisqr", MARKER_CIRCLE)

    if not _is_integer(lower_limit):
        raise TypeError(f"Expected an integer instead of {lower_limit=}")
    elif not _is_integer(upper_limit):
        raise TypeError(f"Expected an integer instead of {upper_limit=}")

    if tests is not None:
        x: List[int] = []
        y: List[float] = []
        for t in sorted(tests, key=lambda _: _.num_RC):
            x.append(t.num_RC)
            y.append(t.pseudo_chisqr)

        included_kwargs: dict = {
            "color": color_chisqr,
            "marker": marker_chisqr,
            "label": r"$\chi^2_{\rm ps.}$",
        }
        if lower_limit > 0 or upper_limit > 0:
            excluded_kwargs: dict = {
                "edgecolor": color_chisqr,
                "facecolor": "none",
                "marker": marker_chisqr,
                "label": included_kwargs["label"] + ", excl.",
            }
            included_kwargs["label"] += ", incl."

            if lower_limit > 0:
                axis.scatter(
                    [_ for _ in x if _ <= lower_limit],
                    [_ for i, _ in enumerate(y) if x[i] <= lower_limit],
                    **excluded_kwargs,
                )
                del excluded_kwargs["label"]

            if upper_limit > 0:
                axis.scatter(
                    [_ for _ in x if _ >= upper_limit],
                    [_ for i, _ in enumerate(y) if x[i] >= upper_limit],
                    **excluded_kwargs,
                )

            if lower_limit > 0 and upper_limit > 0:
                if lower_limit >= upper_limit:
                    raise ValueError(
                        f"Expected {lower_limit=} to be less than {upper_limit=}"
                    )
                axis.scatter(
                    [_ for _ in x if lower_limit <= _ <= upper_limit],
                    [_ for i, _ in enumerate(y) if lower_limit <= x[i] <= upper_limit],
                    **included_kwargs,
                )
            elif lower_limit > 0:
                axis.scatter(
                    [_ for _ in x if lower_limit <= _],
                    [_ for i, _ in enumerate(y) if lower_limit <= x[i]],
                    **included_kwargs,
                )
            elif upper_limit > 0:
                axis.scatter(
                    [_ for _ in x if _ <= upper_limit],
                    [_ for i, _ in enumerate(y) if x[i] <= upper_limit],
                    **included_kwargs,
                )
        else:
            axis.scatter(x, y, **included_kwargs)

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

        axis.set_xlim(min_x, max_x)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))

        axis.set_xlabel(r"$N_\tau$")
        axis.set_ylabel(r"$\chi^{2}_{\rm ps.}$")

        _configure_log_scale(axis, y=True)
        _configure_log_limits(axis, y=True)

    if not _is_boolean(legend):
        raise TypeError(f"Expected a boolean instead of {legend=}")
    elif legend and tests is not None:
        axis.legend()

    if not _is_boolean(colored_axes):
        raise TypeError(f"Expected a boolean instead of {colored_axes=}")
    elif colored_axes:
        _color_axis(axis, color_chisqr, left=True)

    return (
        figure,
        axes,
    )
