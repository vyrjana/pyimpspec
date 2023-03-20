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

from pyimpspec.analysis import (
    TestResult,
)
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
)
from pyimpspec.plot.colors import (
    COLOR_MAGENTA,
    COLOR_RED,
    COLOR_TEAL,
)
from pyimpspec.plot.mpl.markers import (
    MARKER_CIRCLE,
    MARKER_SQUARE,
)
from pyimpspec.plot.mpl.utility import (
    _color_axis,
    _combine_legends,
    _configure_log_limits,
    _configure_log_scale,
    _get_marker_color_args,
)


def plot_mu_xps(
    tests: List[TestResult],
    mu_criterion: float,
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
    Plot the |mu| values and |pseudo chi-squared| values of Kramers-Kronig test results.

    Parameters
    ----------
    tests: List[TestResult]
        The results to plot.

    mu_criterion: float
        The |mu|-criterion to apply.

    colors: Optional[Dict[str, str]], optional
        The colors of the markers or lines. Valid keys: 'mu', 'xps', 'criterion'.

    markers: Optional[Dict[str, str]], optional
        The markers to use when not plotting lines. Valid keys: 'mu', 'xps'.

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
    from matplotlib.figure import Figure

    assert isinstance(tests, list)
    assert all(map(lambda _: isinstance(_, TestResult), tests))
    if colors is None:
        colors = {}
    if markers is None:
        markers = {}
    assert isinstance(colors, dict), colors
    assert isinstance(markers, dict), markers
    assert isinstance(figure, Figure) or figure is None, figure
    if figure is None:
        assert axes is None
        axis: Axes
        figure, axis = plt.subplots()
        axes = [axis, axis.twinx()]
    assert isinstance(axes, list)
    assert len(axes) == 2, axes
    assert all(map(lambda _: isinstance(_, Axes), axes))
    color_mu: str = colors.get("mu", COLOR_MAGENTA)
    color_xps: str = colors.get("xps", COLOR_TEAL)
    color_criterion: str = colors.get("criterion", COLOR_RED)
    marker_mu: str = markers.get("mu", MARKER_CIRCLE)
    marker_xps: str = markers.get("xps", MARKER_SQUARE)
    x: List[int] = []
    y1: List[float] = []
    y2: List[float] = []
    for test in sorted(tests, key=lambda _: _.num_RC):
        x.append(test.num_RC)
        y1.append(test.mu)
        y2.append(test.pseudo_chisqr)
    tests = list(
        sorted(tests, key=lambda _: _.calculate_score(mu_criterion), reverse=True)
    )
    axes[0].axvline(
        tests[0].num_RC,
        color="black",
        linestyle=":",
        alpha=0.5,
        label=f"#RC={tests[0].num_RC}",
    )
    axes[0].axhline(
        mu_criterion,
        color=color_criterion,
        alpha=0.5,
        label=r"$\mu$-crit.",
    )
    axes[0].scatter(
        x,
        y1,
        marker=marker_mu,
        **_get_marker_color_args(marker_mu, color_mu),
        label=r"$\mu$",
    )
    axes[1].scatter(
        x,
        y2,
        marker=marker_xps,
        **_get_marker_color_args(marker_xps, color_xps),
        label=r"$\chi^2_{\rm ps.}$",
    )
    axes[0].plot(
        x,
        y1,
        color=color_mu,
        linestyle="--",
    )
    axes[1].plot(
        x,
        y2,
        color=color_xps,
        linestyle=":",
    )
    if adjust_axes:
        axes[0].set_xlabel(r"Number of (RC) elements")
        axes[0].set_ylabel(r"$\mu$")
        axes[1].set_ylabel(r"$\chi^{2}_{\rm ps.}$")
        _configure_log_scale(axes[1], y=True)
        _configure_log_limits(axes[1], y=True)
        axes[0].set_ylim(-0.1, 1.1)
    if legend is True:
        axes[1].legend(*_combine_legends(axes), loc=1)
    if colored_axes is True:
        _color_axis(axes[0], color_mu, left=True)
        _color_axis(axes[1], color_mu, left=True)
        _color_axis(axes[1], color_xps, right=True)
    return (
        figure,
        axes,
    )
