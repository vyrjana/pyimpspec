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
    FitResult,
)
from pyimpspec.analysis.drt import DRTResult
from numpy import ceil
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)
from pyimpspec.typing import (
    Frequencies,
    Residuals,
)
from pyimpspec.plot.colors import (
    COLOR_MAGENTA,
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


def plot_residuals(
    result: Union[TestResult, FitResult, DRTResult],
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
    Plot the residuals of a result.

    Parameters
    ----------
    result: Union[TestResult, FitResult, DRTResult]
        The result to plot.

    colors: Optional[Dict[str, str]], optional
        The colors of the markers or lines. Valid keys: 'real', 'imaginary'.

    markers: Optional[Dict[str, str]], optional
        The markers to use when not plotting lines. Valid keys: 'real', 'imaginary'.

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

    assert hasattr(result, "get_residuals_data")
    assert callable(result.get_residuals_data)
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
    if not axes[0].lines:
        axes[0].axhline(0, color="black", alpha=0.25)
    color_real: str = colors.get("real", COLOR_TEAL)
    color_imaginary: str = colors.get("imaginary", COLOR_MAGENTA)
    marker_real: str = markers.get("real", MARKER_CIRCLE)
    marker_imaginary: str = markers.get("imaginary", MARKER_SQUARE)
    x: Frequencies
    y1: Residuals
    y2: Residuals
    x, y1, y2 = result.get_residuals_data()
    axes[0].scatter(
        x,
        y1,
        marker=marker_real,
        **_get_marker_color_args(marker_real, color_real),
        label=r"$\Delta_{\rm Re(Z)}$",
    )
    axes[1].scatter(
        x,
        y2,
        marker=marker_imaginary,
        **_get_marker_color_args(marker_imaginary, color_imaginary),
        label=r"$\Delta_{\rm Im(Z)}$",
    )
    axes[0].plot(
        x,
        y1,
        color=color_real,
        linestyle="--",
    )
    axes[1].plot(
        x,
        y2,
        color=color_imaginary,
        linestyle=":",
    )
    if adjust_axes:
        axes[0].set_xlabel(r"$f$ (Hz)")
        axes[0].set_ylabel(r"$\Delta_{\rm Re(Z)}\ (\%)$")
        axes[1].set_ylabel(r"$\Delta_{\rm Im(Z)}\ (\%)$")
        _configure_log_scale(axes[0], x=True)
        _configure_log_limits(axes[0], x=True)
        limit: float = max(  # type: ignore
            map(
                abs,
                [
                    min(y1),
                    max(y1),
                    min(y2),
                    max(y2),
                ],
            )
        )
        if limit < 0.5:
            limit = 0.5
        else:
            limit = ceil(limit)
        axes[0].set_ylim(-limit, limit)
        axes[1].set_ylim(-limit, limit)
    if legend is True:
        axes[1].legend(*_combine_legends(axes), loc=9, ncols=2)
    if colored_axes is True:
        _color_axis(axes[0], color_real, left=True)
        _color_axis(axes[1], color_real, left=True)
        _color_axis(axes[1], color_imaginary, right=True)
    return (
        figure,
        axes,
    )
