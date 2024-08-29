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
    FitResult,
)
from pyimpspec.analysis.drt import DRTResult
from numpy import ceil
from pyimpspec.typing import (
    Frequencies,
    Residuals,
)
from pyimpspec.typing.helpers import (
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    _is_boolean,
    _is_integer,
)
from pyimpspec.plot.colors import (
    COLOR_MAGENTA,
    COLOR_TEAL,
)
from .helpers import (
    _color_axis,
    _combine_legends,
    _configure_log_limits,
    _configure_log_scale,
    _get_marker_color_args,
    _initialize_figure,
    _validate_figure,
)


def plot_residuals(
    result: Union[KramersKronigResult, FitResult, DRTResult],
    colors: Optional[Dict[str, str]] = None,
    markers: Optional[Dict[str, str]] = None,
    figure: Optional["Figure"] = None,  # noqa: F821
    legend: bool = True,
    axes: Optional[List["Axes"]] = None,  # noqa: F821
    adjust_axes: bool = True,
    colored_axes: bool = False,
    limit: float = -1.0,
    moving_average_width: int = 0,
    **kwargs,
) -> Tuple["Figure", List["Axes"]]:  # noqa: F821
    """
    Plot the residuals of a result.

    Parameters
    ----------
    result: Union[KramersKronigResult, FitResult, DRTResult]
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

    limit: float, optional
        The absolute value of the positive and negative limits to apply to each y-axis.
        If equal to or less than zero, then the limits are adjusted automatically.

    moving_average_width: int, optional
        The width of the moving average. Must be an odd integer number greater than or equal to three. Otherwise, the moving averages are not plotted.

    **kwargs

    Returns
    -------
    Tuple[|Figure|, List[|Axes|]]
    """
    from pandas import Series
    from matplotlib.ticker import MaxNLocator

    if figure is None:
        figure, axes = _initialize_figure(num_rows=1, num_cols=1)
        axes = [axes[0], axes[0].twinx()]
    assert axes is not None

    _validate_figure(figure, axes, num_axes=2)

    if not axes[0].lines:
        axes[0].axhline(0, color="black", alpha=0.25)

    if colors is None:
        colors = {}
    elif not isinstance(colors, dict):
        raise TypeError(f"Expected a dictionary or None instead of {colors=}")
    color_real: str = colors.get("real", COLOR_TEAL)
    color_imaginary: str = colors.get("imaginary", COLOR_MAGENTA)

    if markers is None:
        markers = {}
    elif not isinstance(markers, dict):
        raise TypeError(f"Expected a dictionary or None instead of {markers=}")
    marker_real: str = markers.get("real", "")
    marker_imaginary: str = markers.get("imaginary", "")

    label_real: str = r"$\Delta_{\rm Re(Z)}$"
    label_imaginary: str = r"$\Delta_{\rm Im(Z)}$"

    x: Frequencies
    y1: Residuals
    y2: Residuals
    x, y1, y2 = result.get_residuals_data()
    if len(x) > 0:
        if marker_real != "":
            axes[0].scatter(
                x,
                y1,
                marker=marker_real,
                **_get_marker_color_args(marker_real, color_real),
                label=label_real,
            )
        if marker_imaginary != "":
            axes[1].scatter(
                x,
                y2,
                marker=marker_imaginary,
                **_get_marker_color_args(marker_imaginary, color_imaginary),
                label=label_imaginary,
            )

        if not _is_integer(moving_average_width):
            raise TypeError(f"Expected an integer instead of {moving_average_width=}")
        elif moving_average_width > 0:
            rolling_kwargs = dict(
                center=True,
                min_periods=1,
            )
            y1_rolling = Series(y1).rolling(moving_average_width, **rolling_kwargs)
            y2_rolling = Series(y2).rolling(moving_average_width, **rolling_kwargs)
            n_sigma: int = 3

            axes[0].plot(
                x,
                y1_rolling.mean(),
                color=color_real,
                linestyle="-",
                label=label_real + ", avg.",
            )
            axes[1].plot(
                x,
                y2_rolling.mean(),
                color=color_imaginary,
                linestyle="-",
                label=label_imaginary + ", avg.",
            )

            axes[0].fill_between(
                x=x,
                y1=y1_rolling.mean() - y1_rolling.std() * n_sigma,
                y2=y1_rolling.mean() + y1_rolling.std() * n_sigma,
                color=color_real,
                alpha=0.25,
                label=label_real + f", {n_sigma}" + r"$\sigma$",
            )
            axes[1].fill_between(
                x=x,
                y1=y2_rolling.mean() - y2_rolling.std() * n_sigma,
                y2=y2_rolling.mean() + y2_rolling.std() * n_sigma,
                color=color_imaginary,
                alpha=0.25,
                label=label_imaginary + f", {n_sigma}" + r"$\sigma$",
            )
        else:
            axes[0].plot(
                x,
                y1,
                color=color_real,
                linestyle="-",
                label=label_real if marker_real == "" else None,
            )
            axes[1].plot(
                x,
                y2,
                color=color_imaginary,
                linestyle="-",
                label=label_imaginary if marker_imaginary == "" else None,
            )

    if not _is_boolean(adjust_axes):
        raise TypeError(f"Expected a boolean instead of {adjust_axes=}")
    elif adjust_axes:
        axes[0].set_xlabel(r"$f$ (Hz)")
        axes[0].set_ylabel(
            label_real[: label_real.rfind("$")] + r"\ (\%\ \mathrm{of}\ |Z|)$"
        )
        axes[1].set_ylabel(
            label_imaginary[: label_imaginary.rfind("$")] + r"\ (\%\ \mathrm{of}\ |Z|)$"
        )
        _configure_log_scale(axes[0], x=True)
        _configure_log_limits(axes[0], x=True)

        if limit <= 0.0:
            limit = (
                max(  # type: ignore
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
                if len(y1) > 0
                else 0.5
            )
            if limit <= 0.5:
                limit = 0.55
            else:
                limit = ceil(limit) + 0.05

        axes[0].set_ylim(-limit, limit)
        axes[1].set_ylim(-limit, limit)

        axes[0].yaxis.set_major_locator(MaxNLocator(nbins=5))
        axes[1].yaxis.set_major_locator(MaxNLocator(nbins=5))

    if not _is_boolean(legend):
        raise TypeError(f"Expected a boolean instead of {legend=}")
    elif legend:
        axes[1].legend(*_combine_legends(axes), loc="upper center", ncols=2)

    if not _is_boolean(colored_axes):
        raise TypeError(f"Expected a boolean instead of {colored_axes=}")
    elif colored_axes:
        _color_axis(axes[0], color_real, left=True)
        _color_axis(axes[1], color_real, left=True)
        _color_axis(axes[1], color_imaginary, right=True)

    return (
        figure,
        axes,
    )
