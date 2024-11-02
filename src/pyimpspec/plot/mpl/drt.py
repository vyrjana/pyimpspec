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

from pyimpspec.data import DataSet
from pyimpspec.analysis.drt import DRTResult
from pyimpspec.typing.helpers import (
    Dict,
    List,
    Optional,
    Tuple,
)
from pyimpspec.plot.colors import (
    COLOR_BLACK,
    COLOR_MAGENTA,
    COLOR_TEAL,
)
from .markers import (
    MARKER_CIRCLE,
    MARKER_DOT,
    MARKER_SQUARE,
)
from .real_imaginary import plot_real_imaginary
from .gamma import plot_gamma
from .residuals import plot_residuals
from .helpers import _validate_figure


def plot_drt(
    drt: DRTResult,
    data: DataSet,
    peak_threshold: float = -1.0,
    label: Optional[str] = None,
    admittance: bool = False,
    colors: Optional[Dict[str, str]] = None,
    markers: Optional[Dict[str, str]] = None,
    figure: Optional["Figure"] = None,  # noqa: F821
    title: Optional[str] = None,
    legend: bool = True,
    axes: Optional[List["Axes"]] = None,  # noqa: F821
    adjust_axes: bool = True,
    colored_axes: bool = False,
    frequency: bool = False,
    **kwargs,
) -> Tuple["Figure", List["Axes"]]:  # noqa: F821
    """
    Plot the result of calculating the distribution of relaxation times (DRT) as a Bode plot, a DRT plot, and a plot of the residuals.

    Parameters
    ----------
    drt: DRTResult
        The result to plot.

    data: DataSet
        The DataSet instance that was used in the DRT calculations.

    peak_threshold: float, optional
        The threshold to use for identifying and marking peaks (0.0 to 1.0, relative to the highest peak).
        Negative values disable marking peaks.

    label: Optional[str], optional
        The optional label to use in the legend.

    admittance: bool, optional
        Plot the admittance representation of the immittance data.

    colors: Optional[Dict[str, str]], optional
        The colors of the markers or lines. Valid keys: 'gamma', 'real', 'imaginary', 'data_real', 'data_imaginary'.

    markers: Optional[Dict[str, str]], optional
        The markers to use when not plotting lines. Valid keys: 'real', 'imaginary', 'data_real', 'data_imaginary'.

    figure: Optional[|Figure|], optional
        The matplotlib.figure.Figure instance to use when plotting the data.

    title: Optional[str], optional
        The title of the figure.
        If no title is provided, then the circuit description code (and label of the DataSet) is used instead.

    legend: bool, optional
        Whether or not to add a legend.

    axes: Optional[List[|Axes|]], optional
        A list of matplotlib.axes.Axes instances to use when plotting the data.

    adjust_axes: bool, optional
        Whether or not to adjust the axes (label, scale, limits, etc.).

    colored_axes: bool, optional
        Color the y-axes.

    frequency: bool, optional
        Plot gamma as a function of frequency.

    **kwargs

    Returns
    -------
    Tuple[|Figure|, List[|Axes|]]
    """
    import matplotlib.pyplot as plt

    if figure is None:
        figure, tmp = plt.subplot_mosaic(
            [["upper left", "upper right"], ["bottom", "bottom"]],
            gridspec_kw={
                "width_ratios": [1, 1],
                "height_ratios": [2, 1],
            },
            constrained_layout=True,
        )
        axes = [
            tmp["upper left"],
            tmp["upper left"].twinx(),
            tmp["upper right"],
            tmp["bottom"],
            tmp["bottom"].twinx(),
        ]
    assert axes is not None

    _validate_figure(figure, axes, num_axes=5)

    if title is None:
        title = f"{data.get_label()}\n{drt.get_label()}"  # type: ignore
    elif not isinstance(title, str):
        raise TypeError(f"Expected a string or None instead of {title=}")

    if title != "":
        figure.suptitle(title)

    if colors is None:
        colors = {}
    elif not isinstance(colors, dict):
        raise TypeError(f"Expected a dictionary or None instead of {colors=}")
    color_gamma: str = colors.get("gamma", COLOR_BLACK)
    color_real: str = colors.get("real", COLOR_TEAL)
    color_imaginary: str = colors.get("imaginary", COLOR_MAGENTA)
    color_data_real: str = colors.get("data_real", COLOR_BLACK)
    color_data_imaginary: str = colors.get("data_imaginary", COLOR_BLACK)

    if markers is None:
        markers = {}
    elif not isinstance(markers, dict):
        raise TypeError(f"Expected a dictionary or None instead of {markers=}")
    marker_real: str = markers.get("real", MARKER_DOT)
    marker_imaginary: str = markers.get("imaginary", MARKER_DOT)
    marker_data_real: str = markers.get("data_real", MARKER_CIRCLE)
    marker_data_imaginary: str = markers.get("data_imaginary", MARKER_SQUARE)

    plot_real_imaginary(
        drt,
        admittance=admittance,
        colors={
            "real": color_real,
            "imaginary": color_imaginary,
        },
        markers={
            "real": marker_real,
            "imaginary": marker_imaginary,
        },
        line=False,
        label="",
        legend=False,
        colored_axes=colored_axes,
        figure=figure,
        axes=axes[0:2],
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
        axes=axes[0:2],
        adjust_axes=adjust_axes,
    )

    plot_real_imaginary(
        drt,
        admittance=admittance,
        colors={
            "real": color_real,
            "imaginary": color_imaginary,
        },
        line=True,
        label="Fit" if title else label,
        legend=legend,
        colored_axes=colored_axes,
        figure=figure,
        axes=axes[0:2],
        adjust_axes=adjust_axes,
    )

    plot_gamma(
        drt,
        peak_threshold=peak_threshold,
        colors={
            "gamma": color_gamma,
        },
        label=label,
        legend=legend,
        colored_axes=colored_axes,
        figure=figure,
        axes=[axes[2]],
        adjust_axes=adjust_axes,
        frequency=frequency,
    )

    plot_residuals(
        drt,
        colors={
            "real": color_real,
            "imaginary": color_imaginary,
        },
        legend=legend,
        colored_axes=colored_axes,
        figure=figure,
        axes=[axes[3], axes[4]],
        adjust_axes=adjust_axes,
    )

    return (
        figure,
        axes,
    )
