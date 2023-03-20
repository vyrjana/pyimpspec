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

from pyimpspec.data import DataSet
from pyimpspec.analysis.drt import DRTResult
from numpy import (
    floating,
    issubdtype,
)
from typing import (
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
from pyimpspec.plot.mpl.markers import (
    MARKER_CIRCLE,
    MARKER_SQUARE,
)
from .complex import plot_complex
from .gamma import plot_gamma
from .residuals import plot_residuals


def plot_drt(
    drt: DRTResult,
    data: DataSet,
    peak_threshold: float = -1.0,
    label: Optional[str] = None,
    colors: Optional[Dict[str, str]] = None,
    markers: Optional[Dict[str, str]] = None,
    figure: Optional["Figure"] = None,  # noqa: F821
    title: Optional[str] = None,
    legend: bool = True,
    axes: Optional[List["Axes"]] = None,  # noqa: F821
    adjust_axes: bool = True,
    colored_axes: bool = False,
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

    **kwargs

    Returns
    -------
    Tuple[|Figure|, List[|Axes|]]
    """
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    assert hasattr(drt, "get_label") and callable(drt.get_label)
    assert hasattr(drt, "get_frequencies") and callable(drt.get_frequencies)
    assert hasattr(drt, "get_impedances") and callable(drt.get_impedances)
    assert hasattr(drt, "get_residuals_data") and callable(drt.get_residuals_data)
    assert hasattr(data, "get_frequencies") and callable(data.get_frequencies)
    assert hasattr(data, "get_impedances") and callable(data.get_impedances)
    assert (
        issubdtype(type(peak_threshold), floating) and peak_threshold <= 1.0
    ), peak_threshold
    if colors is None:
        colors = {}
    if markers is None:
        markers = {}
    assert isinstance(colors, dict), colors
    assert isinstance(markers, dict), markers
    assert isinstance(title, str) or title is None, title
    assert isinstance(label, str) or label is None, label
    assert isinstance(legend, bool), legend
    assert isinstance(colored_axes, bool), colored_axes
    assert isinstance(figure, Figure) or figure is None, figure
    assert isinstance(adjust_axes, bool), adjust_axes
    if figure is None:
        assert axes is None
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
        if title is None:
            title = f"{data.get_label()}\n{drt.get_label()}"  # type: ignore
        if title != "":
            figure.suptitle(title)
    assert isinstance(axes, list)
    assert len(axes) == 5, axes
    assert all(map(lambda _: isinstance(_, Axes), axes))
    if label is None:
        if hasattr(drt, "get_label") and callable(drt.get_label):
            label = drt.get_label()
    color_gamma: str = colors.get("gamma", COLOR_BLACK)
    color_real: str = colors.get("real", COLOR_TEAL)
    color_imaginary: str = colors.get("imaginary", COLOR_MAGENTA)
    color_data_real: str = colors.get("data_real", COLOR_BLACK)
    color_data_imaginary: str = colors.get("data_imaginary", COLOR_BLACK)
    marker_real: str = markers.get("real", MARKER_CIRCLE)
    marker_imaginary: str = markers.get("imaginary", MARKER_SQUARE)
    marker_data_real: str = markers.get("data_real", MARKER_CIRCLE)
    marker_data_imaginary: str = markers.get("data_imaginary", MARKER_SQUARE)
    plot_complex(
        data,
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
    plot_complex(
        drt,
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
    )
    plot_residuals(
        drt,
        colors={
            "real": color_real,
            "imaginary": color_imaginary,
        },
        markers={
            "real": marker_real,
            "imaginary": marker_imaginary,
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
