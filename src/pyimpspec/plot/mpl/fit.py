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
from pyimpspec.analysis import (
    TestResult,
    FitResult,
)
from pyimpspec.analysis.drt import DRTResult
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)
from pyimpspec.plot.colors import (
    COLOR_BLACK,
    COLOR_BLUE,
    COLOR_MAGENTA,
    COLOR_ORANGE,
    COLOR_RED,
    COLOR_TEAL,
)
from pyimpspec.plot.mpl.markers import (
    MARKER_CIRCLE,
    MARKER_SQUARE,
)
from .bode import plot_bode
from .residuals import plot_residuals
from .nyquist import plot_nyquist


def plot_fit(
    fit: Union[TestResult, FitResult, DRTResult],
    data: DataSet,
    label: Optional[str] = None,
    colors: Optional[Dict[str, str]] = None,
    markers: Optional[Dict[str, str]] = None,
    num_per_decade: int = 100,
    figure: Optional["Figure"] = None,  # noqa: F821
    title: Optional[str] = None,
    legend: bool = True,
    axes: Optional[List["Axes"]] = None,  # noqa: F821
    adjust_axes: bool = True,
    colored_axes: bool = False,
    **kwargs,
) -> Tuple["Figure", List["Axes"]]:  # noqa: F821
    """
    Plot the result of a fit as a Nyquist plot, a Bode plot, and a plot of the residuals.

    Parameters
    ----------
    fit: Union[TestResult, FitResult, DRTResult]
        The circuit fit or test result.

    data: DataSet
        The DataSet instance that a circuit was fitted to.

    label: Optional[str], optional
        The optional label to use in the legend.

    colors: Optional[Dict[str, str]], optional
        The colors of the markers or lines. Valid keys: 'real', 'imaginary', 'data_impedance', 'impedance', 'data_magnitude', 'data_phase', 'magnitude', 'phase'.

    markers: Optional[Dict[str, str]], optional
        The markers to use when not plotting lines. Valid keys: 'real', 'imaginary', 'data_impedance', 'data_magnitude', 'data_phase'.

    num_per_decade: int, optional
        If the data being plotted is not a DataSet instance (e.g. a TestResult instance), then this parameter can be used to change how many points are used to draw the line (i.e. how smooth or angular the line looks).

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

    assert hasattr(fit, "get_label") and callable(fit.get_label)
    assert hasattr(fit, "get_frequencies") and callable(fit.get_frequencies)
    assert hasattr(fit, "get_impedances") and callable(fit.get_impedances)
    assert hasattr(fit, "get_residuals_data") and callable(fit.get_residuals_data)
    assert hasattr(data, "get_frequencies") and callable(data.get_frequencies)
    assert hasattr(data, "get_impedances") and callable(data.get_impedances)
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
    assert isinstance(num_per_decade, int), num_per_decade
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
            tmp["upper right"],
            tmp["upper right"].twinx(),
            tmp["bottom"],
            tmp["bottom"].twinx(),
        ]
        if title is None:
            if isinstance(fit, TestResult):
                title = (
                    fit.circuit.to_string()
                    .replace("K", r"$\rm (RC)_" + f"{{{str(fit.num_RC)}}}$", 1)  # type: ignore
                    .replace("K", "")
                )
            elif isinstance(fit, FitResult):
                title = fit.circuit.to_string()
            elif hasattr(fit, "get_label") and callable(fit.get_label):
                title = fit.get_label()  # type: ignore
            title = f"{data.get_label()}\n{title}"
        if title != "":
            figure.suptitle(title)
    assert isinstance(axes, list)
    assert len(axes) == 5, axes
    assert all(map(lambda _: isinstance(_, Axes), axes))
    color_real: str = colors.get("real", COLOR_TEAL)
    color_imaginary: str = colors.get("imaginary", COLOR_MAGENTA)
    color_data_impedance: str = colors.get("data_impedance", COLOR_BLACK)
    color_impedance: str = colors.get("impedance", COLOR_RED)
    color_data_magnitude: str = colors.get("data_magnitude", COLOR_BLACK)
    color_data_phase: str = colors.get("data_phase", COLOR_BLACK)
    color_magnitude: str = colors.get("magnitude", COLOR_BLUE)
    color_phase: str = colors.get("phase", COLOR_ORANGE)
    marker_real: str = markers.get("real", MARKER_CIRCLE)
    marker_imaginary: str = markers.get("imaginary", MARKER_SQUARE)
    marker_data_impedance: str = markers.get("data_impedance", MARKER_CIRCLE)
    marker_data_magnitude: str = markers.get("data_magnitude", MARKER_CIRCLE)
    marker_data_phase: str = markers.get("data_phase", MARKER_SQUARE)
    plot_nyquist(
        data,
        colors={
            "impedance": color_data_impedance,
        },
        markers={
            "impedance": marker_data_impedance,
        },
        label="Data" if title else label,
        legend=False,
        figure=figure,
        axes=[axes[0]],
        adjust_axes=adjust_axes,
    )
    plot_bode(
        data,
        colors={
            "magnitude": color_data_magnitude,
            "phase": color_data_phase,
        },
        markers={
            "magnitude": marker_data_magnitude,
            "phase": marker_data_phase,
        },
        label="Data" if title else label,
        legend=False,
        figure=figure,
        axes=[axes[1], axes[2]],
        adjust_axes=adjust_axes,
    )
    plot_nyquist(
        fit,
        colors={
            "impedance": color_impedance,
        },
        line=True,
        label="Fit" if title else label,
        legend=legend,
        colored_axes=colored_axes,
        figure=figure,
        axes=[axes[0]],
        adjust_axes=adjust_axes,
        num_per_decade=num_per_decade,
    )
    plot_bode(
        fit,
        colors={
            "magnitude": color_magnitude,
            "phase": color_phase,
        },
        line=True,
        label="Fit" if title else label,
        legend=legend,
        colored_axes=colored_axes,
        figure=figure,
        axes=[axes[1], axes[2]],
        adjust_axes=adjust_axes,
        num_per_decade=num_per_decade,
    )
    plot_residuals(
        fit,
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
