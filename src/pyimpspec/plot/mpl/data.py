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
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
)
from pyimpspec.plot.colors import (
    COLOR_BLUE,
    COLOR_ORANGE,
    COLOR_RED,
)
from pyimpspec.plot.mpl.markers import (
    MARKER_CIRCLE,
    MARKER_SQUARE,
)
from .bode import plot_bode
from .nyquist import plot_nyquist


def plot_data(
    data: DataSet,
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
    Plot a DataSet instance as both a Nyquist and a Bode plot.

    Parameters
    ----------
    data: DataSet
        The DataSet instance to plot.

    label: Optional[str], optional
        The optional label to use in the legend.

    colors: Optional[Dict[str, str]], optional
        The colors of the markers or lines. Valid keys: 'impedance', 'magnitude', 'phase'.

    markers: Optional[Dict[str, str]], optional
        The markers to use when not plotting lines. Valid keys: 'impedance', 'magnitude', 'phase'.

    figure: Optional[|Figure|], optional
        The matplotlib.figure.Figure instance to use when plotting the data.

    title: Optional[str], optional
        The title of the figure.
        If not title is provided, then the label of the DataSet is used instead.

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
    if figure is None:
        assert axes is None
        figure, tmp = plt.subplots(1, 2)
        axes = [
            tmp[0],
            tmp[1],
            tmp[1].twinx(),
        ]
        if title is None:
            title = data.get_label()
        if title != "":
            figure.suptitle(title)
    assert isinstance(axes, list)
    assert len(axes) == 3, axes
    assert all(map(lambda _: isinstance(_, Axes), axes))
    color_impedance: str = colors.get("impedance", COLOR_RED)
    color_magnitude: str = colors.get("magnitude", COLOR_BLUE)
    color_phase: str = colors.get("phase", COLOR_ORANGE)
    marker_impedance: str = markers.get("impedance", MARKER_CIRCLE)
    marker_magnitude: str = markers.get("magnitude", MARKER_CIRCLE)
    marker_phase: str = markers.get("phase", MARKER_SQUARE)
    plot_nyquist(
        data,
        colors={
            "impedance": color_impedance,
        },
        markers={
            "impedance": marker_impedance,
        },
        label=label if not title else "",
        legend=False,
        colored_axes=colored_axes,
        figure=figure,
        axes=[axes[0]],
        adjust_axes=adjust_axes,
    )
    plot_bode(
        data,
        colors={
            "magnitude": color_magnitude,
            "phase": color_phase,
        },
        markers={
            "magnitude": marker_magnitude,
            "phase": marker_phase,
        },
        label=label if not title else "",
        legend=legend,
        colored_axes=colored_axes,
        figure=figure,
        axes=[axes[1], axes[2]],
        adjust_axes=adjust_axes,
    )
    return (
        figure,
        axes,
    )
