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
from pyimpspec.typing.helpers import (
    Dict,
    List,
    Optional,
    Tuple,
)
from pyimpspec.plot.colors import (
    COLOR_BLACK,
    COLOR_BLUE,
    COLOR_ORANGE,
)
from .markers import (
    MARKER_CIRCLE,
    MARKER_SQUARE,
)
from .helpers import _validate_figure
from .bode import plot_bode
from .nyquist import plot_nyquist


def plot_data(
    data: DataSet,
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

    admittance: bool, optional
        Plot the admittance representation of the immittance data.

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

    if colors is None:
        colors = {}
    elif not isinstance(colors, dict):
        raise TypeError(f"Expected a dictionary or None instead of {colors=}")
    color_impedance: str = colors.get("impedance", COLOR_BLACK)
    color_magnitude: str = colors.get("magnitude", COLOR_BLUE)
    color_phase: str = colors.get("phase", COLOR_ORANGE)

    if markers is None:
        markers = {}
    elif not isinstance(markers, dict):
        raise TypeError(f"Expected a dictionary or None instead of {markers=}")
    marker_impedance: str = markers.get("impedance", MARKER_CIRCLE)
    marker_magnitude: str = markers.get("magnitude", MARKER_CIRCLE)
    marker_phase: str = markers.get("phase", MARKER_SQUARE)

    if figure is None:
        figure, axes = plt.subplots(1, 2)
        axes = [
            axes[0],
            axes[1],
            axes[1].twinx(),
        ]
    assert axes is not None

    _validate_figure(figure, axes, num_axes=3)

    if title is None:
        title = data.get_label()
    elif not isinstance(title, str):
        raise TypeError(f"Expected a string or None instead of {title=}")

    if title != "":
        figure.suptitle(title)

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
        admittance=admittance,
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
        admittance=admittance,
    )

    return (
        figure,
        axes,
    )
