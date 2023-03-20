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
    COLOR_BLUE,
    COLOR_ORANGE,
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
)
from .magnitude import plot_magnitude
from .phase import plot_phase


def plot_bode(
    data: Union[DataSet, TestResult, FitResult, DRTResult],
    label: Optional[str] = None,
    colors: Optional[Dict[str, str]] = None,
    markers: Optional[Dict[str, str]] = None,
    line: bool = False,
    num_per_decade: int = 100,
    figure: Optional["Figure"] = None,  # noqa: F821
    legend: bool = True,
    axes: Optional[List["Axes"]] = None,  # noqa: F821
    adjust_axes: bool = True,
    colored_axes: bool = False,
    **kwargs,
) -> Tuple["Figure", List["Axes"]]:  # noqa: F821
    """
    Plot some data as a Bode plot (Mod(Z) vs f and Phase(Z) vs f).

    Parameters
    ----------
    data: Union[DataSet, TestResult, FitResult, DRTResult]
        The data to plot.

    label: Optional[str], optional
        The optional label to use in the legend.

    colors: Optional[Dict[str, str]], optional
        The colors of the markers or lines. Valid keys: 'magnitude', 'phase'.

    markers: Optional[Dict[str, str]], optional
        The markers to use when not plotting lines. Valid keys: 'magnitude', 'phase'.

    line: bool, optional
        Whether or not lines should be used instead of markers.

    num_per_decade: int, optional
        If the data being plotted is not a DataSet instance (e.g. a TestResult instance), then this parameter can be used to change how many points are used to draw the line (i.e. how smooth or angular the line looks).

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

    assert hasattr(data, "get_frequencies") and callable(data.get_frequencies)
    assert hasattr(data, "get_impedances") and callable(data.get_impedances)
    if colors is None:
        colors = {}
    if markers is None:
        markers = {}
    assert isinstance(colors, dict), colors
    assert isinstance(markers, dict), markers
    assert isinstance(line, bool), line
    assert isinstance(label, str) or label is None, label
    assert isinstance(legend, bool), legend
    assert isinstance(colored_axes, bool), colored_axes
    assert isinstance(figure, Figure) or figure is None, figure
    if figure is None:
        assert axes is None
        axis: Axes
        figure, axis = plt.subplots()
        axes = [axis, axis.twinx()]
    assert isinstance(axes, list), axes
    assert len(axes) == 2, axes
    assert all(map(lambda _: isinstance(_, Axes), axes))
    mag_suffix: str = r"Mod($Z$)"
    phase_suffix: str = r"Phase($Z$)"
    if label is None:
        if hasattr(data, "get_label") and callable(data.get_label):
            label = data.get_label()
        else:
            label = ""
    label_1: str = f"{label}, {mag_suffix}" if label != "" else mag_suffix
    label_2: str = f"{label}, {phase_suffix}" if label != "" else phase_suffix
    color_magnitude: str = colors.get("magnitude", COLOR_BLUE)
    color_phase: str = colors.get("phase", COLOR_ORANGE)
    marker_magnitude: str = markers.get("magnitude", MARKER_CIRCLE)
    marker_phase: str = markers.get("phase", MARKER_SQUARE)
    plot_magnitude(
        data,
        colors={
            "magnitude": color_magnitude,
        },
        markers={
            "magnitude": marker_magnitude,
        },
        line=line,
        label=label_1,
        legend=False,
        figure=figure,
        axes=[axes[0]],
        adjust_axes=adjust_axes,
        num_per_decade=num_per_decade,
    )
    plot_phase(
        data,
        colors={
            "phase": color_phase,
        },
        markers={
            "phase": marker_phase,
        },
        line=line,
        label=label_2,
        legend=False,
        figure=figure,
        axes=[axes[1]],
        adjust_axes=adjust_axes,
        num_per_decade=num_per_decade,
    )
    if adjust_axes:
        _configure_log_scale(axes[0], x=True, y=True)
        _configure_log_limits(axes[0], x=True, y=True)
    if legend is True:
        axes[1].legend(*_combine_legends(axes))
    if colored_axes is True:
        _color_axis(axes[0], color_magnitude, left=True)
        _color_axis(axes[1], color_magnitude, left=True)
        _color_axis(axes[1], color_phase, right=True)
    return (
        figure,
        axes,
    )
