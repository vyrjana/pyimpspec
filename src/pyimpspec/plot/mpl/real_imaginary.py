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
from pyimpspec.analysis import (
    KramersKronigResult,
    FitResult,
)
from pyimpspec.analysis.drt import DRTResult
from pyimpspec.typing.helpers import (
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    _is_boolean,
)
from pyimpspec.plot.colors import (
    COLOR_MAGENTA,
    COLOR_TEAL,
)
from .markers import (
    MARKER_CIRCLE,
    MARKER_SQUARE,
)
from .helpers import (
    _color_axis,
    _combine_legends,
    _configure_log_limits,
    _configure_log_scale,
    _format_coord_two_y_axes,
    _initialize_figure,
    _validate_figure,
)
from .real import plot_real
from .imaginary import plot_imaginary


def plot_real_imaginary(
    data: Optional[Union[DataSet, KramersKronigResult, FitResult, DRTResult]],
    label: Optional[str] = None,
    admittance: bool = False,
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
    Plot the real and imaginary parts of the impedance of some data (Re(Z) vs f and -Im(Z) vs f).

    Parameters
    ----------
    data: Optional[Union[DataSet, KramersKronigResult, FitResult, DRTResult]]
        The data to plot.

    label: Optional[str], optional
        The optional label to use in the legend.

    admittance: bool, optional
        Plot the admittance representation of the immittance data.

    colors: Optional[Dict[str, str]], optional
        The colors of the markers or lines. Valid keys: 'real', 'imaginary'.

    markers: Optional[Dict[str, str]], optional
        The markers to use when not plotting lines. Valid keys: 'real', 'imaginary'.

    line: bool, optional
        Whether or not lines should be used instead of markers.

    num_per_decade: int, optional
        If the data being plotted is not a DataSet instance (e.g. a KramersKronigResult instance), then this parameter can be used to change how many points are used to draw the line (i.e. how smooth or angular the line looks).

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
    if figure is None:
        figure, axes = _initialize_figure(num_rows=1, num_cols=1)
        axes = [axes[0], axes[0].twinx()]
    assert axes is not None

    _validate_figure(figure, axes, num_axes=2)

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
    marker_real: str = markers.get("real", MARKER_CIRCLE)
    marker_imaginary: str = markers.get("imaginary", MARKER_SQUARE)

    if label is None:
        if hasattr(data, "get_label") and callable(data.get_label):
            label = data.get_label()
    elif not isinstance(label, str):
        raise TypeError(f"Expected a string or None instead of {label=}")

    real_suffix: str = r"Re($Y$)" if admittance else r"Re($Z$)"
    imag_suffix: str = r"Im($Y$)" if admittance else r"$-$Im($Z$)"
    label_1: str = (
        f"{label}, {real_suffix}" if label != "" else (real_suffix if legend else "")
    )
    label_2: str = (
        f"{label}, {imag_suffix}" if label != "" else (imag_suffix if legend else "")
    )

    plot_real(
        data,
        admittance=admittance,
        colors={
            "real": color_real,
        },
        markers={
            "real": marker_real,
        },
        line=line,
        label=label_1,
        legend=False,
        figure=figure,
        axes=[axes[0]],
        adjust_axes=adjust_axes,
        num_per_decade=num_per_decade,
    )
    plot_imaginary(
        data,
        admittance=admittance,
        colors={
            "imaginary": color_imaginary,
        },
        markers={
            "imaginary": marker_imaginary,
        },
        line=line,
        label=label_2,
        legend=False,
        figure=figure,
        axes=[axes[1]],
        adjust_axes=adjust_axes,
        num_per_decade=num_per_decade,
    )

    if not _is_boolean(adjust_axes):
        raise TypeError(f"Expected a boolean instead of {adjust_axes=}")
    elif adjust_axes:
        _configure_log_scale(axes[0], x=True)
        _configure_log_limits(axes[0], x=True)
        axes[1].format_coord = _format_coord_two_y_axes(
            ax1=axes[0],
            ax2=axes[1],
        )

    if not _is_boolean(legend):
        raise TypeError(f"Expected a boolean instead of {legend=}")
    elif legend:
        axes[1].legend(*_combine_legends(axes))

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
