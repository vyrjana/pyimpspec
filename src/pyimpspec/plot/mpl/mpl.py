# pyimpspec is licensed under the GPLv3 or later (https://www.gnu.org/licenses/gpl-3.0.html).
# Copyright 2022 pyimpspec developers
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

from inspect import signature
from pyimpspec.circuit import Circuit
from pyimpspec.data import DataSet
from pyimpspec.analysis import (
    TestResult,
    FitResult,
)
from pyimpspec.analysis.drt import DRTResult
from pyimpspec.analysis.fitting import _interpolate
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
from numpy import (
    angle,
    ceil,
    floating,
    floor,
    inf,
    issubdtype,
    log10 as log,
    ndarray,
)
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)


# Vibrant color scheme from https://personal.sron.nl/~pault/
_FILLED_MARKERS: Set[str] = set(Line2D.filled_markers)
_UNFILLED_MARKERS: Set[str] = (
    set(m for m, f in Line2D.markers.items() if f != "nothing") - _FILLED_MARKERS
)


def _combine_legends(axes: List[Axes]) -> Tuple[list, list]:
    lines: list = []
    labels: list = []
    ax: Axes
    for ax in axes:
        lin, lab = ax.get_legend_handles_labels()
        lines.extend(lin)
        labels.extend(lab)
    return (
        lines,
        labels,
    )


def _configure_log_limits(axis: Axes, x: bool = False, y: bool = False):
    if x:
        x_min: float
        x_max: float
        x_min, x_max = axis.get_xlim()
        if log(x_max) - log(x_min) < 1.0:
            x_min = 10 ** floor(log(x_min))
            x_max = 10 ** ceil(log(x_max))
            axis.set_xlim(x_min, x_max)
    if y:
        y_min: float
        y_max: float
        y_min, y_max = axis.get_ylim()
        if log(y_max) - log(y_min) < 1.0:
            y_min = 10 ** floor(log(y_min))
            y_max = 10 ** ceil(log(y_max))
            axis.set_ylim(y_min, y_max)


def _configure_log_scale(axis: Axes, x: bool = False, y: bool = False):
    formatter: FormatStrFormatter = FormatStrFormatter("")
    if x:
        axis.set_xscale("log")
        axis.xaxis.set_minor_formatter(formatter)
    if y:
        axis.set_yscale("log")
        axis.yaxis.set_minor_formatter(formatter)


def _get_marker_color_args(marker: str, color: Any) -> Dict[str, Any]:
    if marker in _FILLED_MARKERS:
        return {"edgecolor": color, "facecolor": "none"}
    return {"color": color}


def plot_real_impedance(
    data: Union[DataSet, TestResult, FitResult, DRTResult],
    color: Any = "black",
    marker: str = "o",
    line: bool = False,
    label: Optional[str] = None,
    legend: bool = True,
    fig: Optional[Figure] = None,
    axis: Optional[Axes] = None,
    num_per_decade: int = 100,
    adjust_axes: bool = True,
) -> Tuple[Figure, Axes]:
    """
    Plot the real impedance of some data (Zre vs f).

    Parameters
    ----------
    data: Union[DataSet, TestResult, FitResult, DRTResult]
        The data to plot.
        DataSet instances are plotted using markers (optionally as a line) while all other types of data are plotted as lines.

    color: Any = "black"
        The color of the marker or line.

    marker: str = "o"
        The marker.

    line: bool = False
        Whether or not a DataSet instance should be plotted as a line instead.

    label: Optional[str] = None
        The optional label to use in the legend.

    legend: bool = True
        Whether or not to add a legend.

    fig: Optional[Figure] = None
        The matplotlib.figure.Figure instance to use when plotting the data.

    axis: Optional[Axes] = None
        The matplotlib.axes.Axes instance to use when plotting the data.

    num_per_decade: int = 100
        If the data being plotted is not a DataSet instance (e.g. a TestResult instance), then this parameter can be used to change how many points are used to draw the line (i.e. how smooth or angular the line looks).

    adjust_axes: bool = True
        Whether or not to adjust the axes (label, scale, limits, etc.).
    """
    assert (
        hasattr(data, "get_frequency")
        and callable(data.get_frequency)
        and hasattr(data, "get_impedance")
        and callable(data.get_impedance)
    ), data
    assert type(marker) is str, marker
    assert type(line) is bool, line
    assert type(label) is str or label is None, label
    assert type(legend) is bool, legend
    assert type(fig) is Figure or fig is None, fig
    if fig is None:
        fig, axis = plt.subplots()
    assert axis is not None, axis
    if label is None:
        if hasattr(data, "get_label") and callable(data.get_label):
            label = data.get_label()
        elif isinstance(data, TestResult):
            label = "KK"
        elif isinstance(data, FitResult):
            label = str(data.circuit)
        else:
            label = ""
    x: ndarray
    y: ndarray
    if (
        "num_per_decade" in signature(data.get_frequency).parameters
        and "num_per_decade" in signature(data.get_impedance).parameters
    ):
        x = data.get_frequency(num_per_decade=num_per_decade)
        y = data.get_impedance(num_per_decade=num_per_decade).real
    else:
        x = data.get_frequency()
        y = data.get_impedance().real
    if line is True:
        axis.plot(
            x,
            y,
            color=color,
            linestyle="--",
            label=label if label != "" else None,
        )
    else:
        axis.scatter(
            x,
            y,
            marker=marker,
            **_get_marker_color_args(marker, color),
            label=label if label != "" else None,
        )
    if adjust_axes:
        axis.set_xlabel(r"$f\ (\rm Hz)$")
        axis.set_ylabel(r"$Z_{\rm re}\ (\Omega)$")
        _configure_log_scale(axis, x=True)
        _configure_log_limits(axis, x=True)
    if legend is True:
        axis.legend()
    return (
        fig,
        axis,
    )


def plot_imaginary_impedance(
    data: Union[DataSet, TestResult, FitResult, DRTResult],
    color: Any = "black",
    marker: str = "s",
    line: bool = False,
    label: Optional[str] = None,
    legend: bool = True,
    fig: Optional[Figure] = None,
    axis: Optional[Axes] = None,
    num_per_decade: int = 100,
    adjust_axes: bool = True,
) -> Tuple[Figure, Axes]:
    """
    Plot the imaginary impedance of some data (-Zim vs f).

    Parameters
    ----------
    data: Union[DataSet, TestResult, FitResult, DRTResult]
        The data to plot.
        DataSet instances are plotted using markers (optionally as a line) while all other types of data are plotted as lines.

    color: Any = "black"
        The color of the marker or line.

    marker: str = "s"
        The marker.

    line: bool = False
        Whether or not a DataSet instance should be plotted as a line instead.

    label: Optional[str] = None
        The optional label to use in the legend.

    legend: bool = True
        Whether or not to add a legend.

    fig: Optional[Figure] = None
        The matplotlib.figure.Figure instance to use when plotting the data.

    axis: Optional[Axes] = None
        The matplotlib.axes.Axes instance to use when plotting the data.

    num_per_decade: int = 100
        If the data being plotted is not a DataSet instance (e.g. a TestResult instance), then this parameter can be used to change how many points are used to draw the line (i.e. how smooth or angular the line looks).

    adjust_axes: bool = True
        Whether or not to adjust the axes (label, scale, limits, etc.).
    """
    assert (
        hasattr(data, "get_frequency")
        and callable(data.get_frequency)
        and hasattr(data, "get_impedance")
        and callable(data.get_impedance)
    ), data
    assert type(marker) is str, marker
    assert type(line) is bool, line
    assert type(label) is str or label is None, label
    assert type(legend) is bool, legend
    assert type(fig) is Figure or fig is None, fig
    if fig is None:
        fig, axis = plt.subplots()
    assert axis is not None, axis
    if label is None:
        if hasattr(data, "get_label") and callable(data.get_label):
            label = data.get_label()
        elif isinstance(data, TestResult):
            label = "KK"
        elif isinstance(data, FitResult):
            label = str(data.circuit)
        else:
            label = ""
    x: ndarray
    y: ndarray
    if (
        "num_per_decade" in signature(data.get_frequency).parameters
        and "num_per_decade" in signature(data.get_impedance).parameters
    ):
        x = data.get_frequency(num_per_decade=num_per_decade)
        y = -data.get_impedance(num_per_decade=num_per_decade).imag
    else:
        x = data.get_frequency()
        y = -data.get_impedance().imag
    if line is True:
        axis.plot(
            x,
            y,
            color=color,
            linestyle=":",
            label=label if label != "" else None,
        )
    else:
        axis.scatter(
            x,
            y,
            marker=marker,
            **_get_marker_color_args(marker, color),
            label=label if label != "" else None,
        )
    if adjust_axes:
        axis.set_xlabel(r"$f\ (\rm Hz)$")
        axis.set_ylabel(r"$-Z_{\rm im}\ (\Omega)$")
        _configure_log_scale(axis, x=True)
        _configure_log_limits(axis, x=True)
    if legend is True:
        axis.legend()
    return (
        fig,
        axis,
    )


def plot_complex_impedance(
    data: Union[DataSet, TestResult, FitResult, DRTResult],
    color_real: str = "black",
    color_imaginary: str = "black",
    marker_real: str = "o",
    marker_imaginary: str = "s",
    line: bool = False,
    label: Optional[str] = None,
    legend: bool = True,
    fig: Optional[Figure] = None,
    axes: List[Axes] = [],
    num_per_decade: int = 100,
    adjust_axes: bool = True,
) -> Tuple[Figure, List[Axes]]:
    """
    Plot the real and imaginary parts of the impedance of some data.

    Parameters
    ----------
    data: Union[DataSet, TestResult, FitResult, DRTResult]
        The data to plot.
        DataSet instances are plotted using markers (optionally as a line) while all other types of data are plotted as lines.

    color_real: str = "black"
        The color of the marker or line for the real part of the impedance.

    color_imaginary: str = "black"
        The color of the marker or line for the imaginary part of the impedance.

    marker_real: str = "o"
        The marker for the real part of the impedance.

    marker_imaginary: str = "s"
        The marker for the imaginary part of the impedance.

    line: bool = False
        Whether or not a DataSet instance should be plotted as a line instead.

    label: Optional[str] = None
        The optional label to use in the legend.

    legend: bool = True
        Whether or not to add a legend.

    fig: Optional[Figure] = None
        The matplotlib.figure.Figure instance to use when plotting the data.

    axes: List[Axes] = []
        A list of matplotlib.axes.Axes instances to use when plotting the data.

    num_per_decade: int = 100
        If the data being plotted is not a DataSet instance (e.g. a TestResult instance), then this parameter can be used to change how many points are used to draw the line (i.e. how smooth or angular the line looks).

    adjust_axes: bool = True
        Whether or not to adjust the axes (label, scale, limits, etc.).
    """
    assert (
        hasattr(data, "get_frequency")
        and callable(data.get_frequency)
        and hasattr(data, "get_impedance")
        and callable(data.get_impedance)
    ), data
    assert type(color_real) is str, color_real
    assert type(color_imaginary) is str, color_imaginary
    assert type(marker_real) is str, marker_real
    assert type(marker_imaginary) is str, marker_imaginary
    assert type(line) is bool, line
    assert type(label) is str or label is None, label
    assert type(legend) is bool, legend
    assert type(fig) is Figure or fig is None, fig
    assert type(axes) is list, axes
    real_suffix: str = r"$Z_{\rm re}$"
    imag_suffix: str = r"$Z_{\rm im}$"
    if fig is None:
        axis: Axes
        fig, axis = plt.subplots()
        axes = [axis, axis.twinx()]
    assert len(axes) == 2, axes
    if label is None:
        if hasattr(data, "get_label") and callable(data.get_label):
            label = data.get_label()
        elif isinstance(data, TestResult):
            label = "KK"
        elif isinstance(data, FitResult):
            label = str(data.circuit)
    label_1: str = f"{label}, {real_suffix}" if label != "" else real_suffix
    label_2: str = f"{label}, {imag_suffix}" if label != "" else imag_suffix
    plot_real_impedance(
        data,
        color=color_real,
        marker=marker_real,
        line=line,
        label=label_1,
        legend=False,
        fig=fig,
        axis=axes[0],
        num_per_decade=num_per_decade,
    )
    plot_imaginary_impedance(
        data,
        color=color_imaginary,
        marker=marker_imaginary,
        line=line,
        label=label_2,
        legend=False,
        fig=fig,
        axis=axes[1],
        num_per_decade=num_per_decade,
    )
    if adjust_axes:
        _configure_log_scale(axes[0], x=True)
        _configure_log_limits(axes[0], x=True)
    if legend is True:
        axes[1].legend(*_combine_legends(axes))
    return (
        fig,
        axes,
    )


def plot_nyquist(
    data: Union[DataSet, TestResult, FitResult, DRTResult],
    color: Any = "black",
    marker: str = "o",
    line: bool = False,
    label: Optional[str] = None,
    legend: bool = True,
    fig: Optional[Figure] = None,
    axis: Optional[Axes] = None,
    num_per_decade: int = 100,
    adjust_axes: bool = True,
) -> Tuple[Figure, Axes]:
    """
    Plot some data as a Nyquist plot (-Z" vs Z').

    Parameters
    ----------
    data: Union[DataSet, TestResult, FitResult, DRTResult]
        The data to plot.
        DataSet instances are plotted using markers (optionally as a line) while all other types of data are plotted as lines.

    color: Any = "black"
        The color of the marker or line.

    marker: str = "o"
        The marker.

    line: bool = False
        Whether or not a DataSet instance should be plotted as a line instead.

    label: Optional[str] = None
        The optional label to use in the legend.

    legend: bool = True
        Whether or not to add a legend.

    fig: Optional[Figure] = None
        The matplotlib.figure.Figure instance to use when plotting the data.

    axis: Optional[Axes] = None
        The matplotlib.axes.Axes instance to use when plotting the data.

    num_per_decade: int = 100
        If the data being plotted is not a DataSet instance (e.g. a TestResult instance), then this parameter can be used to change how many points are used to draw the line (i.e. how smooth or angular the line looks).

    adjust_axes: bool = True
        Whether or not to adjust the axes (label, scale, limits, etc.).
    """
    assert hasattr(data, "get_impedance") and callable(data.get_impedance), data
    assert type(line) is bool, line
    assert type(label) is str or label is None, label
    assert type(legend) is bool, legend
    assert type(fig) is Figure or fig is None, fig
    if fig is None:
        fig, axis = plt.subplots()
    assert axis is not None, axis
    if label is None:
        if hasattr(data, "get_label") and callable(data.get_label):
            label = data.get_label()
        elif isinstance(data, TestResult):
            label = "KK"
        elif isinstance(data, FitResult):
            label = str(data.circuit)
        else:
            label = ""
    Z: ndarray
    if "num_per_decade" in signature(data.get_impedance).parameters:
        Z = data.get_impedance(num_per_decade=num_per_decade)
    else:
        Z = data.get_impedance()
    x: ndarray = Z.real
    y: ndarray = -Z.imag
    if line is True:
        axis.plot(
            x,
            y,
            color=color,
            linestyle="--",
            label=label if label != "" else None,
        )
    else:
        axis.scatter(
            x,
            y,
            marker=marker,
            **_get_marker_color_args(marker, color),
            label=label if label != "" else None,
        )
    if adjust_axes:
        axis.set_xlabel(r"$Z_{\rm re}\ (\Omega)$")
        axis.set_ylabel(r"$-Z_{\rm im}\ (\Omega)$")
        axis.set_aspect("equal", adjustable="datalim")
    if legend is True:
        axis.legend()
    return (
        fig,
        axis,
    )


def plot_impedance_magnitude(
    data: Union[DataSet, TestResult, FitResult, DRTResult],
    color: Any = "black",
    marker: str = "o",
    line: bool = False,
    label: Optional[str] = None,
    legend: bool = True,
    fig: Optional[Figure] = None,
    axis: Optional[Axes] = None,
    num_per_decade: int = 100,
    adjust_axes: bool = True,
) -> Tuple[Figure, Axes]:
    """
    Plot the absolute magnitude of the impedance of some data (|Z| vs f).

    Parameters
    ----------
    data: Union[DataSet, TestResult, FitResult, DRTResult]
        The data to plot.
        DataSet instances are plotted using markers (optionally as a line) while all other types of data are plotted as lines.

    color: Any = "black"
        The color of the marker or line.

    marker: str = "o"
        The marker.

    line: bool = False
        Whether or not a DataSet instance should be plotted as a line instead.

    label: Optional[str] = None
        The optional label to use in the legend.

    legend: bool = True
        Whether or not to add a legend.

    fig: Optional[Figure] = None
        The matplotlib.figure.Figure instance to use when plotting the data.

    axis: Optional[Axes] = None
        The matplotlib.axes.Axes instance to use when plotting the data.

    num_per_decade: int = 100
        If the data being plotted is not a DataSet instance (e.g. a TestResult instance), then this parameter can be used to change how many points are used to draw the line (i.e. how smooth or angular the line looks).

    adjust_axes: bool = True
        Whether or not to adjust the axes (label, scale, limits, etc.).
    """
    assert (
        hasattr(data, "get_frequency")
        and callable(data.get_frequency)
        and hasattr(data, "get_impedance")
        and callable(data.get_impedance)
    ), data
    assert type(marker) is str, marker
    assert type(line) is bool, line
    assert type(label) is str or label is None, label
    assert type(legend) is bool, legend
    assert type(fig) is Figure or fig is None, fig
    if fig is None:
        fig, axis = plt.subplots()
    assert axis is not None, axis
    if label is None:
        if hasattr(data, "get_label") and callable(data.get_label):
            label = data.get_label()
        elif isinstance(data, TestResult):
            label = "KK"
        elif isinstance(data, FitResult):
            label = str(data.circuit)
        else:
            label = ""
    x: ndarray
    y: ndarray
    if (
        "num_per_decade" in signature(data.get_frequency).parameters
        and "num_per_decade" in signature(data.get_impedance).parameters
    ):
        x = data.get_frequency(num_per_decade=num_per_decade)
        y = abs(data.get_impedance(num_per_decade=num_per_decade))
    else:
        x = data.get_frequency()
        y = abs(data.get_impedance())
    if line is True:
        axis.plot(
            x,
            y,
            color=color,
            linestyle="--",
            label=label if label != "" else None,
        )
    else:
        axis.scatter(
            x,
            y,
            marker=marker,
            **_get_marker_color_args(marker, color),
            label=label if label != "" else None,
        )
    if adjust_axes:
        axis.set_xlabel(r"$f\ (\rm Hz)$")
        axis.set_ylabel(r"$|Z|\ (\Omega)$")
        _configure_log_scale(axis, x=True, y=True)
        _configure_log_limits(axis, x=True, y=True)
    if legend is True:
        axis.legend()
    return (
        fig,
        axis,
    )


def plot_impedance_phase(
    data: Union[DataSet, TestResult, FitResult, DRTResult],
    color: Any = "black",
    marker: str = "o",
    line: bool = False,
    label: Optional[str] = None,
    legend: bool = True,
    fig: Optional[Figure] = None,
    axis: Optional[Axes] = None,
    num_per_decade: int = 100,
    adjust_axes: bool = True,
) -> Tuple[Figure, Axes]:
    """
    Plot the phase shift of the impedance of some data (phi vs f).

    Parameters
    ----------
    data: Union[DataSet, TestResult, FitResult, DRTResult]
        The data to plot.
        DataSet instances are plotted using markers (optionally as a line) while all other types of data are plotted as lines.

    color: Any = "black"
        The color of the marker or line.

    marker: str = "o"
        The marker.

    line: bool = False
        Whether or not a DataSet instance should be plotted as a line instead.

    label: Optional[str] = None
        The optional label to use in the legend.

    legend: bool = True
        Whether or not to add a legend.

    fig: Optional[Figure] = None
        The matplotlib.figure.Figure instance to use when plotting the data.

    axis: Optional[Axes] = None
        The matplotlib.axes.Axes instance to use when plotting the data.

    num_per_decade: int = 100
        If the data being plotted is not a DataSet instance (e.g. a TestResult instance), then this parameter can be used to change how many points are used to draw the line (i.e. how smooth or angular the line looks).

    adjust_axes: bool = True
        Whether or not to adjust the axes (label, scale, limits, etc.).
    """
    assert (
        hasattr(data, "get_frequency")
        and callable(data.get_frequency)
        and hasattr(data, "get_impedance")
        and callable(data.get_impedance)
    ), data
    assert type(marker) is str, marker
    assert type(line) is bool, line
    assert type(label) is str or label is None, label
    assert type(legend) is bool, legend
    assert type(fig) is Figure or fig is None, fig
    if fig is None:
        fig, axis = plt.subplots()
    assert axis is not None, axis
    if label is None:
        if hasattr(data, "get_label") and callable(data.get_label):
            label = data.get_label()
        elif isinstance(data, TestResult):
            label = "KK"
        elif isinstance(data, FitResult):
            label = str(data.circuit)
        else:
            label = ""
    x: ndarray
    y: ndarray
    if (
        "num_per_decade" in signature(data.get_frequency).parameters
        and "num_per_decade" in signature(data.get_impedance).parameters
    ):
        x = data.get_frequency(num_per_decade=num_per_decade)
        y = -angle(data.get_impedance(num_per_decade=num_per_decade), deg=True)
    else:
        x = data.get_frequency()
        y = -angle(data.get_impedance(), deg=True)
    if line is True:
        axis.plot(
            x,
            y,
            color=color,
            linestyle=":",
            label=label if label != "" else None,
        )
    else:
        axis.scatter(
            x,
            y,
            marker=marker,
            **_get_marker_color_args(marker, color),
            label=label if label != "" else None,
        )
    if adjust_axes:
        axis.set_xlabel(r"$f\ (\rm Hz)$")
        axis.set_ylabel(r"$-\phi\ (^\circ)$")
        _configure_log_scale(axis, x=True)
        _configure_log_limits(axis, x=True)
    if legend is True:
        axis.legend()
    return (
        fig,
        axis,
    )


def plot_bode(
    data: Union[DataSet, TestResult, FitResult, DRTResult],
    color_magnitude: str = "black",
    color_phase: str = "black",
    marker_magnitude: str = "o",
    marker_phase: str = "s",
    line: bool = False,
    label: Optional[str] = None,
    legend: bool = True,
    fig: Optional[Figure] = None,
    axes: List[Axes] = [],
    num_per_decade: int = 100,
    adjust_axes: bool = True,
) -> Tuple[Figure, List[Axes]]:
    """
    Plot some data as a Bode plot (|Z| and phi vs f).

    Parameters
    ----------
    data: Union[DataSet, TestResult, FitResult, DRTResult]
        The data to plot.
        DataSet instances are plotted using markers (optionally as a line) while all other types of data are plotted as lines.

    color_magnitude: str = "black"
        The color of the marker or line for the absolute magnitude of the impedance.

    color_phase: str = "black"
        The color of the marker or line) for the phase shift of the impedance.

    marker_magnitude: str = "o"
        The marker for the absolute magnitude of the impedance.

    marker_phase: str = "s"
        The marker for the phase shift of the impedance.

    line: bool = False
        Whether or not a DataSet instance should be plotted as a line instead.

    label: Optional[str] = None
        The optional label to use in the legend.

    legend: bool = True
        Whether or not to add a legend.

    fig: Optional[Figure] = None
        The matplotlib.figure.Figure instance to use when plotting the data.

    axes: List[Axes] = []
        A list of matplotlib.axes.Axes instances to use when plotting the data.

    num_per_decade: int = 100
        If the data being plotted is not a DataSet instance (e.g. a TestResult instance), then this parameter can be used to change how many points are used to draw the line (i.e. how smooth or angular the line looks).

    adjust_axes: bool = True
        Whether or not to adjust the axes (label, scale, limits, etc.).
    """
    assert (
        hasattr(data, "get_frequency")
        and callable(data.get_frequency)
        and hasattr(data, "get_impedance")
        and callable(data.get_impedance)
    ), data
    assert type(color_magnitude) is str, color_magnitude
    assert type(color_phase) is str, color_phase
    assert type(line) is bool, line
    assert type(label) is str or label is None, label
    assert type(legend) is bool, legend
    assert type(fig) is Figure or fig is None, fig
    assert type(axes) is list, axes
    if fig is None:
        axis: Axes
        fig, axis = plt.subplots()
        axes = [axis, axis.twinx()]
    assert len(axes) == 2, axes
    mag_suffix: str = r"$|Z|$"
    phase_suffix: str = r"$\phi$"
    if label is None:
        if hasattr(data, "get_label") and callable(data.get_label):
            label = data.get_label()
        elif isinstance(data, TestResult):
            label = "KK"
        elif isinstance(data, FitResult):
            label = str(data.circuit)
        else:
            label = ""
    label_1: str = f"{label}, {mag_suffix}" if label != "" else mag_suffix
    label_2: str = f"{label}, {phase_suffix}" if label != "" else phase_suffix
    plot_impedance_magnitude(
        data,
        color=color_magnitude,
        marker=marker_magnitude,
        line=line,
        label=label_1,
        legend=False,
        fig=fig,
        axis=axes[0],
        num_per_decade=num_per_decade,
    )
    plot_impedance_phase(
        data,
        color=color_phase,
        marker=marker_phase,
        line=line,
        label=label_2,
        legend=False,
        fig=fig,
        axis=axes[1],
        num_per_decade=num_per_decade,
    )
    if adjust_axes:
        _configure_log_scale(axes[0], x=True, y=True)
        _configure_log_limits(axes[0], x=True, y=True)
    if legend is True:
        axes[1].legend(*_combine_legends(axes))
    return (
        fig,
        axes,
    )


def plot_residual(
    result: Union[TestResult, FitResult, DRTResult],
    color_real: str = "black",
    color_imaginary: str = "black",
    legend: bool = True,
    fig: Optional[Figure] = None,
    axes: List[Axes] = [],
    adjust_axes: bool = True,
) -> Tuple[Figure, List[Axes]]:
    """
    Plot the residuals of a result.

    Parameters
    ----------
    result: Union[TestResult, FitResult, DRTResult]
        The result to plot.

    color_real: str = "black"
        The color of the markers and line for the residuals of the real parts of the impedances.

    color_imaginary: str = "black"
        The color of the markers and line for the residuals of the imaginary parts of the impedances.

    legend: bool = True
        Whether or not to add a legend.

    fig: Optional[Figure] = None
        The matplotlib.figure.Figure instance to use when plotting the data.

    axes: List[Axes] = []
        A list of matplotlib.axes.Axes instances to use when plotting the data.

    adjust_axes: bool = True
        Whether or not to adjust the axes (label, scale, limits, etc.).
    """
    assert hasattr(result, "get_residual_data") and callable(
        result.get_residual_data
    ), result
    assert type(color_real) is str, color_real
    assert type(color_imaginary) is str, color_imaginary
    assert type(fig) is Figure or fig is None, fig
    assert type(axes) is list, axes
    if fig is None:
        axis: Axes
        fig, axis = plt.subplots()
        axes = [axis, axis.twinx()]
    assert len(axes) == 2, axes
    if not axes[0].lines:
        axes[0].axhline(0, color="black", alpha=0.25)
    x: ndarray
    y1: ndarray
    y2: ndarray
    x, y1, y2 = result.get_residual_data()
    axes[0].scatter(
        x,
        y1,
        marker="o",
        facecolor="none",
        edgecolor=color_real,
        label=r"$\Delta_{\rm re}$",
    )
    axes[1].scatter(
        x,
        y2,
        marker="s",
        facecolor="none",
        edgecolor=color_imaginary,
        label=r"$\Delta_{\rm im}$",
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
        axes[0].set_ylabel(r"$\Delta_{\rm re}\ (\%)$")
        axes[1].set_ylabel(r"$\Delta_{\rm im}\ (\%)$")
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
        axes[1].legend(*_combine_legends(axes))
    return (
        fig,
        axes,
    )


def plot_mu_xps(
    tests: List[TestResult],
    mu_criterion: float,
    color_mu: str = "black",
    color_xps: str = "black",
    color_criterion: str = "black",
    legend: bool = True,
    fig: Optional[Figure] = None,
    axes: List[Axes] = [],
    adjust_axes: bool = True,
) -> Tuple[Figure, List[Axes]]:
    """
    Plot the mu-values and pseudo chi-squared values of Kramers-Kronig test results.

    Parameters
    ----------
    tests: List[TestResult]
        The results to plot.

    mu_criterion: float
        The mu-criterion to apply.

    color_mu: str = "black"
        The color of the markers and line for the mu-values.

    color_xps: str = "black"
        The color of the markers and line for the pseudo chi-squared values.

    color_criterion: str = "black"
        The color of the line for the mu-criterion.

    legend: bool = True
        Whether or not to add a legend.

    fig: Optional[Figure] = None
        The matplotlib.figure.Figure instance to use when plotting the data.

    axes: List[Axes] = []
        A list of matplotlib.axes.Axes instances to use when plotting the data.

    adjust_axes: bool = True
        Whether or not to adjust the axes (label, scale, limits, etc.).
    """
    assert type(tests) is list and all(
        map(lambda _: isinstance(_, TestResult), tests)
    ), tests
    assert type(color_mu) is str, color_mu
    assert type(color_xps) is str, color_xps
    assert type(color_criterion) is str, color_criterion
    assert type(fig) is Figure or fig is None, fig
    assert type(axes) is list, axes
    if fig is None:
        axis: Axes
        fig, axis = plt.subplots()
        axes = [axis, axis.twinx()]
    assert len(axes) == 2, axes
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
        label=f"#RC = {tests[0].num_RC}",
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
        marker="o",
        facecolor="none",
        edgecolor=color_mu,
        label=r"$\mu$",
    )
    axes[1].scatter(
        x,
        y2,
        marker="s",
        facecolor="none",
        edgecolor=color_xps,
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
        axes[0].set_xlabel(r"number of RC elements")
        axes[0].set_ylabel(r"$\mu$")
        axes[1].set_ylabel(r"$\chi^{2}_{\rm ps}$")
        _configure_log_scale(axes[1], y=True)
        _configure_log_limits(axes[1], y=True)
        axes[0].set_ylim(-0.1, 1.1)
    if legend is True:
        axes[1].legend(*_combine_legends(axes), loc=1)
    return (
        fig,
        axes,
    )


def plot_circuit(
    circuit: Circuit,
    f: Union[List[float], ndarray] = [],
    min_f: float = 1e-1,
    max_f: float = 1e5,
    color_nyquist: str = "#CC3311",
    color_bode_magnitude: str = "#CC3311",
    color_bode_phase: str = "#009988",
    data: Optional[DataSet] = None,
    visible_data: bool = False,
    title: Optional[str] = None,
    label: Optional[str] = None,
    legend: bool = True,
    fig: Optional[Figure] = None,
    axes: List[Axes] = [],
) -> Tuple[Figure, List[Axes]]:
    """
    Plot the simulated impedance response of a circuit as both a Nyquist and a Bode plot.

    Parameters
    ----------
    circuit: Circuit
        The circuit to use when simulating the impedance response.

    f: Union[List[float], ndarray] = []
        The frequencies (in hertz) to use when simulating the impedance response.
        If no frequencies are provided, then the range defined by the min_f and max_f parameters will be used instead.
        Alternatively, a DataSet instance can be provided via the data parameter.

    min_f: float = 0.1
        The lower limit of the frequency range to use if a list of frequencies is not provided.

    max_f: float = 100000.0
        The upper limit of the frequency range to use if a list of frequencies is not provided.

    color_nyquist: str = "#CC3311"
        The color to use in the Nyquist plot.

    color_bode_magnitude: str = "#CC3311"
        The color to use for the magnitude in the Bode plot.

    color_bode_phase: str = "#009988"
        The color to use for the phase shift in the Bode plot.

    data: Optional[DataSet] = None
        An optional DataSet instance.
        If provided, then the frequencies of this instance will be used when simulating the impedance spectrum of the circuit.

    visible_data: bool = False
        Whether or not the optional DataSet instance should also be plotted alongside the simulated impedance spectrum of the circuit.

    title: Optional[str] = None
        The title of the figure.
        If not title is provided, then the circuit description code of the circuit is used instead.

    label: Optional[str] = None
        The optional label to use in the legend.

    legend: bool = True
        Whether or not to add a legend.

    fig: Optional[Figure] = None
        The matplotlib.figure.Figure instance to use when plotting the data.

    axes: List[Axes] = []
        A list of matplotlib.axes.Axes instances to use when plotting the data.
    """
    assert hasattr(circuit, "impedances") and callable(circuit.impedances), circuit
    assert type(f) is list or type(f) is ndarray, f
    assert min_f > 0 and max_f < inf and min_f < max_f, (
        min_f,
        max_f,
    )
    assert type(label) is str or label is None, label
    assert type(color_nyquist) is str, color_nyquist
    assert type(color_bode_magnitude) is str, color_bode_magnitude
    assert type(color_bode_phase) is str, color_bode_phase
    assert data is None or (
        hasattr(data, "get_frequency")
        and callable(data.get_frequency)
        and hasattr(data, "get_impedance")
        and callable(data.get_impedance)
    ), data
    assert type(visible_data) is bool, visible_data
    assert type(title) is str or title is None, title
    assert type(fig) is Figure or fig is None, fig
    assert type(axes) is list, axes
    if fig is None:
        assert len(axes) == 0, axes
        fig, tmp = plt.subplots(1, 2)
        axes = [
            tmp[0],
            tmp[1],
            tmp[1].twinx(),
        ]
        if title is None:
            title = circuit.to_string()
        if title != "":
            fig.suptitle(title)
    assert len(axes) == 3, axes
    if data is not None and visible_data is True:
        plot_nyquist(
            data,
            color="#0077BB",
            legend=False,
            fig=fig,
            axis=axes[0],
        )
        plot_bode(
            data,
            color_magnitude="#0077BB",
            color_phase="#EE7733",
            legend=False,
            fig=fig,
            axes=[axes[1], axes[2]],
        )
    spectrum: DataSet
    if len(f) == 0:
        if data is not None:
            f = _interpolate(
                [min(data.get_frequency()), max(data.get_frequency())], 100
            )
        else:
            f = _interpolate([min_f, max_f], 100)
    Z: ndarray = circuit.impedances(f)
    spectrum = DataSet.from_dict(
        {
            "frequency": f,
            "real": Z.real,
            "imaginary": Z.imag,
            "label": label or str(circuit),
        }
    )
    plot_nyquist(
        spectrum,
        color=color_nyquist,
        line=True,
        legend=legend,
        fig=fig,
        axis=axes[0],
    )
    plot_bode(
        spectrum,
        color_magnitude=color_bode_magnitude,
        color_phase=color_bode_phase,
        line=True,
        legend=legend,
        fig=fig,
        axes=[axes[1], axes[2]],
    )
    return (
        fig,
        axes,
    )


def plot_data(
    data: DataSet,
    title: Optional[str] = None,
    label: Optional[str] = None,
    legend: bool = True,
    fig: Optional[Figure] = None,
    axes: List[Axes] = [],
) -> Tuple[Figure, List[Axes]]:
    """
    Plot a DataSet instance as both a Nyquist and a Bode plot.

    Parameters
    ----------
    data: DataSet
        The DataSet instance to plot.

    title: Optional[str] = None
        The title of the figure.
        If not title is provided, then the label of the DataSet is used instead.

    label: Optional[str] = None
        The optional label to use in the legend.

    legend: bool = True
        Whether or not to add a legend.

    fig: Optional[Figure] = None
        The matplotlib.figure.Figure instance to use when plotting the data.

    axes: List[Axes] = []
        A list of matplotlib.axes.Axes instances to use when plotting the data.
    """
    assert (
        hasattr(data, "get_frequency")
        and callable(data.get_frequency)
        and hasattr(data, "get_impedance")
        and callable(data.get_impedance)
    ), data
    assert type(title) is str or title is None, title
    assert type(label) is str or label is None, label
    assert type(legend) is bool, legend
    assert type(fig) is Figure or fig is None, fig
    assert type(axes) is list, axes
    if fig is None:
        assert len(axes) == 0, axes
        fig, tmp = plt.subplots(1, 2)
        axes = [
            tmp[0],
            tmp[1],
            tmp[1].twinx(),
        ]
        if title is None:
            title = data.get_label()
        if title != "":
            fig.suptitle(title)
    assert len(axes) == 3, axes
    plot_nyquist(
        data,
        color="#0077BB",
        label=label,
        legend=legend,
        fig=fig,
        axis=axes[0],
    )
    plot_bode(
        data,
        color_magnitude="#0077BB",
        color_phase="#EE7733",
        label=label,
        legend=legend,
        fig=fig,
        axes=[axes[1], axes[2]],
    )
    return (
        fig,
        axes,
    )


def plot_exploratory_tests(
    tests: List[TestResult],
    mu_criterion: float,
    data: DataSet,
    title: Optional[str] = None,
    legend: bool = True,
    fig: Optional[Figure] = None,
    axes: List[Axes] = [],
) -> Tuple[Figure, List[Axes]]:
    """
    Plot the results of an exploratory Kramers-Kronig test as a Nyquist plot, a Bode plot, a plot of the residuals, and a plot of the mu- and pseudo chi-squared values.

    Parameters
    ----------
    tests: List[TestResult]
        The results to plot.

    mu_criterion: float
        The mu-criterion to apply.

    data: DataSet
        The DataSet instance that was tested.

    title: Optional[str] = None
        The title of the figure.
        If no title is provided, then the label of the DataSet is used instead.

    legend: bool = True
        Whether or not to add a legend.

    fig: Optional[Figure] = None
        The matplotlib.figure.Figure instance to use when plotting the data.

    axes: List[Axes] = []
        A list of matplotlib.axes.Axes instances to use when plotting the data.
    """
    assert type(tests) is list and all(
        map(lambda _: isinstance(_, TestResult), tests)
    ), tests
    assert issubdtype(type(mu_criterion), floating), mu_criterion
    assert (
        hasattr(data, "get_frequency")
        and callable(data.get_frequency)
        and hasattr(data, "get_impedance")
        and callable(data.get_impedance)
    ), data
    assert type(title) is str or title is None, title
    assert type(fig) is Figure or fig is None, fig
    assert type(axes) is list, axes
    if fig is None:
        assert len(axes) == 0, axes
        fig, tmp = plt.subplots(2, 2)
        axes = [
            tmp[0][0],
            tmp[0][0].twinx(),
            tmp[0][1],
            tmp[0][1].twinx(),
            tmp[1][0],
            tmp[1][1],
            tmp[1][1].twinx(),
        ]
        if title is None:
            title = data.get_label()
        if title != "":
            fig.suptitle(title)
    assert len(axes) == 7, axes
    tests = list(
        sorted(tests, key=lambda _: _.calculate_score(mu_criterion), reverse=True)
    )
    test: TestResult = tests[0]
    plot_mu_xps(
        tests,
        mu_criterion,
        color_mu="#EE3377",
        color_xps="#009988",
        color_criterion="#CC3311",
        legend=legend,
        fig=fig,
        axes=[axes[0], axes[1]],
    )
    plot_residual(
        test,
        color_real="#EE3377",
        color_imaginary="#009988",
        legend=legend,
        fig=fig,
        axes=[axes[2], axes[3]],
    )
    plot_nyquist(
        data,
        color="#0077BB",
        legend=False,
        fig=fig,
        axis=axes[4],
    )
    plot_nyquist(
        test,
        color="#CC3311",
        line=True,
        legend=legend,
        fig=fig,
        axis=axes[4],
    )
    plot_bode(
        data,
        color_magnitude="#0077BB",
        color_phase="#EE7733",
        legend=False,
        fig=fig,
        axes=[axes[5], axes[6]],
    )
    plot_bode(
        test,
        color_magnitude="#CC3311",
        color_phase="#009988",
        line=True,
        legend=legend,
        fig=fig,
        axes=[axes[5], axes[6]],
    )
    return (
        fig,
        axes,
    )


def plot_fit(
    fit: Union[TestResult, FitResult, DRTResult],
    data: Optional[DataSet] = None,
    title: Optional[str] = None,
    label: Optional[str] = None,
    legend: bool = True,
    fig: Optional[Figure] = None,
    axes: List[Axes] = [],
    num_per_decade: int = 100,
) -> Tuple[Figure, List[Tuple[Axes]]]:
    """
    Plot the result of a fit as a Nyquist plot, a Bode plot, and a plot of the residuals.

    Parameters
    ----------
    fit: Union[TestResult, FitResult, DRTResult]
        The circuit fit or test result.

    data: Optional[DataSet] = None
        The DataSet instance that a circuit was fitted to.

    title: Optional[str] = None
        The title of the figure.
        If no title is provided, then the circuit description code (and label of the DataSet) is used instead.

    label: Optional[str] = None
        The optional label to use in the legend.

    legend: bool = True
        Whether or not to add a legend.

    fig: Optional[Figure] = None
        The matplotlib.figure.Figure instance to use when plotting the data.

    axes: List[Axes] = []
        A list of matplotlib.axes.Axes instances to use when plotting the data.

    num_per_decade: int = 100
        If the data being plotted is not a DataSet instance (e.g. a TestResult instance), then this parameter can be used to change how many points are used to draw the line (i.e. how smooth or angular the line looks).
    """
    assert (
        (
            (hasattr(fit, "get_label") and callable(fit.get_label))
            or hasattr(fit, "circuit")
        )
        and hasattr(fit, "get_frequency")
        and callable(fit.get_frequency)
        and hasattr(fit, "get_impedance")
        and callable(fit.get_impedance)
        and hasattr(fit, "get_residual_data")
        and callable(fit.get_residual_data)
    ), fit
    assert (
        hasattr(data, "get_frequency")
        and callable(data.get_frequency)
        and hasattr(data, "get_impedance")
        and callable(data.get_impedance)
    ), data
    assert type(title) is str or title is None, title
    assert type(label) is str or label is None, label
    assert type(legend) is bool, legend
    assert type(fig) is Figure or fig is None, fig
    assert type(axes) is list, axes
    if fig is None:
        assert len(axes) == 0, axes
        fig, tmp = plt.subplot_mosaic(
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
            if hasattr(fit, "circuit") and hasattr(fit, "num_RC"):
                title = (
                    fit.circuit.get_label()
                    .replace("K", r"$\rm (RC)_" + f"{{{str(fit.num_RC)}}}$", 1)  # type: ignore
                    .replace("K", "")
                )
                if data is not None:
                    title = f"{data.get_label()}\n{title}"
                if hasattr(fit, "get_label"):
                    title += f" {fit.get_label()}"  # type: ignore
            elif hasattr(fit, "circuit"):
                if data is not None:
                    title = f"{data.get_label()}\n{fit.circuit.get_label()}"  # type: ignore
                else:
                    title = fit.get_label()  # type: ignore
            elif hasattr(fit, "get_label") and callable(fit.get_label):
                if data is not None:
                    title = f"{data.get_label()}\n{fit.get_label()}"  # type: ignore
                else:
                    title = fit.get_label()  # type: ignore
        if title != "":
            fig.suptitle(title)
    assert len(axes) == 5, axes
    if data is not None:
        plot_nyquist(
            data,
            color="#0077BB",
            legend=False,
            fig=fig,
            axis=axes[0],
        )
        plot_bode(
            data,
            color_magnitude="#0077BB",
            color_phase="#EE7733",
            legend=False,
            fig=fig,
            axes=[axes[1], axes[2]],
        )
    plot_nyquist(
        fit,
        color="#CC3311",
        line=True,
        label=label,
        legend=legend,
        fig=fig,
        axis=axes[0],
        num_per_decade=num_per_decade,
    )
    plot_bode(
        fit,
        color_magnitude="#CC3311",
        color_phase="#009988",
        line=True,
        label=label,
        legend=legend,
        fig=fig,
        axes=[axes[1], axes[2]],
        num_per_decade=num_per_decade,
    )
    plot_residual(
        fit,
        color_real="#EE3377",
        color_imaginary="#009988",
        legend=legend,
        fig=fig,
        axes=[axes[3], axes[4]],
    )
    return (
        fig,
        axes,
    )


def plot_gamma(
    drt: DRTResult,
    peak_threshold: float = -1.0,
    color: Any = "black",
    bounds_alpha: float = 0.3,
    label: Optional[str] = None,
    legend: bool = True,
    fig: Optional[Figure] = None,
    axis: Optional[Axes] = None,
    adjust_axes: bool = True,
) -> Tuple[Figure, Axes]:
    """
    Plot the distribution of relaxation times (gamma vs tau).

    Parameters
    ----------
    drt: DRTResult
        The result to plot.

    peak_threshold: float = -1.0
        The threshold to use for identifying and marking peaks (0.0 to 1.0, relative to the highest peak).
        Negative values disable marking peaks.

    color: Any = "black"
        The color to use to plot the data.

    bounds_alpha: float = 0.3
        The alpha to use when plotting the bounds of the Bayesian credible intervals (if they are included in the data).

    label: Optional[str] = None
        The optional label to use in the legend.

    legend: bool = True
        Whether or not to add a legend.

    fig: Optional[Figure] = None
        The matplotlib.figure.Figure instance to use when plotting the data.

    axis: Optional[Axes] = None
        The matplotlib.axes.Axes instance to use when plotting the data.

    adjust_axes: bool = True
        Whether or not to adjust the axes (label, scale, limits, etc.).
    """
    assert hasattr(drt, "get_drt_data") and callable(drt.get_drt_data), drt
    assert (
        issubdtype(type(bounds_alpha), floating) and 0.0 <= bounds_alpha <= 1.0
    ), bounds_alpha
    assert (
        issubdtype(type(peak_threshold), floating) and peak_threshold <= 1.0
    ), peak_threshold
    assert type(label) is str or label is None, label
    assert type(legend) is bool, legend
    assert type(fig) is Figure or fig is None, fig
    if fig is None:
        fig, axis = plt.subplots()
    assert axis is not None, axis
    if label is None:
        if hasattr(drt, "get_label") and callable(drt.get_label):
            label = drt.get_label()
        else:
            label = ""
    x: ndarray
    y1: ndarray
    y2: ndarray
    y3: ndarray
    if hasattr(drt, "get_drt_credible_intervals") and callable(
        drt.get_drt_credible_intervals
    ):
        x, y1, y2, y3 = drt.get_drt_credible_intervals()
        if y1.any() and y2.any() and y3.any():
            mean_label: Optional[str] = None
            if label.strip() != "":
                if label != "":
                    mean_label = f"{label}, mean"
                else:
                    mean_label = "mean"
            axis.plot(
                x,
                y1,
                color=color,
                linestyle="--",
                label=mean_label,
            )
            axis.fill_between(
                x,
                y2,
                y3,
                color=color,
                alpha=bounds_alpha,
            )
    x, y1 = drt.get_drt_data()
    _, y2 = drt.get_drt_data(imaginary=True)
    real_label: Optional[str] = None
    imaginary_label: Optional[str] = None
    if label.strip() != "":
        if label != "":
            if y2.any():
                real_label = f"{label}, real"
            else:
                real_label = label
            imaginary_label = f"{label}, imag."
        else:
            real_label = "real"
            imaginary_label = "imag."
    axis.plot(
        x,
        y1,
        color=color,
        label=real_label,
    )
    if y2.any():
        axis.plot(
            x,
            y2,
            color=color,
            linestyle=":",
            label=imaginary_label,
        )
    if peak_threshold >= 0.0 and hasattr(drt, "get_peaks") and callable(drt.get_peaks):
        x, y1 = drt.get_peaks(threshold=peak_threshold)
        for _x, _y in zip(x, y1):
            axis.plot(
                [_x, _x],
                [0, _y],
                linestyle=":",
                alpha=0.5,
                color=color,
            )
        if y2.any():
            x, y2 = drt.get_peaks(threshold=peak_threshold, imaginary=True)
            for _x, _y in zip(x, y2):
                axis.plot(
                    [_x, _x],
                    [0, _y],
                    linestyle=":",
                    alpha=0.5,
                    color=color,
                )
    if adjust_axes:
        axis.set_xlabel(r"$\tau\ (\rm s)$")
        axis.set_ylabel(r"$\gamma\ (\Omega)$")
        _configure_log_scale(axis, x=True)
        _configure_log_limits(axis, x=True)
    if legend is True:
        axis.legend()
    return (
        fig,
        axis,
    )


def plot_drt(
    drt: DRTResult,
    data: Optional[DataSet] = None,
    peak_threshold: float = -1.0,
    title: Optional[str] = None,
    label: Optional[str] = None,
    legend: bool = True,
    fig: Optional[Figure] = None,
    axes: List[Axes] = [],
) -> Tuple[Figure, List[Tuple[Axes]]]:
    """
    Plot the result of calculating the distribution of relaxation times (DRT) as a Bode plot, a DRT plot, and a plot of the residuals.

    Parameters
    ----------
    drt: DRTResult
        The result to plot.

    data: Optional[DataSet] = None
        The DataSet instance that was used in the DRT calculations.

    peak_threshold: float = -1.0
        The threshold to use for identifying and marking peaks (0.0 to 1.0, relative to the highest peak).
        Negative values disable marking peaks.

    title: Optional[str] = None
        The title of the figure.
        If no title is provided, then the circuit description code (and label of the DataSet) is used instead.

    label: Optional[str] = None
        The optional label to use in the legend.

    legend: bool = True
        Whether or not to add a legend.

    fig: Optional[Figure] = None
        The matplotlib.figure.Figure instance to use when plotting the data.

    axes: List[Axes] = []
        A list of matplotlib.axes.Axes instances to use when plotting the data.
    """
    assert (
        hasattr(drt, "get_label")
        and callable(drt.get_label)
        and hasattr(drt, "get_frequency")
        and callable(drt.get_frequency)
        and hasattr(drt, "get_impedance")
        and callable(drt.get_impedance)
        and hasattr(drt, "get_residual_data")
        and callable(drt.get_residual_data)
    ), drt
    assert data is None or (
        hasattr(data, "get_frequency")
        and callable(data.get_frequency)
        and hasattr(data, "get_impedance")
        and callable(data.get_impedance)
    ), data
    assert (
        issubdtype(type(peak_threshold), floating) and peak_threshold <= 1.0
    ), peak_threshold
    assert type(title) is str or title is None, title
    assert type(label) is str or label is None, label
    assert type(legend) is bool, legend
    assert type(fig) is Figure or fig is None, fig
    assert type(axes) is list, axes
    if fig is None:
        assert len(axes) == 0, axes
        fig, tmp = plt.subplot_mosaic(
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
            if data is not None:
                title = f"{data.get_label()}\n{drt.get_label()}"  # type: ignore
            else:
                title = drt.get_label()  # type: ignore
        if title != "":
            fig.suptitle(title)
    assert len(axes) == 5, axes
    if label is None:
        if hasattr(drt, "get_label") and callable(drt.get_label):
            label = drt.get_label()
    if data is not None:
        plot_complex_impedance(
            data,
            color_real="#0077BB",
            color_imaginary="#EE7733",
            legend=False,
            fig=fig,
            axes=axes[0:2],
        )
    plot_complex_impedance(
        drt,
        color_real="#CC3311",
        color_imaginary="#009988",
        line=True,
        label=label,
        legend=legend,
        fig=fig,
        axes=axes[0:2],
    )
    plot_gamma(
        drt,
        peak_threshold=peak_threshold,
        color="black",
        label=label,
        legend=legend,
        fig=fig,
        axis=axes[2],
    )
    plot_residual(
        drt,
        color_real="#EE3377",
        color_imaginary="#009988",
        legend=legend,
        fig=fig,
        axes=[axes[3], axes[4]],
    )
    return (
        fig,
        axes,
    )
