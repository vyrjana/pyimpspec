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

from inspect import signature
from pyimpspec.circuit.circuit import Circuit
from pyimpspec.data import DataSet
from pyimpspec.analysis import (
    KramersKronigResult,
    FitResult,
)
from pyimpspec.analysis.utility import _interpolate
from pyimpspec.analysis.drt import (
    DRTResult,
)
from pyimpspec.typing.helpers import (
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    _is_boolean,
)
from numpy import (
    complex128,
    isclose,
)
from numpy.typing import NDArray
from pyimpspec.typing import Impedances
from pyimpspec.plot.colors import COLOR_RED
from .markers import MARKER_CIRCLE
from .helpers import (
    _get_marker_color_args,
    _initialize_figure,
    _validate_figure,
)


def plot_nyquist(
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
    **kwargs,
) -> Tuple["Figure", List["Axes"]]:  # noqa: F821
    """
    Plot some data as a Nyquist plot (-Im(Z) vs Re(Z)).

    Parameters
    ----------
    data: Optional[Union[DataSet, KramersKronigResult, FitResult, DRTResult]]
        The data to plot.

    label: Optional[str], optional
        The optional label to use in the legend.

    admittance: bool, optional
        Plot the admittance representation of the immittance data.

    colors: Optional[Dict[str, str]], optional
        The colors of the markers or lines. Valid keys: 'impedance'.

    markers: Optional[Dict[str, str]], optional
        The markers to use when not plotting lines. Valid keys: 'impedance'.

    line: bool, optional
        Whether or not a line should be used instead of markers.

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

    **kwargs

    Returns
    -------
    Tuple[|Figure|, List[|Axes|]]
    """
    from matplotlib.ticker import EngFormatter
    from matplotlib.axes import Axes

    if figure is None:
        figure, axes = _initialize_figure(num_rows=1, num_cols=1)
    assert axes is not None

    _validate_figure(figure, axes, num_axes=1)
    axis: Axes = axes[0]

    if colors is None:
        colors = {}
    elif not isinstance(colors, dict):
        raise TypeError(f"Expected a dictionary or None instead of {colors=}")
    color: str = colors.get("impedance", COLOR_RED)

    if markers is None:
        markers = {}
    elif not isinstance(markers, dict):
        raise TypeError(f"Expected a dictionary or None instead of {markers=}")
    marker: str = markers.get("impedance", MARKER_CIRCLE)

    if label is None:
        if hasattr(data, "get_label") and callable(data.get_label):
            label = data.get_label()
        else:
            label = ""
    elif not isinstance(label, str):
        raise TypeError(f"Expected a string or None instead of {label=}")

    if not _is_boolean(admittance):
        raise TypeError(f"Expected a boolean instead of {admittance=}")

    if not _is_boolean(line):
        raise TypeError(f"Expected a boolean instead of {line=}")

    if data is not None:
        X: NDArray[complex128]
        if line and ("num_per_decade" in signature(data.get_impedances).parameters):
            X = data.get_impedances(num_per_decade=num_per_decade) ** (
                -1 if admittance else 1
            )
        elif line and hasattr(data, "circuit") and isinstance(data.circuit, Circuit):
            X = data.circuit.get_impedances(
                _interpolate(
                    data.get_frequencies(),
                    num_per_decade=num_per_decade,
                ),
            ) ** (-1 if admittance else 1)
        else:
            X = data.get_impedances() ** (-1 if admittance else 1)

        x: Impedances = X.real
        y: Impedances = X.imag * (1 if admittance else -1)

        if isclose(x, x[0]).all() and isclose(y, y[0]).all():
            # Can happen when, e.g., fitting just a resistance.
            line = False

        if line:
            axis.plot(
                x,
                y,
                color=color,
                linestyle=kwargs.get("linestyle", "-"),
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

    if not _is_boolean(adjust_axes):
        raise TypeError(f"Expected a boolean instead of {adjust_axes=}")
    elif adjust_axes:
        axis.set_xlabel(r"${\rm Re}(Y)\ (S)$" if admittance else r"${\rm Re}(Z)\ (\Omega)$")
        axis.set_ylabel(r"${\rm Im}(Y)\ (S)$" if admittance else r"$-{\rm Im}(Z)\ (\Omega)$")
        axis.xaxis.set_major_formatter(EngFormatter(unit=""))
        axis.yaxis.set_major_formatter(EngFormatter(unit=""))
        axis.set_aspect("equal", adjustable="datalim")

    if not _is_boolean(legend):
        raise TypeError(f"Expected a boolean instead of {legend=}")
    elif legend and data is not None:
        axis.legend()

    return (
        figure,
        axes,
    )
