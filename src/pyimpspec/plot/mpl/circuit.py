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

from pyimpspec.circuit import Circuit
from pyimpspec.data import DataSet
from pyimpspec.typing import (
    ComplexImpedances,
    Frequencies,
)
from pyimpspec.typing.helpers import (
    Dict,
    List,
    Optional,
    Tuple,
    _is_boolean,
)
from pyimpspec.plot.colors import (
    COLOR_BLACK,
    COLOR_BLUE,
    COLOR_ORANGE,
)
from pyimpspec.analysis.utility import _interpolate
from .bode import plot_bode
from .nyquist import plot_nyquist
from .helpers import (
    _initialize_figure,
    _validate_figure,
)


def plot_circuit(
    circuit: Circuit,
    frequencies: Frequencies,
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
    Plot the simulated impedance response of a circuit as both a Nyquist and a Bode plot.

    Parameters
    ----------
    circuit: Circuit
        The circuit to use when simulating the impedance response.

    frequencies: Frequencies
        The frequencies (in hertz) to use when simulating the impedance response.

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
        If not title is provided, then the circuit description code of the circuit is used instead.

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
        figure, axes = _initialize_figure(
            num_rows=1,
            num_cols=2,
        )
        axes = [axes[0], axes[1], axes[1].twinx()]
    assert axes is not None

    _validate_figure(figure, axes, num_axes=3)

    if colors is None:
        colors = {}
    elif not isinstance(colors, dict):
        raise TypeError(f"Expected a dictionary or None instead of {colors=}")
    color_nyquist: str = colors.get("impedance", COLOR_BLACK)
    color_bode_magnitude: str = colors.get("magnitude", COLOR_BLUE)
    color_bode_phase: str = colors.get("phase", COLOR_ORANGE)

    if markers is None:
        markers = {}
    elif not isinstance(markers, dict):
        raise TypeError(f"Expected a dictionary or None instead of {markers=}")

    if not _is_boolean(adjust_axes):
        raise TypeError(f"Expected a boolean instead of {adjust_axes=}")

    if title is None:
        title = circuit.to_string()
    elif not isinstance(title, str):
        raise TypeError(f"Expected a string or None instead of {title=}")

    if title != "":
        figure.suptitle(title)

    frequencies = _interpolate([max(frequencies), min(frequencies)], 100)
    Z: ComplexImpedances = circuit.get_impedances(frequencies)

    if label is None:
        label = circuit.to_string()
    elif not isinstance(label, str):
        raise TypeError(f"Expected a string or None instead of {label=}")

    spectrum: DataSet = DataSet(
        frequencies=frequencies,
        impedances=Z,
        label=label,
    )

    plot_nyquist(
        spectrum,
        colors={"impedance": color_nyquist},
        line=True,
        legend=legend,
        colored_axes=colored_axes,
        figure=figure,
        axes=[axes[0]],
        adjust_axes=adjust_axes,
    )

    plot_bode(
        spectrum,
        colors={
            "magnitude": color_bode_magnitude,
            "phase": color_bode_phase,
        },
        line=True,
        legend=legend,
        figure=figure,
        axes=[axes[1], axes[2]],
        adjust_axes=adjust_axes,
        colored_axes=colored_axes,
    )

    return (
        figure,
        axes,
    )
