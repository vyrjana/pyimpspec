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

from typing import (
    Dict,
    List,
    Optional,
    Tuple,
)
from numpy import ndarray
from pyimpspec.circuit import Circuit
from pyimpspec.data import DataSet
from pyimpspec.typing import (
    ComplexImpedances,
    Frequencies,
)
from pyimpspec.plot.colors import (
    COLOR_BLUE,
    COLOR_ORANGE,
    COLOR_RED,
)
from pyimpspec.analysis.utility import _interpolate
from .bode import plot_bode
from .nyquist import plot_nyquist


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
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    assert hasattr(circuit, "get_impedances") and callable(
        circuit.get_impedances
    ), circuit
    assert isinstance(frequencies, ndarray), frequencies
    assert len(frequencies) >= 2, len(frequencies)
    if colors is None:
        colors = {}
    if markers is None:
        markers = {}
    assert isinstance(colors, dict), colors
    assert isinstance(markers, dict), markers
    assert isinstance(label, str) or label is None, label
    assert isinstance(title, str) or title is None, title
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
            title = circuit.to_string()
        if title != "":
            figure.suptitle(title)
    assert isinstance(axes, list), axes
    assert len(axes) == 3, axes
    assert all(map(lambda _: isinstance(_, Axes), axes))
    color_nyquist: str = colors.get("impedance", COLOR_RED)
    color_bode_magnitude: str = colors.get("magnitude", COLOR_BLUE)
    color_bode_phase: str = colors.get("phase", COLOR_ORANGE)
    spectrum: DataSet
    frequencies = _interpolate([max(frequencies), min(frequencies)], 100)
    Z: ComplexImpedances = circuit.get_impedances(frequencies)
    spectrum = DataSet(
        frequencies=frequencies, impedances=Z, label=label or str(circuit)
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
