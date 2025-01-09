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
from pyimpspec.analysis.kramers_kronig.algorithms.utility.common import (
    _is_admittance_test_circuit,
)
from pyimpspec.circuit.kramers_kronig import (
    KramersKronigAdmittanceRC,
    KramersKronigRC,
)
from pyimpspec.typing.helpers import (
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    _is_boolean,
    _is_integer,
)
from pyimpspec.plot.colors import (
    COLOR_BLACK,
    COLOR_BLUE,
    COLOR_MAGENTA,
    COLOR_ORANGE,
    COLOR_RED,
    COLOR_TEAL,
)
from .helpers import _validate_figure
from .markers import (
    MARKER_CIRCLE,
    MARKER_SQUARE,
    MARKER_DOT,
)
from .bode import plot_bode
from .residuals import plot_residuals
from .nyquist import plot_nyquist


def plot_fit(
    fit: Union[KramersKronigResult, FitResult, DRTResult],
    data: DataSet,
    label: Optional[str] = None,
    admittance: bool = False,
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
    fit: Union[KramersKronigResult, FitResult, DRTResult]
        The circuit fit or test result.

    data: DataSet
        The DataSet instance that a circuit was fitted to.

    label: Optional[str], optional
        The optional label to use in the legend.

    admittance: bool, optional
        Plot the admittance representation of the immittance data.

    colors: Optional[Dict[str, str]], optional
        The colors of the markers or lines. Valid keys: 'real', 'imaginary', 'data_impedance', 'impedance', 'data_magnitude', 'data_phase', 'magnitude', 'phase'.

    markers: Optional[Dict[str, str]], optional
        The markers to use when not plotting lines. Valid keys: 'real', 'imaginary', 'data_impedance', 'data_magnitude', 'data_phase'.

    num_per_decade: int, optional
        If the data being plotted is not a DataSet instance (e.g. a KramersKronigResult instance), then this parameter can be used to change how many points are used to draw the line (i.e. how smooth or angular the line looks).

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

    if colors is None:
        colors = {}
    elif not isinstance(colors, dict):
        raise TypeError(f"Expected a dictionary or None instead of {colors=}")
    color_real: str = colors.get("real", COLOR_TEAL)
    color_imaginary: str = colors.get("imaginary", COLOR_MAGENTA)
    color_data_impedance: str = colors.get("data_impedance", COLOR_BLACK)
    color_impedance: str = colors.get("impedance", COLOR_RED)
    color_data_magnitude: str = colors.get("data_magnitude", COLOR_BLACK)
    color_data_phase: str = colors.get("data_phase", COLOR_BLACK)
    color_magnitude: str = colors.get("magnitude", COLOR_BLUE)
    color_phase: str = colors.get("phase", COLOR_ORANGE)

    if markers is None:
        markers = {}
    elif not isinstance(markers, dict):
        raise TypeError(f"Expected a dictionary or None instead of {markers=}")
    marker_data_impedance: str = markers.get("data_impedance", MARKER_CIRCLE)
    marker_data_magnitude: str = markers.get("data_magnitude", MARKER_CIRCLE)
    marker_data_phase: str = markers.get("data_phase", MARKER_SQUARE)
    marker_fit_impedance: str = markers.get("fit_impedance", MARKER_DOT)
    marker_fit_magnitude: str = markers.get("fit_magnitude", MARKER_DOT)
    marker_fit_phase: str = markers.get("fit_phase", MARKER_DOT)

    if not (isinstance(title, str) or title is None):
        raise TypeError(f"Expected a string or None instead of {title=}")

    if not (isinstance(label, str) or label is None):
        raise TypeError(f"Expected a string or None instead of {label=}")

    if not _is_boolean(legend):
        raise TypeError(f"Expected a boolean instead of {legend=}")

    if not _is_boolean(colored_axes):
        raise TypeError(f"Expected a boolean instead of {colored_axes=}")

    if not _is_integer(num_per_decade):
        raise TypeError(f"Expected an integer instead of {num_per_decade=}")

    if not _is_boolean(adjust_axes):
        raise TypeError(f"Expected a boolean instead of {adjust_axes=}")

    if figure is None:
        figure, axes = plt.subplot_mosaic(
            [["upper left", "upper right"], ["bottom", "bottom"]],
            gridspec_kw={
                "width_ratios": [1, 1],
                "height_ratios": [2, 1],
            },
            constrained_layout=True,
        )
        axes = [
            axes["upper left"],
            axes["upper right"],
            axes["upper right"].twinx(),
            axes["bottom"],
            axes["bottom"].twinx(),
        ]
    assert axes is not None

    _validate_figure(figure, axes, num_axes=5)

    if title is None:
        if isinstance(fit, KramersKronigResult):
            old_symbol: str = (
                KramersKronigAdmittanceRC.get_symbol()
                if _is_admittance_test_circuit(fit.circuit)
                else KramersKronigRC.get_symbol()
            )
            title = (
                fit.circuit.to_string()
                .replace(old_symbol, r"$\rm (RC)_" + f"{{{str(fit.num_RC)}}}$", 1)  # type: ignore
                .replace(old_symbol, "")
            )
        elif isinstance(fit, FitResult):
            title = fit.circuit.to_string()
        elif hasattr(fit, "get_label") and callable(fit.get_label):
            title = fit.get_label()  # type: ignore
        title = f"{data.get_label()}\n{title}"
    elif not isinstance(title, str):
        raise TypeError(f"Expected a string or None instead of {title=}")

    if title != "":
        figure.suptitle(title)

    plot_nyquist(
        fit,
        admittance=admittance,
        colors={
            "impedance": color_impedance,
        },
        markers={
            "impedance": marker_fit_impedance,
        },
        line=False,
        label="",
        legend=False,
        colored_axes=colored_axes,
        figure=figure,
        axes=[axes[0]],
        adjust_axes=adjust_axes,
    )

    plot_nyquist(
        data,
        admittance=admittance,
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

    plot_nyquist(
        fit,
        admittance=admittance,
        colors={
            "impedance": color_impedance,
        },
        markers={
            "impedance": marker_fit_impedance,
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
        admittance=admittance,
        colors={
            "magnitude": color_magnitude,
            "phase": color_phase,
        },
        markers={
            "magnitude": marker_fit_magnitude,
            "phase": marker_fit_phase,
        },
        line=False,
        label="",
        colored_axes=colored_axes,
        legend=False,
        figure=figure,
        axes=[axes[1], axes[2]],
        adjust_axes=adjust_axes,
    )

    plot_bode(
        data,
        admittance=admittance,
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

    plot_bode(
        fit,
        admittance=admittance,
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
