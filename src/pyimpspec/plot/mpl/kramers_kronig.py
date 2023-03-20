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
from pyimpspec.analysis import TestResult
from numpy import (
    floating,
    issubdtype,
)
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
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
from .mu_xps import plot_mu_xps
from .nyquist import plot_nyquist
from .residuals import plot_residuals


def plot_tests(
    tests: List[TestResult],
    mu_criterion: float,
    data: DataSet,
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
    Plot the results of an exploratory Kramers-Kronig test as a Nyquist plot, a Bode plot, a plot of the residuals, and a plot of the |mu| and |pseudo chi-squared| values.

    Parameters
    ----------
    tests: List[TestResult]
        The results to plot.

    mu_criterion: float
        The |mu|-criterion to apply.

    data: DataSet
        The DataSet instance that was tested.

    colors: Optional[Dict[str, str]], optional
        The colors of the markers or lines. Valid keys: 'mu', 'xps', 'criterion', 'real', 'imaginary', 'data_impedance', 'impedance', 'data_magnitude', 'data_phase', 'magnitude', 'phase'.

    markers: Optional[Dict[str, str]], optional
        The markers to use when not plotting lines. Valid keys: 'mu', 'xps', 'real', 'imaginary', 'data_impedance', 'data_magnitude', 'data_phase'.

    figure: Optional[|Figure|], optional
        The matplotlib.figure.Figure instance to use when plotting the data.

    title: Optional[str], optional
        The title of the figure.
        If no title is provided, then the label of the DataSet is used instead.

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

    assert isinstance(tests, list)
    assert all(map(lambda _: isinstance(_, TestResult), tests))
    assert issubdtype(type(mu_criterion), floating), mu_criterion
    assert hasattr(data, "get_frequencies") and callable(data.get_frequencies)
    assert hasattr(data, "get_impedances") and callable(data.get_impedances)
    if colors is None:
        colors = {}
    if markers is None:
        markers = {}
    assert isinstance(colors, dict), colors
    assert isinstance(markers, dict), markers
    assert isinstance(title, str) or title is None, title
    assert isinstance(figure, Figure) or figure is None, figure
    assert isinstance(adjust_axes, bool), adjust_axes
    if figure is None:
        assert axes is None
        figure, tmp = plt.subplots(2, 2)
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
            figure.suptitle(title)
    assert isinstance(axes, list)
    assert len(axes) == 7, axes
    assert all(map(lambda _: isinstance(_, Axes), axes))
    color_mu: str = colors.get("mu", COLOR_MAGENTA)
    color_xps: str = colors.get("xps", COLOR_TEAL)
    color_criterion: str = colors.get("criterion", COLOR_RED)
    color_real: str = colors.get("real", COLOR_TEAL)
    color_imaginary: str = colors.get("imaginary", COLOR_MAGENTA)
    color_data_impedance: str = colors.get("data_impedance", COLOR_BLACK)
    color_impedance: str = colors.get("impedance", COLOR_RED)
    color_data_magnitude: str = colors.get("data_magnitude", COLOR_BLACK)
    color_data_phase: str = colors.get("data_phase", COLOR_BLACK)
    color_magnitude: str = colors.get("magnitude", COLOR_BLUE)
    color_phase: str = colors.get("phase", COLOR_ORANGE)
    marker_mu: str = markers.get("mu", MARKER_CIRCLE)
    marker_xps: str = markers.get("xps", MARKER_SQUARE)
    marker_real: str = markers.get("real", MARKER_CIRCLE)
    marker_imaginary: str = markers.get("imaginary", MARKER_SQUARE)
    marker_data_impedance: str = markers.get("data_impedance", MARKER_CIRCLE)
    marker_data_magnitude: str = markers.get("data_magnitude", MARKER_CIRCLE)
    marker_data_phase: str = markers.get("data_phase", MARKER_SQUARE)
    test: TestResult = tests[0]
    plot_mu_xps(
        tests,
        mu_criterion,
        colors={
            "mu": color_mu,
            "xps": color_xps,
            "criterion": color_criterion,
        },
        markers={
            "mu": marker_mu,
            "xps": marker_xps,
        },
        legend=legend,
        colored_axes=colored_axes,
        figure=figure,
        axes=[axes[0], axes[1]],
        adjust_axes=adjust_axes,
    )
    plot_residuals(
        test,
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
        axes=[axes[2], axes[3]],
        adjust_axes=adjust_axes,
    )
    plot_nyquist(
        data,
        colors={
            "impedance": color_data_impedance,
        },
        markers={
            "impedance": marker_data_impedance,
        },
        label="Data" if title else None,
        legend=False,
        figure=figure,
        axes=[axes[4]],
        adjust_axes=adjust_axes,
    )
    plot_nyquist(
        test,
        colors={
            "impedance": color_impedance,
        },
        line=True,
        label="Fit" if title else None,
        legend=legend,
        colored_axes=colored_axes,
        figure=figure,
        axes=[axes[4]],
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
        label="Data" if title else None,
        legend=False,
        figure=figure,
        axes=[axes[5], axes[6]],
        adjust_axes=adjust_axes,
    )
    plot_bode(
        test,
        colors={
            "magnitude": color_magnitude,
            "phase": color_phase,
        },
        line=True,
        label="Fit" if title else None,
        colored_axes=colored_axes,
        legend=legend,
        figure=figure,
        axes=[axes[5], axes[6]],
        adjust_axes=adjust_axes,
    )
    return (
        figure,
        axes,
    )
