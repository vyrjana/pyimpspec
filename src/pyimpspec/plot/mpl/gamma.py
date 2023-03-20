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

from pyimpspec.analysis.drt import DRTResult
from numpy import (
    complex128,
    floating,
    isnan,
    issubdtype,
)
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)
from pyimpspec.typing import (
    Gamma,
    Gammas,
    TimeConstant,
    TimeConstants,
)
from pyimpspec.plot.colors import COLOR_BLACK
from pyimpspec.plot.mpl.utility import (
    _color_axis,
    _configure_log_limits,
    _configure_log_scale,
)


def _plot_credible_intervals(
    drt: DRTResult,
    label: str,
    color: str,
    bounds_alpha: float,
    axis: "Axes",  # noqa: F821
):
    x: TimeConstants
    y1: Gammas
    y2: Gammas
    y3: Gammas
    x, y1, y2, y3 = drt.get_drt_credible_intervals_data()
    if y1.size == y2.size == y3.size > 0:
        mean_label: Optional[str] = None
        ci_label: Optional[str] = None
        if label.strip() != "":
            if label != "":
                mean_label = f"{label}, mean"
                ci_label = f"{label}, " + r"$3\sigma$ CI"
            else:
                mean_label = "mean"
                ci_label = r"$3\sigma$ CI"
        axis.fill_between(
            x,
            y2,
            y3,
            color=color,
            alpha=bounds_alpha,
            label=ci_label,
        )
        axis.plot(
            x,
            y1,
            color=color,
            linestyle="--",
            label=mean_label,
        )


def _plot_gammas(
    drt: DRTResult,
    label: str,
    color: str,
    axis: "Axes",  # noqa: F821
):
    x: TimeConstants
    real_y: Gammas = None
    imag_y: Gammas = None
    x, *y = drt.get_drt_data()
    if len(y) == 2:
        real_y, imag_y = y
    else:
        y = y[0]
        if y.dtype == complex128:
            real_y = y.real
            imag_y = y.imag
        else:
            real_y = y
    real_label: Optional[str] = None
    imaginary_label: Optional[str] = None
    if label.strip() != "":
        if label != "":
            if imag_y is not None:
                real_label = f"{label}, real"
            else:
                real_label = label
            imaginary_label = f"{label}, imag."
        else:
            real_label = "real"
            imaginary_label = "imag."
    axis.plot(
        x,
        real_y,
        color=color,
        label=real_label,
    )
    if imag_y is not None and imag_y.size > 0:
        axis.plot(
            x,
            imag_y,
            color=color,
            linestyle=":",
            label=imaginary_label,
        )


def _plot_peaks(drt, peak_threshold: float, color, axis):
    x: TimeConstant
    y: Gamma
    if peak_threshold >= 0.0 and hasattr(drt, "get_peaks") and callable(drt.get_peaks):
        # Most DRT results return a tuple with a length of two. However, BHTResult
        # returns a tuple with a length of four (real tau, real gamma, imaginary tau,
        # and imaginary gamma).
        peaks: tuple = drt.get_peaks(threshold=peak_threshold)
        for x, y in zip(peaks[0], peaks[1]):
            if isnan(x) or isnan(y):
                continue
            axis.plot(
                [x, x],
                [0, y],
                linestyle=":",
                alpha=0.5,
                color=color,
            )
        if len(peaks) == 4:
            for x, y in zip(peaks[2], peaks[3]):
                if isnan(x) or isnan(y):
                    continue
                axis.plot(
                    [x, x],
                    [0, y],
                    linestyle=":",
                    alpha=0.5,
                    color=color,
                )


def plot_gamma(
    drt: DRTResult,
    peak_threshold: float = -1.0,
    label: Optional[str] = None,
    bounds_alpha: float = 0.3,
    colors: Optional[Dict[str, str]] = None,
    figure: Optional["Figure"] = None,  # noqa: F821
    legend: bool = True,
    axes: Optional[List["Axes"]] = None,  # noqa: F821
    adjust_axes: bool = True,
    colored_axes: bool = False,
    **kwargs,
) -> Tuple["Figure", List["Axes"]]:  # noqa: F821
    """
    Plot the distribution of relaxation times (gamma vs tau).

    Parameters
    ----------
    drt: DRTResult
        The result to plot.

    peak_threshold: float, optional
        The threshold to use for identifying and marking peaks (0.0 to 1.0, relative to the highest peak).
        Negative values disable marking peaks.

    label: Optional[str], optional
        The optional label to use in the legend.

    bounds_alpha: float, optional
        The alpha to use when plotting the bounds of the Bayesian credible intervals (if they are included in the data).

    colors: Optional[Dict[str, str]], optional
        The colors of the lines. Valid keys: 'gamma'.

    figure: Optional[|Figure|], optional
        The matplotlib.figure.Figure instance to use when plotting the data.

    legend: bool, optional
        Whether or not to add a legend.

    axes: Optional[List[|Axes|]], optional
        The matplotlib.axes.Axes instance to use when plotting the data.

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

    assert hasattr(drt, "get_drt_data") and callable(drt.get_drt_data)
    assert (
        issubdtype(type(bounds_alpha), floating) and 0.0 <= bounds_alpha <= 1.0
    ), bounds_alpha
    assert (
        issubdtype(type(peak_threshold), floating) and peak_threshold <= 1.0
    ), peak_threshold
    if colors is None:
        colors = {}
    assert isinstance(colors, dict), colors
    assert isinstance(label, str) or label is None, label
    assert isinstance(legend, bool), legend
    assert isinstance(colored_axes, bool), colored_axes
    assert isinstance(figure, Figure) or figure is None, figure
    axis: Axes
    if figure is None:
        assert axes is None
        figure, axis = plt.subplots()
        axes = [axis]
    assert isinstance(axes, list)
    assert len(axes) == 1
    assert all(map(lambda _: isinstance(_, Axes), axes))
    axis = axes[0]
    assert axis is not None, axis
    if label is None:
        if hasattr(drt, "get_label") and callable(drt.get_label):
            label = drt.get_label()
        else:
            label = ""
    color: str = colors.get("gamma", COLOR_BLACK)
    if hasattr(drt, "get_drt_credible_intervals_data") and callable(
        drt.get_drt_credible_intervals_data
    ):
        _plot_credible_intervals(drt, label, color, bounds_alpha, axis)
    _plot_peaks(drt, peak_threshold, color, axis)
    _plot_gammas(drt, label, color, axis)
    if adjust_axes:
        axis.set_xlabel(r"$\tau\ (\rm s)$")
        axis.set_ylabel(r"$\gamma\ (\Omega)$")
        _configure_log_scale(axis, x=True)
        _configure_log_limits(axis, x=True)
    if legend is True:
        axis.legend()
    if colored_axes is True:
        _color_axis(axis, color, left=True, right=True)
    return (
        figure,
        axes,
    )
