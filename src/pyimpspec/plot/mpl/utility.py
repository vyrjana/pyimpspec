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
    Any,
    Dict,
    List,
    Set,
    Tuple,
)
from numpy import (
    ceil,
    floor,
    log10 as log,
)


def _combine_legends(
    axes: List["Axes"],  # noqa: F821
) -> Tuple[list, list]:
    lines: list = []
    labels: list = []
    ax: "Axes"  # noqa: F821
    for ax in axes:
        lin, lab = ax.get_legend_handles_labels()
        lines.extend(lin)
        labels.extend(lab)
    return (
        lines,
        labels,
    )


def _configure_log_limits(
    axis: "Axes",  # noqa: F821
    x: bool = False,
    y: bool = False,
):
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


def _configure_log_scale(
    axis: "Axes",  # noqa: F821
    x: bool = False,
    y: bool = False,
):
    from matplotlib.ticker import FormatStrFormatter

    formatter: FormatStrFormatter = FormatStrFormatter("")
    if x:
        axis.set_xscale("log")
        axis.xaxis.set_minor_formatter(formatter)
    if y:
        axis.set_yscale("log")
        axis.yaxis.set_minor_formatter(formatter)


def _color_axis(
    axis: "Axes",  # noqa: F821
    color: Any,
    left: bool = False,
    right: bool = False,
):
    if left:
        axis.spines["left"].set_color(color)
    if right:
        axis.spines["right"].set_color(color)
    axis.tick_params(axis="y", colors=color, which="both")
    axis.yaxis.label.set_color(color)


# Vibrant color scheme from https://personal.sron.nl/~pault/
_FILLED_MARKERS: Set[str] = set()
_UNFILLED_MARKERS: Set[str] = set()


def _get_marker_color_args(marker: str, color: Any) -> Dict[str, Any]:
    global _FILLED_MARKERS
    global _UNFILLED_MARKERS
    if len(_FILLED_MARKERS) == 0 or len(_UNFILLED_MARKERS) == 0:
        from matplotlib.lines import Line2D

        _FILLED_MARKERS.update(Line2D.filled_markers)
        _UNFILLED_MARKERS.update(
            [m for m, f in Line2D.markers.items() if f != "nothing"]
        )
        _UNFILLED_MARKERS.difference_update(_FILLED_MARKERS)
        assert len(_FILLED_MARKERS) > 0
        assert len(_UNFILLED_MARKERS) > 0
    if marker in _FILLED_MARKERS:
        return {"edgecolor": color, "facecolor": "none"}
    return {"color": color}
