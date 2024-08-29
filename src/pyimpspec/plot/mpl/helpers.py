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

from pyimpspec.typing.helpers import (
    Any,
    Callable,
    Dict,
    List,
    Set,
    Tuple,
    Union,
)
from numpy import (
    ceil,
    floor,
    log10 as log,
    ndarray,
)


def _validate_figure(
    figure: "Figure",  # noqa: F821
    axes: List["Axes"],  # noqa: F821
    num_axes: int,
):
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    if not isinstance(figure, Figure):
        raise TypeError(f"Expected a Figure instead of {figure=}")

    if not (
        isinstance(axes, list)
        and len(axes) == num_axes
        and all(map(lambda axis: isinstance(axis, Axes), axes))
    ):
        raise TypeError(f"Expected a list of {num_axes=} {Axes=} instead of {axes=}")


def _initialize_figure(
    num_rows: int,
    num_cols: int,
) -> Tuple["Figure", List["Axes"]]:  # noqa: F821
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(num_rows, num_cols)
    if isinstance(axes, ndarray):
        axes = axes.flatten().tolist()

    if not isinstance(axes, list):
        axes = [axes]

    return (
        figure,
        axes,
    )


def _combine_legends(
    axes: List["Axes"],  # noqa: F821
) -> Tuple[list, list]:
    lines: list = []
    labels: list = []

    axis: "Axes"  # noqa: F821
    for axis in axes:
        lin, lab = axis.get_legend_handles_labels()
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
    z: bool = False,
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

    if z:
        z_min: float
        z_max: float
        z_min, z_max = axis.get_zlim()
        if log(z_max) - log(z_min) < 1.0:
            z_min = 10 ** floor(log(z_min))
            z_max = 10 ** ceil(log(z_max))
            axis.set_zlim(z_min, z_max)


def _configure_log_scale(
    axis: "Axes",  # noqa: F821
    x: bool = False,
    y: bool = False,
    z: bool = False,
):
    from matplotlib.ticker import FormatStrFormatter

    formatter: FormatStrFormatter = FormatStrFormatter("")

    if x:
        axis.set_xscale("log")
        axis.xaxis.set_minor_formatter(formatter)

    if y:
        axis.set_yscale("log")
        axis.yaxis.set_minor_formatter(formatter)

    if z:
        axis.set_zscale("log")
        axis.zaxis.set_minor_formatter(formatter)


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
_UNFILLED_MARKERS: Set[Union[str, int]] = set()


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
        if not (len(_FILLED_MARKERS) > 0):
            raise ValueError(
                f"Expected at least one filled marker instead of {_FILLED_MARKERS=}"
            )

        if not (len(_UNFILLED_MARKERS) > 0):
            raise ValueError(
                f"Expected at least one unfilled marker instead of {_UNFILLED_MARKERS=}"
            )

    if marker in _FILLED_MARKERS:
        return {"edgecolor": color, "facecolor": "none"}

    return {"color": color}


def _format_coord_two_y_axes(
    ax2: "Axes",  # noqa: F821
    ax1: "Axes",  # noqa: F821
    x_fmt: Union[str, "Formatter"] = "",  # noqa: F821
    y1_fmt: Union[str, "Formatter"] = "",  # noqa: F821
    y2_fmt: Union[str, "Formatter"] = "",  # noqa: F821
    x_prefix: str = "x=",
    y1_prefix: str = "y1=",
    y2_prefix: str = "y2=",
) -> Callable:
    x_fmt = x_fmt or ax2.xaxis.get_major_formatter()
    y1_fmt = y1_fmt or ax1.yaxis.get_major_formatter()
    y2_fmt = y2_fmt or ax2.yaxis.get_major_formatter()

    def formatter(*xy2: Tuple[float, float]) -> str:
        xy1: Tuple[float, float] = ax1.transData.inverted().transform(
            ax2.transData.transform(xy2)
        )

        values: List[str] = []

        prefix: str
        fmt: str
        v: float
        for prefix, fmt, v in (
            (x_prefix, x_fmt, xy2[0]),
            (y1_prefix, y1_fmt, xy1[1]),
            (y2_prefix, y2_fmt, xy2[1]),
        ):
            if fmt == "none":
                continue

            try:
                values.append(
                    prefix
                    + (
                        fmt.format(v)
                        if isinstance(fmt, str)
                        else fmt.format_data_short(v)
                    )
                )
            except Exception as e:  # TODO
                return f"{type(e)}: {e}"

        return " ".join(values)

    return formatter
