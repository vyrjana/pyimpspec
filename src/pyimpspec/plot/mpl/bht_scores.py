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

from pyimpspec.analysis.drt import BHTResult
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
)
from pyimpspec.plot.colors import (
    COLOR_MAGENTA,
    COLOR_TEAL,
)


def plot_bht_scores(
    drt: BHTResult,
    colors: Optional[Dict[str, str]] = None,
    label_bars: bool = False,
    figure: Optional["Figure"] = None,  # noqa: F821
    title: Optional[str] = None,
    legend: bool = True,
    axes: Optional[List["Axes"]] = None,  # noqa: F821
    adjust_axes: bool = True,
    **kwargs,
) -> Tuple["Figure", List["Axes"]]:  # noqa: F821
    """
    Plot the scores of a BHTResult as a bar chart.

    Parameters
    ----------
    drt: BHTResult
        The result to plot.

    colors: Optional[Dict[str, str]], optional
        The colors of the markers or lines. Valid keys: 'real', 'imaginary'.

    label_bars: bool, optional
        Whether or not to add labels above the bars.

    figure: Optional[|Figure|], optional
        The matplotlib.figure.Figure instance to use when plotting the data.

    title: Optional[str], optional
        The title of the figure.

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
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from pandas import DataFrame

    assert hasattr(drt, "to_scores_dataframe") and callable(
        drt.to_scores_dataframe
    ), drt
    if colors is None:
        colors = {}
    assert isinstance(colors, dict), colors
    assert isinstance(title, str) or title is None, title
    assert isinstance(legend, bool), legend
    assert isinstance(figure, Figure) or figure is None, figure
    assert isinstance(adjust_axes, bool), adjust_axes
    axis: Axes
    if figure is None:
        assert axes is None
        figure, axis = plt.subplots()
        axes = [axis]
        if title:
            figure.suptitle(title)
    assert isinstance(axes, list), axes
    assert len(axes) == 1, axes
    assert all(map(lambda _: isinstance(_, Axes), axes))
    axis = axes[0]
    color_real: str = colors.get("real", COLOR_TEAL)
    color_imaginary: str = colors.get("imaginary", COLOR_MAGENTA)
    df: DataFrame = drt.to_scores_dataframe(
        columns=[
            r"Score",
            r"Real (\%)",
            r"Imag. (\%)",
        ],
        rows=[
            r"$s_\mu$",
            r"$s_{1\sigma}$",
            r"$s_{2\sigma}$",
            r"$s_{3\sigma}$",
            r"$s_{\rm HD}$",
            r"$s_{\rm JSD}$",
        ],
    )
    assert isinstance(df, DataFrame), type(df)
    width: float = 0.5
    ticks: List[float] = []
    ticklabels: List[str] = []
    i: int
    label: str
    real: float
    imag: float
    for i, (label, real, imag) in df.iterrows():
        x: float = 2.0 * i
        axis.bar(x - width / 2, real, width=width, color=color_real)
        axis.bar(x + width / 2, imag, width=width, color=color_imaginary)
        ticks.append(x)
        ticklabels.append(label)
    if legend is True:
        axis.bar(0.0, -1.0, width=0.0, color=color_real, label="Real")
        axis.bar(0.0, -1.0, width=0.0, color=color_imaginary, label="Imag.")
        axis.legend()
    if label_bars is True:
        for i, bar in enumerate(axis.patches):
            if bar.get_height() < 0.0:
                continue
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 3.0,
                "re." if i % 2 == 0 else "im.",
                ha="center",
                va="bottom",
            )
    if adjust_axes is True:
        axis.grid(axis="y")
        axis.set_xlabel("")
        axis.set_xticks(ticks)
        axis.set_xticklabels(ticklabels)
        axis.set_ylabel("Score (%)")
        axis.set_ylim(0.0, 120.0 if label_bars else 100.0)
    return (
        figure,
        axes,
    )
