# Copyright 2022 pyimpspec developers
# pyimpspec is licensed under the GPLv3 or later (https://www.gnu.org/licenses/gpl-3.0.html).
# The licenses of pyimpspec's dependencies and/or sources of portions of code are included in
# the LICENSES folder.

from pyimpspec.circuit import Circuit
from pyimpspec.data import DataSet
from pyimpspec.analysis import KramersKronigResult, FittingResult
from pyimpspec.analysis.fitting import _interpolate
from inspect import signature
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from numpy import inf, ceil, log10 as log, ndarray
from typing import List, Optional, Tuple, Union


# Vibrant color scheme from https://personal.sron.nl/~pault/
COLOR_BLACK: str = "black"
COLOR_BLUE: str = "#0077BB"
COLOR_MAGENTA: str = "#EE3377"
COLOR_ORANGE: str = "#EE7733"
COLOR_RED: str = "#CC3311"
COLOR_TEAL: str = "#009988"


def plot_nyquist(
    data: Union[DataSet, KramersKronigResult, FittingResult],
    color: str = COLOR_BLACK,
    line: bool = False,
    fig: Optional[Figure] = None,
    axis: Optional[Axes] = None,
) -> Tuple[Figure, Axes]:
    assert (
        type(data) is DataSet
        or type(data) is KramersKronigResult
        or type(data) is FittingResult
        or (hasattr(data, "get_nyquist_data") and hasattr(data, "get_label"))
    )
    assert type(color) is str
    assert type(line) is bool
    assert type(fig) is Figure or fig is None
    if fig is None:
        fig, axis = plt.subplots()
    assert axis is not None
    x: ndarray
    y: ndarray
    if type(data) is DataSet:
        x, y = data.get_nyquist_data()
        if line is True:
            axis.plot(
                x,
                y,
                color=color,
                label=data.get_label(),
            )
        else:
            axis.scatter(
                x,
                y,
                marker="o",
                facecolor="none",
                edgecolor=color,
                label=data.get_label(),
            )
    else:
        if "num_per_decade" in signature(data.get_nyquist_data).parameters:
            x, y = data.get_nyquist_data(num_per_decade=100)  # type: ignore
        else:
            x, y = data.get_nyquist_data()
        label: str
        if type(data) is KramersKronigResult:
            label = "KK"
        elif type(data) is FittingResult:
            label = str(data.circuit)
        else:
            label = data.get_label()  # type: ignore
        axis.plot(
            x,
            y,
            color=color,
            linestyle="--",
            label=label,
        )
    axis.set_xlabel(r"$Z_{\rm re}\ (\Omega)$")
    axis.set_ylabel(r"$-Z_{\rm im}\ (\Omega)$")
    axis.set_aspect("equal", adjustable="datalim")
    return (
        fig,
        axis,
    )


def plot_bode(
    data: Union[DataSet, KramersKronigResult, FittingResult],
    color_mag: str = COLOR_BLACK,
    color_phase: str = COLOR_BLACK,
    line: bool = False,
    fig: Optional[Figure] = None,
    axes: List[Axes] = [],
) -> Tuple[Figure, List[Axes]]:
    assert (
        type(data) is DataSet
        or type(data) is KramersKronigResult
        or type(data) is FittingResult
        or (hasattr(data, "get_bode_data") and hasattr(data, "get_label"))
    )
    assert type(color_mag) is str
    assert type(color_phase) is str
    assert type(line) is bool
    assert type(fig) is Figure or fig is None
    assert type(axes) is list
    mag_suffix: str = r" ($|Z|$)"
    phase_suffix: str = r" ($\phi$)"
    if fig is None:
        axis: Axes
        fig, axis = plt.subplots()
        axes = [axis, axis.twinx()]
    assert len(axes) == 2
    x: ndarray
    y1: ndarray
    y2: ndarray
    if type(data) is DataSet:
        x, y1, y2 = data.get_bode_data()
        if line is True:
            axes[0].plot(
                x,
                y1,
                color=color_mag,
                label=(data.get_label() or "Data") + mag_suffix,
            )
            axes[1].plot(
                x,
                y2,
                color=color_phase,
                label=(data.get_label() or "Data") + phase_suffix,
            )
        else:
            axes[0].scatter(
                x,
                y1,
                marker="o",
                facecolor="none",
                edgecolor=color_mag,
                label=(data.get_label() or "Data") + mag_suffix,
            )
            axes[1].scatter(
                x,
                y2,
                marker="s",
                facecolor="none",
                edgecolor=color_phase,
                label=(data.get_label() or "Data") + phase_suffix,
            )
    else:
        if "num_per_decade" in signature(data.get_bode_data).parameters:
            x, y1, y2 = data.get_bode_data(num_per_decade=100)  # type: ignore
        else:
            x, y1, y2 = data.get_bode_data()
        label: str
        if type(data) is KramersKronigResult:
            label = "KK"
        elif type(data) is FittingResult:
            label = str(data.circuit)
        else:
            label = data.get_label()  # type: ignore
        axes[0].plot(
            x,
            y1,
            color=color_mag,
            linestyle="--",
            label=label + mag_suffix,
        )
        axes[1].plot(
            x,
            y2,
            color=color_phase,
            linestyle=":",
            label=label + phase_suffix,
        )
    axes[0].set_xlabel(r"$\log{f}$")
    axes[0].set_ylabel(r"$\log{|Z|}$")
    axes[1].set_ylabel(r"$-\phi\ (^\circ)$")
    return (
        fig,
        axes,
    )


def plot_residual(
    result: Union[KramersKronigResult, FittingResult],
    color_re: str = COLOR_BLACK,
    color_im: str = COLOR_BLACK,
    fig: Optional[Figure] = None,
    axes: List[Axes] = [],
) -> Tuple[Figure, List[Axes]]:
    assert (
        type(result) is KramersKronigResult
        or type(result) is FittingResult
        or hasattr(result, "get_residual_data")
    )
    assert type(color_re) is str
    assert type(color_im) is str
    assert type(fig) is Figure or fig is None
    assert type(axes) is list
    if fig is None:
        axis: Axes
        fig, axis = plt.subplots()
        axes = [axis, axis.twinx()]
    assert len(axes) == 2
    if not axes[0].lines:
        axes[0].axhline(0, color=COLOR_BLACK, alpha=0.25)
    x: ndarray
    y1: ndarray
    y2: ndarray
    x, y1, y2 = result.get_residual_data()
    axes[0].scatter(
        x,
        y1,
        marker="o",
        facecolor="none",
        edgecolor=color_re,
        label=r"$\Delta_{\rm re}$",
    )
    axes[1].scatter(
        x,
        y2,
        marker="s",
        facecolor="none",
        edgecolor=color_im,
        label=r"$\Delta_{\rm im}$",
    )
    axes[0].plot(x, y1, color=color_re, linestyle="--")
    axes[1].plot(x, y2, color=color_im, linestyle=":")
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
    axes[0].set_xlabel(r"$\log{f}$")
    axes[0].set_ylabel(r"$\Delta_{\rm re}\ (\%)$")
    axes[1].set_ylabel(r"$\Delta_{\rm im}\ (\%)$")
    return (
        fig,
        axes,
    )


def plot_mu_xps(
    scored_tests: List[Tuple[float, KramersKronigResult]],
    mu_criterion: float,
    color_mu: str = COLOR_BLACK,
    color_xps: str = COLOR_BLACK,
    color_criterion: str = COLOR_BLACK,
    fig: Optional[Figure] = None,
    axes: List[Axes] = [],
) -> Tuple[Figure, List[Axes]]:
    assert type(scored_tests) is list
    assert type(color_mu) is str
    assert type(color_xps) is str
    assert type(color_criterion) is str
    assert type(fig) is Figure or fig is None
    assert type(axes) is list
    if fig is None:
        axis: Axes
        fig, axis = plt.subplots()
        axes = [axis, axis.twinx()]
    assert len(axes) == 2
    x: List[int] = []
    y1: List[float] = []
    y2: List[float] = []
    for _, test in sorted(scored_tests, key=lambda _: _[1].num_RC):
        x.append(test.num_RC)
        y1.append(test.mu)
        y2.append(log(test.pseudo_chisqr))
    axes[0].axvline(scored_tests[0][1].num_RC, color=color_criterion, alpha=0.5)
    axes[0].axhline(mu_criterion, color=color_criterion, alpha=0.5)
    axes[0].scatter(x, y1, marker="o", facecolor="none", edgecolor=color_mu)
    axes[1].scatter(x, y2, marker="s", facecolor="none", edgecolor=color_xps)
    axes[0].plot(x, y1, color=color_mu, linestyle="--")
    axes[1].plot(x, y2, color=color_xps, linestyle=":")
    axes[0].set_ylim(-0.1, 1.1)
    axes[0].set_xlabel(r"num. RC")
    axes[0].set_ylabel(r"$\mu$")
    axes[1].set_ylabel(r"$\log{\chi^{2}_{\rm ps}}$")
    return (
        fig,
        axes,
    )


def plot_circuit(
    circuit: Circuit,
    freq: Union[List[float], ndarray] = [],
    min_f: float = 1e-1,
    max_f: float = 1e5,
    data: Optional[DataSet] = None,
    visible_data: bool = False,
    title: Optional[str] = None,
    fig: Optional[Figure] = None,
    axes: List[Axes] = [],
) -> Tuple[Figure, List[Axes]]:
    assert type(circuit) is Circuit
    assert type(freq) is list or type(freq) is ndarray
    assert min_f > 0 and max_f < inf and min_f < max_f
    assert type(data) is DataSet or data is None
    assert type(visible_data) is bool
    assert type(title) is str or title is None
    assert type(fig) is Figure or fig is None
    assert type(axes) is list
    if fig is None:
        assert len(axes) == 0
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
    assert len(axes) == 3
    nyquist_color: str = COLOR_BLUE
    bode_mag_color: str = COLOR_BLUE
    bode_phase_color: str = COLOR_ORANGE
    if data is not None and visible_data is True:
        plot_nyquist(data, color=COLOR_BLUE, fig=fig, axis=axes[0])
        plot_bode(
            data,
            color_mag=COLOR_BLUE,
            color_phase=COLOR_ORANGE,
            fig=fig,
            axes=[axes[1], axes[2]],
        )
        nyquist_color = COLOR_RED
        bode_mag_color = COLOR_RED
        bode_phase_color = COLOR_TEAL
    spectrum: DataSet
    if len(freq) == 0:
        if data is not None:
            freq = _interpolate(
                [min(data.get_frequency()), max(data.get_frequency())], 100
            )
        else:
            freq = _interpolate([min_f, max_f], 100)
    Z: ndarray = circuit.impedances(freq)
    spectrum = DataSet.from_dict(
        {
            "frequency": freq,
            "real": Z.real,
            "imaginary": Z.imag,
            "label": str(circuit),
        }
    )
    plot_nyquist(spectrum, color=nyquist_color, line=True, fig=fig, axis=axes[0])
    plot_bode(
        spectrum,
        color_mag=bode_mag_color,
        color_phase=bode_phase_color,
        line=True,
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
    fig: Optional[Figure] = None,
    axes: List[Axes] = [],
) -> Tuple[Figure, List[Axes]]:
    assert type(data) is DataSet
    assert type(title) is str or title is None
    assert type(fig) is Figure or fig is None
    assert type(axes) is list
    if fig is None:
        assert len(axes) == 0
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
    assert len(axes) == 3
    plot_nyquist(data, color=COLOR_BLUE, fig=fig, axis=axes[0])
    plot_bode(
        data,
        color_mag=COLOR_BLUE,
        color_phase=COLOR_ORANGE,
        fig=fig,
        axes=[axes[1], axes[2]],
    )
    return (
        fig,
        axes,
    )


def plot_exploratory_tests(
    scored_tests: List[Tuple[float, KramersKronigResult]],
    mu_criterion: float,
    data: DataSet,
    title: Optional[str] = None,
    fig: Optional[Figure] = None,
    axes: List[Axes] = [],
) -> Tuple[Figure, List[Axes]]:
    assert type(scored_tests) is list and all(
        map(lambda _: type(_) is tuple, scored_tests)
    )
    assert type(mu_criterion) is float
    assert type(data) is DataSet
    assert type(title) is str or title is None
    assert type(fig) is Figure or fig is None
    assert type(axes) is list
    if fig is None:
        assert len(axes) == 0
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
    assert len(axes) == 7
    test: KramersKronigResult = scored_tests[0][1]
    plot_mu_xps(
        scored_tests,
        mu_criterion,
        color_mu=COLOR_MAGENTA,
        color_xps=COLOR_TEAL,
        color_criterion=COLOR_RED,
        fig=fig,
        axes=[axes[0], axes[1]],
    )
    plot_residual(
        test,
        color_re=COLOR_MAGENTA,
        color_im=COLOR_TEAL,
        fig=fig,
        axes=[axes[2], axes[3]],
    )
    plot_nyquist(data, color=COLOR_BLUE, fig=fig, axis=axes[4])
    plot_nyquist(test, color=COLOR_RED, fig=fig, axis=axes[4])
    plot_bode(
        data,
        color_mag=COLOR_BLUE,
        color_phase=COLOR_ORANGE,
        fig=fig,
        axes=[axes[5], axes[6]],
    )
    plot_bode(
        test,
        color_mag=COLOR_RED,
        color_phase=COLOR_TEAL,
        fig=fig,
        axes=[axes[5], axes[6]],
    )
    return (
        fig,
        axes,
    )


def plot_fit(
    fit: Union[FittingResult, KramersKronigResult],
    data: Optional[DataSet] = None,
    title: Optional[str] = None,
    fig: Optional[Figure] = None,
    axes: List[Axes] = [],
) -> Tuple[Figure, List[Tuple[Axes]]]:
    assert (
        type(fit) is FittingResult
        or type(fit) is KramersKronigResult
        or (
            hasattr(fit, "get_label")
            and hasattr(fit, "get_nyquist_data")
            and hasattr(fit, "get_bode_data")
            and hasattr(fit, "get_residual_data")
        )
    ), fit
    assert type(data) is DataSet or data is None
    assert type(title) is str or title is None
    assert type(fig) is Figure or fig is None
    assert type(axes) is list
    if fig is None:
        assert len(axes) == 0
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
            if type(fit) is FittingResult:
                if data is not None:
                    title = f"{data.get_label()}\n{fit.circuit.get_label()}"
                else:
                    title = fit.circuit.get_label()
            elif type(fit) is KramersKronigResult or (
                hasattr(fit, "circuit") and hasattr(fit, "num_RC")
            ):
                title = (
                    fit.circuit.get_label()
                    .replace("K", r"$\rm (RC)_" + f"{{{str(fit.num_RC)}}}$", 1)  # type: ignore
                    .replace("K", "")
                )
                if data is not None:
                    title = f"{data.get_label()}\n{title}"
                if hasattr(fit, "get_label"):
                    title += f" {fit.get_label()}"  # type: ignore
            else:
                if data is not None:
                    title = f"{data.get_label()}\n{fit.get_label()}"  # type: ignore
                else:
                    title = fit.get_label()  # type: ignore
        if title != "":
            fig.suptitle(title)
    assert len(axes) == 5
    if data is not None:
        plot_nyquist(data, color=COLOR_BLUE, fig=fig, axis=axes[0])
    if data is not None:
        plot_bode(
            data,
            color_mag=COLOR_BLUE,
            color_phase=COLOR_ORANGE,
            fig=fig,
            axes=[axes[1], axes[2]],
        )
    plot_nyquist(fit, color=COLOR_RED, fig=fig, axis=axes[0])
    plot_bode(
        fit,
        color_mag=COLOR_RED,
        color_phase=COLOR_TEAL,
        fig=fig,
        axes=[axes[1], axes[2]],
    )
    plot_residual(
        fit,
        color_re=COLOR_MAGENTA,
        color_im=COLOR_TEAL,
        fig=fig,
        axes=[axes[3], axes[4]],
    )
    return (
        fig,
        axes,
    )
