# pyimpspec is licensed under the GPLv3 or later (https://www.gnu.org/licenses/gpl-3.0.html).
# Copyright 2022 pyimpspec developers
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
from pyimpspec.analysis import KramersKronigResult, FittingResult
from pyimpspec.analysis.fitting import _interpolate
from inspect import signature
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from numpy import inf, ceil, log10 as log, ndarray
from typing import List, Optional, Tuple, Union


# Vibrant color scheme from https://personal.sron.nl/~pault/


def plot_nyquist(
    data: Union[DataSet, KramersKronigResult, FittingResult],
    color: str = "black",
    line: bool = False,
    fig: Optional[Figure] = None,
    axis: Optional[Axes] = None,
    num_per_decade: int = 100,
) -> Tuple[Figure, Axes]:
    """
    Plot some data as a Nyquist plot (-Z" vs Z').

    Parameters
    ----------
    data: Union[DataSet, KramersKronigResult, FittingResult]
        The data to plot.
        DataSet instances are plotted using markers (optionally as a line) while all other types of data are plotted as lines.

    color: str = "black"
        The color of the marker (and line).

    line: bool = False
        Whether or not a DataSet instance should be plotted as a line instead.

    fig: Optional[Figure] = None
        The matplotlib.figure.Figure instance to use when plotting the data.

    axis: Optional[Axes] = None
        The matplotlib.axes.Axes instance to use when plotting the data.

    num_per_decade: int = 100
        If the data being plotted is not a DataSet instance (e.g. a KramersKronigResult instance), then this parameter can be used to change how many points are used to draw the line (i.e. how smooth or angular the line looks).
    """
    assert (
        isinstance(data, DataSet)
        or type(data) is KramersKronigResult
        or type(data) is FittingResult
        or (hasattr(data, "get_nyquist_data") and hasattr(data, "get_label"))
    ), data
    assert type(color) is str, color
    assert type(line) is bool, line
    assert type(fig) is Figure or fig is None, fig
    if fig is None:
        fig, axis = plt.subplots()
    assert axis is not None, axis
    x: ndarray
    y: ndarray
    if isinstance(data, DataSet):
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
            x, y = data.get_nyquist_data(num_per_decade=num_per_decade)  # type: ignore
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
    color_mag: str = "black",
    color_phase: str = "black",
    line: bool = False,
    fig: Optional[Figure] = None,
    axes: List[Axes] = [],
    num_per_decade: int = 100,
) -> Tuple[Figure, List[Axes]]:
    """
    Plot some data as a Bode plot (log |Z| and phi vs log f).

    Parameters
    ----------
    data: Union[DataSet, KramersKronigResult, FittingResult]
        The data to plot.
        DataSet instances are plotted using markers (optionally as a line) while all other types of data are plotted as lines.

    color_mag: str = "black"
        The color of the marker (and line) for the logarithm of the absolute magnitude.

    color_phase: str = "black"
        The color of the marker (and line) for the logarithm of the phase shift.

    line: bool = False
        Whether or not a DataSet instance should be plotted as a line instead.

    fig: Optional[Figure] = None
        The matplotlib.figure.Figure instance to use when plotting the data.

    axes: List[Axes] = []
        A list of matplotlib.axes.Axes instances to use when plotting the data.

    num_per_decade: int = 100
        If the data being plotted is not a DataSet instance (e.g. a KramersKronigResult instance), then this parameter can be used to change how many points are used to draw the line (i.e. how smooth or angular the line looks).
    """
    assert (
        isinstance(data, DataSet)
        or type(data) is KramersKronigResult
        or type(data) is FittingResult
        or (hasattr(data, "get_bode_data") and hasattr(data, "get_label"))
    ), data
    assert type(color_mag) is str, color_mag
    assert type(color_phase) is str, color_phase
    assert type(line) is bool, line
    assert type(fig) is Figure or fig is None, fig
    assert type(axes) is list, axes
    mag_suffix: str = r" ($|Z|$)"
    phase_suffix: str = r" ($\phi$)"
    if fig is None:
        axis: Axes
        fig, axis = plt.subplots()
        axes = [axis, axis.twinx()]
    assert len(axes) == 2, axes
    x: ndarray
    y1: ndarray
    y2: ndarray
    if isinstance(data, DataSet):
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
                linestyle=":",
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
            x, y1, y2 = data.get_bode_data(num_per_decade=num_per_decade)  # type: ignore
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
    color_re: str = "black",
    color_im: str = "black",
    fig: Optional[Figure] = None,
    axes: List[Axes] = [],
) -> Tuple[Figure, List[Axes]]:
    """
    Plot the residuals of a test or fit result.

    Parameters
    ----------
    result: Union[KramersKronigResult, FittingResult]
        The result to plot.

    color_re: str = "black"
        The color of the markers and line for the residuals of the real parts of the impedances.

    color_im: str = "black"
        The color of the markers and line for the residuals of the imaginary parts of the impedances.

    fig: Optional[Figure] = None
        The matplotlib.figure.Figure instance to use when plotting the data.

    axes: List[Axes] = []
        A list of matplotlib.axes.Axes instances to use when plotting the data.
    """
    assert (
        type(result) is KramersKronigResult
        or type(result) is FittingResult
        or hasattr(result, "get_residual_data")
    ), result
    assert type(color_re) is str, color_re
    assert type(color_im) is str, color_im
    assert type(fig) is Figure or fig is None, fig
    assert type(axes) is list, axes
    if fig is None:
        axis: Axes
        fig, axis = plt.subplots()
        axes = [axis, axis.twinx()]
    assert len(axes) == 2, axes
    if not axes[0].lines:
        axes[0].axhline(0, color="black", alpha=0.25)
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
    color_mu: str = "black",
    color_xps: str = "black",
    color_criterion: str = "black",
    fig: Optional[Figure] = None,
    axes: List[Axes] = [],
) -> Tuple[Figure, List[Axes]]:
    """
    Plot the mu-values and pseudo chi-squared values of exploratory Kramers-Kronig test results.

    Parameters
    ----------
    scored_tests: List[Tuple[float, KramersKronigResult]]
        The scored test results as returned by the pyimpspec.analysis.kramers_kronig.score_test_results function.

    mu_criterion: float
        The mu-criterion that was used when performing the tests.

    color_mu: str = "black"
        The color of the markers and line for the mu-values.

    color_xps: str = "black"
        The color of the markers and line for the pseudo chi-squared values.

    color_criterion: str = "black"
        The color of the line for the mu-criterion.

    fig: Optional[Figure] = None
        The matplotlib.figure.Figure instance to use when plotting the data.

    axes: List[Axes] = []
        A list of matplotlib.axes.Axes instances to use when plotting the data.
    """
    assert type(scored_tests) is list, scored_tests
    assert type(color_mu) is str, color_mu
    assert type(color_xps) is str, color_xps
    assert type(color_criterion) is str, color_criterion
    assert type(fig) is Figure or fig is None, fig
    assert type(axes) is list, axes
    if fig is None:
        axis: Axes
        fig, axis = plt.subplots()
        axes = [axis, axis.twinx()]
    assert len(axes) == 2, axes
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
    axes[0].set_xlabel(r"number of RC elements")
    axes[0].set_ylabel(r"$\mu$")
    axes[1].set_ylabel(r"$\log{\chi^{2}_{\rm ps}}$")
    return (
        fig,
        axes,
    )


def plot_circuit(
    circuit: Circuit,
    f: Union[List[float], ndarray] = [],
    min_f: float = 1e-1,
    max_f: float = 1e5,
    data: Optional[DataSet] = None,
    visible_data: bool = False,
    title: Optional[str] = None,
    fig: Optional[Figure] = None,
    axes: List[Axes] = [],
) -> Tuple[Figure, List[Axes]]:
    """
    Plot the simulated impedance response of a circuit as both a Nyquist and a Bode plot.

    Parameters
    ----------
    circuit: Circuit
        The circuit to use when simulating the impedance response.

    f: Union[List[float], ndarray] = []
        The frequencies (in hertz) to use when simulating the impedance response.
        If no frequencies are provided, then the range defined by the min_f and max_f parameters will be used instead.
        Alternatively, a DataSet instance can be provided via the data parameter.

    min_f: float = 0.1
        The lower limit of the frequency range to use if a list of frequencies is not provided.

    max_f: float = 100000.0
        The upper limit of the frequency range to use if a list of frequencies is not provided.

    data: Optional[DataSet] = None
        An optional DataSet instance.
        If provided, then the frequencies of this instance will be used when simulating the impedance spectrum of the circuit.

    visible_data: bool = False
        Whether or not the optional DataSet instance should also be plotted alongside the simulated impedance spectrum of the circuit.

    title: Optional[str] = None
        The title of the figure.
        If not title is provided, then the circuit description code of the circuit is used instead.

    fig: Optional[Figure] = None
        The matplotlib.figure.Figure instance to use when plotting the data.

    axes: List[Axes] = []
        A list of matplotlib.axes.Axes instances to use when plotting the data.
    """
    assert type(circuit) is Circuit, circuit
    assert type(f) is list or type(f) is ndarray, f
    assert min_f > 0 and max_f < inf and min_f < max_f, (
        min_f,
        max_f,
    )
    assert isinstance(data, DataSet) or data is None, data
    assert type(visible_data) is bool, visible_data
    assert type(title) is str or title is None, title
    assert type(fig) is Figure or fig is None, fig
    assert type(axes) is list, axes
    if fig is None:
        assert len(axes) == 0, axes
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
    assert len(axes) == 3, axes
    nyquist_color: str = "#0077BB"
    bode_mag_color: str = "#0077BB"
    bode_phase_color: str = "#EE7733"
    if data is not None and visible_data is True:
        plot_nyquist(data, color="#0077BB", fig=fig, axis=axes[0])
        plot_bode(
            data,
            color_mag="#0077BB",
            color_phase="#EE7733",
            fig=fig,
            axes=[axes[1], axes[2]],
        )
        nyquist_color = "#CC3311"
        bode_mag_color = "#CC3311"
        bode_phase_color = "#009988"
    spectrum: DataSet
    if len(f) == 0:
        if data is not None:
            f = _interpolate(
                [min(data.get_frequency()), max(data.get_frequency())], 100
            )
        else:
            f = _interpolate([min_f, max_f], 100)
    Z: ndarray = circuit.impedances(f)
    spectrum = DataSet.from_dict(
        {
            "frequency": f,
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
    """
    Plot a DataSet instance as both a Nyquist and a Bode plot.

    Parameters
    ----------
    data: DataSet
        The DataSet instance to plot.

    title: Optional[str] = None
        The title of the figure.
        If not title is provided, then the label of the DataSet is used instead.

    fig: Optional[Figure] = None
        The matplotlib.figure.Figure instance to use when plotting the data.

    axes: List[Axes] = []
        A list of matplotlib.axes.Axes instances to use when plotting the data.
    """
    assert isinstance(data, DataSet), data
    assert type(title) is str or title is None, title
    assert type(fig) is Figure or fig is None, fig
    assert type(axes) is list, axes
    if fig is None:
        assert len(axes) == 0, axes
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
    assert len(axes) == 3, axes
    plot_nyquist(data, color="#0077BB", fig=fig, axis=axes[0])
    plot_bode(
        data,
        color_mag="#0077BB",
        color_phase="#EE7733",
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
    """
    Plot the results of an exploratory Kramers-Kronig test and the tested DataSet as a Nyquist plot, a Bode plot, a plot of the residuals, and a plot of the mu-values and pseudo chi-squared values.

    Parameters
    ----------
    scored_tests: List[Tuple[float, KramersKronigResult]]
        The scored test results as returned by the pyimpspec.analysis.kramers_kronig.score_test_results function.

    mu_criterion: float
        The mu-criterion that was used when performing the tests.

    data: DataSet
        The DataSet instance that was tested.

    title: Optional[str] = None
        The title of the figure.
        If no title is provided, then the label of the DataSet is used instead.

    fig: Optional[Figure] = None
        The matplotlib.figure.Figure instance to use when plotting the data.

    axes: List[Axes] = []
        A list of matplotlib.axes.Axes instances to use when plotting the data.
    """
    assert type(scored_tests) is list and all(
        map(lambda _: type(_) is tuple, scored_tests)
    ), scored_tests
    assert type(mu_criterion) is float, mu_criterion
    assert isinstance(data, DataSet), data
    assert type(title) is str or title is None, title
    assert type(fig) is Figure or fig is None, fig
    assert type(axes) is list, axes
    if fig is None:
        assert len(axes) == 0, axes
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
    assert len(axes) == 7, axes
    test: KramersKronigResult = scored_tests[0][1]
    plot_mu_xps(
        scored_tests,
        mu_criterion,
        color_mu="#EE3377",
        color_xps="#009988",
        color_criterion="#CC3311",
        fig=fig,
        axes=[axes[0], axes[1]],
    )
    plot_residual(
        test,
        color_re="#EE3377",
        color_im="#009988",
        fig=fig,
        axes=[axes[2], axes[3]],
    )
    plot_nyquist(data, color="#0077BB", fig=fig, axis=axes[4])
    plot_nyquist(test, color="#CC3311", fig=fig, axis=axes[4])
    plot_bode(
        data,
        color_mag="#0077BB",
        color_phase="#EE7733",
        fig=fig,
        axes=[axes[5], axes[6]],
    )
    plot_bode(
        test,
        color_mag="#CC3311",
        color_phase="#009988",
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
    """
    Plot a the result of a circuit fit as a Nyquist plot, a Bode plot, and a plot of the residuals.

    Parameters
    ----------
    fit: Union[FittingResult, KramersKronigResult]
        The circuit fit or test result.

    data: Optional[DataSet] = None
        The DataSet instance that a circuit was fitted to.

    title: Optional[str] = None
        The title of the figure.
        If no title is provided, then the circuit description code (and label of the DataSet) is used instead.

    fig: Optional[Figure] = None
        The matplotlib.figure.Figure instance to use when plotting the data.

    axes: List[Axes] = []
        A list of matplotlib.axes.Axes instances to use when plotting the data.
    """
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
    assert isinstance(data, DataSet) or data is None, data
    assert type(title) is str or title is None, title
    assert type(fig) is Figure or fig is None, fig
    assert type(axes) is list, axes
    if fig is None:
        assert len(axes) == 0, axes
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
    assert len(axes) == 5, axes
    if data is not None:
        plot_nyquist(data, color="#0077BB", fig=fig, axis=axes[0])
    if data is not None:
        plot_bode(
            data,
            color_mag="#0077BB",
            color_phase="#EE7733",
            fig=fig,
            axes=[axes[1], axes[2]],
        )
    plot_nyquist(fit, color="#CC3311", fig=fig, axis=axes[0])
    plot_bode(
        fit,
        color_mag="#CC3311",
        color_phase="#009988",
        fig=fig,
        axes=[axes[1], axes[2]],
    )
    plot_residual(
        fit,
        color_re="#EE3377",
        color_im="#009988",
        fig=fig,
        axes=[axes[3], axes[4]],
    )
    return (
        fig,
        axes,
    )
