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

from dataclasses import dataclass
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)
from numpy import (
    angle,
    complex128,
    float64,
    log as ln,
    log10 as log,
    pi,
)
from numpy.typing import NDArray
from pyimpspec.typing import (
    ComplexImpedances,
    ComplexResiduals,
    Frequencies,
    Impedances,
    Phases,
    Residuals,
)
from pyimpspec.typing.helpers import (
    _is_boolean,
    _is_floating,
    _is_floating_array,
    _is_integer,
)
from pyimpspec.data import DataSet
from pyimpspec.analysis.utility import (
    _calculate_residuals,
    get_default_num_procs,
)
from pyimpspec.exceptions import ZHITError
from pyimpspec.progress import Progress
from .weights import (
    _WINDOW_FUNCTIONS,
    _generate_window_options,
    _initialize_window_functions,
)
from .smoothing import _generate_smoothing_options
from .interpolation import _generate_interpolation_options
from .reconstruction import _reconstruct_modulus_data
from .offset import _adjust_modulus_offset


@dataclass(frozen=True)
class ZHITResult:
    """
    An object representing the results of reconstructing the modulus of the impedance using the Z-HIT algorithm.

    Parameters
    ----------
    frequencies: |Frequencies|
        The frequencies used to perform the reconstruction.

    impedances: |ComplexImpedances|
        The reconstructed impedances.

    residuals: |ComplexResiduals|
        The residuals for the real (eq. 15 in Schönleber et al., 2014) and imaginary (eq. 16 in Schönleber et al., 2014) parts of the reconstruction.

    pseudo_chisqr: float
        The pseudo chi-squared value (|pseudo chi-squared|, eq. 14 in Boukamp, 1995).

    smoothing: str
        The smoothing algorithm that was used.

    interpolation: str
        The spline that was used for interpolation.

    window: str
        The window function that was used.
    """

    frequencies: Frequencies
    impedances: ComplexImpedances
    residuals: ComplexResiduals
    pseudo_chisqr: float
    smoothing: str
    interpolation: str
    window: str

    def __repr__(self) -> str:
        return f"ZHITResult ({hex(id(self))})"

    def get_label(self) -> str:
        """
        Get the label for this result.

        Returns
        -------
        str
        """
        return "Z-HIT"

    def get_frequencies(self) -> Frequencies:
        """
        Get the frequencies in the fitted frequency range.

        Returns
        -------
        |Frequencies|
        """
        return self.frequencies

    def get_impedances(self) -> ComplexImpedances:
        """
        Get the impedance response of the fitted circuit.

        Returns
        -------
        |ComplexImpedances|
        """
        return self.impedances

    def get_nyquist_data(self) -> Tuple[Impedances, Impedances]:
        """
        Get the data necessary to plot this FitResult as a Nyquist plot: the real and the negative imaginary parts of the impedances.

        Returns
        -------
        Tuple[|Impedances|, |Impedances|]
        """
        return (
            self.impedances.real,
            -self.impedances.imag,
        )

    def get_bode_data(self) -> Tuple[Frequencies, Impedances, Phases]:
        """
        Get the data necessary to plot this FitResult as a Bode plot: the frequencies, the absolute magnitudes of the impedances, and the negative phase angles/shifts of the impedances in degrees.

        Returns
        -------
        Tuple[|Frequencies|, |Impedances|, |Phases|]
        """
        return (
            self.frequencies,
            abs(self.impedances),
            -angle(self.impedances, deg=True),
        )

    def get_residuals_data(
        self,
    ) -> Tuple[Frequencies, Residuals, Residuals]:
        """
        Get the data necessary to plot the relative residuals for this result: the frequencies, the relative residuals for the real parts of the impedances in percents, and the relative residuals for the imaginary parts of the impedances in percents.

        Returns
        -------
        Tuple[|Frequencies|, |Residuals|, |Residuals|]
        """
        return (
            self.frequencies,
            self.residuals.real * 100,
            self.residuals.imag * 100,
        )

    def to_statistics_dataframe(self) -> "DataFrame":  # noqa: F821
        """
        Get the statistics related to the modulus reconstruction as a `pandas.DataFrame`_ object.

        Returns
        -------
        `pandas.DataFrame`_
        """
        from pandas import DataFrame

        statistics: Dict[str, Union[int, float, str]] = {
            "Log pseudo chi-squared": log(self.pseudo_chisqr),
            "Smoothing": self.smoothing,
            "Interpolation": self.interpolation,
            "Window": self.window,
        }
        return DataFrame.from_dict(
            {
                "Label": list(statistics.keys()),
                "Value": list(statistics.values()),
            }
        )


def perform_zhit(
    data: DataSet,
    smoothing: str = "modsinc",
    interpolation: str = "makima",
    window: str = "auto",
    num_points: int = 3,
    polynomial_order: int = 2,
    num_iterations: int = 3,
    center: float = 1.5,
    width: float = 3.0,
    weights: Optional[NDArray[float64]] = None,
    admittance: bool = False,
    num_procs: int = -1,
) -> ZHITResult:
    r"""
    Performs a reconstruction of the modulus data of an impedance spectrum based on the phase data of that impedance spectrum using the Z-HIT algorithm described by Ehm et al. (2000).
    The results can be used to, e.g., check the validity of an impedance spectrum by detecting non-steady state issues like drift at low frequencies.
    See the references below for more information about the algorithm and its applications.
    The algorithm involves an offset adjustment of the reconstructed modulus data, which is done by fitting the reconstructed modulus data to the experimental modulus data in a frequency range that is unaffected (or minimally affected) by artifacts.
    This frequency range is typically around 1 Hz to 1000 Hz, which is why the default window function is a "boxcar" window that is centered around :math:`\log{f} = 1.5` and has a width of 3.0.
    Multiple window functions are supported and a custom array of weights can also be used.

    References:

    - Ehm, W., Göhr, H., Kaus, R., Röseler, B., and Schiller, C.A., 2000, Acta Chimica Hungarica, 137 (2-3), 145-157.
    - Ehm, W., Kaus, R., Schiller, C.A., and Strunz, W., 2001, in "New Trends in Electrochemical Impedance Spectroscopy and Electrochemical Noise Analysis".
    - `Schiller, C.A., Richter, F., Gülzow, E., and Wagner, N., 2001, 3, 374-378 <https://doi.org/10.1039/B007678N>`_

    Parameters
    ----------
    data: |DataSet|
        The data set for which the modulus of the impedance should be reconstructed.

    smoothing: str, optional
        The type of smoothing to apply: "none", "lowess" (`Locally Weighted Scatterplot Smoothing <https://www.statsmodels.org/dev/generated/statsmodels.nonparametric.smoothers_lowess.lowess.html#statsmodels.nonparametric.smoothers_lowess.lowess>`_), "modsinc" (`modified sinc kernel <https://pubs.acs.org/doi/full/10.1021/acsmeasuresciau.1c00054>`_), "savgol" (`Savitzky-Golay <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html>`_), "whithend" (`Whittaker-Henderson <https://doi.org/10.1021/acsmeasuresciau.1c00054>`_) or "auto".

    interpolation: str, optional
        The type of interpolation to apply: "akima" (`Akima spline <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.Akima1DInterpolator.html#scipy.interpolate.Akima1DInterpolator>`_), "cubic" (`cubic spline <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html#scipy.interpolate.CubicSpline>`_), "pchip" (`Piecewise Cubic Hermite Interpolating Polynomial <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html#scipy.interpolate.PchipInterpolator>`_), or "auto".

    window: str, optional
        The name of the window function. See `scipy.signal.windows <https://docs.scipy.org/doc/scipy/reference/signal.windows.html>`_ for window functions with only two parameters (``M`` and ``sym``). For example, "boxcar", "cosine", and "triang" are valid values for this parameter. All of these window functions can also be tested by using the value "auto".

    num_points: int, optional
        The number of points to take into account while smoothing any given point.

    polynomial_order: int, optional
        The order of the polynomial used when smoothing (Savitzky-Golay and Whittaker-Henderson only).

    num_iterations: int, optional
        The number of iterations to perform while smoothing (LOWESS only).

    center: float, optional
        The center of the window on a logarithmic frequency scale (e.g., centered between 1 Hz and 1000 Hz would mean a value of (log(1000) - log(1)) / 2 = 1.5).

    width: float, optional
        The width of the window on a logarithmic frequency scale. For example, 3.0 to cover the range from 1 Hz (log(1) = 0) to 1000 Hz (log(1000) = 3) when centered at log f = 1.5.

    weights: Optional[NDArray[float64]], optional
        If the desired weights can not be implemented using the ``window``, ``center``, and ``width`` parameters, then this parameter can be used to provide custom weights.

    admittance: bool, optional
        Use the admittance representation of the data instead of the impedance representation.

    num_procs: int, optional
        The maximum number of parallel processes to use when smoothing algorithm, interpolation spline, and/or window function are set to "auto".
        A value less than 1 results in an attempt to figure out a suitable value based on, e.g., the number of cores detected.
        Additionally, a negative value can be used to reduce the number of processes by that much (e.g., to leave one core for a GUI thread).

    Returns
    -------
    |ZHITResult|
    """
    if not isinstance(smoothing, str):
        raise TypeError(f"Expected a string instead of {smoothing=}")

    if not isinstance(interpolation, str):
        raise TypeError(f"Expected a string instead of {interpolation=}")

    if not _is_integer(num_points):
        raise TypeError(f"Expected an integer instead of {num_points=}")
    elif num_points < 1:
        raise ValueError(f"Expected {num_points=} > 0")

    if not _is_integer(polynomial_order):
        raise TypeError(f"Expected an integer instead of {polynomial_order=}")

    if not _is_integer(num_iterations):
        raise TypeError(f"Expected an integer instead of {num_iterations=}")
    elif num_iterations < 1:
        raise ValueError(f"Expected {num_iterations=} > 0")

    if not isinstance(window, str):
        raise TypeError(f"Expected a string instead of {window=}")

    if not _is_floating(center):
        raise TypeError(f"Expected a float instead of {center=}")

    if not _is_floating(width):
        raise TypeError(f"Expected a float instead of {width=}")
    elif width <= 0.0:
        raise ValueError(f"Expected {width=} > 0.0")

    if not (weights is None or _is_floating_array(weights)):
        raise TypeError(f"Expected an array of floats or None instead of {weights=}")

    if not _is_boolean(admittance):
        raise TypeError(f"Expected a boolean instead of {admittance=}")

    if not _is_integer(num_procs):
        raise TypeError(f"Expected an integer instead of {num_procs=}")

    if num_procs < 1:
        num_procs = max((get_default_num_procs() - abs(num_procs), 1))

    if smoothing in ("auto", "savgol", "whithend"):
        if num_points < 2:
            raise ValueError(f"Expected {num_points=} > 1")
        if not (0 < polynomial_order < num_points):
            raise ValueError(f"Expected 0 < {polynomial_order=} < {num_points=}")

    if len(_WINDOW_FUNCTIONS) == 0:
        _initialize_window_functions()

    f: Frequencies = data.get_frequencies()
    if not (len(f) > 0):
        raise ValueError(
            f"There are no unmasked data points in the '{data.get_label()}' data set parsed from '{data.get_path()}'"
        )

    log_f: NDArray[float64] = log(f)
    ln_omega: NDArray[float64] = ln(2 * pi * f)

    X_exp: NDArray[complex128] = data.get_impedances() ** (-1 if admittance else 1)
    offset: float = 0.0
    # TODO: Apply also when admittance==False?
    if admittance and min(X_exp.real) < 0.0:
        offset = abs(min(X_exp.real))
        X_exp += offset

    ln_modulus_exp: NDArray[float64] = ln(abs(X_exp))
    phase_exp: Phases = angle(X_exp)

    num_smoothing: int = 5 if smoothing == "auto" else 1
    num_interpolation: int = 4 if interpolation == "auto" else 1
    num_window: int = len(_WINDOW_FUNCTIONS) if window == "auto" else 1

    num_steps: int = 0
    # Generate weights
    num_steps += num_window
    # Smoothing
    num_steps += num_smoothing
    # Interpolation
    num_steps += num_smoothing * num_interpolation
    # Reconstruction
    num_steps += num_smoothing * num_interpolation
    # Offset adjustment
    num_steps += num_window * (num_smoothing * num_interpolation)

    prog: Progress
    with Progress("Performing Z-HIT", total=num_steps + 1) as prog:
        window_options: Dict[str, NDArray[float64]] = _generate_window_options(
            weights,
            log_f,
            window,
            center,
            width,
            prog,
        )

        smoothing_options: Dict[str, Phases] = _generate_smoothing_options(
            smoothing,
            num_points,
            polynomial_order,
            num_iterations,
            ln_omega,
            phase_exp,
            prog,
        )

        interpolation_options: Dict[str, Dict[str, Callable]]
        simulated_phase: Dict[str, Dict[str, Phases]]
        interpolation_options, simulated_phase = _generate_interpolation_options(
            interpolation,
            ln_omega,
            smoothing_options,
            prog,
        )

        reconstructions: List[Tuple[NDArray[float64], Phases, str, str]]
        reconstructions = _reconstruct_modulus_data(
            interpolation_options,
            simulated_phase,
            ln_omega,
            admittance,
            num_procs,
            prog,
        )

        results: List[Tuple[float, NDArray[complex128], str, str, str]]
        results = _adjust_modulus_offset(
            reconstructions,
            window_options,
            ln_modulus_exp,
            X_exp,
            admittance,
            num_procs,
            prog,
        )

    pseudo_chisqr: float
    X_fit: NDArray[complex128]
    pseudo_chisqr, X_fit, smoothing, interpolation, window = results[0]
    X_fit -= offset

    Z_fit: ComplexImpedances = X_fit ** (-1 if admittance else 1)
    residuals: ComplexResiduals = _calculate_residuals(
        Z_exp=data.get_impedances(),
        Z_fit=Z_fit,
    )

    return ZHITResult(
        frequencies=f,
        impedances=Z_fit,
        residuals=residuals,
        pseudo_chisqr=pseudo_chisqr,
        smoothing=smoothing,
        interpolation=interpolation,
        window=window,
    )
