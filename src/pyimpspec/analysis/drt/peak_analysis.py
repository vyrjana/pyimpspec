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

# This module implements analysis of peaks in DRT spectra by fitting skew
# normal distributions.See 10.3390/batteries5030053 for more information.

# TODO: Implement tests for this feature

from pyimpspec.progress import Progress
from dataclasses import dataclass
from scipy.signal import find_peaks
from numpy import (
    allclose,
    argmin,
    argwhere,
    copy,
    exp,
    float64,
    isclose,
    log10 as log,
    logspace,
    sign,
    zeros,
)
from numpy.typing import NDArray
from pyimpspec.typing import (
    Gamma,
    Gammas,
    Indices,
    TimeConstants,
)
from pyimpspec.typing.helpers import (
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    _is_boolean,
    _is_floating_array,
    _is_floating_list,
    _is_integer,
    _is_integer_array,
    _is_integer_list,
)


def _skew_normal(x, h, p, a, s):
    dp = x - p
    den = 2 * s**2
    num = (dp * (1 + a * sign(dp)))**2

    return h * exp(-num/den)


def _function(
    x,
    parameters: "Parameters",  # noqa: F821
) -> NDArray[float64]:
    params = parameters.valuesdict()
    assert len(params) % 2 == 0, (len(params), params.keys())

    gammas: NDArray[float64] = zeros(len(x), dtype=float64)

    i = 0
    while True:
        try:
            h = params[f"h_{i}"]
        except KeyError:
            break
        p = params[f"p_{i}"]
        alpha = params[f"alpha_{i}"]
        sigma = params[f"sigma_{i}"]

        gammas += _skew_normal(x, h, p, alpha, sigma)
        i += 1

    return gammas


def _residual(
    parameters: "Parameters",  # noqa: F821
    x: NDArray[float64],
    y: NDArray[float64],
):
    return _function(x, parameters) - y


@dataclass(frozen=True)
class DRTPeak:
    """
    An object that represents a peak in a distribution of relaxation times.
    The peak is modeled using a (skew) normal distribution.

    Parameters
    ----------
    position: float64
        The relative position of the (skew) normal distribution.

    height: float64
        The relative height of the (skew) normal distribution.

    alpha: float64
        The skew of the (skew) normal distribution.

    sigma: float64
        The standard deviation of the (skew) normal distribution.

    x_offset: float64
        The offset used to translate time constants to the relative x-axis.

    x_scale: float64
        The scaling factor used to translate time constants to the relative x-axis.

    y_offset: float64
        The offset used to translate the relative y-axis values to gamma (ohm).

    y_scale: float64
        The scaling factor used to translate the relative y-axis values to gamma (ohm).
    """
    position: float64  # Relative x-axis
    height: float64  # Relative y-axis
    alpha: float64  # Skew
    sigma: float64  # Standard deviation
    x_offset: float64  # For translating from time constant to relative x-axis
    x_scale: float64  # For translating from time constant to relative x-axis
    y_offset: float64  # For translating relative y-axis to gamma (ohm)
    y_scale: float64  # For translating relative y-axis to gamma (ohm)

    def get_gammas(
        self,
        time_constants: TimeConstants,
    ) -> Gammas:
        """
        Calculate the gamma values (ohm) corresponding to the given time constants.

        Parameters
        ----------
        time_constants: |TimeConstants|

        Returns
        -------
        |Gammas|
        """
        x: NDArray[float64] = (log(time_constants) - self.x_offset) / self.x_scale

        return (_skew_normal(
            x=x,
            h=self.height,
            p=self.position,
            a=self.alpha,
            s=self.sigma,
        ) * self.y_scale) + self.y_offset

    def _get_gamma(self, log_tau: float64) -> float64:
        x: float64 = (log_tau - self.x_offset) / self.x_scale

        return (_skew_normal(
            x=x,
            h=self.height,
            p=self.position,
            a=self.alpha,
            s=self.sigma,
        ) * self.y_scale) + self.y_offset

    def get_area(
        self,
        time_constants: TimeConstants,
    ) -> float64:
        """
        Calculate the area (ohms) of this peak for a given set of time constants.

        Parameters
        ----------
        time_constants: |TimeConstants|

        Returns
        -------
        float64
        """
        from scipy.integrate import quad

        return quad(
            func=self._get_gamma,
            a=log(min(time_constants)),
            b=log(max(time_constants)),
        )[0] / log(exp(1))



@dataclass(frozen=True)
class DRTPeaks:
    """
    An object that represents a collection of peaks in a distribution of relaxation times.
    The peaks are modeled using a (skew) normal distribution.

    Parameters
    ----------
    time_constants: |TimeConstants|
        The time constants of the DRT.

    peaks: List[|DRTPeak|
        A list of the (skew) normal distributions fitted to the DRT peaks.

    suffix: str
        The suffix used in the labels when plotting the results.
    """
    time_constants: TimeConstants
    peaks: List[DRTPeak]
    suffix: str

    def __iter__(self):
        return iter(self.peaks)

    def get_num_peaks(self) -> int:
        """
        Get the number of peaks.

        Returns
        -------
        int
        """
        return len(self.peaks)

    def get_time_constants(
        self,
        num_per_decade: int = 100,
    ) -> TimeConstants:
        """
        Get either the original time constants (``num_per_decade < 1``) or an interpolated set of time constants.

        Parameters
        ----------
        num_per_decade: int, optional
            The number of points per decade to use when generating the time constants.

        Returns
        -------
        |TimeConstants|
        """
        if not _is_integer(num_per_decade):
            raise TypeError(f"Expected an integer instead of {num_per_decade=}")

        if num_per_decade < 1:
            return self.time_constants

        log_tau_min = log(min(self.time_constants))
        log_tau_max = log(max(self.time_constants))

        return logspace(
            log_tau_min,
            log_tau_max,
            num=int(round(log_tau_max - log_tau_min) * num_per_decade) + 1,
        )

    def get_gammas(
        self,
        peak_indices: Optional[List[int]] = None,
        num_per_decade: int = 100,
    ) -> Gammas:
        """
        Calculate the gamma values (ohm) for one or more peaks.

        Parameters
        ----------
        peak_indices: Optional[List[int]], optional
            If indices are specified, then only those peaks will be used.

        num_per_decade: int, optional
            The number of points per decade to use when generating the gamma values.

        Returns
        -------
        |Gammas|
        """
        num_peaks: int = self.get_num_peaks()
        if peak_indices is None:
            pass
        elif not (_is_integer_list(peak_indices) or _is_integer_array(peak_indices)):
            raise TypeError(f"Expected an array of integers instead of {peak_indices=}")
        elif not all(map(lambda i: 0 <= i < num_peaks, peak_indices)):
            raise ValueError(f"Expected 0 <= {peak_indices=} < {num_peaks}")

        time_constants: TimeConstants = self.get_time_constants(
            num_per_decade=num_per_decade,
        )

        gammas: Gammas = zeros(time_constants.shape, dtype=Gamma)

        i: int
        peak: DRTPeak
        for i, peak in enumerate(self.peaks):
            if peak_indices and i not in peak_indices:
                continue

            gammas += peak.get_gammas(time_constants=time_constants)

        return gammas

    def get_peak_area(
        self,
        index: int,
    ) -> float64:
        """
        Calculate the area of a peak.

        Parameters
        ----------
        index: int
            The index (zero-based) of the peak for which to calculate the area.

        Returns
        -------
        float64
        """
        from scipy.integrate import quad

        num_peaks: int = self.get_num_peaks()
        if not _is_integer(index):
            raise TypeError(f"Expected an integer instead of {index=}")
        elif not (0 <= index < num_peaks):
            raise ValueError(f"Expected 0 <= {index=} < {num_peaks}")

        peak: DRTPeak = self.peaks[index]

        return quad(
            func=lambda x: peak.get_gammas(10**x),
            a=log(min(self.time_constants)),
            b=log(max(self.time_constants)),
        )[0] / log(exp(1))

    def to_peaks_dataframe(
        self,
        peak_indices: Optional[List[int]] = None,
    ) -> "DataFrame":  # noqa: F821
        """
        Generate a `pandas.DataFrame`_ object containing information about the peaks.

        Parameters
        ----------
        peak_indices: Optional[List[int]], optional
            If indices are specified, then only those peaks will be used.

        Returns
        -------
        `pandas.DataFrame`_
        """
        from pandas import DataFrame

        num_peaks: int = self.get_num_peaks()
        if peak_indices is None:
            pass
        elif not (_is_integer_list(peak_indices) or _is_integer_array(peak_indices)):
            raise TypeError(f"Expected an array of integers instead of {peak_indices=}")
        elif not all(map(lambda i: 0 <= i < num_peaks, peak_indices)):
            raise ValueError(f"Expected 0 <= {peak_indices=} < {num_peaks}")

        positions: List[float64] = []
        heights: List[float64] = []
        areas: List[float64] = []

        i: int
        peak: DRTPeak
        for i, peak in enumerate(self.peaks):
            if peak_indices and i not in peak_indices:
                continue

            positions.append(10**((peak.position * peak.x_scale) + peak.x_offset))
            heights.append((peak.height * peak.y_scale) + peak.y_offset)
            areas.append(self.get_peak_area(i))

        suffix: str = ""
        if self.suffix != "":
            suffix = f", {self.suffix}"

        return DataFrame.from_dict({
            "tau" + suffix + " (s)": positions,
            "gamma" + suffix + " (ohm)": heights,
            "R_peak" + suffix + " (ohm)": areas,
        })


def _generate_parameters(
    peaks: List[Tuple[float64, float64]],
    disallow_skew: bool,
) -> Tuple["Parameters", int]:  # noqa: F821
    from lmfit.parameter import Parameters

    if len(peaks) < 1:
        raise ValueError(f"Expected to have at least one peak to analyze!")

    parameters: Parameters = Parameters()
    num_variables: int = 0

    i: int
    x: float64
    y: float64
    for i, (x, y) in enumerate(peaks):
        assert 0.0 <= x <= 1.0, x
        parameters.add(
            name=f"h_{i}",
            value=y,
            min=0.0,
            max=2.0,
        )

        assert 0.0 <= x <= 1.0
        kw = dict(
            name=f"p_{i}",
            value=x,
            min=x * 0.99,
            max=x * 1.01,
        )
        
        if isclose(kw["min"], 0.0):
            kw["min"] = -1e-10

        if isclose(kw["max"], 1.0):
            kw["max"] = 1.0 + 1e-10

        parameters.add(**kw)

        parameters.add(
            name=f"alpha_{i}",
            value=0.0,
            min=-1.0,
            max=1.0,
            vary=not disallow_skew,
        )

        parameters.add(
            name=f"sigma_{i}",
            value=0.05,
            min=1e-10,
            max=1e2,
        )

        num_variables += 3 if disallow_skew else 4

    return (parameters, num_variables)


def _analyze_peaks(
    time_constants: TimeConstants,
    gammas: Gammas,
    num_peaks: int,
    peak_positions: Optional[Union[List[float], NDArray[float64]]],
    disallow_skew: bool = False,
    suffix: str = "",
) -> DRTPeaks:
    from lmfit.minimizer import minimize, MinimizerResult
    from lmfit.parameter import Parameters

    if not _is_floating_array(time_constants):
        raise TypeError(f"Expected an array of floats instead of {time_constants=}")
    elif not _is_floating_array(gammas):
        raise TypeError(f"Expected an array of floats instead of {gammas=}")
    elif not _is_integer(num_peaks):
        raise TypeError(f"Expected an integer instead of {num_peaks=}")
    elif not (
        peak_positions is None
        or _is_floating_array(peak_positions)
        or _is_floating_list(peak_positions)
    ):
        raise TypeError(f"Expected either None or a list/array of floats instead of {peak_positions=}")
    elif not _is_boolean(disallow_skew):
        raise TypeError(f"Expected a boolean instead of {disallow_skew=}")
    elif not isinstance(suffix, str):
        raise TypeError(f"Expected a string instead of {suffix=}")

    if allclose(gammas, 0.0):
        return DRTPeaks(
            time_constants=time_constants,
            peaks=[],
            suffix=suffix,
        )

    prog: Progress
    with Progress(
        "Scaling data",
        total=4,
    ) as prog:
        x: NDArray[float64] = log(time_constants)
        x_offset: float64 = min(x)
        x -= x_offset
        x_scale: float64 = max(x)
        x /= x_scale
        assert isclose(min(x), 0.0)
        assert isclose(max(x), 1.0)
        prog.increment()

        y: NDArray[float64] = copy(gammas)
        indices: Indices = argwhere(y < 0.0).flatten()
        y[indices] = 0.0
        y_offset: float64 = min(y)
        y = y - y_offset
        y_scale: float64 = max(y)
        y /= y_scale
        assert isclose(min(y), 0.0), min(y)
        assert isclose(max(y), 1.0), max(y)
        prog.increment()

        prog.set_message("Detecting peaks")
        peaks: List[Tuple[float64, float64]] = []
        i: int

        if peak_positions is not None and len(peak_positions) > 0:
            pos: float
            for pos in peak_positions:
                i = argmin(abs(time_constants - pos))
                peaks.append((x[i], y[i]))
        else:
            y_ext: NDArray[float64] = zeros(len(y) + 2, dtype=float64)
            y_ext[1:-1] += y

            for i in find_peaks(y_ext)[0]:
                i -= 1
                peaks.append((x[i], y[i]))

            if num_peaks > 0:
                peaks.sort(key=lambda t: t[1], reverse=True)
                peaks = peaks[:num_peaks]

            peaks.sort(key=lambda t: t[0])

        parameters: Parameters
        num_variables: int
        parameters, num_variables = _generate_parameters(
            peaks,
            disallow_skew=disallow_skew,
        )
        if num_peaks < 1 and peak_positions is None:
            while len(peaks) > 1 and num_variables > x.size:
                peaks.sort(key=lambda t: t[1], reverse=True)
                peaks.pop()
                peaks.sort(key=lambda t: t[0])
                parameters, num_variables = _generate_parameters(
                    peaks,
                    disallow_skew=disallow_skew,
                )

        prog.increment()

        prog.set_message(f"Fitting {len(peaks)} peaks")
        fit: MinimizerResult = minimize(
            _residual,
            parameters,
            method="leastsq",
            args=(x, y),
        )
        params: Dict[str, float64] = fit.params.valuesdict()

        drt_peaks: List[DRTPeak] = []

        for i in range(0, len(peaks)):
            drt_peaks.append(DRTPeak(
                position=params[f"p_{i}"],
                height=params[f"h_{i}"],
                alpha=params[f"alpha_{i}"],
                sigma=params[f"sigma_{i}"],
                x_offset=x_offset,
                x_scale=x_scale,
                y_offset=y_offset,
                y_scale=y_scale,
            ))

    return DRTPeaks(
        time_constants=time_constants,
        peaks=drt_peaks,
        suffix=suffix,
    )
