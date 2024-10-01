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

from abc import (
    ABC,
    abstractmethod,
)
from dataclasses import dataclass
from numpy import (
    angle,
    array,
    int64,
    zeros,
)
from .peak_analysis import (
    DRTPeaks,
    DRTPeak,
    _analyze_peaks,
)
from pyimpspec.typing import (
    ComplexImpedances,
    ComplexResiduals,
    Frequencies,
    Gammas,
    Impedances,
    Indices,
    Phases,
    Residuals,
    TimeConstants,
)
from pyimpspec.typing.helpers import (
    List,
    NDArray,
    Optional,
    Tuple,
    Union,
    float64,
    _is_boolean,
    _is_floating,
    _is_floating_array,
)


@dataclass(frozen=True)
class DRTResult(ABC):
    """
    The base class for objects representing the results of calculating the distribution of relaxation times in a data set.
    Each method implements a subclass for its results based on this class.

    Parameters
    ----------
    time_constants: |TimeConstants|

    frequencies: |Frequencies|

    impedances: |ComplexImpedances|

    residuals: |ComplexResiduals|

    pseudo_chisqr: float
    """

    time_constants: TimeConstants
    frequencies: Frequencies
    impedances: ComplexImpedances
    residuals: ComplexResiduals
    pseudo_chisqr: float

    @abstractmethod
    def get_label(self) -> str:
        """
        The label includes information such as the method that was used.

        Returns
        -------
        str
        """
        pass

    def get_frequencies(self) -> Frequencies:
        """
        Get the frequencies (in hertz) of the data set.

        Returns
        -------
        |Frequencies|
        """
        return self.frequencies

    def get_impedances(self) -> ComplexImpedances:
        """
        Get the complex impedance of the model.

        Returns
        -------
        |ComplexImpedances|
        """
        return self.impedances

    def get_time_constants(self) -> TimeConstants:
        """
        Get the time constants.

        Returns
        -------
        |TimeConstants|
        """
        return self.time_constants

    @abstractmethod
    def to_peaks_dataframe(
        self,
        threshold: float = 0.0,
        columns: Optional[List[str]] = None,
    ) -> "DataFrame":  # noqa: F821
        """
        Get the peaks as a `pandas.DataFrame`_ object that can be used to generate, e.g., a Markdown table.

        Parameters
        ----------
        threshold: float, optional
            The minimum peak height threshold (relative to the height of the tallest peak) for a peak to be included.

        columns: Optional[List[str]], optional
            The labels for the column headers.

        Returns
        -------
        `pandas.DataFrame`_
        """
        pass

    def _get_peak_indices(self, threshold: float, gammas: Gammas) -> Indices:
        from scipy.signal import find_peaks

        if not _is_floating(threshold):
            raise TypeError(f"Expected a float instead of {threshold=}")
        elif not (0.0 <= threshold <= 1.0):
            raise ValueError(f"Expected a value in the range [0.0, 1.0] instead of {threshold=}")

        padded_gammas: Gammas = zeros(gammas.size + 2, dtype=gammas.dtype)
        padded_gammas[1:-1] = gammas

        indices: Indices = find_peaks(padded_gammas)[0]
        if not indices.any():
            return array([], dtype=int64)

        max_g: float = max(gammas)
        if max_g == 0.0:
            return array([], dtype=int64)

        indices -= 1  # Because of the padding

        return array(
            list(
                filter(
                    lambda _: gammas[_] / max_g > threshold and gammas[_] > 0.0, indices
                )
            ),
            dtype=int64,
        )

    def get_nyquist_data(self) -> Tuple[Impedances, Impedances]:
        """
        Get the data necessary to plot this result as a Nyquist plot: the real and the negative imaginary parts of the impedances.

        Returns
        -------
        Tuple[|Impedances|, |Impedances|]
        """
        return (
            self.impedances.real,  # type: ignore
            -self.impedances.imag,  # type: ignore
        )

    def get_bode_data(self) -> Tuple[Frequencies, Impedances, Phases]:
        """
        Get the data necessary to plot this result as a Bode plot: the frequencies, the absolute magnitudes of the impedances, and the negative phase angles/shifts of the impedances in degrees.

        Returns
        -------
        Tuple[|Frequencies|, |Impedances|, |Phases|]
        """
        return (
            self.frequencies,  # type: ignore
            abs(self.impedances),  # type: ignore
            -angle(self.impedances, deg=True),  # type: ignore
        )

    def get_residuals_data(self) -> Tuple[Frequencies, Residuals, Residuals]:
        """
        Get the data necessary to plot the relative residuals for this result: the frequencies, the relative residuals for the real parts of the impedances in percents, and the relative residuals for the imaginary parts of the impedances in percents.

        Returns
        -------
        Tuple[|Frequencies|, |Residuals|, |Residuals|]
        """
        return (
            self.frequencies,  # type: ignore
            self.residuals.real * 100,  # type: ignore
            self.residuals.imag * 100,  # type: ignore
        )

    @abstractmethod
    def to_statistics_dataframe(
        self,
    ) -> "DataFrame":  # noqa: F821
        """
        Get the statistics related to the DRT as a `pandas.DataFrame`_ object.

        Returns
        -------
        `pandas.DataFrame`_
        """
        pass

    def analyze_peaks(
        self,
        num_peaks: int = 0,
        peak_positions: Optional[Union[List[float], NDArray[float64]]] = None,
        disallow_skew: bool = False,
    ) -> DRTPeaks:
        """
        Analyze the peaks present in a distribution of relaxation times using skew normal distributions.

        Parameters
        ----------
        num_peaks: int, optional
            If greater than zero, then analyze only that number of peaks (sorted from highest to lowest gamma values).

        peak_positions: Optional[Union[List[float], NDArray[float64]]], optional
            Analyze only the peaks at the provided positions.

        disallow_skew: bool, optional
            If true, then normal distributions are used instead of skew normal distributions.

        Returns
        -------
        |DRTPeaks|
        """
        args = self.get_drt_data()
        if not isinstance(args, tuple):
            raise TypeError(f"Expected a tuple instead of {args=}")
        elif len(args) != 2:
            raise ValueError(f"Expected a tuple with two values instead of {args=}")
        elif not _is_floating_array(args[0]):
            raise TypeError(f"Expected an array of floats instead of {args[0]=}")
        elif not _is_floating_array(args[1]):
            raise TypeError(f"Expected an array of floats instead of {args[1]=}")
        elif not _is_boolean(disallow_skew):
            raise TypeError(f"Expected a boolean instead of {disallow_skew=}")

        peaks: DRTPeaks = _analyze_peaks(
            *args,
            num_peaks=num_peaks,
            peak_positions=peak_positions,
            disallow_skew=disallow_skew,
        )
        object.__setattr__(self, "_peak_analysis", peaks)

        return peaks
