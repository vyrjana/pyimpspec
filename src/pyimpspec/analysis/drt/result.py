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

from abc import (
    ABC,
    abstractmethod,
)
from dataclasses import dataclass
from typing import (
    List,
    Optional,
    Tuple,
)
from numpy import (
    angle,
    array,
    floating,
    int64,
    issubdtype,
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
        Get the peaks as a |DataFrame| object that can be used to generate, e.g., a Markdown table.

        Parameters
        ----------
        threshold: float, optional
            The minimum peak height threshold (relative to the height of the tallest peak) for a peak to be included.

        columns: Optional[List[str]], optional
            The labels for the column headers.

        Returns
        -------
        |DataFrame|
        """
        pass

    def _get_peak_indices(self, threshold: float, gamma: Gammas) -> Indices:
        from scipy.signal import find_peaks

        assert issubdtype(type(threshold), floating), threshold
        assert 0.0 <= threshold <= 1.0, threshold
        indices: Indices
        indices, _ = find_peaks(gamma)
        if not indices.any():
            return array([], dtype=int64)
        max_g: float = max(gamma)
        if max_g == 0.0:
            return array([], dtype=int64)
        return array(
            list(
                filter(
                    lambda _: gamma[_] / max_g > threshold and gamma[_] > 0.0, indices
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
        Get the statistics related to the DRT as a |DataFrame| object.

        Returns
        -------
        |DataFrame|
        """
        pass
