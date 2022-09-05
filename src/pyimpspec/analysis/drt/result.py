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

from dataclasses import dataclass
from typing import (
    Dict,
    Optional,
    Tuple,
)
from numpy import (
    angle,
    array,
    floating,
    issubdtype,
    ndarray,
    sqrt,
    sum as array_sum,
)
from pandas import DataFrame
from scipy.signal import find_peaks


class DRTError(Exception):
    pass


@dataclass(frozen=True)
class DRTResult:
    """
    An object representing the results of calculating the distribution of relaxation times in a  data set.

    Parameters
    ----------
    label: str
        Includes information such as the method used.

    tau: ndarray
        The time constants.

    gamma: ndarray
        The gamma values.
        These values are the real gamma values when the BHT method has been used.

    frequency: ndarray
        The frequencies of the impedance spectrum.

    impedance: ndarray
        The impedance produced by the model.

    real_residual: ndarray
        The residuals of the real parts of the model and the data set.

    imaginary_residual: ndarray
        The residuals of the imaginary parts of the model and the data set.

    mean_gamma: ndarray
        The mean gamma values of the Bayesian credible intervals.

    lower_bound: ndarray
        The lower bound gamma values of the Bayesian credible intervals.

    upper_bound: ndarray
        The upper bound gamma values of the Bayesian credible intervals.

    imaginary_gamma: ndarray
        The imaginary gamma values produced by the BHT method.

    scores: Dict[str, complex]
        The scores calculated by the BHT method.

    chisqr: float
        The chi-squared value of the modeled impedance.

    lambda_value: float
        The lambda value that was ultimately used.
    """

    label: str
    tau: ndarray
    gamma: ndarray
    frequency: ndarray
    impedance: ndarray
    real_residual: ndarray
    imaginary_residual: ndarray
    mean_gamma: ndarray
    lower_bound: ndarray
    upper_bound: ndarray
    imaginary_gamma: ndarray
    scores: Dict[str, complex]
    chisqr: float
    lambda_value: float

    def get_label(self) -> str:
        """
        The label includes information such as the method that was used.

        Returns
        -------
        str
        """
        return self.label

    def get_frequency(self) -> ndarray:
        """
        Get the frequencies (in hertz) of the data set.

        Returns
        -------
        ndarray
        """
        return self.frequency

    def get_impedance(self) -> ndarray:
        """
        Get the complex impedance of the model.

        Returns
        -------
        ndarray
        """
        return self.impedance

    def get_tau(self) -> ndarray:
        """
        Get the time constants.

        Returns
        -------
        ndarray
        """
        return self.tau

    def get_gamma(self, imaginary: bool = False) -> ndarray:
        """
        Get the gamma values.

        Parameters
        ----------
        imaginary: bool = False
            Get the imaginary gamma (non-empty only when using the BHT method).

        Returns
        -------
        ndarray
        """
        if imaginary is True:
            return self.imaginary_gamma
        return self.gamma

    def to_dataframe(
        self,
        threshold: float = 0.0,
        imaginary: bool = False,
        latex_labels: bool = False,
        include_frequency: bool = False,
    ) -> DataFrame:
        """
        Get the peaks as a pandas.DataFrame object that can be used to generate, e.g., a Markdown table.

        Parameters
        ----------
        threshold: float = 0.0
            The threshold for the peaks (0.0 to 1.0 relative to the highest peak).

        imaginary: bool = False
            Use the imaginary gamma (non-empty only when using the BHT method).

        latex_labels: bool = False
            Whether or not to use LaTeX macros in the labels.

        include_frequency: bool = False
            Whether or not to also include a column with the frequencies corresponding to the time constants.

        Returns
        -------
        DataFrame
        """
        tau: ndarray
        gamma: ndarray
        tau, gamma = self.get_peaks(threshold=threshold, imaginary=imaginary)
        f: ndarray = 1 / tau
        dictionary: dict = {}
        dictionary["tau (s)" if not latex_labels else r"$\tau$ (s)"] = tau
        if include_frequency is True:
            dictionary["f (Hz)" if not latex_labels else r"$f$ (Hz)"] = f
        dictionary[
            "gamma (ohms)" if not latex_labels else r"$\gamma\ (\Omega)$"
        ] = gamma
        return DataFrame.from_dict(dictionary)

    def get_peaks(
        self,
        threshold: float = 0.0,
        imaginary: bool = False,
    ) -> Tuple[ndarray, ndarray]:
        """
        Get the time constants (in seconds) and gamma (in ohms) of peaks with magnitudes greater than the threshold.
        The threshold and the magnitudes are all relative to the magnitude of the highest peak.

        Parameters
        ----------
        threshold: float = 0.0
            The threshold for the relative magnitude (0.0 to 1.0).

        imaginary: bool = False
            Use the imaginary gamma (non-empty only when using the BHT method).

        Returns
        -------
        Tuple[ndarray, ndarray]
        """
        assert (
            issubdtype(type(threshold), floating) and 0.0 <= threshold <= 1.0
        ), threshold
        assert type(imaginary) is bool, imaginary
        gamma: ndarray = self.gamma if not imaginary else self.imaginary_gamma
        assert type(gamma) is ndarray, gamma
        if not gamma.any():
            return (
                array([]),
                array([]),
            )
        indices: ndarray
        indices, _ = find_peaks(gamma)
        if not indices.any():
            return (
                array([]),
                array([]),
            )
        max_g: float = max(gamma)
        if max_g == 0.0:
            return (
                array([]),
                array([]),
            )
        indices = array(
            list(
                filter(
                    lambda _: gamma[_] / max_g > threshold and gamma[_] > 0.0, indices
                )
            )
        )
        if indices.any():
            return (
                self.tau[indices],
                gamma[indices],
            )
        return (
            array([]),
            array([]),
        )

    def get_nyquist_data(self) -> Tuple[ndarray, ndarray]:
        """
        Get the data necessary to plot this DataSet as a Nyquist plot: the real and the negative imaginary parts of the impedances.

        Returns
        -------
        Tuple[ndarray, ndarray]
        """
        return (
            self.impedance.real,
            -self.impedance.imag,
        )

    def get_bode_data(self) -> Tuple[ndarray, ndarray, ndarray]:
        """
        Get the data necessary to plot this DataSet as a Bode plot: the frequencies, the absolute magnitudes of the impedances, and the negative phase angles/shifts of the impedances in degrees.

        Returns
        -------
        Tuple[ndarray, ndarray, ndarray]
        """
        return (
            self.frequency,
            abs(self.impedance),
            -angle(self.impedance, deg=True),
        )

    def get_drt_data(self, imaginary: bool = False) -> Tuple[ndarray, ndarray]:
        """
        Get the data necessary to plot this DRTResult as a DRT plot: the time constants and the corresponding gamma values.

        Parameters
        ----------
        imaginary: bool = False
            Get the imaginary gamma (non-empty only when using the BHT method).

        Returns
        -------
        Tuple[ndarray, ndarray]
        """
        gamma: ndarray = self.gamma if not imaginary else self.imaginary_gamma
        if not gamma.any():
            return (
                array([]),
                array([]),
            )
        return (
            self.tau,
            gamma,
        )

    def get_drt_credible_intervals(self) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
        """
        Get the data necessary to plot the Bayesian credible intervals for this DRTResult: the time constants, the mean gamma values, the lower bound gamma values, and the upper bound gamma values.

        Returns
        -------
        Tuple[ndarray, ndarray, ndarray, ndarray]
        """
        if not self.mean_gamma.any():
            return (
                array([]),
                array([]),
                array([]),
                array([]),
            )
        return (
            self.tau,
            self.mean_gamma,
            self.lower_bound,
            self.upper_bound,
        )

    def get_residual_data(self) -> Tuple[ndarray, ndarray, ndarray]:
        """
        Get the data necessary to plot the relative residuals for this DRTResult: the frequencies, the relative residuals for the real parts of the impedances in percents, and the relative residuals for the imaginary parts of the impedances in percents.

        Returns
        -------
        Tuple[ndarray, ndarray, ndarray]
        """
        return (
            self.frequency,
            self.real_residual * 100,
            self.imaginary_residual * 100,
        )

    def get_scores(self) -> Dict[str, complex]:
        """
        Get the scores (BHT method) for the data set.
        The scores are represented as complex values where the real and imaginary parts have magnitudes ranging from 0.0 to 1.0.
        A consistent impedance spectrum should score high.

        Returns
        -------
        Dict[str, complex]
        """
        return self.scores

    def get_score_dataframe(self, latex_labels: bool = False) -> Optional[DataFrame]:
        """
        Get the scores (BHT) method for the data set as a pandas.DataFrame object that can be used to generate, e.g., a Markdown table.

        Parameters
        ----------
        latex_labels: bool = False
            Whether or not to use LaTeX macros in the labels.

        Returns
        -------
        Optional[DataFrame]
        """
        if not self.scores:
            return None
        return DataFrame.from_dict(
            {
                "Score": [
                    "Mean" if not latex_labels else r"$s_\mu$",
                    "Residuals, 1 sigma" if not latex_labels else r"$s_{1\sigma}$",
                    "Residuals, 2 sigma" if not latex_labels else r"$s_{2\sigma}$",
                    "Residuals, 3 sigma" if not latex_labels else r"$s_{3\sigma}$",
                    "Hellinger distance" if not latex_labels else r"$s_{\rm HD}$",
                    "Jensen-Shannon distance" if not latex_labels else r"$s_{\rm JSD}$",
                ],
                ("Real (%)" if not latex_labels else r"Real (\%)"): [
                    self.scores["mean"].real * 100,
                    self.scores["residuals_1sigma"].real * 100,
                    self.scores["residuals_2sigma"].real * 100,
                    self.scores["residuals_3sigma"].real * 100,
                    self.scores["hellinger_distance"].real * 100,
                    self.scores["jensen_shannon_distance"].real * 100,
                ],
                ("Imaginary (%)" if not latex_labels else r"Imaginary (\%)"): [
                    self.scores["mean"].imag * 100,
                    self.scores["residuals_1sigma"].imag * 100,
                    self.scores["residuals_2sigma"].imag * 100,
                    self.scores["residuals_3sigma"].imag * 100,
                    self.scores["hellinger_distance"].imag * 100,
                    self.scores["jensen_shannon_distance"].imag * 100,
                ],
            }
        )


def _calculate_chisqr(Z_exp: ndarray, Z_fit: ndarray) -> float:
    z: complex = array_sum((Z_fit - Z_exp) ** 2 / Z_exp)
    return sqrt(z.real**2 + z.imag**2)
