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
from functools import cached_property
from numpy import (
    abs,
    angle,
    array,
    float64,
    isclose,
    isnan,
    log10 as log,
    logical_and,
    mean,
    nan,
    pi,
    std,
    sum as array_sum,
)
from numpy.typing import NDArray
from pyimpspec.analysis.utility import _interpolate
from pyimpspec.analysis.kramers_kronig.utility import (
    _estimate_pct_noise,
    _format_log_F_ext_for_latex,
)
from pyimpspec.circuit.circuit import Circuit
from pyimpspec.circuit.kramers_kronig import (
    Element,
    KramersKronigRC,
    KramersKronigAdmittanceRC,
)
from pyimpspec.circuit.connections import (
    Parallel,
    Series,
)
from pyimpspec.circuit.elements import (
    Capacitor,
    Resistor,
    Inductor,
)
from pyimpspec.typing import (
    ComplexImpedances,
    ComplexResiduals,
    Frequencies,
    Impedances,
    Phases,
    Residuals,
    TimeConstants,
)
from pyimpspec.typing.helpers import (
    Dict,
    List,
    Tuple,
    Type,
    Union,
    _is_integer,
    _is_floating,
)


@dataclass(frozen=True)
class KramersKronigResult:
    """
    An object representing the results of a linear Kramers-Kronig test applied to a data set.

    Parameters
    ----------
    circuit: Circuit
        The fitted circuit.

    pseudo_chisqr: float
        The pseudo chi-squared value (|pseudo chi-squared|, eq. 14 in Boukamp, 1995).

    frequencies: |Frequencies|
        The frequencies used to perform the test.

    impedances: |ComplexImpedances|
        The impedances produced by the fitted circuit at each of the tested frequencies.

    residuals: |ComplexResiduals|
        The residuals for the real (eq. 15 in Schönleber et al., 2014) and imaginary (eq. 16 in Schönleber et al., 2014) parts of the fit.

    test: str
        The type of test (and implementation) that was performed:

        - 'complex'
        - 'real'
        - 'imaginary'
        - 'complex-inv'
        - 'real-inv'
        - 'imaginary-inv'
        - 'cnls'
    """

    circuit: Circuit
    pseudo_chisqr: float
    frequencies: Frequencies
    impedances: ComplexImpedances
    residuals: ComplexResiduals
    test: str

    def __repr__(self) -> str:
        return (
            "KramersKronigResult ("
            + ", ".join(
                (
                    f"X={'Y' if self.admittance else 'Z'}",
                    f"log_F_ext={self.log_F_ext:.3f}",
                    f"num_RC={self.num_RC}",
                    f"{hex(id(self))}",
                )
            )
            + ")"
        )

    @cached_property
    def num_RC(self) -> int:
        return self.get_num_RC()

    @cached_property
    def admittance(self) -> bool:
        return self.was_tested_on_admittance()

    @cached_property
    def label(self) -> str:
        return self.get_label()

    @cached_property
    def time_constants(self) -> TimeConstants:
        return self.get_time_constants()

    @cached_property
    def log_F_ext(self) -> float:
        return self.get_log_F_ext()

    @cached_property
    def series_resistance(self) -> float:
        return self.get_series_resistance()

    @cached_property
    def series_capacitance(self) -> float:
        return self.get_series_capacitance()

    @cached_property
    def series_inductance(self) -> float:
        return self.get_series_inductance()

    @cached_property
    def parallel_resistance(self) -> float:
        return self.get_parallel_resistance()

    @cached_property
    def parallel_capacitance(self) -> float:
        return self.get_parallel_capacitance()

    @cached_property
    def parallel_inductance(self) -> float:
        return self.get_parallel_inductance()

    def get_num_RC(self) -> int:
        Class: Type[Element] = (
            KramersKronigAdmittanceRC if self.admittance else KramersKronigRC
        )

        return len(
            [
                element
                for element in self.circuit.get_elements(recursive=True)
                if isinstance(element, Class)
            ]
        )

    def was_tested_on_admittance(self) -> bool:
        element: Element
        for element in self.circuit.get_elements(recursive=True):
            if isinstance(element, KramersKronigAdmittanceRC):
                return True

        return False

    def get_label(self) -> str:
        """
        Get the label of this result.

        Returns
        -------
        str
        """
        label: str = "Y" if self.admittance else "Z"
        cdc: str = self.circuit.to_string()

        if "C" in cdc:
            label += ", C"
            if "L" in cdc:
                label += "+L"
        elif "L" in cdc:
            label += ", L"

        label += r", $N_\tau = " + str(self.num_RC) + "$"

        log_F_ext: float = self.log_F_ext
        if log_F_ext != 0.0:
            formatted_extension: str = _format_log_F_ext_for_latex(log_F_ext)

            label += (
                r", $\tau \in "
                + r"[\frac{1}{f_{\rm max} \times F_{\rm ext}},"
                + r"\frac{F_{\rm ext}}{f_{\rm min}}],"
                + r"\log{F_{\rm ext}} = "
                + formatted_extension
                + r"$"
            )
        else:
            label += r", $\tau \in [\frac{1}{f_{\rm max}}, \frac{1}{f_{\rm min}}]$"

        return label

    def get_frequencies(self, num_per_decade: int = -1) -> Frequencies:
        """
        Get the frequencies in the tested frequency range.

        Parameters
        ----------
        num_per_decade: int, optional
            The number of points per decade.
            A positive value results in frequencies being calculated within the original frequency range.
            Otherwise, only the original frequencies are used.

        Returns
        -------
        |Frequencies|
        """
        if not _is_integer(num_per_decade):
            raise TypeError(f"Expected an integer instead of {num_per_decade=}")

        if num_per_decade > 0:
            return _interpolate(self.frequencies, num_per_decade)

        return self.frequencies

    def get_impedances(self, num_per_decade: int = -1) -> ComplexImpedances:
        """
        Get the fitted circuit's impedance response within the tested frequency range.

        Parameters
        ----------
        num_per_decade: int, optional
            The number of points per decade.
            A positive value results in data points being calculated using the fitted circuit within the original frequency range.
            Otherwise, only the original frequencies are used.

        Returns
        -------
        |ComplexImpedances|
        """
        if not _is_integer(num_per_decade):
            raise TypeError(f"Expected an integer instead of {num_per_decade=}")

        if num_per_decade > 0:
            return self.circuit.get_impedances(self.get_frequencies(num_per_decade))

        return self.impedances

    def get_nyquist_data(
        self,
        num_per_decade: int = -1,
    ) -> Tuple[Impedances, Impedances]:
        """
        Get the data necessary to plot this KramersKronigResult as a Nyquist plot: the real and the negative imaginary parts of the impedances.

        Parameters
        ----------
        num_per_decade: int, optional
            The number of points per decade.
            A positive value results in data points being calculated using the fitted circuit within the original frequency range.
            Otherwise, only the original frequencies are used.

        Returns
        -------
        Tuple[|Impedances|, |Impedances|]
        """
        if not _is_integer(num_per_decade):
            raise TypeError(f"Expected an integer instead of {num_per_decade=}")

        if num_per_decade > 0:
            Z: ComplexImpedances = self.get_impedances(num_per_decade)
            return (
                Z.real,
                -Z.imag,
            )

        return (
            self.impedances.real,
            -self.impedances.imag,
        )

    def get_bode_data(
        self,
        num_per_decade: int = -1,
    ) -> Tuple[Frequencies, Impedances, Phases]:
        """
        Get the data necessary to plot this KramersKronigResult as a Bode plot: the frequencies, the absolute magnitudes of the impedances, and the negative phase angles/shifts of the impedances in degrees.

        Parameters
        ----------
        num_per_decade: int, optional
            The number of points per decade.
            A positive value results in data points being calculated using the fitted circuit within the original frequency range.
            Otherwise, only the original frequencies are used.

        Returns
        -------
        Tuple[|Frequencies|, |Impedances|, |Phases|]
        """
        if not _is_integer(num_per_decade):
            raise TypeError(f"Expected an integer instead of {num_per_decade=}")

        if num_per_decade > 0:
            f: Frequencies = self.get_frequencies(num_per_decade)
            Z: ComplexImpedances = self.circuit.get_impedances(f)
            return (
                f,
                abs(Z),
                -angle(Z, deg=True),
            )

        return (
            self.frequencies,
            abs(self.impedances),
            -angle(self.impedances, deg=True),
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

    def get_time_constants(self) -> TimeConstants:
        """
        Get the time constants that were used during fitting.

        Returns
        -------
        |TimeConstants|
        """
        time_constants: List[float] = []

        element: Element
        for element in self.circuit.get_elements(recursive=True):
            if not (
                isinstance(element, KramersKronigRC)
                or isinstance(element, KramersKronigAdmittanceRC)
            ):
                continue

            time_constants.append(element.get_value(key="tau"))

        return array(sorted(time_constants))

    def get_log_F_ext(self) -> float:
        """
        Get the value of |log F_ext|, which affects the range of time constants.

        Returns
        -------
        float
        """
        time_constants: TimeConstants = self.time_constants
        if len(time_constants) < 2:
            return 0.0

        minimum_time_constant: float64 = min(time_constants)
        maximum_time_constant: float64 = max(time_constants)

        f: Frequencies = self.get_frequencies()
        low_end_extension: float64 = log(1 / (2 * pi * max(f))) - log(
            minimum_time_constant
        )
        high_end_extension: float64 = log(maximum_time_constant) - log(
            1 / (2 * pi * min(f))
        )

        if not isclose(low_end_extension, high_end_extension):
            raise ValueError(f"Expected {low_end_extension=} ≃ {high_end_extension=}")

        return float(low_end_extension)

    def perform_lilliefors_test(self) -> Tuple[float, float]:
        """
        Perform the Lilliefors test for normality on the residuals of the real and imaginary parts.

        Returns
        -------
        Tuple[float, float]
            The p-values for the tests performed on the residuals of the real and imaginary parts.
            The null hypothesis is that the residuals come from a normal distribution.
        """
        from statsmodels.stats.diagnostic import lilliefors

        real: NDArray[float64]
        imag: NDArray[float64]
        _, real, imag = self.get_residuals_data()

        return tuple(
            map(
                lambda samples: lilliefors(samples)[1],
                (real, imag),
            )
        )

    def perform_shapiro_wilk_test(self) -> Tuple[float, float]:
        """
        Perform the Shapiro-Wilk test for normality on the residuals of the real and imaginary parts.

        Returns
        -------
        Tuple[float, float]
            The p-values for the tests performed on the residuals of the real and imaginary parts.
            The null hypothesis is that the residuals come from a normal distribution.
        """
        from scipy.stats import shapiro

        real: NDArray[float64]
        imag: NDArray[float64]
        _, real, imag = self.get_residuals_data()

        return tuple(
            map(
                lambda samples: shapiro(samples)[1],
                (real, imag),
            )
        )

    def perform_kolmogorov_smirnov_test(
        self,
        standard_deviation: float = 0.0,
    ) -> Tuple[float, float]:
        """
        Perform one-sample Kolmogorov-Smirnov test on the residuals of the real and imaginary parts.
        The residuals are tested against a normal distribution with a mean that is assumed to be zero and a standard deviation that can either be provided or estimated automatically.

        Parameters
        ----------
        standard_deviation: float, optional
            If greater than zero, then the provided value is used.
            Otherwise, the standard deviation estimated based on the pseudo chi-squared and the number of frequencies is used.

        Returns
        -------
        Tuple[float, float]
            The p-values for the tests performed on the residuals of the real and imaginary parts.
            The null hypothesis is that the distributions of the residuals are identical to the normal distribution with a mean of zero and the provided (or estimated) standard deviation.
        """
        from scipy.stats import kstest

        if not _is_floating(standard_deviation):
            raise TypeError(f"Expected a float instead of {standard_deviation=}")
        elif standard_deviation <= 0.0:
            standard_deviation = self.get_estimated_percent_noise()

        real: NDArray[float64]
        imag: NDArray[float64]
        _, real, imag = self.get_residuals_data()

        return tuple(
            map(
                lambda samples: kstest(
                    rvs=samples,
                    cdf="norm",
                    args=(0.0, standard_deviation),
                ).pvalue,
                (real, imag),
            )
        )

    def _calculate_residuals_statistics(self, level: int) -> Dict[str, float]:
        results: Dict[str, List[float]] = {}

        real: NDArray[float64]
        imag: NDArray[float64]
        _, real, imag = self.get_residuals_data()
        order: List[Tuple[NDArray[float64], str]] = [
            (real, "real"),
            (imag, "imag."),
        ]

        lilliefors: Tuple[float, float] = self.perform_lilliefors_test()
        shapiro_wilk: Tuple[float, float] = self.perform_shapiro_wilk_test()
        kolmogorov_smirnov: Tuple[float, float] = self.perform_kolmogorov_smirnov_test()

        samples: NDArray[float64]
        if level >= 1:
            for samples, _ in order:
                label: str = "Mean of residuals, %part% (% of |Z|)"
                if label not in results:
                    results[label] = []

                sample_mean: float64 = mean(samples)
                results[label].append(sample_mean)

                sample_sd: float64 = std(samples, ddof=1)
                label = "SD of residuals, %part% (% of |Z|)"
                if label not in results:
                    results[label] = []

                results[label].append(sample_sd)

                i: int
                for i in range(0, 3):
                    label = f"Residuals within {i + 1} SD, %part% (%)"
                    if label not in results:
                        results[label] = []

                    pct: float = (
                        array_sum(
                            logical_and(
                                samples < sample_mean + (i + 1) * sample_sd,
                                samples > sample_mean - (i + 1) * sample_sd,
                            )
                        )
                        / len(samples)
                        * 100
                    )
                    results[label].append(pct)

        if level >= 2:
            p: float
            for p in lilliefors:
                label = "Lilliefors test p-value, %part%"
                if label not in results:
                    results[label] = []
                results[label].append(p)

            for p in shapiro_wilk:
                label = "Shapiro-Wilk test p-value, %part%"
                if label not in results:
                    results[label] = []
                results[label].append(p)

        if level >= 3:
            for p in kolmogorov_smirnov:
                label = "One-sample Kolmogorov-Smirnov test p-value, %part%"
                if label not in results:
                    results["Estimated SD of Gaussian noise (% of |Z|)"] = (
                        self.get_estimated_percent_noise()
                    )
                    results[label] = []
                results[label].append(p)

        output: Dict[str, str] = {}

        for key in results.keys():
            if isinstance(results[key], list):
                value: Union[str, float]
                for i, value in enumerate(results[key]):
                    if "%part%" not in key:
                        raise ValueError(
                            f"Expected the substring '%part%' to exist in {key=}"
                        )

                    label = key.replace("%part%", order[i][1])
                    output[label] = value
            else:
                value = results[key]
                output[key] = value

        if not all(map(lambda key: isinstance(key, str), output.keys())):
            raise TypeError(f"Expected only string keys in {output=}")

        if not all(map(lambda value: _is_floating(value), output.values())):
            raise TypeError(f"Expected only float values in {output=}")

        return output

    def to_statistics_dataframe(
        self,
        extended_statistics: int = 3,
    ) -> "DataFrame":  # noqa: F821
        r"""
        Get the statistics related to the test as a `pandas.DataFrame`_ object.

        Parameters
        ----------
        extended_statistics: int, optional
            Include different amounts of additional statistics depending on the chosen level.
            Level 1 includes:

            - The estimated equivalent standard deviation of a Gaussian noise calculated based on the pseudo chi-squared value assuming that the noise in the real and imaginary parts of the impedance are independent, have a Gaussian distribution, a mean of zero, and the same standard deviation.
            - The means of the real and imaginary residuals.
            - The sample standard deviations of the real and imaginary residuals.
            - The percentage of points found within 1, 2, or 3 standard deviations.

            Level 2 includes:

            - The p-values of normality tests performed on the real or imaginary residuals. These tests include: Lilliefors and Shapiro-Wilk.

            Level 3 includes:

            - The p-values for one-sample Kolmogorov-Smirnov tests comparing the real or imaginary residuals against a normal distribution with a mean of zero and a standard deviation (as a percentage of :math:`|Z|`) equal to the approximation obtained with :math:`{\rm SD}_{\rm est} \approx \sqrt{\chi^2_{\rm ps} \times 5000 / N_\omega}` where :math:`\chi^2_{\rm ps}` is the pseudo chi-squared value of the fit and :math:`N_\omega` is the number of excitation frequencies. This approximation assumes that the error is spread evenly across the real and imaginary parts of the immittance spectrum.

        Returns
        -------
        `pandas.DataFrame`_
        """
        from pandas import DataFrame

        if not _is_integer(extended_statistics):
            raise TypeError(f"Expected an integer instead of {extended_statistics=}")
        elif not (0 <= extended_statistics <= 3):
            raise ValueError(
                f"Expected an integer in the range [0, 3] instead of {extended_statistics=}"
            )

        statistics: Dict[str, Union[int, float, str]] = {
            "Log pseudo chi-squared": log(self.pseudo_chisqr),
            "Number of RC elements": self.num_RC,
        }

        statistics["Log Fext (extension factor for time constant range)"] = (
            self.log_F_ext
        )

        R: float = (
            self.parallel_resistance if self.admittance else self.series_resistance
        )
        C: float = (
            self.parallel_capacitance if self.admittance else self.series_capacitance
        )
        L: float = (
            self.parallel_inductance if self.admittance else self.series_inductance
        )

        connection_type: str = "Parallel" if self.admittance else "Series"

        if not isnan(R):
            statistics[f"{connection_type} resistance (ohm)"] = R

        if not isnan(C):
            statistics[f"{connection_type} capacitance (F)"] = C

        if not isnan(L):
            statistics[f"{connection_type} inductance (H)"] = L

        if extended_statistics > 0:
            statistics.update(
                self._calculate_residuals_statistics(level=extended_statistics)
            )

        return DataFrame.from_dict(
            {
                "Label": list(statistics.keys()),
                "Value": list(statistics.values()),
            }
        )

    def get_series_resistance(self) -> float:
        """
        Get the value of the series resistance.

        Returns
        -------
        float
        """
        if not self.admittance:
            series: Series = self.circuit.get_connections(recursive=False)[0]
            if not isinstance(series, Series):
                raise TypeError(f"Expected a Series instead of {series=}")

            for element in series.get_elements(recursive=False):
                if isinstance(element, Resistor):
                    return element.get_value("R")

        return nan

    def get_series_capacitance(self) -> float:
        """
        Get the value of the series capacitance (or numpy.nan if not included in the circuit).

        Returns
        -------
        float
        """
        if not self.admittance:
            series: Series = self.circuit.get_connections(recursive=False)[0]
            if not isinstance(series, Series):
                raise TypeError(f"Expected a Series instead of {series=}")

            for element in series.get_elements(recursive=False):
                if isinstance(element, Capacitor):
                    return element.get_value("C")

        return nan

    def get_series_inductance(self) -> float:
        """
        Get the value of the series inductance (or numpy.nan if not included in the circuit).

        Returns
        -------
        float
        """
        if not self.admittance:
            series: Series = self.circuit.get_connections(recursive=False)[0]
            if not isinstance(series, Series):
                raise TypeError(f"Expected a Series instead of {series=}")

            for element in series.get_elements(recursive=False):
                if isinstance(element, Inductor):
                    return element.get_value("L")

        return nan

    def get_parallel_resistance(self) -> float:
        """
        Get the value of the parallel resistance.

        Returns
        -------
        float
        """
        if self.admittance:
            parallel: Parallel = self.circuit.get_connections(recursive=True)[1]
            if not isinstance(parallel, Parallel):
                raise TypeError(f"Expected a Parallel instead of {parallel=}")

            for element in parallel.get_elements(recursive=False):
                if isinstance(element, Resistor):
                    return element.get_value("R")

        return nan

    def get_parallel_capacitance(self) -> float:
        """
        Get the value of the parallel capacitance (or numpy.nan if not included in the circuit).

        Returns
        -------
        float
        """
        if self.admittance:
            parallel: Parallel = self.circuit.get_connections(recursive=True)[1]
            if not isinstance(parallel, Parallel):
                raise TypeError(f"Expected a Parallel instead of {parallel=}")

            for element in parallel.get_elements(recursive=False):
                if isinstance(element, Capacitor):
                    return element.get_value("C")

        return nan

    def get_parallel_inductance(self) -> float:
        """
        Get the value of the parallel inductance (or numpy.nan if not included in the circuit).

        Returns
        -------
        float
        """
        if self.admittance:
            parallel: Parallel = self.circuit.get_connections(recursive=True)[1]
            if not isinstance(parallel, Parallel):
                raise TypeError(f"Expected a Parallel instead of {parallel=}")

            for element in parallel.get_elements(recursive=False):
                if isinstance(element, Inductor):
                    return element.get_value("L")

        return nan

    def get_estimated_percent_noise(self) -> float:
        r"""
        Estimate the standard deviation of the noise (as a percentage of :math:`|Z|`) using the approximation :math:`{\rm SD}_{\rm est} \approx \sqrt{\chi^2_{\rm ps} \times 5000 / N_\omega}` where :math:`\chi^2_{\rm ps}` is the pseudo chi-squared value of the fit and :math:`N_\omega` is the number of excitation frequencies. This approximation assumes that the error is spread evenly across the real and imaginary parts of the immittance spectrum.

        References:

        - `V. Yrjänä and J. Bobacka, 2024, Electrochim. Acta, 504, 144951 <https://doi.org/10.1016/j.electacta.2024.144951>`_

        Returns
        -------
        float
        """
        return _estimate_pct_noise(self.impedances, self.pseudo_chisqr)
