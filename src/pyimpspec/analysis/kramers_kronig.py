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

from dataclasses import dataclass
from multiprocessing import (
    Pool,
    Value,
)
from typing import (
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Tuple,
    Union,
)
from numpy import (
    abs,
    angle,
    array,
    float64,
    floating,
    inf,
    integer,
    isinf,
    issubdtype,
    log10 as log,
    min,
    max,
    nan,
    ndarray,
    pi,
    sum as array_sum,
    zeros,
)
from numpy.linalg import (
    inv,
    pinv,
)
from numpy.typing import NDArray
from pyimpspec.exceptions import KramersKronigError
from pyimpspec.analysis.fitting import (
    _METHODS,
    _from_lmfit,
    _to_lmfit,
)
from pyimpspec.analysis.utility import (
    _boukamp_weight,
    _calculate_pseudo_chisqr,
    _calculate_residuals,
    _interpolate,
    _get_default_num_procs,
)
from pyimpspec.circuit import parse_cdc
from pyimpspec.circuit.base import Element
from pyimpspec.circuit.circuit import Circuit
from pyimpspec.circuit.connections import Series
from pyimpspec.circuit.elements import (
    Capacitor,
    Resistor,
    Inductor,
)
from pyimpspec.data import DataSet
from pyimpspec.progress import Progress
from pyimpspec.typing import (
    ComplexImpedances,
    ComplexResiduals,
    Frequencies,
    Impedances,
    Phases,
    Residuals,
)


@dataclass(frozen=True)
class TestResult:
    """
    An object representing the results of a linear Kramers-Kronig test applied to a data set.

    Parameters
    ----------
    circuit: Circuit
        The fitted circuit.

    num_RC: int
        The final number of RC elements in the fitted model (Boukamp, 1995).

    mu: float
        The |mu| value of the final fit (eq. 21 in Schönleber et al., 2014).

    pseudo_chisqr: float
        The pseudo chi-squared value (|pseudo chi-squared|, eq. 14 in Boukamp, 1995).

    frequencies: |Frequencies|
        The frequencies used to perform the test.

    impedances: |ComplexImpedances|
        The impedances produced by the fitted circuit at each of the tested frequencies.

    residuals: |ComplexResiduals|
        The residuals for the real (eq. 15 in Schönleber et al., 2014) and imaginary (eq. 16 in Schönleber et al., 2014) parts of the fit.
    """

    circuit: Circuit
    num_RC: int
    mu: float
    pseudo_chisqr: float
    frequencies: Frequencies
    impedances: ComplexImpedances
    residuals: ComplexResiduals

    def __repr__(self) -> str:
        return f"TestResult (num_RC={self.num_RC}, {hex(id(self))})"

    def get_label(self) -> str:
        """
        Get the label of this result

        Returns
        -------
        str
        """
        label: str = f"#(RC)={self.num_RC}"
        cdc: str = self.circuit.to_string()
        if "C" in cdc:
            label += ", C"
            if "L" in cdc:
                label += "+L"
        elif "L" in cdc:
            label += ", L"
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
        assert issubdtype(type(num_per_decade), integer), num_per_decade
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
        assert issubdtype(type(num_per_decade), integer), num_per_decade
        if num_per_decade > 0:
            return self.circuit.get_impedances(self.get_frequencies(num_per_decade))
        return self.impedances

    def get_nyquist_data(
        self,
        num_per_decade: int = -1,
    ) -> Tuple[Impedances, Impedances]:
        """
        Get the data necessary to plot this TestResult as a Nyquist plot: the real and the negative imaginary parts of the impedances.

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
        assert issubdtype(type(num_per_decade), integer), num_per_decade
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
        Get the data necessary to plot this TestResult as a Bode plot: the frequencies, the absolute magnitudes of the impedances, and the negative phase angles/shifts of the impedances in degrees.

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
        assert issubdtype(type(num_per_decade), integer), num_per_decade
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

    def calculate_score(self, mu_criterion: float) -> float:
        """
        Calculate a score based on the provided |mu|-criterion and the statistics of the test result.
        This calculation is part of the modified implementation of the algorithm described by Schönleber et al. (2014).
        A test result with |mu| greater than or equal to the |mu|-criterion will get a score of -numpy.inf.

        Parameters
        ----------
        mu_criterion: float
            The |mu|-criterion to apply (see |perform_test| for details).

        Returns
        -------
        float
        """
        return (
            -inf
            if self.mu >= mu_criterion
            else -log(self.pseudo_chisqr) / (abs(mu_criterion - self.mu) ** 0.75)
        )

    def to_statistics_dataframe(self) -> "DataFrame":  # noqa: F821
        """
        Get the statistics related to the test as a |DataFrame| object.

        Returns
        -------
        |DataFrame|
        """
        from pandas import DataFrame

        statistics: Dict[str, Union[int, float, str]] = {
            "Log pseudo chi-squared": log(self.pseudo_chisqr),
            "Mu": self.mu,
            "Number of parallel RC elements": self.num_RC,
            "Series resistance (ohm)": self.get_series_resistance(),
            "Series capacitance (F)": self.get_series_capacitance(),
            "Series inductance (H)": self.get_series_inductance(),
        }
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
        series: Series = self.circuit.get_elements(flattened=False)[0]
        assert isinstance(series, Series)
        for elem_con in series.get_elements(flattened=False):
            if isinstance(elem_con, Resistor):
                return elem_con.get_value("R")
        return nan

    def get_series_capacitance(self) -> float:
        """
        Get the value of the series capacitance (or numpy.nan if not included in the circuit).

        Returns
        -------
        float
        """
        series: Series = self.circuit.get_elements(flattened=False)[0]
        assert isinstance(series, Series)
        for elem_con in series.get_elements(flattened=False):
            if isinstance(elem_con, Capacitor):
                return elem_con.get_value("C")
        return nan

    def get_series_inductance(self) -> float:
        """
        Get the value of the series inductance (or numpy.nan if not included in the circuit).

        Returns
        -------
        float
        """
        series: Series = self.circuit.get_elements(flattened=False)[0]
        assert isinstance(series, Series)
        for elem_con in series.get_elements(flattened=False):
            if isinstance(elem_con, Inductor):
                return elem_con.get_value("L")
        return nan


def _calculate_tau(i: int, num_RC: int, tau_min: float64, tau_max: float64) -> float:
    # Calculate time constants according to eq. 12 in Schönleber et al. (2014)
    assert issubdtype(type(i), integer), i
    assert issubdtype(type(num_RC), integer), num_RC
    assert issubdtype(type(tau_min), floating), tau_min
    assert issubdtype(type(tau_max), floating), tau_max
    return pow(10, (log(tau_min) + (i - 1) / (num_RC - 1) * log(tau_max / tau_min)))


def _generate_time_constants(w: NDArray[float64], num_RC: int) -> NDArray[float64]:
    assert isinstance(w, ndarray), w
    assert issubdtype(type(num_RC), integer), num_RC
    taus: NDArray[float64] = zeros(shape=(num_RC,), dtype=float64)
    tau_min: float64 = 1 / max(w)
    tau_max: float64 = 1 / min(w)
    taus[0] = tau_min
    taus[-1] = tau_max
    i: int
    if num_RC > 1:
        for i in range(2, num_RC):
            taus[i - 1] = _calculate_tau(i, num_RC, tau_min, tau_max)
    return taus


def _calculate_mu(params: "Parameters") -> float:  # noqa: F821
    from lmfit import Parameters

    assert isinstance(params, Parameters), params
    # Calculates the mu-value based on the fitted parameters according to eq. 21 in
    # Schönleber et al. (2014)
    R_neg: List[float64] = []
    R_pos: List[float64] = []
    name: str
    value: float64
    for name, value in params.valuesdict().items():
        if not name.startswith("R"):
            continue
        if value < 0:
            R_neg.append(abs(value))
        else:
            R_pos.append(value)
    neg_sum: float64 = sum(R_neg)
    pos_sum: float64 = sum(R_pos)
    if pos_sum == 0:
        return 0.0
    mu: float64 = 1.0 - neg_sum / pos_sum
    if mu < 0.0:
        return 0.0
    elif mu > 1.0:
        return 1.0
    return float(mu)


def _elements_to_parameters(
    elements: NDArray[float64],
    taus: NDArray[float64],
    add_capacitance: bool,
    circuit: Circuit,
) -> "Parameters":  # noqa: F821
    from lmfit import Parameters

    assert isinstance(elements, ndarray), elements
    assert isinstance(taus, ndarray), taus
    assert isinstance(add_capacitance, bool), add_capacitance
    assert isinstance(circuit, Circuit), circuit
    parameters: Parameters = Parameters()
    R0: float
    R0, elements = elements[0], elements[1:]
    parameters.add(f"R_{0}", R0, vary=False)
    L: float
    L, elements = elements[-1], elements[:-1]
    C: float
    if add_capacitance:
        C, elements = elements[-1], elements[:-1]
        if C == 0.0:
            # Impedance due to the series capacitance is negligible.
            C = 1e50
        else:
            C = 1 / C
    for i, (R, t) in enumerate(zip(elements, taus), start=1):
        parameters.add(f"R_{i}", R, vary=False)
        parameters.add(f"tau_{i}", t, vary=False)
    if add_capacitance:
        i += 1
        parameters.add(f"C_{i}", C)
    i += 1
    parameters.add(f"L_{i}", L, vary=False)
    return parameters


def _complex_residual(
    params: "Parameters",  # noqa: F821
    circuit: Circuit,
    f: Frequencies,
    Z_exp: ComplexImpedances,
    weight: NDArray[float64],
    identifiers: Dict[int, Element],
) -> NDArray[float64]:
    from lmfit import Parameters

    assert isinstance(params, Parameters), params
    assert isinstance(circuit, Circuit), circuit
    assert isinstance(f, ndarray), f
    assert isinstance(Z_exp, ndarray), Z_exp
    assert isinstance(weight, ndarray), weight
    assert isinstance(identifiers, dict), identifiers
    _from_lmfit(params, identifiers)
    Z_fit: ComplexImpedances = circuit.get_impedances(f)
    return array(
        [
            (weight * (Z_exp.real - Z_fit.real)) ** 2,
            (weight * (Z_exp.imag - Z_fit.imag)) ** 2,
        ]
    )


def _cnls_test(args: tuple) -> Tuple[int, float, Optional[Circuit], float]:
    from lmfit import minimize
    from lmfit.minimizer import MinimizerResult

    f: Frequencies
    Z_exp: ComplexImpedances
    weight: NDArray[float64]
    num_RC: int
    add_capacitance: bool
    add_inductance: bool
    method: str
    max_nfev: int
    (
        f,
        Z_exp,
        weight,
        num_RC,
        add_capacitance,
        add_inductance,
        method,
        max_nfev,
    ) = args
    assert isinstance(f, ndarray), f
    assert isinstance(Z_exp, ndarray), Z_exp
    assert isinstance(weight, ndarray), weight
    assert issubdtype(type(num_RC), integer), num_RC
    assert isinstance(add_capacitance, bool), add_capacitance
    assert isinstance(add_inductance, bool), add_inductance
    assert isinstance(method, str), method
    assert issubdtype(type(max_nfev), integer), max_nfev
    w: NDArray[float64] = 2 * pi * f
    taus: NDArray[float64] = _generate_time_constants(w, num_RC)
    circuit: Circuit = _generate_circuit(taus, add_capacitance, add_inductance)
    identifiers: Dict[int, Element] = {
        v: k for k, v in circuit.generate_element_identifiers(running=True).items()
    }
    fit: MinimizerResult
    fit = minimize(
        _complex_residual,
        _to_lmfit(identifiers),
        method,
        args=(
            circuit,
            f,
            Z_exp,
            weight,
            identifiers,
        ),
        max_nfev=None if max_nfev < 1 else max_nfev,
    )
    _from_lmfit(fit.params, identifiers)
    mu: float = _calculate_mu(fit.params)
    Z_fit: ComplexImpedances = circuit.get_impedances(f)
    Xps: float = _calculate_pseudo_chisqr(Z_exp, Z_fit, weight)
    return (
        num_RC,
        mu,
        circuit,
        Xps,
    )


# A shared integer value, which represents the smallest number of RC elements that has resulted
# in a mu-value that is less than the chosen mu-criterion. If this value is negative, then an
# appropriate number of RC elements has not yet been found. However, if the value is greater than
# the number of RC elements currently being evaluated by the process accessing the value, then
# the process should continue performing the current fitting. Otherwise, any ongoing fitting should
# be terminated as soon as possible and no further attempts should be made with a greater number of
# RC elements.
pool_optimal_num_RC = None  # multiprocessing.Value


# Initializer for the pool of processes that are used when performing the complex variant of the
# linear Kramers-Kronig test.
def _pool_init(args):
    global pool_optimal_num_RC
    pool_optimal_num_RC = args


def _cnls_mu_process(args: tuple) -> Tuple[int, float, Optional[Circuit], float]:
    from lmfit import minimize
    from lmfit.minimizer import MinimizerResult

    global pool_optimal_num_RC
    f: Frequencies
    Z_exp: ComplexImpedances
    weight: NDArray[float64]
    mu_criterion: float
    num_RC: int
    add_capacitance: bool
    add_inductance: bool
    method: str
    max_nfev: int
    (
        f,
        Z_exp,
        weight,
        mu_criterion,
        num_RC,
        add_capacitance,
        add_inductance,
        method,
        max_nfev,
    ) = args
    assert isinstance(f, ndarray), f
    assert isinstance(Z_exp, ndarray), Z_exp
    assert isinstance(weight, ndarray), weight
    assert issubdtype(type(mu_criterion), floating), mu_criterion
    assert issubdtype(type(num_RC), integer), num_RC
    assert isinstance(add_capacitance, bool), add_capacitance
    assert isinstance(add_inductance, bool), add_inductance
    assert isinstance(method, str), method
    assert issubdtype(type(max_nfev), integer), max_nfev

    def exit_early(
        params: "Parameters",  # noqa: F821
        i: int,
        _,
        *args,
        **kwargs,
    ):
        if 0 <= pool_optimal_num_RC.value < num_RC:  # type: ignore
            return True
        return None

    w: NDArray[float64] = 2 * pi * f
    taus: NDArray[float64] = _generate_time_constants(w, num_RC)
    circuit: Circuit = _generate_circuit(taus, add_capacitance, add_inductance)
    if 0 <= pool_optimal_num_RC.value < num_RC:  # type: ignore
        return (num_RC, -1.0, None, -1.0)
    identifiers: Dict[int, Element] = {
        v: k for k, v in circuit.generate_element_identifiers(running=True).items()
    }
    fit: MinimizerResult
    fit = minimize(
        _complex_residual,
        _to_lmfit(identifiers),
        method,
        args=(
            circuit,
            f,
            Z_exp,
            weight,
            identifiers,
        ),
        max_nfev=None if max_nfev < 1 else max_nfev,
        iter_cb=exit_early,
    )
    if 0 <= pool_optimal_num_RC.value < num_RC:  # type: ignore
        return (num_RC, -1.0, None, -1.0)
    _from_lmfit(fit.params, identifiers)
    mu: float = _calculate_mu(fit.params)
    Z_fit: ComplexImpedances = circuit.get_impedances(f)
    Xps: float = _calculate_pseudo_chisqr(Z_exp, Z_fit, weight)
    if mu < mu_criterion:
        with pool_optimal_num_RC.get_lock():  # type: ignore
            pool_optimal_num_RC.value = num_RC  # type: ignore
    return (
        num_RC,
        mu,
        circuit,
        Xps,
    )


def _generate_variable_matrices(
    w: NDArray[float64],
    num_RC: int,
    taus: NDArray[float64],
    add_capacitance: bool,
    abs_Z_exp: Impedances,
) -> Tuple[NDArray[float64], NDArray[float64]]:
    assert isinstance(w, ndarray), w
    assert issubdtype(type(num_RC), integer), num_RC
    assert isinstance(taus, ndarray), taus
    assert isinstance(add_capacitance, bool), add_capacitance
    assert isinstance(abs_Z_exp, ndarray), abs_Z_exp
    # Generate matrices with the following columns (top to bottom is left to right)
    # - R0, series resistance
    # - Ri, resistance in parallel with taus[i - 1] where 0 < i <= num_RC
    # - (C, optional series capacitance)
    # - L, series inductance
    if add_capacitance is True:
        a_re = zeros((w.size, num_RC + 3), dtype=float64)
        a_im = zeros((w.size, num_RC + 3), dtype=float64)
        # Series capacitance
        a_im[:, -2] = -1 / (w * abs_Z_exp)  # No real part
    else:
        a_re = zeros((w.size, num_RC + 2), dtype=float64)
        a_im = zeros((w.size, num_RC + 2), dtype=float64)
    # Series resistance
    a_re[:, 0] = 1 / abs_Z_exp  # No imaginary part
    # Series inductance
    a_im[:, -1] = w / abs_Z_exp  # No real part
    # RC elements
    for i, tau in enumerate(taus):
        a_re[:, i + 1] = (1 / (1 + 1j * w * tau)).real / abs_Z_exp
        a_im[:, i + 1] = (1 / (1 + 1j * w * tau)).imag / abs_Z_exp
    return (
        a_re,
        a_im,
    )


def _generate_circuit(
    taus: NDArray[float64],
    add_capacitance: bool,
    add_inductance: bool,
) -> Circuit:
    assert isinstance(taus, ndarray), taus
    assert isinstance(add_capacitance, bool), add_capacitance
    assert isinstance(add_inductance, bool), add_inductance
    cdc: List[str] = ["R{R=1}"]
    t: float
    for t in taus:
        cdc.append(f"K{{R=1,tau={t}F}}")
    if add_capacitance is True:
        cdc.append("C{C=1e-6}")
    if add_inductance is True:
        cdc.append("L{L=1e-3}")
    circuit: Circuit = parse_cdc("".join(cdc))
    for element in circuit.get_elements():
        assert isinstance(element, Element)
        keys: List[str] = list(element.get_values().keys())
        element.set_lower_limits(**{_: -inf for _ in keys})
        element.set_upper_limits(**{_: inf for _ in keys})
    return circuit


def _real_test(
    a_re: NDArray[float64],
    a_im: NDArray[float64],
    Z_exp: ComplexImpedances,
    abs_Z_exp: Impedances,
    w: NDArray[float64],
    f: Frequencies,
    taus: NDArray[float64],
    add_capacitance: bool,
    circuit: Circuit,
    identifiers: Dict[int, Element],
):
    assert isinstance(a_re, ndarray), a_re
    assert isinstance(a_im, ndarray), a_im
    assert isinstance(Z_exp, ndarray), Z_exp
    assert isinstance(abs_Z_exp, ndarray), abs_Z_exp
    assert isinstance(w, ndarray), w
    assert isinstance(f, ndarray), f
    assert isinstance(taus, ndarray), taus
    assert isinstance(add_capacitance, bool), add_capacitance
    assert isinstance(circuit, Circuit), circuit
    assert isinstance(identifiers, dict), identifiers
    # Fit using the real part
    elements: NDArray[float64] = pinv(a_re).dot(Z_exp.real / abs_Z_exp)
    # Fit using the imaginary part to fix the series inductance (and capacitance)
    a_im = zeros((w.size, 2))
    a_im[:, -1] = w / abs_Z_exp
    if add_capacitance:
        a_im[:, -2] = -1 / (w * abs_Z_exp)
        elements[-2] = 1e-18  # Nullifies the series capacitance without dividing by 0
    _from_lmfit(
        _elements_to_parameters(
            elements,
            taus,
            add_capacitance,
            circuit,
        ),
        identifiers,
    )
    Z_fit: ComplexImpedances = circuit.get_impedances(f)
    coefs: NDArray[float64] = pinv(a_im).dot((Z_exp.imag - Z_fit.imag) / abs_Z_exp)
    # Extract the corrected series inductance (and capacitance)
    if add_capacitance:
        elements[-2:] = coefs
    else:
        elements[-1] = coefs[-1]
    _from_lmfit(
        _elements_to_parameters(
            elements,
            taus,
            add_capacitance,
            circuit,
        ),
        identifiers,
    )


def _imaginary_test(
    a_im: NDArray[float64],
    Z_exp: ComplexImpedances,
    abs_Z_exp: Impedances,
    f: Frequencies,
    taus: NDArray[float64],
    add_capacitance: bool,
    weight: NDArray[float64],
    circuit: Circuit,
    identifiers: Dict[int, Element],
):
    assert isinstance(a_im, ndarray), a_im
    assert isinstance(Z_exp, ndarray), Z_exp
    assert isinstance(abs_Z_exp, ndarray), abs_Z_exp
    assert isinstance(f, ndarray), f
    assert isinstance(taus, ndarray), taus
    assert isinstance(add_capacitance, bool), add_capacitance
    assert isinstance(weight, ndarray), weight
    assert isinstance(circuit, Circuit), circuit
    assert isinstance(identifiers, dict), identifiers
    # Fit using the imaginary part
    elements: NDArray[float64] = pinv(a_im).dot(Z_exp.imag / abs_Z_exp)
    # Estimate the series resistance
    _from_lmfit(
        _elements_to_parameters(
            elements,
            taus,
            add_capacitance,
            circuit,
        ),
        identifiers,
    )
    Z_fit: ComplexImpedances = circuit.get_impedances(f)
    elements[0] = array_sum(weight * (Z_exp.real - Z_fit.real)) / array_sum(weight)
    _from_lmfit(
        _elements_to_parameters(
            elements,
            taus,
            add_capacitance,
            circuit,
        ),
        identifiers,
    )


def _complex_test(
    a_re: NDArray[float64],
    a_im: NDArray[float64],
    Z_exp: ComplexImpedances,
    abs_Z_exp: Impedances,
    taus: NDArray[float64],
    add_capacitance: bool,
    circuit: Circuit,
    identifiers: Dict[int, Element],
):
    assert isinstance(a_re, ndarray), a_re
    assert isinstance(a_im, ndarray), a_im
    assert isinstance(Z_exp, ndarray), Z_exp
    assert isinstance(abs_Z_exp, ndarray), abs_Z_exp
    assert isinstance(taus, ndarray), taus
    assert isinstance(add_capacitance, bool), add_capacitance
    assert isinstance(circuit, Circuit), circuit
    assert isinstance(identifiers, dict), identifiers
    # Fit using the complex impedance
    x: NDArray[float64] = inv(a_re.T.dot(a_re) + a_im.T.dot(a_im))
    y: NDArray[float64] = a_re.T.dot(Z_exp.real / abs_Z_exp) + a_im.T.dot(
        Z_exp.imag / abs_Z_exp
    )
    elements: NDArray[float64] = x.dot(y)
    _from_lmfit(
        _elements_to_parameters(
            elements,
            taus,
            add_capacitance,
            circuit,
        ),
        identifiers,
    )


def _test_wrapper(args: tuple) -> Tuple[int, float, Optional[Circuit], float]:
    test: str
    f: Frequencies
    Z_exp: ComplexImpedances
    weight: NDArray[float64]
    num_RC: int
    add_capacitance: bool
    test, f, Z_exp, weight, num_RC, add_capacitance = args
    assert isinstance(test, str), test
    assert isinstance(f, ndarray), f
    assert isinstance(Z_exp, ndarray), Z_exp
    assert isinstance(weight, ndarray), weight
    assert issubdtype(type(num_RC), integer), num_RC
    assert isinstance(add_capacitance, bool), add_capacitance
    abs_Z_exp: Impedances = abs(Z_exp)
    w: NDArray[float64] = 2 * pi * f
    taus: NDArray[float64] = _generate_time_constants(w, num_RC)
    a_re: NDArray[float64]
    a_im: NDArray[float64]
    a_re, a_im = _generate_variable_matrices(
        w, num_RC, taus, add_capacitance, abs_Z_exp
    )
    circuit: Circuit = _generate_circuit(taus, add_capacitance, True)
    identifiers: Dict[int, Element] = {
        v: k for k, v in circuit.generate_element_identifiers(running=True).items()
    }
    # Solve the set of linear equations and update the circuit's parameters
    if test == "real":
        _real_test(
            a_re,
            a_im,
            Z_exp,
            abs_Z_exp,
            w,
            f,
            taus,
            add_capacitance,
            circuit,
            identifiers,
        )
    elif test == "imaginary":
        _imaginary_test(
            a_im,
            Z_exp,
            abs_Z_exp,
            f,
            taus,
            add_capacitance,
            weight,
            circuit,
            identifiers,
        )
    elif test == "complex":
        _complex_test(
            a_re,
            a_im,
            Z_exp,
            abs_Z_exp,
            taus,
            add_capacitance,
            circuit,
            identifiers,
        )
    # Calculate return values
    mu: float = _calculate_mu(_to_lmfit(identifiers))
    Z_fit: ComplexImpedances = circuit.get_impedances(f)
    Xps: float = _calculate_pseudo_chisqr(Z_exp, Z_fit, weight)
    return (
        num_RC,
        mu,
        circuit,
        Xps,
    )


def _perform_single_cnls_test(
    f: Frequencies,
    Z_exp: ComplexImpedances,
    weight: NDArray[float64],
    num_RC: int,
    add_capacitance: bool,
    add_inductance: bool,
    method: str,
    max_nfev: int,
    num_procs: int,
) -> Tuple[int, float, Optional[Circuit], float]:
    with Progress("Performing test"):
        args = (
            (
                f,
                Z_exp,
                weight,
                num_RC,
                add_capacitance,
                add_inductance,
                method,
                max_nfev,
            )
            for _ in range(1)
        )
        fits: List[Tuple[int, float, Optional[Circuit], float]]
        if num_procs > 1:
            # To prevent the GUI thread of DearEIS from locking up
            with Pool(1) as pool:
                fits = pool.map(_cnls_test, args)
        else:
            fits = list(map(_cnls_test, args))
    return fits[0]


def _perform_multiple_cnls_tests(
    f: Frequencies,
    Z_exp: ComplexImpedances,
    weight: NDArray[float64],
    num_RC: int,
    mu_criterion: float,
    add_capacitance: bool,
    add_inductance: bool,
    method: str,
    max_nfev: int,
    num_procs: int,
) -> Tuple[int, float, Optional[Circuit], float]:
    prog: Progress
    with Progress("Preparing arguments") as prog:
        num_RC = abs(num_RC) or len(f)
        num_RCs: List[int] = list(range(1, num_RC + 1))
        args = (
            (
                f,
                Z_exp,
                weight,
                mu_criterion,
                num_RC,
                add_capacitance,
                add_inductance,
                method,
                max_nfev,
            )
            for num_RC in num_RCs
        )
        fits: List[Tuple[int, float, Optional[Circuit], float]] = []
        prog.set_message("Performing test(s)", i=0, total=len(num_RCs) + 1)
        i: int
        res: Tuple[int, float, Optional[Circuit], float]
        if num_procs > 1:
            with Pool(
                num_procs,
                initializer=_pool_init,
                initargs=(Value("i", -1),),
            ) as pool:
                for i, res in enumerate(pool.imap_unordered(_cnls_mu_process, args)):
                    prog.increment()
                    if res[2] is not None:
                        fits.append(res)
        else:
            _pool_init(Value("i", -1))
            for i, res in enumerate(map(_cnls_mu_process, args)):
                prog.increment()
                if res[2] is not None:
                    fits.append(res)
        fits.sort(key=lambda _: _[0])
        mu: float
        circuit: Circuit
        Xps: float
        for i, (num_RC, mu, circuit, Xps) in enumerate(fits):
            if mu < mu_criterion:
                break
        return fits[i]


def _perform_linear_tests(
    test: str,
    f: Frequencies,
    Z_exp: ComplexImpedances,
    weight: NDArray[float64],
    num_RC: int,
    mu_criterion: float,
    add_capacitance: bool,
    num_procs: int,
) -> Tuple[int, float, Optional[Circuit], float]:
    prog: Progress
    with Progress("Preparing arguments") as prog:
        supported_tests: List[str] = [
            "complex",
            "real",
            "imaginary",
        ]
        assert test in supported_tests, f"Unsupported test: '{test}'!"
        num_RCs: List[int]
        if num_RC > 0:
            num_RCs = [num_RC]
        else:
            num_RC = abs(num_RC)
            if num_RC <= 1:
                num_RC = len(f)
            num_RCs = list(range(1, num_RC + 1))
        args = (
            (
                test,
                f,
                Z_exp,
                weight,
                num_RC,
                add_capacitance,
            )
            for num_RC in num_RCs
        )
        fits: List[Tuple[int, float, Optional[Circuit], float]] = []
        prog.set_message("Performing test(s)", i=0, total=len(num_RCs) + 1)
        for i, res in enumerate(map(_test_wrapper, args)):
            prog.increment()
            fits.append(res)
    if len(fits) == 1:
        return fits[0]
    fits.sort(key=lambda _: _[0])
    for i, (num_RC, mu, circuit, Xps) in enumerate(fits):
        if mu < mu_criterion:
            break
    return fits[i]


def perform_test(
    data: DataSet,
    test: str = "complex",
    num_RC: int = 0,
    mu_criterion: float = 0.85,
    add_capacitance: bool = False,
    add_inductance: bool = False,
    method: str = "leastsq",
    max_nfev: int = -1,
    num_procs: int = 0,
) -> TestResult:
    """
    Performs a linear Kramers-Kronig test as described by Boukamp (1995).
    The results can be used to check the validity of an impedance spectrum before performing equivalent circuit fitting.
    If the number of RC elements is less than two, then a suitable number of RC elements is determined using the procedure described by Schönleber et al. (2014) based on a criterion for the calculated |mu| (0.0 to 1.0).
    A |mu| of 1.0 represents underfitting and a |mu| of 0.0 represents overfitting.

    References:

    - Boukamp, B.A., 1995, J. Electrochem. Soc., 142, 1885-1894 (https://doi.org/10.1149/1.2044210)
    - Schönleber, M., Klotz, D., and Ivers-Tiffée, E., 2014, Electrochim. Acta, 131, 20-27 (https://doi.org/10.1016/j.electacta.2014.01.034)

    Parameters
    ----------
    data: DataSet
        The data set to be tested.

    test: str, optional
        Supported values include "complex", "imaginary", "real", and "cnls".
        The "complex", "imaginary", and "real" tests perform the complex, imaginary, and real tests, respectively, according to Boukamp (1995).
        The "cnls" test, which is slower than the other three tests, performs a complex non-linear least squares fit using `lmfit.minimize`_.

    num_RC: int, optional
        The number of RC elements to use.
        A value greater than or equal to one results in the specific number of RC elements being tested.
        A value less than one results in the use of the procedure described by Schönleber et al. (2014) based on the chosen |mu|-criterion.
        If the provided value is negative, then the maximum number of RC elements to test is equal to the absolute value of the provided value.
        If the provided value is zero, then the maximum number of RC elements to test is equal to the number of frequencies in the data set.

    mu_criterion: float, optional
        The chosen |mu|-criterion. See Schönleber et al. (2014) for more information.

    add_capacitance: bool, optional
        Add an additional capacitance in series with the rest of the circuit.

    add_inductance: bool, optional
        Add an additional inductance in series with the rest of the circuit.
        Applies only to the "cnls" test.

    method: str, optional
        The fitting method to use when performing a "cnls" test.
        See the list of methods that are listed in the documentation for the lmfit package.
        Methods that do not require providing bounds for all parameters or a function to calculate the Jacobian should work.

    max_nfev: int, optional
        The maximum number of function evaluations when fitting.
        A value less than one equals no limit.
        Applies only to the "cnls" test.

    num_procs: int, optional
        The maximum number of parallel processes to use when performing a test.
        A value less than 1 results in an attempt to figure out a suitable value based on, e.g., the number of cores detected.
        Additionally, a negative value can be used to reduce the number of processes by that much (e.g., to leave one core for a GUI thread).
        Applies only to the "cnls" test.

    Returns
    -------
    TestResult
    """
    assert hasattr(data, "get_frequencies") and callable(data.get_frequencies)
    assert hasattr(data, "get_impedances") and callable(data.get_impedances)
    assert isinstance(test, str), (
        type(test),
        test,
    )
    assert issubdtype(type(num_RC), integer), (
        type(num_RC),
        num_RC,
    )
    num_points: int = len(data.get_frequencies())
    if num_RC > num_points:
        raise KramersKronigError(
            "The value of num_RC must be less than or equal to the number of data points"
        )
    assert issubdtype(type(mu_criterion), floating), (
        type(mu_criterion),
        mu_criterion,
    )
    if num_RC <= 0 and not (0.0 <= mu_criterion <= 1.0):
        raise KramersKronigError(
            "The value of mu_criterion must be between 0.0 and 1.0 (inclusive)"
        )
    assert isinstance(add_capacitance, bool), (
        type(add_capacitance),
        add_capacitance,
    )
    assert isinstance(add_inductance, bool), (
        type(add_inductance),
        add_inductance,
    )
    assert isinstance(method, str), (
        type(method),
        method,
    )
    if method not in _METHODS:
        raise KramersKronigError("Valid method values: '" + "', '".join(_METHODS) + "'")
    assert issubdtype(type(max_nfev), integer), (
        type(max_nfev),
        max_nfev,
    )
    assert issubdtype(type(num_procs), integer), (
        type(num_procs),
        num_procs,
    )
    if num_procs < 1:
        num_procs = _get_default_num_procs() - abs(num_procs)
        if num_procs < 1:
            num_procs = 1
    f: Frequencies = data.get_frequencies()
    Z_exp: ComplexImpedances = data.get_impedances()
    weight: NDArray[float64] = _boukamp_weight(Z_exp)
    mu: float
    circuit: Optional[Circuit] = None
    Xps: float  # pseudo chi-squared
    if test == "cnls":
        assert method in _METHODS, f"Unsupported method: '{method}'!"
        if num_RC > 0:
            # Perform the test with a specific number of RC elements
            num_RC, mu, circuit, Xps = _perform_single_cnls_test(
                f,
                Z_exp,
                weight,
                num_RC,
                add_capacitance,
                add_inductance,
                method,
                max_nfev,
                num_procs,
            )
        else:
            # Find an appropriate number of RC elements based on the calculated mu-value and the
            # provided threshold value. Use multiple parallel processes if possible.
            num_RC, mu, circuit, Xps = _perform_multiple_cnls_tests(
                f,
                Z_exp,
                weight,
                num_RC,
                mu_criterion,
                add_capacitance,
                add_inductance,
                method,
                max_nfev,
                num_procs,
            )
    else:
        num_RC, mu, circuit, Xps = _perform_linear_tests(
            test,
            f,
            Z_exp,
            weight,
            num_RC,
            mu_criterion,
            add_capacitance,
            num_procs,
        )
    # ========== Result ==========
    assert circuit is not None
    Z_fit: ComplexImpedances = circuit.get_impedances(f)
    return TestResult(
        circuit=circuit,
        num_RC=num_RC,
        mu=mu,
        pseudo_chisqr=Xps,
        frequencies=f,
        impedances=Z_fit,
        # Residuals calculated according to eqs. 15 and 16
        # in Schönleber et al. (2014)
        residuals=_calculate_residuals(Z_exp, Z_fit),
    )


def _prepare_exploratory_function_and_arguments(
    test: str,
    f: Frequencies,
    Z_exp: ComplexImpedances,
    weight: NDArray[float64],
    num_RCs: List[int],
    add_capacitance: bool,
    add_inductance: bool,
    method: str,
    max_nfev: int,
) -> Tuple[Callable, Generator]:
    if test == "cnls":
        return (
            _cnls_test,
            (
                (
                    f,
                    Z_exp,
                    weight,
                    num_RC,
                    add_capacitance,
                    add_inductance,
                    method,
                    max_nfev,
                )
                for num_RC in num_RCs
            ),
        )
    supported_tests: List[str] = [
        "complex",
        "real",
        "imaginary",
    ]
    assert test in supported_tests, f"Unsupported test: '{test}'!"
    return (
        _test_wrapper,
        (
            (
                test,
                f,
                Z_exp,
                weight,
                num_RC,
                add_capacitance,
            )
            for num_RC in num_RCs
        ),
    )


def perform_exploratory_tests(
    data: DataSet,
    test: str = "complex",
    num_RCs: Optional[List[int]] = None,
    mu_criterion: float = 0.85,
    add_capacitance: bool = False,
    add_inductance: bool = False,
    method: str = "leastsq",
    max_nfev: int = -1,
    num_procs: int = 0,
) -> List[TestResult]:
    """
    Performs a batch of linear Kramers-Kronig tests (Boukamp, 1995), which are then scored and sorted from best to worst before they are returned.
    Based on the algorithm described by Schönleber et al. (2014).
    However, the selection of the number of RC elements takes into account factors other than just the applied |mu|-criterion and the |mu| values of the test results.
    This custom scoring system in combination with the ability to plot the intermediate results (i.e., all test results and corresponding |mu| versus the number of RC elements) should help to avoid false negatives that could otherwise occur in some cases.

    References:

    - B.A. Boukamp, 1995, J. Electrochem. Soc., 142, 1885-1894 (https://doi.org/10.1149/1.2044210)
    - M. Schönleber, D. Klotz, and E. Ivers-Tiffée, 2014, Electrochim. Acta, 131, 20-27 (https://doi.org/10.1016/j.electacta.2014.01.034)

    Parameters
    ----------
    data: DataSet
        The data set to be tested.

    test: str, optional
        See |perform_test| for details.

    num_RCs: Optional[List[int]], optional
        A list of integers representing the various number of RC elements to test.
        An empty list results in all possible numbers of RC elements up to the total number of frequencies being tested.

    mu_criterion: float, optional
        See |perform_test| for details.

    add_capacitance: bool, optional
        See |perform_test| for details.

    add_inductance: bool, optional
        See |perform_test| for details.

    method: str, optional
        See |perform_test| for details.

    max_nfev: int, optional
        See |perform_test| for details.

    num_procs: int, optional
        See |perform_test| for details.

    Returns
    -------
    List[TestResult]
    """
    assert hasattr(data, "get_frequencies") and callable(data.get_frequencies)
    assert hasattr(data, "get_impedances") and callable(data.get_impedances)
    assert isinstance(test, str), (
        type(test),
        test,
    )
    if num_RCs is None:
        num_RCs = []
    assert isinstance(num_RCs, list), (
        type(num_RCs),
        num_RCs,
    )
    assert all(map(lambda _: issubdtype(type(_), integer), num_RCs))
    num_points: int = len(data.get_frequencies())
    if len(num_RCs) > 0 and max(num_RCs) > num_points:
        raise KramersKronigError(
            "The maximum value of num_RCs must be less than or equal to the number of data points"
        )
    assert issubdtype(type(mu_criterion), floating), (
        type(mu_criterion),
        mu_criterion,
    )
    if not (0.0 <= mu_criterion <= 1.0):
        raise KramersKronigError(
            "The value of mu_criterion must be between 0.0 and 1.0 (inclusive)"
        )
    assert isinstance(add_capacitance, bool), (
        type(add_capacitance),
        add_capacitance,
    )
    assert isinstance(add_inductance, bool), (
        type(add_inductance),
        add_inductance,
    )
    assert isinstance(method, str), (
        type(method),
        method,
    )
    if method not in _METHODS:
        raise KramersKronigError("Valid method values: '" + "', '".join(_METHODS) + "'")
    assert issubdtype(type(max_nfev), integer), (
        type(max_nfev),
        max_nfev,
    )
    assert issubdtype(type(num_procs), integer), (
        type(num_procs),
        num_procs,
    )
    results: List[TestResult] = []
    if num_procs < 1:
        num_procs = _get_default_num_procs() - abs(num_procs)
        if num_procs < 1:
            num_procs = 1
    f: Frequencies = data.get_frequencies()
    Z_exp: ComplexImpedances = data.get_impedances()
    if len(num_RCs) == 0:
        num_RCs = list(range(1, len(f)))
    num_steps: int = len(num_RCs)
    num_steps += 2  # Calculating weight and preparing arguments
    prog: Progress
    with Progress("Preparing arguments", total=num_steps + 1) as prog:
        weight: NDArray[float64] = _boukamp_weight(Z_exp)
        prog.increment()
        num_RC: int
        func: Callable
        func, args = _prepare_exploratory_function_and_arguments(
            test,
            f,
            Z_exp,
            weight,
            num_RCs,
            add_capacitance,
            add_inductance,
            method,
            max_nfev,
        )
        prog.increment()
        fits: List[Tuple[int, float, Optional[Circuit], float]] = []
        prog.set_message("Performing test(s)")
        if test == "cnls" and num_procs > 1:
            with Pool(num_procs) as pool:
                for res in pool.imap_unordered(func, args):
                    fits.append(res)
                    prog.increment()
        else:
            for res in map(func, args):
                fits.append(res)
                prog.increment()
    mu: float
    circuit: Optional[Circuit]
    Xps: float
    for (num_RC, mu, circuit, Xps) in fits:
        assert circuit is not None
        Z_fit: ComplexImpedances = circuit.get_impedances(f)
        results.append(
            TestResult(
                circuit=circuit,
                num_RC=num_RC,
                mu=mu,
                pseudo_chisqr=Xps,
                frequencies=f,
                impedances=Z_fit,
                # Residuals calculated according to eqs. 15 and 16
                # in Schönleber et al. (2014)
                residuals=_calculate_residuals(Z_exp, Z_fit),
            )
        )
    scores: List[float] = [_.calculate_score(mu_criterion) for _ in results]
    if all(map(isinf, scores)):
        results.sort(key=lambda _: abs(-6 - log(_.pseudo_chisqr)))
    else:
        results.sort(
            key=lambda _: _.calculate_score(mu_criterion),
            reverse=True,
        )
    return results
