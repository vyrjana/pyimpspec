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
from multiprocessing import (
    Pool,
    Value,
    cpu_count,
)
from typing import (
    Callable,
    List,
    Tuple,
    Optional,
)
from traceback import format_exc
from lmfit import (
    Parameters,
    minimize,
)
from lmfit.minimizer import MinimizerResult
from numpy import (
    abs,
    angle,
    array,
    float64,
    floating,
    inf,
    integer,
    issubdtype,
    log10 as log,
    min,
    max,
    ndarray,
    pi,
    sum as array_sum,
    zeros,
)
from numpy.linalg import (
    inv,
    pinv,
)
from pyimpspec.analysis.fitting import (
    _from_lmfit,
    _to_lmfit,
    _interpolate,
    _calculate_residuals,
    FittingError,
)
from pyimpspec.circuit import parse_cdc
from pyimpspec.circuit.base import Connection
from pyimpspec.circuit.circuit import Circuit
from pyimpspec.data import DataSet
import pyimpspec.progress as progress


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
        The mu-value of the final fit (eq. 21 in Schönleber et al., 2014).

    pseudo_chisqr: float
        The pseudo chi-squared fit value (eq. 14 in Boukamp, 1995).

    frequency: ndarray
        The frequencies used to perform the test.

    impedance: ndarray
        The impedance produced by the fitted circuit at each of the tested frequencies.

    real_residual: ndarray
        The residuals for the real parts (eq. 15 in Schönleber et al., 2014).

    imaginary_residual: ndarray
        The residuals for the imaginary parts (eq. 16 in Schönleber et al., 2014).
    """

    circuit: Circuit
    num_RC: int
    mu: float
    pseudo_chisqr: float
    frequency: ndarray
    impedance: ndarray
    real_residual: ndarray
    imaginary_residual: ndarray

    def __repr__(self) -> str:
        return f"TestResult (num_RC={self.num_RC}, {hex(id(self))})"

    def get_frequency(self, num_per_decade: int = -1) -> ndarray:
        assert issubdtype(type(num_per_decade), integer), num_per_decade
        if num_per_decade > 0:
            return _interpolate(self.frequency, num_per_decade)
        return self.frequency

    def get_impedance(self, num_per_decade: int = -1) -> ndarray:
        assert issubdtype(type(num_per_decade), integer), num_per_decade
        if num_per_decade > 0:
            return self.circuit.impedances(self.get_frequency(num_per_decade))
        return self.impedance

    def get_nyquist_data(self, num_per_decade: int = -1) -> Tuple[ndarray, ndarray]:
        """
        Get the data necessary to plot this TestResult as a Nyquist plot: the real and the negative imaginary parts of the impedances.

        Parameters
        ----------
        num_per_decade: int = -1

        Returns
        -------
        Tuple[ndarray, ndarray]
        """
        assert issubdtype(type(num_per_decade), integer), num_per_decade
        if num_per_decade > 0:
            Z: ndarray = self.get_impedance(num_per_decade)
            return (
                Z.real,
                -Z.imag,
            )
        return (
            self.impedance.real,
            -self.impedance.imag,
        )

    def get_bode_data(
        self, num_per_decade: int = -1
    ) -> Tuple[ndarray, ndarray, ndarray]:
        """
        Get the data necessary to plot this TestResult as a Bode plot: the frequencies, the absolute magnitudes of the impedances, and the negative phase angles/shifts of the impedances in degrees.

        Parameters
        ----------
        num_per_decade: int = -1

        Returns
        -------
        Tuple[ndarray, ndarray, ndarray]
        """
        assert issubdtype(type(num_per_decade), integer), num_per_decade
        if num_per_decade > 0:
            freq: ndarray = self.get_frequency(num_per_decade)
            Z: ndarray = self.circuit.impedances(freq)
            return (
                freq,
                abs(Z),
                -angle(Z, deg=True),
            )
        return (
            self.frequency,
            abs(self.impedance),
            -angle(self.impedance, deg=True),
        )

    def get_residual_data(self) -> Tuple[ndarray, ndarray, ndarray]:
        """
        Get the data necessary to plot the relative residuals for this TestResult: the frequencies, the relative residuals for the real parts of the impedances in percents, and the relative residuals for the imaginary parts of the impedances in percents.

        Returns
        -------
        Tuple[ndarray, ndarray, ndarray]
        """
        return (
            self.frequency,
            self.real_residual * 100,
            self.imaginary_residual * 100,
        )

    def calculate_score(self, mu_criterion: float) -> float:
        """
        Calculate a score based on the provided mu-criterion and the statistics of the result.
        A result with a mu-value greater than or equal to the mu-criterion will get a score of -numpy.inf.

        Parameters
        ----------
        mu_criterion: float
            The mu-criterion to apply.
            See perform_test for details.

        Returns
        -------
        float
        """
        return (
            -inf
            if self.mu >= mu_criterion
            else -log(self.pseudo_chisqr) / (abs(mu_criterion - self.mu) ** 0.75)
        )


def _calculate_tau(i: int, num_RC: int, tau_min: float64, tau_max: float64) -> float:
    # Calculate time constants according to eq. 12 in Schönleber et al. (2014)
    assert issubdtype(type(i), integer), i
    assert issubdtype(type(num_RC), integer), num_RC
    assert issubdtype(type(tau_min), floating), tau_min
    assert issubdtype(type(tau_max), floating), tau_max
    return pow(10, (log(tau_min) + (i - 1) / (num_RC - 1) * log(tau_max / tau_min)))


def _generate_time_constants(w: ndarray, num_RC: int) -> ndarray:
    assert type(w) is ndarray, w
    assert issubdtype(type(num_RC), integer), num_RC
    taus: ndarray = zeros(shape=(num_RC,))
    tau_min: float64 = 1 / max(w)
    tau_max: float64 = 1 / min(w)
    taus[0] = tau_min
    taus[-1] = tau_max
    i: int
    if num_RC > 1:
        for i in range(2, num_RC):
            taus[i - 1] = _calculate_tau(i, num_RC, tau_min, tau_max)
    return taus


def _generate_weight(Z_exp: ndarray) -> ndarray:
    assert type(Z_exp) is ndarray, Z_exp
    # See eq. 13 in Boukamp (1995)
    return (Z_exp.real**2 + Z_exp.imag**2) ** -1


def _calculate_pseudo_chisqr(Z_exp: ndarray, Z_fit: ndarray, weight: ndarray) -> float:
    assert type(Z_exp) is ndarray, Z_exp
    assert type(Z_fit) is ndarray, Z_fit
    assert type(weight) is ndarray, weight
    # See eq. 14 in Boukamp (1995)
    return float(
        array_sum(
            weight * ((Z_exp.real - Z_fit.real) ** 2 + (Z_exp.imag - Z_fit.imag) ** 2)
        )
    )


def _calculate_mu(params: Parameters) -> float:
    assert type(params) is Parameters, params
    # Calculates the mu-value based on the fitted parameters according to eq. 21 in
    # Schönleber et al. (2014)
    R_neg: List[float] = []
    R_pos: List[float] = []
    name: str
    value: float
    for name, value in params.valuesdict().items():
        if not name.startswith("R"):
            continue
        elif name == "R0":
            continue
        if value < 0:
            R_neg.append(abs(value))
        else:
            R_pos.append(value)
    neg_sum: float = sum(R_neg)
    pos_sum: float = sum(R_pos)
    if pos_sum == 0:
        return 0.0
    mu: float = 1.0 - neg_sum / pos_sum
    if mu < 0.0:
        return 0.0
    elif mu > 1.0:
        return 1.0
    return mu


def _elements_to_parameters(
    elements: ndarray, taus: ndarray, add_capacitance: bool
) -> Parameters:
    assert type(elements) is ndarray, elements
    assert type(taus) is ndarray, taus
    assert type(add_capacitance) is bool, add_capacitance
    parameters: Parameters = Parameters()
    R0: float
    R0, elements = elements[0], elements[1:]
    parameters.add("R_0", R0, vary=False)
    L: float
    L, elements = elements[-1], elements[:-1]
    C: float
    if add_capacitance:
        C, elements = 1 / elements[-1], elements[:-1]
    for i, (R, t) in enumerate(zip(elements, taus), start=1):
        parameters.add(f"R_{i}", R, vary=False)
        parameters.add(f"t_{i}", t, vary=False)
    if add_capacitance:
        i += 1
        parameters.add(f"C_{i}", C)
    i += 1
    parameters.add(f"L_{i}", L, vary=False)
    return parameters


def _complex_residual(
    params: Parameters, circuit: Circuit, freq: ndarray, Z_exp: ndarray, weight: ndarray
) -> ndarray:
    assert type(params) is Parameters, params
    assert type(circuit) is Circuit, circuit
    assert type(freq) is ndarray, freq
    assert type(Z_exp) is ndarray, Z_exp
    assert type(weight) is ndarray, weight
    circuit.set_parameters(_from_lmfit(params))
    Z_fit: ndarray = circuit.impedances(freq)
    return array(
        [
            (weight * (Z_exp.real - Z_fit.real)) ** 2,
            (weight * (Z_exp.imag - Z_fit.imag)) ** 2,
        ]
    )


def _cnls_test(args: tuple) -> Tuple[int, float, Circuit, float]:
    freq: ndarray
    Z_exp: ndarray
    weight: ndarray
    num_RC: int
    add_capacitance: bool
    add_inductance: bool
    method: str
    max_nfev: int
    (
        freq,
        Z_exp,
        weight,
        num_RC,
        add_capacitance,
        add_inductance,
        method,
        max_nfev,
    ) = args
    assert type(freq) is ndarray, freq
    assert type(Z_exp) is ndarray, Z_exp
    assert type(weight) is ndarray, weight
    assert issubdtype(type(num_RC), integer), num_RC
    assert type(add_capacitance) is bool, add_capacitance
    assert type(add_inductance) is bool, add_inductance
    assert type(method) is str, method
    assert issubdtype(type(max_nfev), integer), max_nfev
    w: ndarray = 2 * pi * freq
    taus: ndarray = _generate_time_constants(w, num_RC)
    circuit: Circuit = _generate_circuit(taus, add_capacitance, add_inductance, True)
    fit: MinimizerResult
    try:
        fit = minimize(
            _complex_residual,
            _to_lmfit(circuit),
            method,
            args=(
                circuit,
                freq,
                Z_exp,
                weight,
            ),
            max_nfev=None if max_nfev < 1 else max_nfev,
        )
    except Exception:
        raise FittingError(format_exc())
    circuit.set_parameters(_from_lmfit(fit.params))
    mu: float = _calculate_mu(fit.params)
    Z_fit: ndarray = circuit.impedances(freq)
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
    global pool_optimal_num_RC
    freq: ndarray
    Z_exp: ndarray
    weight: ndarray
    mu_criterion: float
    num_RC: int
    add_capacitance: bool
    add_inductance: bool
    method: str
    max_nfev: int
    (
        freq,
        Z_exp,
        weight,
        mu_criterion,
        num_RC,
        add_capacitance,
        add_inductance,
        method,
        max_nfev,
    ) = args
    assert type(freq) is ndarray, freq
    assert type(Z_exp) is ndarray, Z_exp
    assert type(weight) is ndarray, weight
    assert issubdtype(type(mu_criterion), floating), mu_criterion
    assert issubdtype(type(num_RC), integer), num_RC
    assert type(add_capacitance) is bool, add_capacitance
    assert type(add_inductance) is bool, add_inductance
    assert type(method) is str, method
    assert issubdtype(type(max_nfev), integer), max_nfev

    def exit_early(params: Parameters, i: int, _, *args, **kwargs):
        if 0 <= pool_optimal_num_RC.value < num_RC:  # type: ignore
            return True
        return None

    w: ndarray = 2 * pi * freq
    taus: ndarray = _generate_time_constants(w, num_RC)
    circuit: Circuit = _generate_circuit(taus, add_capacitance, add_inductance, True)
    if 0 <= pool_optimal_num_RC.value < num_RC:  # type: ignore
        return (num_RC, -1.0, None, -1.0)
    fit: MinimizerResult
    try:
        fit = minimize(
            _complex_residual,
            _to_lmfit(circuit),
            method,
            args=(
                circuit,
                freq,
                Z_exp,
                weight,
            ),
            max_nfev=None if max_nfev < 1 else max_nfev,
            iter_cb=exit_early,
        )
    except Exception:
        raise FittingError(format_exc())
    if 0 <= pool_optimal_num_RC.value < num_RC:  # type: ignore
        return (num_RC, -1.0, None, -1.0)
    circuit.set_parameters(_from_lmfit(fit.params))
    mu: float = _calculate_mu(fit.params)
    Z_fit: ndarray = circuit.impedances(freq)
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
    w: ndarray, num_RC: int, taus: ndarray, add_capacitance: bool, abs_Z_exp: ndarray
) -> Tuple[ndarray, ndarray]:
    assert type(w) is ndarray, w
    assert issubdtype(type(num_RC), integer), num_RC
    assert type(taus) is ndarray, taus
    assert type(add_capacitance) is bool, add_capacitance
    assert type(abs_Z_exp) is ndarray, abs_Z_exp
    # Generate matrices with the following columns (top to bottom is left to right)
    # - R0, series resistance
    # - Ri, resistance in parallel with taus[i - 1] where 0 < i <= num_RC
    # - (C, optional series capacitance)
    # - L, series inductance
    if add_capacitance is True:
        a_re = zeros((w.size, num_RC + 3))
        a_im = zeros((w.size, num_RC + 3))
        # Series capacitance
        a_im[:, -2] = -1 / (w * abs_Z_exp)  # No real part
    else:
        a_re = zeros((w.size, num_RC + 2))
        a_im = zeros((w.size, num_RC + 2))
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
    taus: ndarray, add_capacitance: bool, add_inductance: bool, cnls: bool
) -> Circuit:
    assert type(taus) is ndarray, taus
    assert type(add_capacitance) is bool, add_capacitance
    assert type(add_inductance) is bool, add_inductance
    assert type(cnls) is bool, cnls
    cdc: List[str] = ["R{R=1}"]
    t: float
    for t in taus:
        cdc.append(f"K{{R=1,t={t}F}}")
    if add_capacitance is True:
        cdc.append("C{C=1e-6}")
    if add_inductance is True:
        cdc.append("L{L=1e-3}")
    circuit: Circuit = parse_cdc("".join(cdc))
    if cnls is True:
        for element in circuit.get_elements():
            for parameter in element.get_parameters():
                if isinstance(parameter, Connection):
                    continue
                element.set_lower_limit(parameter, -inf)  # type: ignore
                element.set_upper_limit(parameter, inf)  # type: ignore
    return circuit


def _real_test(
    a_re: ndarray,
    a_im: ndarray,
    Z_exp: ndarray,
    abs_Z_exp: ndarray,
    w: ndarray,
    freq: ndarray,
    taus: ndarray,
    add_capacitance: bool,
    circuit: Circuit,
):
    assert type(a_re) is ndarray, a_re
    assert type(a_im) is ndarray, a_im
    assert type(Z_exp) is ndarray, Z_exp
    assert type(abs_Z_exp) is ndarray, abs_Z_exp
    assert type(w) is ndarray, w
    assert type(freq) is ndarray, freq
    assert type(taus) is ndarray, taus
    assert type(add_capacitance) is bool, add_capacitance
    assert type(circuit) is Circuit, circuit
    # Fit using the real part
    elements: ndarray = pinv(a_re).dot(Z_exp.real / abs_Z_exp)
    # Fit using the imaginary part to fix the series inductance (and capacitance)
    a_im = zeros((w.size, 2))
    a_im[:, -1] = w / abs_Z_exp
    if add_capacitance:
        a_im[:, -2] = -1 / (w * abs_Z_exp)
        elements[-2] = 1e-18  # Nullifies the series capacitance without dividing by 0
    circuit.set_parameters(
        _from_lmfit(_elements_to_parameters(elements, taus, add_capacitance))
    )
    Z_fit: ndarray = circuit.impedances(freq)
    coefs: ndarray = pinv(a_im).dot((Z_exp.imag - Z_fit.imag) / abs_Z_exp)
    # Extract the corrected series inductance (and capacitance)
    if add_capacitance:
        elements[-2:] = coefs
    else:
        elements[-1] = coefs[-1]
    circuit.set_parameters(
        _from_lmfit(_elements_to_parameters(elements, taus, add_capacitance))
    )


def _imaginary_test(
    a_im: ndarray,
    Z_exp: ndarray,
    abs_Z_exp: ndarray,
    freq: ndarray,
    taus: ndarray,
    add_capacitance: bool,
    weight: ndarray,
    circuit: Circuit,
):
    assert type(a_im) is ndarray, a_im
    assert type(Z_exp) is ndarray, Z_exp
    assert type(abs_Z_exp) is ndarray, abs_Z_exp
    assert type(freq) is ndarray, freq
    assert type(taus) is ndarray, taus
    assert type(add_capacitance) is bool, add_capacitance
    assert type(weight) is ndarray, weight
    assert type(circuit) is Circuit, circuit
    # Fit using the imaginary part
    elements: ndarray = pinv(a_im).dot(Z_exp.imag / abs_Z_exp)
    # Estimate the series resistance
    circuit.set_parameters(
        _from_lmfit(_elements_to_parameters(elements, taus, add_capacitance))
    )
    Z_fit: ndarray = circuit.impedances(freq)
    elements[0] = array_sum(weight * (Z_exp.real - Z_fit.real)) / array_sum(weight)
    circuit.set_parameters(
        _from_lmfit(_elements_to_parameters(elements, taus, add_capacitance))
    )


def _complex_test(
    a_re: ndarray,
    a_im: ndarray,
    Z_exp: ndarray,
    abs_Z_exp: ndarray,
    taus: ndarray,
    add_capacitance: bool,
    circuit: Circuit,
):
    assert type(a_re) is ndarray, a_re
    assert type(a_im) is ndarray, a_im
    assert type(Z_exp) is ndarray, Z_exp
    assert type(abs_Z_exp) is ndarray, abs_Z_exp
    assert type(taus) is ndarray, taus
    assert type(add_capacitance) is bool, add_capacitance
    assert type(circuit) is Circuit, circuit
    # Fit using the complex impedance
    x: ndarray = inv(a_re.T.dot(a_re) + a_im.T.dot(a_im))
    y: ndarray = a_re.T.dot(Z_exp.real / abs_Z_exp) + a_im.T.dot(Z_exp.imag / abs_Z_exp)
    elements: ndarray = x.dot(y)
    circuit.set_parameters(
        _from_lmfit(_elements_to_parameters(elements, taus, add_capacitance))
    )


def _test_wrapper(args: tuple) -> Tuple[int, float, Circuit, float]:
    test: str
    freq: ndarray
    Z_exp: ndarray
    weight: ndarray
    num_RC: int
    add_capacitance: bool
    test, freq, Z_exp, weight, num_RC, add_capacitance = args
    assert type(test) is str, test
    assert type(freq) is ndarray, freq
    assert type(Z_exp) is ndarray, Z_exp
    assert type(weight) is ndarray, weight
    assert issubdtype(type(num_RC), integer), num_RC
    assert type(add_capacitance) is bool, add_capacitance
    abs_Z_exp: ndarray = abs(Z_exp)
    w: ndarray = 2 * pi * freq
    taus: ndarray = _generate_time_constants(w, num_RC)
    a_re: ndarray
    a_im: ndarray
    a_re, a_im = _generate_variable_matrices(
        w, num_RC, taus, add_capacitance, abs_Z_exp
    )
    circuit: Circuit = _generate_circuit(taus, add_capacitance, True, False)
    # Solve the set of linear equations and update the circuit's parameters
    if test == "real":
        _real_test(
            a_re, a_im, Z_exp, abs_Z_exp, w, freq, taus, add_capacitance, circuit
        )
    elif test == "imaginary":
        _imaginary_test(
            a_im, Z_exp, abs_Z_exp, freq, taus, add_capacitance, weight, circuit
        )
    elif test == "complex":
        _complex_test(a_re, a_im, Z_exp, abs_Z_exp, taus, add_capacitance, circuit)
    # Calculate return values
    mu: float = _calculate_mu(_to_lmfit(circuit))
    Z_fit: ndarray = circuit.impedances(freq)
    Xps: float = _calculate_pseudo_chisqr(Z_exp, Z_fit, weight)
    return (
        num_RC,
        mu,
        circuit,
        Xps,
    )


def perform_test(
    data: DataSet,
    test: str = "complex",
    num_RC: int = 0,
    mu_criterion: float = 0.85,
    add_capacitance: bool = False,
    add_inductance: bool = False,
    method: str = "leastsq",
    max_nfev: int = -1,
    num_procs: int = -1,
) -> TestResult:
    """
    Performs a linear Kramers-Kronig test as described by Boukamp (1995).
    The results can be used to check the validity of an impedance spectrum before performing equivalent circuit fitting.
    If the number of RC elements is less than two, then a suitable number of RC elements is determined using the procedure described by Schönleber et al. (2014) based on a criterion for the calculated mu-value (zero to one).
    A mu-value of one represents underfitting and a mu-value of zero represents overfitting.

    References:

    - B.A. Boukamp, 1995, J. Electrochem. Soc., 142, 1885-1894 (https://doi.org/10.1149/1.2044210)
    - M. Schönleber, D. Klotz, and E. Ivers-Tiffée, 2014, Electrochim. Acta, 131, 20-27 (https://doi.org/10.1016/j.electacta.2014.01.034)

    Parameters
    ----------
    data: DataSet
        The data set to be tested.

    test: str = "complex"
        Supported values include "cnls", "complex", "imaginary", and "real". The "cnls" test performs a complex non-linear least squares fit using lmfit.minimize, which usually provides a good fit but is also quite slow.
        The "complex", "imaginary", and "real" tests perform the complex, imaginary, and real tests, respectively, according to Boukamp (1995).

    num_RC: int = 0
        The number of RC elements to use.
        A value greater than or equal to one results in the specific number of RC elements being tested.
        A value less than one results in the use of the procedure described by Schönleber et al. (2014) based on the chosen mu-criterion.
        If the provided value is negative, then the maximum number of RC elements to test is equal to the absolute value of the provided value.
        If the provided value is zero, then the maximum number of RC elements to test is equal to the number of frequencies in the data set.

    mu_criterion: float = 0.85
        The chosen mu-criterion. See Schönleber et al. (2014) for more information.

    add_capacitance: bool = False
        Add an additional capacitance in series with the rest of the circuit.

    add_inductance: bool = False
        Add an additional inductance in series with the rest of the circuit.
        Applies only to the "cnls" test.

    method: str = "leastsq"
        The fitting method to use when performing a "cnls" test.
        See the list of methods that are listed in the documentation for the lmfit package.
        Methods that do not require providing bounds for all parameters or a function to calculate the Jacobian should work.

    max_nfev: int = -1
        The maximum number of function evaluations when fitting.
        A value less than one equals no limit.
        Applies only to the "cnls" test.

    num_procs: int = -1
        The maximum number of parallel processes to use when performing a test.
        A value less than one results in using the number of cores returned by multiprocessing.cpu_count.
        Applies only to the "cnls" test.

    Returns
    -------
    TestResult
    """
    assert isinstance(data, DataSet), (
        type(data),
        data,
    )
    assert data.get_num_points() > 0
    assert type(test) is str, (
        type(test),
        test,
    )
    assert issubdtype(type(num_RC), integer), (
        type(num_RC),
        num_RC,
    )
    assert num_RC <= data.get_num_points(), f"{num_RC=} > {data.get_num_points()=}"
    assert issubdtype(type(mu_criterion), floating), (
        type(mu_criterion),
        mu_criterion,
    )
    if num_RC <= 0:
        assert mu_criterion >= 0.0 and mu_criterion <= 1.0, mu_criterion
    assert type(add_capacitance) is bool, (
        type(add_capacitance),
        add_capacitance,
    )
    assert type(add_inductance) is bool, (
        type(add_inductance),
        add_inductance,
    )
    assert type(method) is str, (
        type(method),
        method,
    )
    assert issubdtype(type(max_nfev), integer), (
        type(max_nfev),
        max_nfev,
    )
    assert issubdtype(type(num_procs), integer), (
        type(num_procs),
        num_procs,
    )
    if num_procs < 1:
        num_procs = cpu_count()
    freq: ndarray = data.get_frequency()
    Z_exp: ndarray = data.get_impedance()
    weight: ndarray = _generate_weight(Z_exp)
    params: Parameters
    mu: float
    circuit: Optional[Circuit] = None
    Xps: float  # pseudo chi-squared
    func: Callable
    fits: List[Tuple[int, float, Circuit, float]] = []
    progress_message: str = "Preparing arguments"
    progress.update_every_N_percent(0, message=progress_message)
    i: int
    num_RCs: List[int]
    if test == "cnls":
        if num_RC > 0:
            # Perform the test with a specific number of RC elements
            args = [
                (
                    freq,
                    Z_exp,
                    weight,
                    num_RC,
                    add_capacitance,
                    add_inductance,
                    method,
                    max_nfev,
                )
            ]
            progress_message = "Performing test"
            progress.update_every_N_percent(0, message=progress_message)
            try:
                if num_procs > 1:
                    # To prevent the GUI thread of DearEIS from locking up
                    with Pool(1) as pool:
                        fits = pool.map(_cnls_test, args)
                else:
                    fits = list(map(_cnls_test, args))
            except Exception:
                raise FittingError(format_exc())
            progress.update_every_N_percent(1, message=progress_message)
            num_RC, mu, circuit, Xps = fits[0]
        else:
            num_RC = abs(num_RC) or len(freq)
            num_RCs = list(range(1, num_RC + 1))
            # Find an appropriate number of RC elements based on the calculated mu-value and the
            # provided threshold value. Use multiple parallel processes if possible.
            args = (
                (
                    freq,
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
            progress_message = "Performing test(s)"
            progress.update_every_N_percent(0, message=progress_message)
            try:
                if num_procs > 1:
                    with Pool(
                        num_procs,
                        initializer=_pool_init,
                        initargs=(Value("i", -1),),
                    ) as pool:
                        for i, res in enumerate(
                            pool.imap_unordered(_cnls_mu_process, args)
                        ):
                            progress.update_every_N_percent(
                                i + 1,
                                total=len(num_RCs),
                                message=progress_message,
                            )
                            if res[2] is not None:
                                fits.append(res)
                else:
                    _pool_init(Value("i", -1))
                    for i, res in enumerate(map(_cnls_mu_process, args)):
                        progress.update_every_N_percent(
                            i + 1,
                            total=len(num_RCs),
                            message=progress_message,
                        )
                        if res[2] is not None:
                            fits.append(res)
            except Exception:
                raise FittingError(format_exc())
            for (num_RC, mu, circuit, Xps) in sorted(fits, key=lambda _: _[0]):
                if mu < mu_criterion:
                    break
    else:
        supported_tests: List[str] = [
            "complex",
            "real",
            "imaginary",
        ]
        if test not in supported_tests:
            raise FittingError(f"Unsupported test: '{test}'")
        if num_RC > 0:
            num_RCs = [num_RC]
        else:
            num_RC = abs(num_RC) or len(freq)
            num_RCs = list(range(1, num_RC + 1))
        args = (
            (
                test,
                freq,
                Z_exp,
                weight,
                num_RC,
                add_capacitance,
            )
            for num_RC in num_RCs
        )
        progress_message = "Performing test(s)"
        progress.update_every_N_percent(0, message=progress_message)
        try:
            for i, res in enumerate(map(_test_wrapper, args)):
                progress.update_every_N_percent(
                    i + 1,
                    total=len(num_RCs),
                    message=progress_message,
                )
                fits.append(res)
        except Exception:
            raise FittingError(format_exc())
        if len(fits) == 1:
            num_RC, mu, circuit, Xps = fits[0]
        else:
            for (num_RC, mu, circuit, Xps) in sorted(fits, key=lambda _: _[0]):
                if mu < mu_criterion:
                    break
    # ========== Result ==========
    assert circuit is not None
    Z_fit: ndarray = circuit.impedances(freq)
    return TestResult(
        circuit,
        num_RC,
        mu,
        Xps,
        freq,
        Z_fit,
        # Residuals calculated according to eqs. 15 and 16
        # in Schönleber et al. (2014)
        *_calculate_residuals(Z_exp, Z_fit),
    )


def perform_exploratory_tests(
    data: DataSet,
    test: str = "complex",
    num_RCs: List[int] = [],
    mu_criterion: float = 0.85,
    add_capacitance: bool = False,
    add_inductance: bool = False,
    method: str = "leastsq",
    max_nfev: int = -1,
    num_procs: int = -1,
) -> List[TestResult]:
    """
    Performs a batch of linear Kramers-Kronig tests, which are then scored and sorted from best to worst before they are returned.

    Parameters
    ----------
    data: DataSet
        The data set to be tested.

    test: str = "complex"
        See perform_test for details.

    num_RCs: List[int] = []
        A list of integers representing the various number of RC elements to test.
        An empty list results in all possible numbers of RC elements up to the total number of frequencies being tested.

    mu_criterion: float = 0.85
        See perform_test for details.

    add_capacitance: bool = False
        See perform_test for details.

    add_inductance: bool = False
        See perform_test for details.

    method: str = "leastsq"
        See perform_test for details.

    max_nfev: int = -1
        See perform_test for details.

    num_procs: int = -1
        See perform_test for details.

    Returns
    -------
    List[TestResult]
    """
    assert isinstance(data, DataSet), (
        type(data),
        data,
    )
    assert data.get_num_points() > 0
    assert type(test) is str, (
        type(test),
        test,
    )
    assert type(num_RCs) is list, (
        type(num_RCs),
        num_RCs,
    )
    assert all(map(lambda _: issubdtype(type(_), integer), num_RCs))
    if len(num_RCs) > 0:
        assert (
            max(num_RCs) <= data.get_num_points()
        ), f"{max(num_RCs)=} > {data.get_num_points()=}"
    assert issubdtype(type(mu_criterion), floating), (
        type(mu_criterion),
        mu_criterion,
    )
    assert mu_criterion >= 0.0 and mu_criterion <= 1.0, mu_criterion
    assert type(add_capacitance) is bool, (
        type(add_capacitance),
        add_capacitance,
    )
    assert type(add_inductance) is bool, (
        type(add_inductance),
        add_inductance,
    )
    assert type(method) is str, (
        type(method),
        method,
    )
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
        num_procs = cpu_count()
    freq: ndarray = data.get_frequency()
    Z_exp: ndarray = data.get_impedance()
    weight: ndarray = _generate_weight(Z_exp)
    if len(num_RCs) == 0:
        num_RCs = list(range(1, len(freq)))
    params: Parameters
    num_RC: int
    func: Callable
    progress_message: str = "Preparing arguments"
    progress.update_every_N_percent(0, message=progress_message)
    if test == "cnls":
        func = _cnls_test
        args = (
            (
                freq,
                Z_exp,
                weight,
                num_RC,
                add_capacitance,
                add_inductance,
                method,
                max_nfev,
            )
            for num_RC in num_RCs
        )
    else:
        supported_tests: List[str] = [
            "complex",
            "real",
            "imaginary",
        ]
        if test not in supported_tests:
            raise FittingError(f"Unsupported test: '{test}'")
        func = _test_wrapper
        args = (
            (
                test,
                freq,
                Z_exp,
                weight,
                num_RC,
                add_capacitance,
            )
            for num_RC in num_RCs
        )
    fits: List[Tuple[int, float, Circuit, float]] = []
    progress_message = "Performing test(s)"
    progress.update_every_N_percent(0, message=progress_message)
    try:
        if test == "cnls" and num_procs > 1:
            with Pool(num_procs) as pool:
                for i, res in enumerate(pool.imap_unordered(func, args)):
                    progress.update_every_N_percent(
                        i + 1,
                        total=len(num_RCs),
                        message=progress_message,
                    )
                    fits.append(res)
        else:
            for i, res in enumerate(map(func, args)):
                progress.update_every_N_percent(
                    i + 1,
                    total=len(num_RCs),
                    message=progress_message,
                )
                fits.append(res)
    except Exception:
        raise FittingError(format_exc())
    mu: float
    circuit: Circuit
    Xps: float
    for (num_RC, mu, circuit, Xps) in fits:
        Z_fit: ndarray = circuit.impedances(freq)
        results.append(
            TestResult(
                circuit,
                num_RC,
                mu,
                Xps,
                freq,
                Z_fit,
                # Residuals calculated according to eqs. 15 and 16
                # in Schönleber et al. (2014)
                *_calculate_residuals(Z_exp, Z_fit),
            )
        )
    results.sort(
        key=lambda _: _.calculate_score(mu_criterion),
        reverse=True,
    )
    return results
