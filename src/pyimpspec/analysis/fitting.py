# Copyright 2022 pyimpspec developers
# pyimpspec is licensed under the GPLv3 or later (https://www.gnu.org/licenses/gpl-3.0.html).
# The licenses of pyimpspec's dependencies and/or sources of portions of code are included in
# the LICENSES folder.

from collections import OrderedDict
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from traceback import format_exc
import warnings
from numpy import (
    angle,
    array,
    ceil,
    floor,
    inf,
    log10 as log,
    logspace,
    ndarray,
    ones as ones_array,
    sum as array_sum,
)
from pandas import DataFrame
from lmfit import minimize, Parameters
from lmfit.minimizer import MinimizerResult
from pyimpspec.circuit import Circuit, string_to_circuit
from pyimpspec.circuit.base import Element
from pyimpspec.data import DataSet


class FittingError(Exception):
    pass


VERSION: int = 1


@dataclass(frozen=True)
class FittedParameter:
    """
    An object representing a fitted parameter.

    Properties
    ----------
    value: float
        The fitted value.
    stderr: Optional[float] = None
        The estimated standard error of the fitted value.
    fixed: bool = False
        Whether or not this parameter had a fixed value during the circuit fitting.
    """

    value: float
    stderr: Optional[float] = None
    fixed: bool = False

    @staticmethod
    def _parse_v1(dictionary: dict) -> dict:
        assert type(dictionary) is dict
        return {
            "value": dictionary["value"],
            "stderr": dictionary["stderr"],
            "fixed": dictionary["fixed"],
        }

    @classmethod
    def from_dict(Class, dictionary: dict) -> "FittedParameter":
        assert type(dictionary) is dict
        assert "version" in dictionary
        version: int = dictionary["version"]
        assert version <= VERSION, f"{version=} > {VERSION=}"
        parsers: Dict[int, Callable] = {
            1: Class._parse_v1,
        }
        assert version in parsers, f"{version=} not in {parsers.keys()=}"
        return Class(**parsers[version](dictionary))

    def to_dict(self) -> dict:
        return {
            "version": VERSION,
            "value": self.value,
            "stderr": self.stderr,
            "fixed": self.fixed,
        }


def _interpolate(
    experimental: Union[List[float], ndarray], num_per_decade: int
) -> ndarray:
    assert type(experimental) is list or type(experimental) is ndarray
    assert type(num_per_decade) is int and num_per_decade > 0
    min_f: float = min(experimental)
    max_f: float = max(experimental)
    log_min_f: int = int(floor(log(min_f)))
    log_max_f: int = int(ceil(log(max_f)))
    f: float
    freq: List[float] = [
        f
        for f in logspace(
            log_min_f, log_max_f, num=(log_max_f - log_min_f) * num_per_decade + 1
        )
        if f >= min_f and f <= max_f
    ]
    if min_f not in freq:
        freq.append(min_f)
    if max_f not in freq:
        freq.append(max_f)
    return array(list(sorted(freq, reverse=True)))


@dataclass(frozen=True)
class FittingResult:
    """
    An object representing the results of fitting a circuit to a data set.

    Properties
    ----------
    circuit: Circuit
        The fitted circuit.
    parameters: Dict[str, Dict[str, FittedParameter]]
        Fitted parameters and their estimated standard errors (if possible to estimate).
    pseudo_chisqr: float
        The pseudo chi-squared fit value (eq. 14 in Boukamp, 1995).
    minimizer_result: MinimizerResult
        The results of the fit as provided by the lmfit.minimize function.
    frequency: ndarray
        The frequencies used to perform the fit.
    impedance: ndarray
        The impedance produced by the fitted circuit at each of the fitted frequencies.
    real_residual: ndarray
        The residuals for the real parts (eq. 15 in Schönleber et al., 2014).
    imaginary_residual: ndarray
        The residuals for the imaginary parts (eq. 16 in Schönleber et al., 2014).
    """

    circuit: Circuit
    parameters: Dict[str, Dict[str, FittedParameter]]
    pseudo_chisqr: float
    minimizer_result: MinimizerResult
    frequency: ndarray
    impedance: ndarray
    real_residual: ndarray
    imaginary_residual: ndarray
    method: str
    weight: str

    def __repr__(self) -> str:
        return f"FittingResult ({self.circuit.to_string()}, {hex(id(self))})"

    def get_frequency(self, num_per_decade: int = -1) -> ndarray:
        assert type(num_per_decade) is int
        if num_per_decade > 0:
            return _interpolate(self.frequency, num_per_decade)
        return self.frequency

    def get_impedance(self, num_per_decade: int = -1) -> ndarray:
        assert type(num_per_decade) is int
        if num_per_decade > 0:
            return self.circuit.impedances(self.get_frequency(num_per_decade))
        return self.impedance

    def get_nyquist_data(self, num_per_decade: int = -1) -> Tuple[ndarray, ndarray]:
        """
        Get the data necessary to plot this FittingResult as a Nyquist plot: the real and the
        negative imaginary parts of the impedances.

        Parameters
        ----------
        num_per_decade: int = -1

        Returns
        -------
        Tuple[numpy.ndarray, numpy.ndarray]
        """
        assert type(num_per_decade) is int
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
        Get the data necessary to plot this FittingResult as a Bode plot: the base-10 logarithms of
        the frequencies, the base-10 logarithms of the absolute magnitudes of the impedances, and
        the negative phase angles/shifts of the impedances in degrees.

        Parameters
        ----------
        num_per_decade: int = -1

        Returns
        -------
        Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        """
        assert type(num_per_decade) is int
        if num_per_decade > 0:
            freq: ndarray = self.get_frequency(num_per_decade)
            Z: ndarray = self.circuit.impedances(freq)
            return (
                log(freq),
                log(abs(Z)),
                -angle(Z, deg=True),
            )
        return (
            log(self.frequency),
            log(abs(self.impedance)),
            -angle(self.impedance, deg=True),
        )

    def get_residual_data(self) -> Tuple[ndarray, ndarray, ndarray]:
        """
        Get the data necessary to plot the relative residuals for this FittingResult: the base-10
        logarithms of the frequencies, the relative residuals for the real parts of the impedances
        in percents, and the relative residuals for the imaginary parts of the impedances in
        percents.

        Returns
        -------
        Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        """
        return (
            log(self.frequency),
            self.real_residual * 100,
            self.imaginary_residual * 100,
        )

    def to_dataframe(self) -> DataFrame:
        element_labels: List[str] = []
        parameter_labels: List[str] = []
        fitted_values: List[float] = []
        stderr_values: List[Optional[float]] = []
        fixed: List[str] = []
        element_label: str
        parameters: Dict[str, FittedParameter]
        for element_label, parameters in self.parameters.items():
            parameter_label: str
            parameter: FittedParameter
            for parameter_label, parameter in parameters.items():
                element_labels.append(element_label)
                parameter_labels.append(parameter_label)
                fitted_values.append(parameter.value)
                stderr_values.append(
                    parameter.stderr / parameter.value * 100
                    if parameter.stderr is not None
                    else None
                )
                fixed.append("Yes" if parameter.fixed else "No")
        return DataFrame.from_dict(
            {
                "Element": element_labels,
                "Parameter": parameter_labels,
                "Value": fitted_values,
                "Std. err. (%)": stderr_values,
                "Fixed": fixed,
            }
        )


def _to_lmfit(circuit: Circuit) -> Parameters:
    assert type(circuit) is Circuit
    result: Parameters = Parameters()
    parameters: Dict[int, OrderedDict[str, float]] = circuit.get_parameters()
    ident: int
    params: Dict[str, float]
    for ident, params in parameters.items():
        element: Optional[Element] = circuit.get_element(ident)
        assert element is not None
        element_symbol: str
        symbol: str
        value: float
        for symbol, value in params.items():
            minimum: Any
            maximum: Any
            lower: float = element.get_lower_limit(symbol)
            upper: float = element.get_upper_limit(symbol)
            fixed: bool = element.is_fixed(symbol)
            result.add(f"{symbol}_{ident}", value, min=lower, max=upper, vary=not fixed)
    return result


def _from_lmfit(parameters: Parameters) -> Dict[int, Dict[str, float]]:
    assert type(parameters) is Parameters
    result: Dict[int, Dict[str, float]] = {}
    key: str
    value: float
    for key, value in parameters.valuesdict().items():
        ident: int
        symbol: str
        symbol, ident = key.split("_")  # type: ignore
        ident = int(ident)
        if ident not in result:
            result[ident] = {}
        result[ident][symbol] = float(value)
    return result


def _residual(
    params: Parameters,
    circuit: Circuit,
    freq: ndarray,
    Z_exp: ndarray,
    weight_func: Callable,
) -> ndarray:
    assert type(params) is Parameters
    assert type(circuit) is Circuit
    assert type(freq) is ndarray
    assert type(Z_exp) is ndarray
    circuit.set_parameters(_from_lmfit(params))
    Z_fit: ndarray = circuit.impedances(freq)
    errors: ndarray = array(
        [(Z_exp.real - Z_fit.real) ** 2, (Z_exp.imag - Z_fit.imag) ** 2]
    )
    return weight_func(Z_exp, Z_fit) * errors


def _unity_weight(Z_exp: ndarray, Z_fit: ndarray) -> ndarray:
    assert type(Z_exp) is ndarray
    assert type(Z_fit) is ndarray
    return ones_array(shape=(2, len(Z_exp)))


def _modulus_weight(Z_exp: ndarray, Z_fit: ndarray) -> ndarray:
    assert type(Z_exp) is ndarray
    assert type(Z_fit) is ndarray
    return ones_array(shape=(2, len(Z_exp))) / abs(Z_fit)


def _proportional_weight(Z_exp: ndarray, Z_fit: ndarray) -> ndarray:
    assert type(Z_exp) is ndarray
    assert type(Z_fit) is ndarray
    weight: ndarray = ones_array(shape=(2, len(Z_exp)))
    weight[0] = weight[0] / Z_fit.real**2
    weight[1] = weight[1] / Z_fit.imag**2
    return weight


def _boukamp_weight(Z_exp: ndarray, Z_fit: ndarray) -> ndarray:
    assert type(Z_exp) is ndarray
    assert type(Z_fit) is ndarray
    # See eq. 13 in Boukamp (1995)
    return (Z_exp.real**2 + Z_exp.imag**2) ** -1


_weight_functions: Dict[str, Callable] = {
    "unity": _unity_weight,
    "modulus": _modulus_weight,
    "proportional": _proportional_weight,
    "boukamp": _boukamp_weight,
}


_methods: List[str] = [
    "leastsq",
    "least_squares",
    # "differential_evolution",
    # "brute",
    # "basinhopping",
    # "ampgo",
    "nelder",
    "lbfgsb",
    "powell",
    "cg",
    # "newton",
    # "cobyla",
    "bfgs",
    "tnc",
    # "trust-ncg",
    # "trust-exact",
    # "trust-krylov",
    # "trust-constr",
    # "dogleg",
    "slsqp",
    # "emcee",
    # "shgo",
    # "dual_annealing",
]


def _extract_parameters(
    circuit: Circuit, fit: MinimizerResult
) -> Dict[str, Dict[str, FittedParameter]]:
    assert type(circuit) is Circuit
    assert type(fit) is MinimizerResult
    parameters: Dict[str, Dict[str, FittedParameter]] = {}
    ident: int
    for ident in reversed(circuit.get_parameters()):
        element: Optional[Element] = circuit.get_element(ident)
        assert element is not None
        label: str = element.get_label()
        assert label not in parameters, label
        parameters[label] = {}
        # Parameters that were not fixed
        name: str
        for name in filter(lambda _: _.endswith(f"_{ident}"), fit.var_names):
            par = fit.params[name]
            stderr: Optional[float] = par.stderr if hasattr(par, "stderr") else None
            parameters[label][name[: name.find("_")]] = FittedParameter(
                par.value, stderr
            )
        # Remaining parameters are fixed
        value: float
        for name, value in element.get_parameters().items():
            if name in parameters[label]:
                continue
            parameters[label][name] = FittedParameter(value, None, True)
    return parameters


def _calculate_pseudo_chisqr(Z_exp: ndarray, Z_fit: ndarray) -> float:
    assert type(Z_exp) is ndarray
    assert type(Z_fit) is ndarray
    # See eq. 14 in Boukamp (1995)
    weight: ndarray = _boukamp_weight(Z_exp, Z_fit)
    return float(
        array_sum(
            weight * ((Z_exp.real - Z_fit.real) ** 2 + (Z_exp.imag - Z_fit.imag) ** 2)
        )
    )


def _fit_process(args) -> Tuple[str, Optional[MinimizerResult], float, str, str, str]:
    circuit: Circuit
    freq: ndarray
    Z_exp: ndarray
    method: str
    weight: str
    max_nfev: int
    auto: bool
    circuit, freq, Z_exp, method, weight, max_nfev, auto = args
    assert type(circuit) is Circuit
    assert type(freq) is ndarray
    assert type(Z_exp) is ndarray
    assert type(method) is str
    assert type(weight) is str
    assert type(max_nfev) is int
    assert type(auto) is bool
    weight_func: Callable = _weight_functions[weight]
    with warnings.catch_warnings():
        if auto:
            warnings.filterwarnings("error")
        try:
            fit = minimize(
                _residual,
                _to_lmfit(circuit),
                method,
                args=(circuit, freq, Z_exp, weight_func),
                max_nfev=None if max_nfev < 1 else max_nfev,
            )
        except (Exception, Warning):
            return (
                circuit.to_string(),
                None,
                inf,
                method,
                weight,
                format_exc(),
            )
    if fit.ndata < len(freq) and log(fit.chisqr) < -50:
        return (circuit.to_string(), None, inf, method, weight, "Invalid result!")
    circuit.set_parameters(_from_lmfit(fit.params))
    Z_fit: ndarray = circuit.impedances(freq)
    Xps: float = _calculate_pseudo_chisqr(Z_exp, Z_fit)
    return (
        circuit.to_string(12),
        fit,
        Xps,
        method,
        weight,
        "",
    )


def validate_circuit(circuit: Circuit):
    element_labels: Set[str] = set()
    for element in circuit.get_elements():
        label: str = element.get_label()
        assert (
            label not in element_labels
        ), f"Two or more elements of the same type have the same label ({label})!"
        element_labels.add(label)


def fit_circuit_to_data(
    circuit: Circuit,
    data: DataSet,
    method: str = "auto",
    weight: str = "auto",
    max_nfev: int = -1,
    num_procs: int = -1,
) -> FittingResult:
    """
    Fit a circuit to a data set.

    Parameters
    ----------
    circuit: Circuit
        The circuit to fit to a data set.
    data: DataSet
        The data set that the circuit will be fitted to.
    method: str = "auto"
        The iteration method used during fitting. See lmfit's documentation for valid method
        names. Note that not all methods supported by lmfit are possible in the current
        implementation (e.g. some methods may require a function that calculates a Jacobian).
        The "auto" value results in multiple methods being tested in parallel and the
        best result being returned based on the chi-squared values.
    weight: str = "auto"
        The weight function to use when calculating residuals. Currently supported values:
        "modulus", "proportional", "unity", "boukamp", and "auto". The "auto" value results in
        multiple weights being tested in parallel and the best result being returned based on the
        chi-squared values.
    max_nfev: int = -1
        The maximum number of function evaluations when fitting. A value less than one equals
        no limit.
    num_procs: int = -1
        The maximum number of parallel processes to use when method and/or weight is "auto".

    Returns
    -------
    FittingResult
    """
    assert type(circuit) is Circuit, (
        type(circuit),
        circuit,
    )
    assert type(data) is DataSet, (
        type(data),
        data,
    )
    assert type(method) is str, (
        type(method),
        method,
    )
    assert method in _methods or method == "auto", method
    assert type(weight) is str, (
        type(weight),
        weight,
    )
    assert weight in _weight_functions or weight == "auto", weight
    assert type(max_nfev) is int, (
        type(max_nfev),
        max_nfev,
    )
    assert type(num_procs) is int, (
        type(num_procs),
        num_procs,
    )
    if num_procs < 1:
        num_procs = cpu_count()
    validate_circuit(circuit)
    cdc: str = circuit.to_string(12)
    freq: ndarray = data.get_frequency()
    Z_exp: ndarray = data.get_impedance()
    arguments: List[Tuple[Circuit, ndarray, ndarray, str, str, int, bool]] = []
    fits: List[Tuple[str, Optional[MinimizerResult], float, str, str, str]] = []
    methods: List[str] = [method] if method != "auto" else _methods
    weights: List[str] = (
        [weight] if weight != "auto" else list(_weight_functions.keys())
    )
    for method in methods:
        for weight in weights:
            arguments.append(
                (
                    string_to_circuit(cdc),
                    freq,
                    Z_exp,
                    method,
                    weight,
                    max_nfev,
                    True,
                )
            )
    try:
        if num_procs > 1:
            with Pool(num_procs) as pool:
                fits = pool.map(_fit_process, arguments)
        else:
            fits = list(map(_fit_process, arguments))
        fits.sort(
            key=lambda _: log(_[1].chisqr) + log(_[2]) if _[1] is not None else inf
        )
    except Exception:
        raise FittingError(format_exc())
    if not fits:
        raise FittingError("No valid results generated!")
    fit: Optional[MinimizerResult]
    cdc, fit, Xps, method, weight, error_msg = fits[0]
    if fit is None:
        raise FittingError(error_msg)
    # Return results
    circuit = string_to_circuit(cdc)
    Z_fit: ndarray = circuit.impedances(freq)
    return FittingResult(
        circuit,
        _extract_parameters(circuit, fit),
        Xps,
        fit,
        freq,
        Z_fit,
        # Residuals calculated according to eqs. 15 and 16
        # in Schönleber et al. (2014)
        (Z_exp.real - Z_fit.real) / abs(Z_exp),
        (Z_exp.imag - Z_fit.imag) / abs(Z_exp),
        method,
        weight,
    )
