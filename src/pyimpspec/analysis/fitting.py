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

from copy import deepcopy
from dataclasses import dataclass
from multiprocessing import Pool
from multiprocessing.context import TimeoutError as MPTimeoutError
from types import SimpleNamespace
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)
from traceback import format_exc
from warnings import (
    catch_warnings,
    filterwarnings,
)
from numpy import (
    angle,
    array,
    float64,
    inf,
    isnan,
    log10 as log,
    nan,
    ones,
)
from numpy.typing import NDArray
from pyimpspec.exceptions import FittingError
from pyimpspec.analysis.utility import (
    _calculate_pseudo_chisqr,
    _calculate_residuals,
    _interpolate,
    get_default_num_procs,
)
from pyimpspec.circuit import Circuit
from pyimpspec.circuit.base import Element
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
from pyimpspec.typing.helpers import (
    _is_boolean,
    _is_complex_array,
    _is_floating,
    _is_floating_array,
    _is_integer,
)


class FitIdentifiers(SimpleNamespace):
    """
    An object that maps an |Element|'s parameters to the name/identifier used by the corresponding `lmfit Parameter <https://lmfit.github.io/lmfit-py/parameters.html>`_. The values can be accessed in two ways:

    - ``name: str = identifiers.R``
    - ``name: str = identifiers["R"]``

    In this example, the |FitIdentifiers| instance ``identifiers`` was generated based on a |Resistor| instance. Thus, ``identifiers`` has a property called ``R``.

    It is also possible to do the following as if ``identifiers`` was a dictionary:

    - ``{symbol: identifiers[symbol] for symbol in identifiers}``
    - ``{symbol: name for symbol, name in identifiers.items()}``
    """

    def __iter__(self):
        return iter(self.__dict__)

    def items(self):
        return self.__dict__.items()

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def __getitem__(self, key: str):
        return getattr(self, key)


@dataclass(frozen=True)
class FittedParameter:
    """
    An object representing a fitted parameter.

    Parameters
    ----------
    value: float
        The fitted value.

    stderr: float
        The estimated standard error of the fitted value. If the value is numpy.nan, then the standard error could not be estimated.

    fixed: bool
        Whether or not this parameter had a fixed value during the circuit fitting.

    unit: str
        The parameter's unit.
    """

    value: float
    stderr: float
    fixed: bool
    unit: str

    def __str__(self) -> str:
        string: str = f"{self.value:.6e}"
        if not isnan(self.stderr):
            string += f" +/- {self.stderr:.6e}"

        if self.unit != "":
            string += f" {self.unit}"

        if self.fixed:
            string += " (fixed)"

        return string

    def get_value(self) -> float:
        """
        Get the fitted value of this parameter.

        Returns
        -------
        float
        """
        return self.value

    def get_error(self) -> float:
        """
        Get the estimated absolute standard error of this parameter or numpy.nan if it was not possible to provide an estimate.

        Returns
        -------
        float
        """
        return self.stderr

    def is_fixed(self) -> bool:
        """
        Check whether or not this parameter was fixed during the fitting process.

        Returns
        -------
        bool
        """
        return self.fixed

    def get_unit(self) -> str:
        """
        Get the unit of this parameter if it has one.

        Returns
        -------
        str
        """
        return self.unit

    def get_relative_error(self) -> float:
        """
        Get the estimated relative standard error of this parameter or numpy.nan if it was not possible to estimate.

        Returns
        -------
        float
        """
        if isnan(self.stderr):
            return self.stderr
        elif self.value == 0.0:
            return nan

        return (self.stderr or 0.0) / self.value


@dataclass(frozen=True)
class FitResult:
    """
    An object representing the results of fitting a circuit to a data set.

    Parameters
    ----------
    circuit: |Circuit|
        The fitted circuit.

    parameters: Dict[str, Dict[str, |FittedParameter|]]
        Fitted parameters and their estimated standard errors (if possible to estimate).

    minimizer_result: `lmfit.MinimizerResult`_
        The results of the fit as provided by the `lmfit.minimize`_ function.

    frequencies: |Frequencies|
        The frequencies used to perform the fit.

    impedances: |ComplexImpedances|
        The impedances produced by the fitted circuit at each of the frequencies.

    residuals: |ComplexResiduals|
        The residuals for the real (eq. 15 in Schönleber et al., 2014) and imaginary (eq. 16 in Schönleber et al., 2014) parts of the fit.

    pseudo_chisqr: float
        The pseudo chi-squared value (|pseudo chi-squared|, eq. 14 in Boukamp, 1995).

    method: str
        The iterative method used during the fitting process.

    weight: str
        The weight function used during the fitting process.
    """

    circuit: Circuit
    parameters: Dict[str, Dict[str, FittedParameter]]
    minimizer_result: "MinimizerResult"  # noqa: F821
    frequencies: Frequencies
    impedances: ComplexImpedances
    residuals: ComplexResiduals
    pseudo_chisqr: float
    method: str
    weight: str

    def __repr__(self) -> str:
        return f"FitResult ({self.circuit.to_string()}, {hex(id(self))})"

    def get_label(self) -> str:
        """
        Get the label for this result.

        Returns
        -------
        str
        """
        cdc: str = self.circuit.to_string()
        if cdc.startswith("[") and cdc.endswith("]"):
            cdc = cdc[1:-1]

        return cdc

    def get_frequencies(self, num_per_decade: int = -1) -> Frequencies:
        """
        Get the frequencies in the fitted frequency range.

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
        if num_per_decade > 0:
            return _interpolate(self.frequencies, num_per_decade)

        return self.frequencies

    def get_impedances(self, num_per_decade: int = -1) -> ComplexImpedances:
        """
        Get the impedance response of the fitted circuit.

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
        if num_per_decade > 0:
            return self.circuit.get_impedances(self.get_frequencies(num_per_decade))

        return self.impedances

    def get_nyquist_data(
        self,
        num_per_decade: int = -1,
    ) -> Tuple[Impedances, Impedances]:
        """
        Get the data necessary to plot this FitResult as a Nyquist plot: the real and the negative imaginary parts of the impedances.

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
        Get the data necessary to plot this FitResult as a Bode plot: the frequencies, the absolute magnitudes of the impedances, and the negative phase angles/shifts of the impedances in degrees.

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
        f: Frequencies
        Z: ComplexImpedances
        if num_per_decade > 0:
            f = self.get_frequencies(num_per_decade)
            Z = self.circuit.get_impedances(f)
        else:
            f = self.frequencies
            Z = self.impedances

        return (
            f,
            abs(Z),
            -angle(Z, deg=True),
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

    def get_parameters(self) -> Dict[str, Dict[str, FittedParameter]]:
        """
        Get information about the the fitted parameters as FittedParameter objects.
        The outer dictionary has the labels of the elements as keys and the inner dictionary has the symbols of the parameters as keys.

        Returns
        -------
        Dict[str, Dict[str, FittedParameter]]
        """
        return self.parameters

    def to_parameters_dataframe(
        self,
        running: bool = False,
    ) -> "DataFrame":  # noqa: F821
        """
        Get the fitted parameters and the corresponding estimated errors as a `pandas.DataFrame`_ object.
        Parameters
        ----------
        running: bool, optional
            Whether or not to use running counts as the lower indices of elements.

        Returns
        -------
        `pandas.DataFrame`_
        """
        from pandas import DataFrame

        if not _is_boolean(running):
            raise TypeError(f"Expected a boolean instead of {running=}")

        element_names: List[str] = []
        parameter_labels: List[str] = []
        fitted_values: List[float] = []
        stderr_values: List[Optional[float]] = []
        fixed: List[str] = []
        units: List[str] = []

        element_name: str
        parameters: Dict[str, FittedParameter]
        internal_identifiers: Dict[
            Element, int
        ] = self.circuit.generate_element_identifiers(running=True)
        external_identifiers: Dict[
            Element, int
        ] = self.circuit.generate_element_identifiers(running=False)

        element: Element
        for element, ident in external_identifiers.items():
            element_name = self.circuit.get_element_name(
                element,
                identifiers=external_identifiers,
            )
            parameters = self.parameters[element_name]

            parameter_label: str
            for parameter_label in sorted(parameters.keys()):
                parameter: FittedParameter = parameters[parameter_label]
                element_names.append(
                    self.circuit.get_element_name(
                        element,
                        identifiers=external_identifiers
                        if not running
                        else internal_identifiers,
                    )
                )
                parameter_labels.append(parameter_label)

                fitted_values.append(parameter.value)
                stderr_values.append(
                    abs(parameter.stderr / parameter.value * 100)
                    if (
                        parameter.stderr is not None
                        and not parameter.fixed
                        and parameter.value != 0.0
                    )
                    else nan
                )
                fixed.append("Yes" if parameter.fixed else "No")
                units.append(parameter.unit)

        return DataFrame.from_dict(
            {
                "Element": element_names,
                "Parameter": parameter_labels,
                "Value": fitted_values,
                "Std. err. (%)": stderr_values,
                "Unit": units,
                "Fixed": fixed,
            }
        )

    def to_statistics_dataframe(self) -> "DataFrame":  # noqa: F821
        """
        Get the statistics related to the fit as a `pandas.DataFrame`_ object.

        Returns
        -------
        `pandas.DataFrame`_
        """
        from lmfit.minimizer import MinimizerResult
        from pandas import DataFrame

        result: MinimizerResult = self.minimizer_result
        statistics: Dict[str, Union[int, float, str]] = {
            "Log pseudo chi-squared": log(self.pseudo_chisqr),
            "Log chi-squared": log(result.chisqr),
            "Log chi-squared (reduced)": log(result.redchi),
            "Akaike info. criterion": result.aic,
            "Bayesian info. criterion": result.bic,
            "Degrees of freedom": result.nfree,
            "Number of data points": result.ndata,
            "Number of function evaluations": result.nfev,
            "Method": self.method,
            "Weight": self.weight,
        }

        return DataFrame.from_dict(
            {
                "Label": list(statistics.keys()),
                "Value": list(statistics.values()),
            }
        )


def _to_lmfit(
    identifiers: Dict[Element, FitIdentifiers],
    constraint_expressions: Dict[str, str],
    constraint_variables: Dict[str, Dict[str, Any]],
) -> "Parameters":  # noqa: F821
    from lmfit import Parameters

    result: Parameters = Parameters()

    name: str
    kwargs: dict
    for name, kwargs in constraint_variables.items():
        if "value" in kwargs:
            value: float = kwargs["value"]
            minimum: Union[int, float] = kwargs.get("min", -inf)
            maximum: Union[int, float] = kwargs.get("max", inf)

            if not (minimum <= value <= maximum):
                raise ValueError(
                    f"Expected {minimum} <= {value} <= {maximum} for {name=}"
                )

        result.add(name=name, **kwargs)

    queued_kwargs: List[Dict[str, Any]] = []

    element: Element
    mapping: FitIdentifiers
    for element, mapping in identifiers.items():
        lower_limits: Dict[str, float] = element.get_lower_limits()
        upper_limits: Dict[str, float] = element.get_upper_limits()
        fixed: Dict[str, bool] = element.are_fixed()

        symbol: str
        for symbol, value in element.get_values().items():
            if not (lower_limits[symbol] <= value <= upper_limits[symbol]):
                raise ValueError(
                    f"Expected {lower_limits[symbol]=} <= {value} <= {upper_limits[symbol]=} for {symbol=}"
                )

            name: str = mapping[symbol]
            kwargs = dict(
                name=name,
                value=value,
                min=lower_limits[symbol],
                max=upper_limits[symbol],
                vary=not fixed[symbol],
            )

            # Parameters with constraint expressions may need to be added in a
            # specific order since a NameError may be raised if the expression
            # makes use of a parameter that has not yet been defined.
            if name in constraint_expressions:
                kwargs["expr"] = constraint_expressions[name]
                queued_kwargs.append(kwargs)
            else:
                result.add(**kwargs)

    # Brute force any queued parameters with constraint expressions until
    # either all of them slot nicely into place or we run out of attempts.
    num_attempts_left: int = len(queued_kwargs)

    while num_attempts_left > 0:
        kwargs = queued_kwargs.pop(0)
        try:
            result.add(**kwargs)
            num_attempts_left = len(queued_kwargs)
        except NameError:
            queued_kwargs.append(kwargs)
            num_attempts_left -= 1

    if len(queued_kwargs) > 0:
        raise FittingError(
            "Failed to generate parameters for lmfit. Make sure that all"
            + " constraint expressions are valid and all the required"
            + " variables have been defined!"
        )

    return result


def _from_lmfit(
    parameters: "Parameters",  # noqa: F821
    identifiers: Dict[Element, Dict[str, str]],
):
    from lmfit import Parameters

    if not isinstance(parameters, Parameters):
        raise TypeError(f"Expected a Parameters instead of {parameters=}")

    lookup: Dict[str, Tuple[Element, str]] = {}
    fitted_values: Dict[Element, Dict[str, float]] = {}

    element: Element
    mapping: Dict[str, str]
    for element, mapping in identifiers.items():
        symbol: str
        key: str
        for symbol, key in mapping.items():
            lookup[key] = (element, symbol)
            fitted_values[element] = {}

    value: float
    for key, value in parameters.valuesdict().items():
        if key not in lookup:
            continue

        element, symbol = lookup[key]
        fitted_values[element][symbol] = value

    for element, values in fitted_values.items():
        element.set_values(**values)


def _residual(
    params: "Parameters",  # noqa: F821
    circuit: Circuit,
    f: Frequencies,
    Z_exp: ComplexImpedances,
    weight_func: Callable,
    identifiers: Dict[int, Element],
) -> NDArray[float64]:
    _from_lmfit(params, identifiers)
    Z_fit: ComplexImpedances = circuit.get_impedances(f)
    errors: NDArray[float64] = array(
        [
            (Z_exp.real - Z_fit.real) ** 2,
            (Z_exp.imag - Z_fit.imag) ** 2,
        ],
        dtype=float64,
    )

    return weight_func(Z_exp, Z_fit) * errors


def _unity_weight(
    Z_exp: ComplexImpedances,
    Z_fit: ComplexImpedances,
) -> NDArray[float64]:
    return ones(shape=(2, Z_exp.size), dtype=float64)


def _modulus_weight(
    Z_exp: ComplexImpedances,
    Z_fit: ComplexImpedances,
) -> NDArray[float64]:
    return ones(shape=(2, Z_exp.size), dtype=float64) / abs(Z_fit)


def _proportional_weight(
    Z_exp: ComplexImpedances,
    Z_fit: ComplexImpedances,
) -> NDArray[float64]:
    weight: NDArray[float64] = ones(shape=(2, Z_exp.size), dtype=float64)
    weight[0] = weight[0] / Z_fit.real**2
    weight[1] = weight[1] / Z_fit.imag**2

    return weight


def _boukamp_weight(
    Z_exp: ComplexImpedances,
    Z_fit: ComplexImpedances,
) -> NDArray[float64]:
    # See eq. 13 in Boukamp (1995)
    return (Z_exp.real**2 + Z_exp.imag**2) ** -1  # type: ignore


_WEIGHT_FUNCTIONS: Dict[str, Callable] = {
    "unity": _unity_weight,
    "modulus": _modulus_weight,
    "proportional": _proportional_weight,
    "boukamp": _boukamp_weight,
}


_METHODS: List[str] = [
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
    circuit: Circuit,
    fit: "MinimizerResult",  # noqa: F821
) -> Dict[str, Dict[str, FittedParameter]]:
    from lmfit.minimizer import MinimizerResult

    parameters: Dict[str, Dict[str, FittedParameter]] = {}

    internal_identifiers: Dict[int, Element] = {
        v: k for k, v in circuit.generate_element_identifiers(running=True).items()
    }
    external_identifiers: Dict[Element, int] = circuit.generate_element_identifiers(
        running=False
    )

    internal_id: int
    element: Element
    for internal_id, element in internal_identifiers.items():
        element_name: str = element.get_name()
        symbol: str = element.get_symbol()
        if element_name == symbol:
            element_name = f"{symbol}_{external_identifiers[element]}"

        if element_name in parameters:
            raise KeyError(f"Expected {element_name=} not to exist in {parameters=}")

        parameters[element_name] = {}
        units: Dict[str, str] = element.get_units()

        # Parameters that were not fixed
        variable_name: str
        for variable_name in filter(
            lambda _: _.endswith(f"_{internal_id}"),
            fit.var_names,
        ):
            par = fit.params[variable_name]
            stderr: float = par.stderr if hasattr(par, "stderr") else nan
            try:
                float(stderr)
            except TypeError:
                stderr = nan

            variable_name, _ = variable_name.rsplit("_", 1)
            parameters[element_name][variable_name] = FittedParameter(
                value=par.value,
                stderr=stderr,
                fixed=False,
                unit=units[variable_name],
            )

        # Remaining parameters are fixed
        value: float
        for name, value in element.get_values().items():
            if name in parameters[element_name]:
                continue

            parameters[element_name][name] = FittedParameter(
                value=value,
                stderr=nan,
                fixed=True,
                unit=units[name],
            )

    return parameters


def _fit_process(
    args,
) -> Tuple[Circuit, float, Optional["MinimizerResult"], str, str, str]:  # noqa: F821
    from lmfit import minimize
    from lmfit.minimizer import MinimizerResult

    original_circuit: Circuit
    f: Frequencies
    Z_exp: ComplexImpedances
    method: str
    weight: str
    max_nfev: int
    auto: bool
    constraint_expressions: Dict[str, str]
    constraint_variables: Dict[str, Dict[str, Any]]
    (
        original_circuit,
        f,
        Z_exp,
        method,
        weight,
        max_nfev,
        auto,
        constraint_expressions,
        constraint_variables,
    ) = args

    circuit = deepcopy(original_circuit)
    if not isinstance(circuit, Circuit):
        raise TypeError(f"Expected a Circuit instead of {circuit=}")

    if not _is_floating_array(f):
        raise TypeError(f"Expected an NDArray[float] instead of {f=}")

    if not _is_complex_array(Z_exp):
        raise TypeError(f"Expected an NDArray[complex] instead of {Z_exp=}")

    if not isinstance(method, str):
        raise TypeError(f"Expected a string instead of {method=}")

    if not isinstance(weight, str):
        raise TypeError(f"Expected a string instead of {weight=}")

    if not _is_integer(max_nfev):
        raise TypeError(f"Expected an integer instead of {max_nfev=}")

    if not _is_boolean(auto):
        raise TypeError(f"Expected a boolean instead of {auto=}")

    identifiers: Dict[Element, FitIdentifiers]
    identifiers = generate_fit_identifiers(circuit)

    weight_func: Callable = _WEIGHT_FUNCTIONS[weight]

    with catch_warnings():
        if auto:
            filterwarnings("error")
            filterwarnings("ignore", category=DeprecationWarning)
            filterwarnings("ignore", category=RuntimeWarning)

        try:
            fit: MinimizerResult = minimize(
                _residual,
                _to_lmfit(identifiers, constraint_expressions, constraint_variables),
                method,
                args=(
                    circuit,
                    f,
                    Z_exp,
                    weight_func,
                    identifiers,
                ),
                max_nfev=None if max_nfev < 1 else max_nfev,
            )

        except (Exception, Warning):  # TODO: Be more specific
            return (
                circuit,
                inf,
                None,
                method,
                weight,
                format_exc(),
            )

    if fit.ndata < len(f) and log(fit.chisqr) < -50:
        return (circuit, inf, None, method, weight, "Invalid result!")

    _from_lmfit(fit.params, identifiers)

    return (
        circuit,
        _calculate_pseudo_chisqr(Z_exp=Z_exp, Z_fit=circuit.get_impedances(f)),
        fit,
        method,
        weight,
        "",
    )


def validate_circuit(circuit: Circuit):
    """
    Validate circuits for circuit fitting.

    Parameters
    ----------
    circuit: Circuit
    """
    if circuit.to_string() in ["[]", "()"]:
        raise ValueError("The circuit has no elements!")

    identifiers: Dict[Element, int] = circuit.generate_element_identifiers(
        running=False
    )
    element_names: Set[str] = set()

    element: Element
    ident: int
    for element, ident in identifiers.items():
        name: str = circuit.get_element_name(element, identifiers)
        if name in element_names:
            raise ValueError(
                f"Two or more elements of the same type have the same name ({name=})"
            )
        else:
            element_names.add(name)


def generate_fit_identifiers(circuit: Circuit) -> Dict[Element, FitIdentifiers]:
    """
    Generate the identifiers that are provided to `lmfit`_ when fitting the provided circuit. These identifiers can be used to define optional `constraints <https://lmfit.github.io/lmfit-py/constraints.html>`_ when calling |fit_circuit|.

    Parameters
    ----------
    circuit: |Circuit|
        The circuit to process.

    Returns
    -------
    Dict[|Element|, |FitIdentifiers|]
        A dictionary that maps each element to a |FitIdentifiers| instance that maps each parameter of that element to the identifier that is used by `lmfit`_ during fitting.
    """
    identifiers: Dict[Element, Dict[str, str]] = {}

    element: Element
    ident: int
    for element, ident in circuit.generate_element_identifiers(running=True).items():
        identifiers[element] = {}

        symbol: str
        for symbol in element.get_values().keys():
            identifiers[element][symbol] = f"{symbol}_{ident}"

    return {element: FitIdentifiers(**mapping) for element, mapping in identifiers.items()}


def fit_circuit(
    circuit: Circuit,
    data: DataSet,
    method: Union[str, List[str]] = "auto",
    weight: Union[str, List[str]] = "auto",
    max_nfev: int = -1,
    num_procs: int = -1,
    timeout: int = 0,
    constraint_expressions: Optional[Dict[str, str]] = None,
    constraint_variables: Optional[Dict[str, dict]] = None,
    **kwargs,
) -> FitResult:
    """
    Fit a circuit to a data set.

    Parameters
    ----------
    circuit: |Circuit|
        The circuit to fit to a data set.

    data: |DataSet|
        The data set that the circuit will be fitted to.

    method: Union[str, List[str]], optional
        The iteration method(s) to use during fitting.
        See `lmfit's documentation <https://lmfit.github.io/lmfit-py/>`_ for valid method names.
        Note that not all methods supported by lmfit are possible in the current implementation (e.g. some methods may require a function that calculates a Jacobian).
        If a list of multiple methods (or "auto") is provided, then multiple methods will be tested and the best result will be returned based on the |pseudo chi-squared| value.

    weight: Union[str, List[str]], optional
        The weight function(s) to use when calculating residuals.
        Currently supported values: "modulus", "proportional", "unity", "boukamp", and "auto".
        If a list of multiple weights (or "auto") is provided, then multiple weights will be tested and the best result will be returned based on the |pseudo chi-squared| value.

    max_nfev: int, optional
        The maximum number of function evaluations when fitting.
        A value less than one equals no limit.

    num_procs: int, optional
        The maximum number of parallel processes to use when performing the fitting with multiple methods and/or weights.
        A value less than 1 results in an attempt to figure out a suitable value based on, e.g., the number of cores detected.
        Additionally, a negative value can be used to reduce the number of processes by that much (e.g., to leave one core for a GUI thread).

    timeout: int, optional
        The amount of time in seconds that a single fit is allowed to take before being timed out.
        If this values is less than one, then no time limit is imposed.

    constraint_expressions: Optional[Dict[str, str]], optional
        A mapping of the optional `constraints <https://lmfit.github.io/lmfit-py/constraints.html>`_ to apply.

    constraint_variables: Optional[Dict[str, dict]], optional
        A mapping of variable names to their keyword arguments (see `lmfit.Parameters <https://lmfit.github.io/lmfit-py/parameters.html#lmfit.parameter.Parameter>`_ for more information). These variables may be referenced in the optional `constraints <https://lmfit.github.io/lmfit-py/constraints.html>`_.

    **kwargs

    Returns
    -------
    |FitResult|
    """
    from lmfit.minimizer import MinimizerResult

    if not isinstance(circuit, Circuit):
        raise TypeError(f"Expected a Circuit instead of {circuit=}")
    else:
        validate_circuit(circuit)

    methods: List[str] = []
    if isinstance(method, list) and all(map(lambda m: isinstance(m, str), method)):
        methods.extend(method)
    elif isinstance(method, str):
        if method == "auto":
            methods.extend(_METHODS)
        else:
            methods.append(method)
    else:
        raise TypeError(f"Expected a string or a list of strings instead of {method=}")

    if not all(map(lambda m: m in _METHODS, methods)):
        raise ValueError(
            "Valid methods include: '" + "', '".join(_METHODS) + "', and 'auto'"
        )

    weights: List[str] = []
    if isinstance(weight, list) and all(map(lambda w: isinstance(w, str), weight)):
        weights.extend(weight)
    elif isinstance(weight, str):
        if weight == "auto":
            weights.extend(list(_WEIGHT_FUNCTIONS.keys()))
        else:
            weights.append(weight)
    else:
        raise TypeError(f"Expected a string or a list of strings instead of {weight=}")
    
    if not all(map(lambda w: w in _WEIGHT_FUNCTIONS, weights)):
        raise ValueError(
            "Valid weights include: '" + "', '".join(_WEIGHT_FUNCTIONS) + "', and 'auto'"
        )

    if not _is_integer(max_nfev):
        raise TypeError(f"Expected an integer instead of {max_nfev=}")

    if not _is_integer(num_procs):
        raise TypeError(f"Expected an integer instead of {num_procs=}")
    elif num_procs < 1:
        num_procs = max((get_default_num_procs() - abs(num_procs), 1))

    if not _is_integer(timeout):
        raise TypeError(f"Expected an integer instead of {timeout=}")
    elif timeout < 0:
        raise ValueError(
            f"Expected an integer equal to or greater than zero instead of {timeout=}"
        )

    if constraint_expressions is None:
        pass
    elif not isinstance(constraint_expressions, dict):
        raise TypeError(f"Expected either None or a dictionary instead of {constraint_expressions=}")
    elif not all(map(lambda k: isinstance(k, str), constraint_expressions.keys())):
        raise TypeError(f"Expected all keys to be strings instead of {constraint_expressions=}")
    elif not all(map(lambda v: isinstance(v, str), constraint_expressions.values())):
        raise TypeError(f"Expected all values to be strings instead of {constraint_expressions=}")

    if constraint_variables is None:
        pass
    elif not isinstance(constraint_variables, dict):
        raise TypeError(f"Expected either None or a dictionary instead of {constraint_variables=}")
    elif not all(map(lambda k: isinstance(k, str), constraint_variables.keys())):
        raise TypeError(f"Expected all keys to be strings instead of {constraint_variables=}")
    elif not all(map(lambda v: isinstance(v, dict), constraint_variables.values())):
        raise TypeError(f"Expected all values to be dictionaries instead of {constraint_variables=}")

    num_steps: int = (len(_METHODS) if method == "auto" else 1) * (
        len(_WEIGHT_FUNCTIONS) if weight == "auto" else 1
    )

    prog: Progress
    with Progress("Preparing to fit", total=num_steps + 1) as prog:
        fits: List[
            Tuple[Circuit, float, Optional["MinimizerResult"], str, str, str]
        ] = []

        method_weight_combos: List[Tuple[str, str]] = []
        for method in methods:
            for weight in weights:
                method_weight_combos.append(
                    (
                        method,
                        weight,
                    )
                )

        f: Frequencies = data.get_frequencies()
        if len(f) < 2:
            raise ValueError(
                f"There are fewer than two unmasked data points in the '{data.get_label()}' data set parsed from '{data.get_path()}'"
            )

        Z_exp: ComplexImpedances = data.get_impedances()
        if len(Z_exp) != len(f):
            raise ValueError(f"Expected {len(Z_exp)=} == {len(f)=}")

        args = (
            (
                circuit,
                f,
                Z_exp,
                method,
                weight,
                max_nfev,
                True,
                constraint_expressions or {},
                constraint_variables or {},
            )
            for (method, weight) in method_weight_combos
        )

        prog.set_message(
            "Performing fit" + ("s" if len(method_weight_combos) > 1 else "")
        )
        res: Tuple[Circuit, float, Optional["MinimizerResult"], str, str, str]
        if num_procs > 1 or timeout > 0:
            with Pool(num_procs) as pool:
                iterator = pool.imap(_fit_process, args, 1)
                while True:
                    try:
                        if timeout > 0:
                            res = iterator.next(timeout=timeout)
                        else:
                            res = iterator.next()
                    except MPTimeoutError:
                        if len(method_weight_combos) > 1:
                            raise FittingError(
                                "Timed out before finishing fitting using all combinations of methods and weights! Consider either reducing the number of combinations to test or increasing the time limit."
                            )
                        else:
                            raise FittingError(
                                "Timed out before finishing fitting! Consider increasing the time limit."
                            )
                    except StopIteration:
                        break

                    fits.append(res)
                    prog.increment()

        else:
            for res in map(_fit_process, args):
                fits.append(res)
                prog.increment()

        if not fits:
            raise FittingError("No valid results generated!")

        fits.sort(key=lambda _: log(_[1]) if _[2] is not None else inf)

    return _convert_intermediate_result(fits[0], f, Z_exp)


def _convert_intermediate_result(
    intermediate: Tuple[
        Circuit,
        float,
        Optional["MinimizerResult"],  # noqa: F821
        str,
        str,
        str,
    ],
    f: Frequencies,
    Z_exp: ComplexImpedances,
) -> FitResult:
    from lmfit.minimizer import MinimizerResult

    fit: Optional[MinimizerResult]
    error_msg: str
    Xps: float
    circuit, Xps, fit, method, weight, error_msg = intermediate
    if fit is None:
        raise FittingError(error_msg)

    Z_fit: ComplexImpedances = circuit.get_impedances(f)

    return FitResult(
        circuit=circuit,
        parameters=_extract_parameters(circuit, fit),
        minimizer_result=fit,
        frequencies=f,
        impedances=Z_fit,
        # Residuals calculated according to eqs. 15 and 16
        # in Schönleber et al. (2014)
        residuals=_calculate_residuals(Z_exp, Z_fit),
        pseudo_chisqr=Xps,
        method=method,
        weight=weight,
    )
