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

# This module implements the m(RQ)fit (or multi-(RQ) CNLS-fit) method
# 10.1016/j.electacta.2014.12.059
# 10.1016/j.ssi.2016.10.009

from typing import (
    Dict,
    List,
    Optional,
    OrderedDict,
    Tuple,
    Type,
    Union,
)
from numpy import (
    array,
    cos,
    cosh,
    exp,
    floating,
    integer,
    isclose,
    issubdtype,
    log as ln,
    log10 as log,
    ndarray,
    pi,
    sin,
    sqrt,
    zeros,
)
from pyimpspec.circuit import (
    Circuit,
    Connection,
    Parallel,
    Series,
    Element,
    Capacitor,
    ConstantPhaseElement,
    Resistor,
    parse_cdc,
    simulate_spectrum,
)
from pyimpspec.data import DataSet
from pyimpspec.analysis.fitting import (
    _interpolate,
    _calculate_residuals,
    fit_circuit,
)
from .result import (
    DRTError,
    DRTResult,
    _calculate_chisqr,
)
import pyimpspec.progress as progress


def _validate_elements(elements: List[Union[Element, Connection]]):
    element: Union[Element, Connection]
    for element in elements:
        if not (
            isinstance(element, Resistor)
            or isinstance(element, Capacitor)
            or isinstance(element, ConstantPhaseElement)
        ):
            raise DRTError(
                "Invalid circuit: only resistors, capacitors, and constant phase elements are allowed!"
            )


def _validate_connections(connections: List[Connection]):
    # Top-level series connection
    if not isinstance(connections[0], Series):
        raise DRTError("Invalid circuit: expected a top-level series connection!")
    series: Connection = connections.pop(0)
    elements_or_connections: List[Union[Type[Element], Type[Connection]]]
    elements_or_connections = list(map(type, series.get_elements(flattened=False)))
    if elements_or_connections.count(Resistor) > 1:
        raise DRTError(
            "Invalid circuit: only one optional series resistance is allowed!"
        )
    if elements_or_connections.count(Parallel) < 1:
        raise DRTError("Invalid circuit: expected at least one parallel connection!")
    for elem_or_con in elements_or_connections:
        if elem_or_con is not Resistor and elem_or_con is not Parallel:
            raise DRTError(f"Invalid circuit: unsupported element in the top-level series connection!")
    # Parallel connections (i.e., (RC) and (RQ))
    con: Connection
    for con in connections:
        if not isinstance(con, Parallel):
            raise DRTError(
                "Invalid circuit: no series connections other than the top-level series connection are allowed!"
            )
        if len(con.get_connections()) > 1:
            raise DRTError(
                "Invalid circuit: nested connections are not allowed within the parallel connections!"
            )
        elements: List[Type[Element]] = list(map(type, con.get_elements()))
        if len(elements) != 2:
            raise DRTError(
                "Invalid circuit: the parallel connections may only contain two elements!"
            )
        if Resistor not in elements:
            raise DRTError(
                "Invalid circuit: the parallel connections must contain a resistor!"
            )
        if not (Capacitor in elements or ConstantPhaseElement in elements):
            raise DRTError(
                "Invalid circuit: the parallel connections must contain either a capacitor or a constant phase element!"
            )


def _validate_circuit(circuit: Circuit):
    _validate_elements(circuit.get_elements())
    _validate_connections(circuit.get_connections())


def _is_fitted_circuit(circuit: Circuit) -> bool:
    """
    If all parameters are fixed, then assume that the circuit has been fitted.
    This can be used as a way of manually defining the circuit using a CDC without having to
    perform a fit (i.e., a priori knowledge of the correct parameter values is required).

    If all parameters are set at their default values, then assume that the circuit has not been fitted.
    """
    parameters: Dict[int, OrderedDict[str, float]] = circuit.get_parameters()
    ident: int
    element_parameters: OrderedDict[str, float]
    fixed_values: List[bool] = []
    for ident, element_parameters in parameters.items():
        element: Optional[Element] = circuit.get_element(ident)
        assert element is not None
        default_values: Dict[str, float] = element.get_defaults()
        key: str
        for key in default_values.keys():
            if not isclose(default_values[key], element_parameters[key]):
                return True
            fixed_values.append(element.is_fixed(key))
    return all(fixed_values)


def _adjust_initial_values(circuit: Circuit, data: DataSet) -> Circuit:
    f: ndarray = data.get_frequency()
    Z_exp: ndarray = data.get_impedance()
    connections: List[Connection] = circuit.get_connections()
    series: Connection = connections.pop(0)
    assert type(series) is Series
    element: Union[Element, Connection]
    for element in series.get_elements(flattened=False):
        if isinstance(element, Resistor):
            if not element.is_fixed("R"):
                element.set_parameters({"R": Z_exp[0].real})
    num_parallels: int = len(connections)
    R_frac: float = (max(Z_exp.real) - min(Z_exp.real)) / num_parallels
    i: int = num_parallels + 1
    parallel: Connection
    for parallel in connections:
        assert isinstance(parallel, Parallel)
        for element in parallel.get_elements():
            if isinstance(element, Resistor):
                if not element.is_fixed("R"):
                    element.set_parameters({"R": R_frac})
                continue
            i -= 1
            if isinstance(element, Capacitor):
                if not element.is_fixed("C"):
                    element.set_parameters(
                        {
                            "C": (10 ** -((log(max(f)) * i / (num_parallels + 1))))
                            / R_frac
                        }
                    )
            elif isinstance(element, ConstantPhaseElement):
                if not element.is_fixed("Y"):
                    element.set_parameters(
                        {
                            "Y": (
                                10
                                ** -(
                                    (log(max(f)) * i / (num_parallels + 1))
                                    ** element.get_parameters()["n"]
                                )
                            )
                            / R_frac
                        }
                    )
    return circuit


def _calculate_tau_gamma(
    circuit: Circuit,
    f: ndarray,
    W: float,
    num_per_decade: int,
) -> Tuple[ndarray, ndarray]:
    tau: ndarray = 1 / (2 * pi * _interpolate(f, num_per_decade=num_per_decade))
    gamma: ndarray = zeros(tau.shape)
    connections: List[Connection] = circuit.get_connections()
    assert isinstance(connections.pop(0), Series)
    parallel: Connection
    for parallel in connections:
        assert isinstance(parallel, Parallel)
        parameters: Dict[str, float] = {}
        element_parameters: OrderedDict[str, float]
        for element_parameters in parallel.get_parameters().values():
            parameters.update(element_parameters)
        R: float = parameters["R"]
        Y: Optional[float] = parameters.get("Y", parameters.get("C"))
        assert Y is not None
        n: float = parameters.get("n", 1.0)
        tau_0: float = (R * Y) ** (1.0 / n)
        if isclose(n, 1.0, atol=1e-2):
            gamma += R / (W * sqrt(pi)) * exp(-((ln(tau / tau_0) / W) ** 2))
        else:
            gamma += (
                (R / (2 * pi))
                * (sin((1 - n) * pi))
                / (cosh(n * ln(tau / tau_0)) - cos((1 - n) * pi))
            )
    return (
        tau,
        gamma,
    )


def _generate_label(circuit: Circuit) -> str:
    label_fragments: List[str] = []
    series: Union[Element, Connection] = circuit.get_elements(flattened=False)[0]
    assert isinstance(series, Series)
    elem_or_con: Union[Element, Connection]
    for elem_or_con in series.get_elements(flattened=False):
        if isinstance(elem_or_con, Parallel):
            element: Union[Element, Connection]
            for element in elem_or_con.get_elements():
                if isinstance(element, Capacitor):
                    label_fragments.append("(RC)")
                elif isinstance(element, ConstantPhaseElement):
                    label_fragments.append("(RQ)")
        elif isinstance(elem_or_con, Resistor):
            assert "R" not in label_fragments
            label_fragments.append("R")
    assert label_fragments.count("R") <= 1
    num_RQ: int = label_fragments.count("(RQ)")
    num_RC: int = label_fragments.count("(RC)")
    label: str = "R" * label_fragments.count("R")
    if num_RQ == 1:
        label += "(RQ)"
    elif num_RQ > 1:
        label += f"-{num_RQ}(RQ)"
        if num_RC > 0:
            label += "-"
    if num_RC == 1:
        label += "(RC)"
    elif num_RC > 1:
        label += f"-{num_RC}(RC)"
    return label


def _calculate_drt_mRQfit(
    data: DataSet,
    circuit: Circuit,
    W: float = 0.15,
    num_per_decade: int = 100,
    num_procs: int = -1,
) -> DRTResult:
    assert isinstance(data, DataSet), type(data)
    assert isinstance(circuit, Circuit), type(circuit)
    assert issubdtype(type(W), floating) and W > 0.0, (type(W), W)
    assert issubdtype(type(num_procs), integer), (type(num_procs), num_procs)
    num_steps: int = 3
    progress.update_every_N_percent(
        0,
        total=num_steps,
        message="Validating circuit",
    )
    _validate_circuit(circuit)
    if not _is_fitted_circuit(circuit):
        progress.update_every_N_percent(
            1,
            total=num_steps,
            message="Fitting circuit",
        )
        circuit = fit_circuit(
            _adjust_initial_values(
                parse_cdc(circuit.to_string(12)),
                data,
            ),
            data,
            num_procs=num_procs,
        ).circuit
    progress.update_every_N_percent(
        2,
        total=num_steps,
        message="Calculating DRT",
    )
    tau: ndarray
    gamma: ndarray
    tau, gamma = _calculate_tau_gamma(
        circuit,
        data.get_frequency(),
        W,
        num_per_decade,
    )
    f: ndarray = data.get_frequency()
    Z_fit: ndarray = simulate_spectrum(circuit, f).get_impedance()
    real_res, imag_res = _calculate_residuals(data.get_impedance(), Z_fit)
    return DRTResult(
        _generate_label(circuit),
        tau,
        gamma,
        f,
        Z_fit,
        real_res,
        imag_res,
        array([]),
        array([]),
        array([]),
        array([]),
        {},
        _calculate_chisqr(data.get_impedance(), Z_fit),
        -1.0,
    )
