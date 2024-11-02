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

# This module implements the m(RQ)fit (or multi-(RQ) CNLS-fit) method
# 10.1016/j.electacta.2014.12.059
# 10.1016/j.ssi.2016.10.009

from copy import deepcopy
from dataclasses import dataclass
from numpy import (
    cos,
    cosh,
    exp,
    float64,
    isclose,
    log as ln,
    log10 as log,
    pi,
    sin,
    sqrt,
    zeros,
)
from numpy.typing import NDArray
from pyimpspec.circuit.circuit import Circuit
from pyimpspec.circuit.base import (
    Connection,
    Element,
)
from pyimpspec.circuit.connections import (
    Parallel,
    Series,
)
from pyimpspec.circuit.elements import (
    Capacitor,
    ConstantPhaseElement,
    Resistor,
)
from pyimpspec.circuit import (
    parse_cdc,
    simulate_spectrum,
)
from pyimpspec.data import DataSet
from pyimpspec.analysis.fitting import (
    FitResult,
    fit_circuit,
)
from pyimpspec.analysis.utility import (
    _interpolate,
    _calculate_pseudo_chisqr,
)
from pyimpspec.exceptions import DRTError
from .result import DRTResult
from pyimpspec.progress import Progress
from pyimpspec.typing import (
    ComplexImpedances,
    Frequencies,
    Gammas,
    Indices,
    TimeConstants,
)
from pyimpspec.typing.helpers import (
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    Union,
    _is_floating,
)


@dataclass(frozen=True)
class MRQFitResult(DRTResult):
    """
    An object representing the results of calculating the distribution of relaxation times in a data set using the multi-(RQ)-fit (or m(RQ)fit) method.

    Parameters
    ----------
    time_constants: |TimeConstants|
        The time constants.

    gammas: |Gammas|
        The gamma values.

    frequencies: |Frequencies|
        The frequencies of the impedance spectrum.

    impedances: |ComplexImpedances|
        The impedance produced by the model.

    residuals: |ComplexResiduals|
        The residuals of the impedance of the model and the data set.

    pseudo_chisqr: float
        The pseudo chi-squared value, |pseudo chi-squared|, of the modeled impedance (eq. 14 in Boukamp, 1995).

    circuit: |Circuit|
        The fitted circuit.
    """

    gammas: Gammas
    circuit: Circuit

    @staticmethod
    def _generate_label(circuit: Circuit) -> str:
        label_fragments: List[str] = []
        series: Series = circuit.get_connections()[0]
        if not isinstance(series, Series):
            raise TypeError(f"Expected a Series instead of {series=}")

        elem_or_con: Union[Element, Connection]
        for elem_or_con in series:
            if isinstance(elem_or_con, Parallel):
                element: Union[Element, Connection]
                for element in elem_or_con.get_elements():
                    if isinstance(element, Capacitor):
                        label_fragments.append("(RC)")
                    elif isinstance(element, ConstantPhaseElement):
                        label_fragments.append("(RQ)")
            elif isinstance(elem_or_con, Resistor):
                if "R" in label_fragments:
                    raise ValueError(f"'R' is already in {label_fragments=}")

                label_fragments.append("R")

        if label_fragments.count("R") > 1:
            raise ValueError(
                f"There should only be a maximum of 1 'R' in {label_fragments=}"
            )

        num_RQ: int = label_fragments.count("(RQ)")
        num_RC: int = label_fragments.count("(RC)")

        label: str = "R" * label_fragments.count("R")
        if num_RQ == 1:
            label += "(RQ)"
        elif num_RQ > 1:
            label += f"-{num_RQ}(RQ)"
        if num_RC == 1:
            label += ("-" if num_RQ > 1 else "") + "(RC)"
        elif num_RC > 1:
            label += f"-{num_RC}(RC)"

        return label

    def get_label(self) -> str:
        return self._generate_label(self.circuit)

    def get_gammas(self) -> Gammas:
        """
        Get the gamma values.

        Returns
        -------
        |Gammas|
        """
        return self.gammas

    def get_peaks(self, threshold: float = 0.0) -> Tuple[TimeConstants, Gammas]:
        """
        Get the time constants (in seconds) and gamma (in ohms) of peaks with magnitudes greater than the threshold.
        The threshold and the magnitudes are all relative to the magnitude of the highest peak.

        Parameters
        ----------
        threshold: float, optional
            The minimum peak height threshold (relative to the height of the tallest peak) for a peak to be included.

        Returns
        -------
        Tuple[|TimeConstants|, |Gammas|]
        """
        indices: Indices = self._get_peak_indices(
            threshold,
            self.gammas,  # type: ignore
        )

        return (
            self.time_constants[indices],  # type: ignore
            self.gammas[indices],  # type: ignore
        )

    def get_drt_data(self) -> Tuple[TimeConstants, Gammas]:
        """
        Get the data necessary to plot this DRTResult as a DRT plot: the time constants and the corresponding gamma values.

        Returns
        -------
        Tuple[|TimeConstants|, |Gammas|]
        """
        return (
            self.time_constants,  # type: ignore
            self.gammas,  # type: ignore
        )

    def to_peaks_dataframe(
        self,
        threshold: float = 0.0,
        columns: Optional[List[str]] = None,
    ) -> "DataFrame":  # noqa: F821
        from pandas import DataFrame

        if columns is None:
            columns = [
                "tau (s)",
                "gamma (ohm)",
            ]
        elif not isinstance(columns, list):
            raise TypeError(f"Expected a list of strings instead of {columns=}")
        elif len(columns) != 2:
            raise ValueError(f"Expected a list with 2 items instead of {len(columns)=}")
        elif not all(map(lambda s: isinstance(s, str), columns)):
            raise TypeError(f"Expected a list of strings instead of {columns=}")
        elif len(set(columns)) != 2:
            raise ValueError(
                f"Expected a list of 2 unique strings instead of {columns=}"
            )

        indices: Indices = self._get_peak_indices(
            threshold,
            self.gammas,  # type: ignore
        )

        return DataFrame.from_dict(
            {
                columns[0]: self.time_constants[indices],  # type: ignore
                columns[1]: self.gammas[indices],  # type: ignore
            }
        )

    def to_statistics_dataframe(
        self,
    ) -> "DataFrame":  # noqa: F821
        from pandas import DataFrame

        statistics: Dict[str, Union[int, float, str]] = {
            "Log pseudo chi-squared": log(self.pseudo_chisqr),
        }

        return DataFrame.from_dict(
            {
                "Label": list(statistics.keys()),
                "Value": list(statistics.values()),
            }
        )


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
    elements_or_connections = list(map(type, series))

    if elements_or_connections.count(Resistor) > 1:
        raise DRTError(
            "Invalid circuit: only one optional series resistance is allowed!"
        )

    if elements_or_connections.count(Parallel) < 1:
        raise DRTError("Invalid circuit: expected at least one parallel connection!")

    for elem_or_con in elements_or_connections:
        if elem_or_con is not Resistor and elem_or_con is not Parallel:
            raise DRTError(
                "Invalid circuit: unsupported element in the top-level series connection!"
            )

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


def _adjust_initial_values(circuit: Circuit, data: DataSet) -> Circuit:
    element: Union[Element, Connection]
    for element in circuit.get_elements(recursive=True):
        defaults: Dict[str, float] = element.get_default_values()

        key: str
        value: float
        for key, value in element.get_values().items():
            if not isclose(value, defaults[key]):
                return circuit

    f: Frequencies = data.get_frequencies()
    Z_exp: ComplexImpedances = data.get_impedances()

    connections: List[Connection] = circuit.get_connections()
    series: Connection = connections.pop(0)
    if not isinstance(series, Series):
        raise TypeError(f"Expected a Series instead of {series=}")

    for element in series:
        if isinstance(element, Resistor):
            if not element.is_fixed("R"):
                element.set_values(R=Z_exp[0].real)

    num_parallels: int = len(connections)
    R_frac: float = (max(Z_exp.real) - min(Z_exp.real)) / num_parallels
    i: int = num_parallels + 1

    parallel: Connection
    for parallel in connections:
        if not isinstance(parallel, Parallel):
            raise TypeError(f"Expected a Parallel instead of {parallel=}")

        for element in parallel.get_elements():
            if isinstance(element, Resistor):
                if not element.is_fixed("R"):
                    element.set_values(R=R_frac)
                continue
            i -= 1

            if isinstance(element, Capacitor):
                if not element.is_fixed("C"):
                    element.set_values(
                        C=(10 ** -((log(max(f)) * i / (num_parallels + 1)))) / R_frac
                    )
            elif isinstance(element, ConstantPhaseElement):
                if not element.is_fixed("Y"):
                    element.set_values(
                        Y=(
                            10
                            ** -(
                                (log(max(f)) * i / (num_parallels + 1))
                                ** element.get_values("n")["n"]
                            )
                        )
                        / R_frac
                    )

    return circuit


def _calculate_tau_gamma(
    circuit: Circuit,
    f: Frequencies,
    W: float,
    num_per_decade: int,
) -> Tuple[TimeConstants, Gammas]:
    tau: NDArray[float64] = 1 / (_interpolate(f, num_per_decade=num_per_decade) * 2*pi)
    gamma: NDArray[float64] = zeros(tau.shape, dtype=float64)
    connections: List[Connection] = circuit.get_connections()

    series: Connection = connections.pop(0)
    if not isinstance(series, Series):
        raise TypeError(f"Expected a Series instead of {series=}")

    parallel: Connection
    for parallel in connections:
        if not isinstance(parallel, Parallel):
            raise TypeError(f"Expected a Parallel instead of {parallel=}")

        parameters: Dict[str, float] = {}

        element: Element
        for element in parallel.get_elements(recursive=True):
            parameters.update(element.get_values())

        R: float = parameters["R"]
        Y: Optional[float] = parameters.get("Y", parameters.get("C"))
        if not _is_floating(Y):
            raise TypeError(f"Expected a float instead of {Y=}")

        n: float = parameters.get("n", 1.0)
        tau_0: float = (R * Y) ** (1.0 / n)
        if isclose(abs(n), 1.0, atol=1e-2):
            gamma += (
                R / (W * sqrt(pi)) * exp(-((ln(tau / tau_0) / W) ** 2))
                * (-1 if n < 0.0 else 1)
            )
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
    series: Series = circuit.get_connections()[0]
    if not isinstance(series, Series):
        raise TypeError(f"Expected a Series instead of {series=}")

    elem_or_con: Union[Element, Connection]
    for elem_or_con in series:
        if isinstance(elem_or_con, Parallel):
            element: Union[Element, Connection]
            for element in elem_or_con.get_elements():
                if isinstance(element, Capacitor):
                    label_fragments.append("(RC)")
                elif isinstance(element, ConstantPhaseElement):
                    label_fragments.append("(RQ)")

        elif isinstance(elem_or_con, Resistor):
            if "R" in label_fragments:
                raise ValueError(f"'R' is already in {label_fragments}")

            label_fragments.append("R")

    if label_fragments.count("R") > 1:
        raise ValueError(
            f"There should only be a maximum of 1 'R' in {label_fragments=}"
        )

    num_RQ: int = label_fragments.count("(RQ)")
    num_RC: int = label_fragments.count("(RC)")

    label: str = "R" * label_fragments.count("R")
    if num_RQ == 1:
        label += "(RQ)"
    elif num_RQ > 1:
        label += f"-{num_RQ}(RQ)"

    if num_RC == 1:
        label += ("-" if num_RQ > 1 else "") + "(RC)"
    elif num_RC > 1:
        label += f"-{num_RC}(RC)"

    return label


def calculate_drt_mrq_fit(
    data: DataSet,
    circuit: Circuit,
    fit: Optional[FitResult] = None,
    gaussian_width: float = 0.15,
    num_per_decade: int = 100,
    max_nfev: int = -1,
    num_procs: int = -1,
    **kwargs,
) -> MRQFitResult:
    """
    Calculates the distribution of relaxation times (DRT) for a given data set by fitting a circuit with multiple parallel RQ elements connected in series (m(RQ)fit method).

    References:

    - `Boukamp, B.A., 2015, Electrochim. Acta, 154, 35-46 <https://doi.org/10.1016/j.electacta.2014.12.059>`_
    - `Boukamp, B.A. and Rolle, A, 2017, Solid State Ionics, 302, 12-18 <https://doi.org/10.1016/j.ssi.2016.10.009>`_

    Parameters
    ----------
    data: DataSet
        The data set to use in the calculations.

    circuit: Circuit
        A circuit that contains one or more parallel RQ and/or parallel RC elements connected in series.
        An optional series resistance may also be included.
        For example, a circuit with a CDC representation of "R(RQ)(RQ)(RC)" would be a valid circuit.

    fit: Optional[FitResult], optional
        If a FitResult object is provided, then no fitting will be performed.

    gaussian_width: float, optional
        The width of the Gaussian curve that is used to approximate the DRT of an "(RC)" element.

    num_per_decade: int, optional
        The number of points per decade to use when calculating a DRT.

    max_nfev: int, optional
        The maximum number of function evaluations when fitting anything.

    num_procs: int, optional
        The maximum number of parallel processes to use.
        A value less than 1 results in an attempt to figure out a suitable value based on, e.g., the number of cores detected.
        Additionally, a negative value can be used to reduce the number of processes by that much (e.g., to leave one core for a GUI thread).

    Returns
    -------
    |MRQFitResult|
    """
    if not isinstance(circuit, Circuit):
        raise TypeError(f"Expected a Circuit instead of {circuit=}")

    if fit is not None and not hasattr(fit, "circuit"):
        raise TypeError(
            f"Expected None or an object with a 'circuit' property instead of {fit=}"
        )

    if not _is_floating(gaussian_width):
        raise TypeError(f"Expected a float instead of {gaussian_width=}")
    elif gaussian_width <= 0.0:
        raise ValueError("The Gaussian width must be greater than 0.0!")

    prog: Progress
    with Progress("Validating circuit", total=3) as prog:
        _validate_circuit(circuit)
        prog.increment()
        if fit is not None:
            if fit.circuit is not circuit:
                raise ValueError(
                    f"Expected {circuit=} and {fit.circuit=} to be the same object"
                )
        else:
            fit = fit_circuit(
                _adjust_initial_values(
                    deepcopy(circuit),
                    data,
                ),
                data,
                max_nfev=max_nfev,
                num_procs=num_procs,
            )
            fit = fit_circuit(
                fit.circuit,
                data,
                max_nfev=max_nfev,
                num_procs=num_procs,
            )
            circuit = fit.circuit

        prog.increment()
        prog.set_message("Calculating DRT")
        f: Frequencies = data.get_frequencies()
        if len(f) < 1:
            raise ValueError(
                f"There are no unmasked data points in the '{data.get_label()}' data set parsed from '{data.get_path()}'"
            )

        tau: TimeConstants
        gamma: Gammas
        tau, gamma = _calculate_tau_gamma(
            circuit=circuit,
            f=f,
            W=gaussian_width,
            num_per_decade=num_per_decade,
        )

        Z_fit: ComplexImpedances = simulate_spectrum(circuit, f).get_impedances()

    return MRQFitResult(
        time_constants=tau,
        gammas=gamma,
        frequencies=f,
        impedances=Z_fit,
        residuals=fit.residuals,
        pseudo_chisqr=_calculate_pseudo_chisqr(
            Z_exp=data.get_impedances(), Z_fit=Z_fit
        ),
        circuit=circuit,
    )
