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

from typing import (
    Dict,
    List,
    Type,
    Union,
)
from pandas import DataFrame
from numpy import (
    logspace,
    ndarray,
)
from pyimpspec.data import DataSet
from .base import (
    Connection,
    Element,
)
from .parser import (
    Parser,
    ParsingError,
)
from .circuit import Circuit
from .parallel import Parallel
from .series import Series
from .resistor import Resistor
from .capacitor import Capacitor
from .inductor import Inductor
from .constant_phase_element import ConstantPhaseElement
from .gerischer import Gerischer
from .havriliak_negami import HavriliakNegami
from .warburg import (
    Warburg,
    WarburgOpen,
    WarburgShort,
)
from .de_levie import DeLevieFiniteLength


def get_elements() -> Dict[str, Type[Element]]:
    """
    Returns a mapping of element symbols to the element class.

    Returns
    -------
    Dict[str, Type[Element]]
    """
    return {
        _.get_symbol(): _
        for _ in sorted(
            [
                Resistor,
                Capacitor,
                Inductor,
                ConstantPhaseElement,
                Warburg,
                WarburgShort,
                WarburgOpen,
                DeLevieFiniteLength,
                Gerischer,
                HavriliakNegami,
            ],
            key=lambda _: _.get_symbol(),
        )
    }


def parse_cdc(cdc: str) -> Circuit:
    """
    Generate a Circuit instance from a string that contains a circuit description code (CDC).

    Parameters
    ----------
    cdc: str
        A circuit description code (CDC) corresponding to an equivalent circuit.

    Returns
    -------
    Circuit
    """
    assert type(cdc) is str
    return Parser().process(cdc)


class CircuitBuilder:
    """
    A class for building circuits using context managers:

    with CircuitBuilder() as builder:
        builder.add(Resistor())
    """

    def __init__(self, parallel: bool = False):
        self._is_parallel: bool = parallel
        self._elements: List[Union["CircuitBuilder", Element]] = []

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        if self._is_parallel:
            assert (
                len(self._elements) >= 2
            ), "Parallel connections must contain at least two items (elements and/or other connections)."
        else:
            assert (
                len(self._elements) >= 1
            ), "Series connections must contain at least one item (an element or another connection)."

    def __str__(self) -> str:
        return self.to_string()

    def series(self):
        """
        Create a series connection:

        with CircuitBuilder() as builder:
            with builder.series() as series:
                builder.add(Resistor())
                builder.add(Capacitor())
        """
        self._elements.append(CircuitBuilder(parallel=False))
        return self._elements[-1]

    def parallel(self):
        """
        Create a parallel connection:

        with CircuitBuilder() as builder:
            with builder.parallel() as parallel:
                builder.add(Resistor())
                builder.add(Capacitor())
        """
        self._elements.append(CircuitBuilder(parallel=True))
        return self._elements[-1]

    def add(self, element: Element):
        """
        Add an element to the current context (i.e., connection).

        Parameters
        ----------
        element: Element
            The element to add to the current series or parallel connection.
        """
        assert isinstance(element, Element), element
        self._elements.append(element)

    def _to_string(self, decimals: int = 12) -> str:
        cdc: str = "(" if self._is_parallel else "["
        element: Union["CircuitBuilder", Element]
        for element in self._elements:
            if isinstance(element, Element):
                cdc += element.to_string(decimals=decimals)
            else:
                cdc += element._to_string(decimals=decimals)
        cdc += ")" if self._is_parallel else "]"
        return cdc

    def to_string(self, decimals: int = -1) -> str:
        """
        Generate a circuit description code.

        Parameters
        ----------
        decimals: int = -1
            The number of decimals to include for the current element parameter values and limits.
            -1 means that the CDC is generated using the basic syntax, which omits element labels, parameter values, and parameter limits.

        """
        return self.to_circuit().to_string(decimals=decimals)

    def to_circuit(self) -> Circuit:
        """
        Generate a circuit.
        """
        return parse_cdc(self._to_string())


def simulate_spectrum(
    circuit: Circuit,
    frequencies: Union[List[float], ndarray] = [],
    label: str = "",
) -> DataSet:
    """
    Simulate the impedance spectrum generated by a circuit in a certain frequency range.

    Parameters
    ----------
    circuit: Circuit
        The circuit to use when calculating impedances at various frequencies.

    frequencies: Union[List[float], ndarray] = []
        A list of floats representing frequencies in Hz.
        If no frequencies are provided, then a frequency range of 10 mHz to 100 kHz with 10 points per decade will be used.

    label: str = ""
        The label for the DataSet that is returned.

    Returns
    -------
    DataSet
    """
    assert type(circuit) is Circuit
    assert type(frequencies) is list or type(frequencies) is ndarray
    assert type(label) is str
    if len(frequencies) == 0:
        frequencies = logspace(5, -2, 71)
    columns: dict = {
        "frequency": frequencies,
        "real": [],
        "imaginary": [],
        "label": label,
    }
    f: float
    for f in frequencies:
        z: complex = circuit.impedance(f)
        columns["real"].append(z.real)
        columns["imaginary"].append(z.imag)
    return DataSet.from_dict(columns)
