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

from typing import Dict, List, Union
from pandas import DataFrame
from numpy import logspace, ndarray
from pyimpspec.data import DataSet
from .base import Element, Connection
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
from .warburg import Warburg, WarburgOpen, WarburgShort
from .de_levie import DeLevieFiniteLength


def get_elements() -> Dict[str, Element]:
    """
Returns a mapping of element symbols to the element class.

Returns
-------
Dict[str, Element]
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


def string_to_circuit(cdc: str) -> Circuit:
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


def simulate_spectrum(
    circuit: Circuit, frequencies: Union[List[float], ndarray] = [], label: str = ""
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
