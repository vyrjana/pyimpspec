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

from typing import (
    List,
    Union,
)
from pyimpspec.circuit.base import Element
from pyimpspec.circuit.circuit import Circuit
from pyimpspec.circuit.parser import Parser


class CircuitBuilder:
    """
    A class for building circuits using context managers.

    Parameters
    ----------
    parallel: bool = False
        Whether or not this context/connection is a parallel connection.
    """

    def __init__(self, parallel: bool = False):
        self._is_parallel: bool = parallel
        self._elements: List[Union["CircuitBuilder", Element]] = []

    def __enter__(self) -> "CircuitBuilder":
        return self

    def __exit__(self, *args, **kwargs):
        if self._is_parallel:
            if len(self._elements) < 2:
                raise ValueError("Parallel connections must contain at least two items (elements and/or other connections)")
        else:
            if not (len(self._elements) >= 1):
                raise ValueError("Series connections must contain at least one item (an element or another connection)")

    def __str__(self) -> str:
        return self.to_string()

    def series(self) -> "CircuitBuilder":
        """
        Create a series connection.

        Returns
        -------
        |CircuitBuilder|
        """
        series: "CircuitBuilder" = CircuitBuilder(parallel=False)
        self._elements.append(series)

        return series

    def parallel(self) -> "CircuitBuilder":
        """
        Create a parallel connection.

        Returns
        -------
        |CircuitBuilder|
        """
        parallel: "CircuitBuilder" = CircuitBuilder(parallel=True)
        self._elements.append(parallel)

        return parallel

    def __iadd__(self, element: Element) -> "CircuitBuilder":
        self.add(element)

        return self

    def add(self, element: Element):
        """
        Add an element to the current context (i.e., connection).

        Parameters
        ----------
        element: |Element|
            The element to add to the current series or parallel connection.
        """
        if not isinstance(element, Element):
            raise TypeError(f"Expected an Element instead of {element=}")

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

        Returns
        -------
        str
        """
        return self.to_circuit().to_string(decimals=decimals)

    def to_circuit(self) -> Circuit:
        """
        Generate a circuit.

        Returns
        -------
        |Circuit|
        """
        return Parser().process(self._to_string())
