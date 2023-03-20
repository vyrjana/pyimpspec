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

from typing import (
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)
from .base import (
    Connection,
    Element,
    _calculate_impedances,
)
from .series import Series
from numpy import ndarray
from sympy import (
    Expr,
    latex,
)
from pyimpspec.typing import (
    ComplexImpedances,
    Frequencies,
)


# CDC syntax version
VERSION: int = 1


ElementParameters = Dict[str, Dict[str, str]]


class Circuit:
    """
    A class that represents an equivalent circuit.

    Parameters
    ----------
    elements: Series
        The elements of the circuit wrapped in a Series connection.
    """

    def __init__(self, elements: Connection):
        if not isinstance(elements, Series):
            elements = Series([elements])
        self._elements: Series = elements

    def __copy__(self) -> "Circuit":
        return type(self)(
            self._elements.__copy__(),  # type: ignore
        )

    def __deepcopy__(self, memo: dict) -> "Circuit":
        ident: int = id(self)
        copy: Optional["Circuit"] = memo.get(ident)
        if copy is None:
            copy = type(self)(
                self._elements.__deepcopy__(memo),  # type: ignore
            )
            memo[ident] = copy
        return copy

    def __repr__(self) -> str:
        return f"Circuit ('{self.to_string()}', {hex(id(self))})"

    def __str__(self) -> str:
        return self.to_string()

    def __contains__(self, element_or_connection: Union[Element, Connection]) -> bool:
        return self._elements.contains(element_or_connection, top_level=False)

    def to_stack(self) -> List[Tuple[str, Union[Element, Connection]]]:
        stack: List[Tuple[str, Union[Element, Connection]]] = []
        self._elements.to_stack(stack)
        return stack

    def serialize(self, decimals: int = 12) -> str:
        return (
            "!"
            + "/".join(
                [
                    f"V={VERSION}",  # CDC syntax version
                ]
            )
            + "!"
            + self.to_string(decimals=decimals)
        )

    def to_string(self, decimals: int = -1) -> str:
        """
        Generate the circuit description code (CDC) that represents this circuit.

        Parameters
        ----------
        decimals: int, optional
            The number of decimals to include for the current element parameter values and limits.
            -1 means that the CDC is generated using the basic syntax, which omits element labels, parameter values, and parameter limits.

        Returns
        -------
        str
        """
        return self._elements.to_string(decimals=decimals)

    def get_impedances(self, frequencies: Frequencies) -> ComplexImpedances:
        """
        Calculate the impedance of this circuit at multiple frequencies.

        Parameters
        ----------
        frequencies: |Frequencies|
            The excitation frequencies in hertz.

        Returns
        -------
        |ComplexImpedances|
        """
        assert isinstance(frequencies, ndarray), frequencies
        return _calculate_impedances(self._elements, frequencies)

    def get_connections(self, flattened: bool = True) -> List[Connection]:
        """
        Get the connections in this circuit.

        Parameters
        ----------
        flattened: bool, optional
            Whether or not the connections should be returned as a list of all connections or as a list connections that may also contain more connections.

        Returns
        -------
        List[Connection]
        """
        if flattened is True:
            connections: List[Connection] = self._elements.get_connections(
                flattened=flattened
            )
            connections.insert(0, self._elements)
            return connections
        return [self._elements]

    def get_elements(self, flattened: bool = True) -> List[Union[Element, Connection]]:
        """
        Get the elements in this circuit.

        Parameters
        ----------
        flattened: bool, optional
            Whether or not the elements should be returned as a list of only elements or as a list of connections containing elements.

        Returns
        -------
        List[Union[Element, Connection]]
        """
        if flattened is True:
            return self._elements.get_elements(flattened=flattened)
        return [self._elements]

    def generate_element_identifiers(self, running: bool) -> Dict[Element, int]:
        """
        Generate a mapping of elements to their corresponding integer identifiers.

        Parameters
        ----------
        running: bool
            If true, then the identifiers are simply a running count from 0 to N. Primarily intended for use within pyimpspec.
            If false, then the identifiers represent what number instance of a particular element type an element is (e.g., the second resistor of three resistors would have 2 as its identifier). Primarily intended for use in anything that most users would see (e.g., circuit diagrams and parameter tables).

        Returns
        -------
        Dict[Element, int]
        """
        return self._elements.generate_element_identifiers(running=running)

    def get_element_name(
        self,
        element: Element,
        identifiers: Optional[Dict[Element, int]] = None,
    ) -> str:
        """
        Get the name of the element with consideration for any overriding label assigned to the element or the type-specific count in the context of this circuit.

        Parameters
        ----------
        element: Element
            The element whose name should be returned.

        identifiers: Optional[Dict[Element, int]], optional
            The identifiers to use when determining the name of the provided element.

        Returns
        -------
        str
        """
        return self._elements.get_element_name(element=element, identifiers=identifiers)

    def to_sympy(self, substitute: bool = False) -> Expr:
        """
        Get the |Expr| object corresponding to this circuit's impedance.

        Parameters
        ----------
        substitute: bool, optional
            Whether or not the variables should be substituted with the current values.

        Returns
        -------
        |Expr|
        """
        expr: Expr = self._elements.to_sympy(
            substitute=substitute,
            identifiers=self.generate_element_identifiers(running=True),
        )
        assert isinstance(expr, Expr)
        return expr

    def to_latex(self) -> str:
        """
        Get the LaTeX math expression corresponding to this circuit's impedance.

        Returns
        -------
        str
        """
        return f"Z = {latex(self.to_sympy(substitute=False))}"

    def to_drawing(self) -> "Drawing":  # noqa: F821
        # Dynamically set to pyimpspec.circuit.diagrams.to_drawing
        raise NotImplementedError()

    def to_circuitikz(self) -> str:
        # Dynamically set to pyimpspec.circuit.diagrams.to_circuitikz
        raise NotImplementedError()
