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
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)
from .base import (
    Connection,
    Container,
    Element,
)
from sympy import (
    Expr,
    sympify,
)
from numpy import zeros
from pyimpspec.typing import (
    ComplexImpedance,
    ComplexImpedances,
    Frequencies,
)
from pyimpspec.typing.helpers import _is_boolean


class Series(Connection):
    """
    Elements connected in series.

    Parameters
    ----------
    elements: List[Union[Element, Connection]]
        List of elements (and connections) that are connected in series.
    """

    def to_stack(self, stack: List[Tuple[str, Union[Element, Connection]]]):
        stack.append(
            (
                "[",
                self,
            )
        )

        for element in self._elements:
            if isinstance(element, Connection):
                element.to_stack(stack)
            else:
                stack.append(
                    (
                        element.to_string(),
                        element,
                    )
                )

        stack.append(
            (
                "]",
                self,
            )
        )

    def to_string(self, decimals: int = -1):
        return (
            "["
            + "".join(map(lambda _: _.to_string(decimals=decimals), self._elements))
            + "]"
        )

    def __repr__(self) -> str:
        return f"Series ({hex(id(self))})"

    def _impedance(
        self,
        f: Frequencies,
    ) -> ComplexImpedances:
        if not self._elements:
            return complex(0, 0) * f

        result: ComplexImpedances = zeros(f.shape, dtype=ComplexImpedance)

        elem_con: Union[Element, Connection]
        for elem_con in self._elements:
            Z: ComplexImpedances
            if isinstance(elem_con, Container):
                Z = elem_con._impedance(
                    f,
                    **elem_con.get_values(),
                    **elem_con.get_subcircuits(),
                )
            elif isinstance(elem_con, Element):
                Z = elem_con._impedance(
                    f,
                    **elem_con.get_values(),
                )
            else:
                Z = elem_con._impedance(f)
            result += Z

        return result

    def to_sympy(
        self,
        substitute: bool = False,
        identifiers: Optional[Dict[Element, int]] = None,
    ) -> Expr:
        expr: Expr = sympify("0")

        if not self._elements:
            return expr

        if not _is_boolean(substitute):
            raise TypeError(f"Expected a boolean instead of {substitute=}")

        if identifiers is None:
            identifiers = self.generate_element_identifiers(running=False)

        if not isinstance(identifiers, dict):
            raise TypeError(f"Expected identifiers to be a dictionary instead of {identifiers=}")

        for element in self._elements:
            if isinstance(element, Container) or isinstance(element, Connection):
                expr += element.to_sympy(substitute=substitute, identifiers=identifiers)
            elif isinstance(element, Element):
                expr += element.to_sympy(
                    substitute=substitute, identifier=identifiers[element]
                )

        return expr
