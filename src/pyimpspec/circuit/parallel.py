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
    InfiniteImpedance,
)
from numpy import (
    bool_,
    full,
    isinf,
    where,
    zeros,
)
from numpy.typing import NDArray
from sympy import (
    Expr,
    sympify,
)
from pyimpspec.typing import (
    ComplexImpedance,
    ComplexImpedances,
    Frequencies,
    Indices,
)
from pyimpspec.typing.helpers import _is_boolean


class Parallel(Connection):
    """
    Elements connected in parallel.

    Parameters
    ----------
    elements: List[Union[Element, Connection]]
        List of elements (and connections) that are connected in parallel.
    """

    def to_stack(self, stack: List[Tuple[str, Union[Element, Connection]]]):
        stack.append(
            (
                "(",
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
                ")",
                self,
            )
        )

    def to_string(self, decimals: int = -1):
        return (
            "("
            + "".join(map(lambda _: _.to_string(decimals=decimals), self._elements))
            + ")"
        )

    def __repr__(self) -> str:
        return f"Parallel ({hex(id(self))})"

    def _impedance(self, f: Frequencies) -> ComplexImpedances:
        if not self._elements:
            return complex(0, 0) * f

        shorted: NDArray[bool_] = full(f.shape, False, dtype=bool_)
        path_impedances: List[ComplexImpedances] = []
        num_open_paths: int = 0

        elem_con: Union[Element, Connection]
        for elem_con in self._elements:
            # Calculate the impedances of this element/connection at all of the
            # frequencies.
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
            else:  # Connection
                Z = elem_con._impedance(f)

            # Check for open paths.
            inf_indices: Indices = where(isinf(Z))[0]
            if inf_indices.size == f.size:
                # Infinite impedance at all finite frequencies.
                # Ignore this open pathway.
                num_open_paths += 1
                continue
            elif inf_indices.size > 0:
                # One or more cases of infinite impedances at finite
                # frequencies! f == 0 and f == inf are calculated
                # elsewhere using SymPy, so these infinite impedances
                # are probably caused by some bug in the implementation
                # of an element or connection.
                raise InfiniteImpedance()

            # Check for shorted paths.
            zero_indices: Indices = where(Z == 0.0)[0]
            if zero_indices.size == f.size:
                # Short at all frequencies.
                return complex(0, 0) * f
            elif zero_indices.size > 0:
                # One or more shorts at finite frequencies but in
                # combination with other paths there may also be
                # shorts across all finite frequencies.
                shorted[zero_indices] = True
                if shorted.all():
                    return complex(0, 0) * f

            path_impedances.append(Z)

        if shorted.all():
            return complex(0, 0) * f
        elif num_open_paths == len(self._elements):
            raise InfiniteImpedance()

        results: ComplexImpedances = zeros(f.shape, dtype=ComplexImpedance)

        if shorted.any():
            non_shorted_indices: Indices = where(~shorted)[0]

            for Z in path_impedances:
                results[non_shorted_indices] += 1 / Z[non_shorted_indices]

            results[non_shorted_indices] = 1 / results[non_shorted_indices]
        else:
            for Z in path_impedances:
                results += 1 / Z

            results = 1 / results

        return results

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
                expr += 1 / element.to_sympy(
                    substitute=substitute, identifiers=identifiers
                )
            elif isinstance(element, Element):
                expr += 1 / element.to_sympy(
                    substitute=substitute, identifier=identifiers[element]
                )

        return 1 / expr
