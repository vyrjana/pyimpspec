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
    List,
    Union,
    Tuple,
)
from .base import (
    Connection,
    Element,
)


class Series(Connection):
    """
    Elements connected in series.

    Parameters
    ----------
    elements: List[Union[Element, Connection]]
        List of elements (and connections) that are connected in series.
    """

    def __init__(self, elements: List[Union[Element, Connection]]):
        super().__init__(elements)

    def to_stack(self, stack: List[Tuple[str, Union[Element, Connection]]]):
        stack.append(
            (
                "[",
                self,
            )
        )
        for element in reversed(self._elements):
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
            + "".join(
                map(lambda _: _.to_string(decimals=decimals), reversed(self._elements))
            )
            + "]"
        )

    def get_label(self) -> str:
        return "Series"

    def impedance(self, f: float) -> complex:
        return sum(map(lambda _: _.impedance(f), self._elements)) or complex(0, 0)

    def _str_expr(self, substitute: bool = False) -> str:
        if not self._elements:
            return "0"
        string: str = ""
        for element in reversed(self._elements):
            elem_str: str = element._str_expr(substitute=substitute)
            if string == "":
                string = elem_str
            else:
                string += f" + ({elem_str})"
        return string
