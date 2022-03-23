# Copyright 2022 pyimpspec developers
# pyimpspec is licensed under the GPLv3 or later (https://www.gnu.org/licenses/gpl-3.0.html).
# The licenses of pyimpspec's dependencies and/or sources of portions of code are included in
# the LICENSES folder.

from typing import List, Union, Tuple
from .base import Connection, Element


class Parallel(Connection):
    def __init__(self, elements: List[Union[Element, Connection]]):
        """
        Elements connected in parallel.

        Parameters
        ----------
        elements: List[Union[Element, Connection]]
            List of elements
        """
        super().__init__(elements)

    def to_stack(self, stack: List[Tuple[str, Union[Element, Connection]]]):
        stack.append(
            (
                "(",
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
                ")",
                self,
            )
        )

    def to_string(self, decimals: int = -1):
        return (
            "("
            + "".join(
                map(lambda _: _.to_string(decimals=decimals), reversed(self._elements))
            )
            + ")"
        )

    def get_label(self) -> str:
        return "Parallel"

    def impedance(self, f: float) -> complex:
        if self._elements:
            return 1 / sum(map(lambda _: 1 / _.impedance(f), self._elements))  # type: ignore
        return complex(0, 0)

    def _str_expr(self, substitute: bool = False) -> str:
        if not self._elements:
            return "0"
        string: str = ""
        for element in reversed(self._elements):
            elem_str: str = element._str_expr(substitute=substitute)
            if string == "":
                string = f"1 / ({elem_str})"
            else:
                string += f" + 1 / ({elem_str})"
        return f"1 / ({string})"
