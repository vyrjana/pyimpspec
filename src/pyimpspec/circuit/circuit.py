# Copyright 2022 pyimpspec developers
# pyimpspec is licensed under the GPLv3 or later (https://www.gnu.org/licenses/gpl-3.0.html).
# The licenses of pyimpspec's dependencies and/or sources of portions of code are included in
# the LICENSES folder.

from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union, Type
from .base import Element, Connection
from .series import Series
from .parallel import Parallel
from .resistor import Resistor
from .capacitor import Capacitor
from .inductor import Inductor
from .constant_phase_element import ConstantPhaseElement
from numpy import array, inf, ndarray
from sympy import Expr, latex


ElementParameters = Dict[str, Dict[str, str]]


class Circuit:
    def __init__(self, elements: Series, label: str = ""):
        assert type(elements) is Series
        self._elements: Series = elements
        self._assign_identifiers()
        self._label: str = label or self.to_string()

    def __repr__(self) -> str:
        return f"Circuit ('{self.to_string()}', {hex(id(self))})"

    def __str__(self) -> str:
        return self.to_string()

    def get_label(self) -> str:
        return self._label

    def set_label(self, label: str):
        assert type(label) is str
        self._label = label.strip()

    def _assign_identifiers(self):
        return self._elements._assign_identifier(0)

    def to_stack(self) -> List[Tuple[str, Union[Element, Connection]]]:
        stack: List[Tuple[str, Union[Element, Connection]]] = []
        self._elements.to_stack(stack)
        return stack

    def to_string(self, decimals: int = -1) -> str:
        return self._elements.to_string(decimals=decimals)

    def impedance(self, f: float) -> complex:
        assert f > 0 and f < inf
        return self._elements.impedance(f)

    def impedances(self, freq: Union[list, ndarray]) -> ndarray:
        assert type(freq) is list or type(freq) is ndarray
        assert min(freq) > 0 and max(freq) < inf
        return array(list(map(self.impedance, freq)))

    def get_elements(self, flattened: bool = True) -> List[Union[Element, Connection]]:
        if flattened is True:
            return self._elements.get_elements(flattened=flattened)
        return [self._elements]

    def get_parameters(self) -> Dict[int, OrderedDict[str, float]]:
        return self._elements.get_parameters()

    def set_parameters(self, parameters: Dict[int, Dict[str, float]]):
        self._elements.set_parameters(parameters)

    def get_element(self, ident: int) -> Optional[Element]:
        return self._elements.get_element(ident)

    def to_sympy(self, substitute: bool = False) -> Expr:
        expr: Expr = self._elements.to_sympy(substitute=substitute)
        assert isinstance(expr, Expr)
        return expr

    def to_latex(self) -> str:
        return latex(self.to_sympy(substitute=False))

    def to_circuitikz(
        self,
        node_width: float = 3.0,
        node_height: float = 1.5,
        working_label: str = "WE+WS",
        counter_label: str = "CE+RE",
        hide_labels: bool = False,
    ) -> str:
        assert node_width > 0
        assert node_height > 0
        assert type(working_label) is str
        assert type(counter_label) is str
        assert type(hide_labels) is bool
        if hide_labels:
            working_label = ""
            counter_label = ""
        # Phase 1 - figure out the dimensions of the connections and the positions of elements.
        Short = int
        short_counter: Short = 0
        dimensions: Dict[
            Union[Series, Parallel, Element, Short], Tuple[float, float]
        ] = {}
        positions: Dict[
            Union[Series, Parallel, Element, Short], Tuple[float, float]
        ] = {}
        num_nested_parallels: int = 0

        def short_wire(x: float, y: float) -> Tuple[float, float]:
            nonlocal short_counter
            short_counter += 1
            dimensions[short_counter] = (
                0.25,
                1.0,
            )
            positions[short_counter] = (
                x,
                -y,
            )
            return dimensions[short_counter]

        def phase_1_element(
            element: Element, x: float, y: float
        ) -> Tuple[float, float]:
            dimensions[element] = (
                1.0,
                1.0,
            )
            positions[element] = (
                x,
                -y,
            )
            return dimensions[element]

        def phase_1_series(series: Series, x: float, y: float) -> Tuple[float, float]:
            nonlocal num_nested_parallels
            width: float = 0.0
            height: float = 0.0
            elements: List[Union[Element, Connection]] = series.get_elements(
                flattened=False
            )
            num_elements: int = len(elements)
            i: int
            element_connection: Union[Element, Connection]
            for i, element_connection in enumerate(elements):
                if type(element_connection) is Series:
                    w, h = phase_1_series(element_connection, x + width, y)
                    width += w
                    if h > height:
                        height = h
                elif type(element_connection) is Parallel:
                    if num_nested_parallels > 0 and i == 0:
                        w, h = short_wire(x + width, y)
                        width += w
                        if h > height:
                            height = h
                    w, h = phase_1_parallel(element_connection, x + width, y)
                    width += w
                    if h > height:
                        height = h
                    if num_nested_parallels > 0 and (
                        i == num_elements - 1
                        or (i < num_elements - 1 and type(elements[i + 1]) is Parallel)
                    ):
                        w, h = short_wire(x + width, y)
                        width += w
                        if h > height:
                            height = h
                else:
                    assert isinstance(element_connection, Element)
                    w, h = phase_1_element(element_connection, x + width, y)
                    width += w
            dimensions[series] = (
                max(1, width),
                max(1, height),
            )
            positions[series] = (
                x,
                -y,
            )
            return dimensions[series]

        def phase_1_parallel(
            parallel: Parallel, x: float, y: float
        ) -> Tuple[float, float]:
            nonlocal num_nested_parallels
            num_nested_parallels += 1
            width: float = 0.0
            height: float = 0.0
            for element_connection in parallel.get_elements(flattened=False):
                if type(element_connection) is Series:
                    w, h = phase_1_series(element_connection, x, y + height)
                    if w > width:
                        width = w
                    height += h
                elif type(element_connection) is Parallel:
                    w, h = phase_1_parallel(element_connection, x, y + height)
                    if w > width:
                        width = w
                    height += h
                else:
                    assert isinstance(element_connection, Element)
                    w, h = phase_1_element(element_connection, x, y + height)
                    if w > width:
                        width = w
                    height += h
            dimensions[parallel] = (
                max(1, width),
                max(1, height),
            )
            positions[parallel] = (
                x,
                -y,
            )
            num_nested_parallels -= 1
            return dimensions[parallel]

        assert type(self._elements) is Series
        phase_1_series(self._elements, 0, 0)
        assert set(dimensions.keys()) == set(positions.keys())

        # Phase 2 - generate the LaTeX source for drawing the circuit diagram.
        lines: List[str] = [
            r"\begin{circuitikz}",
            r"\draw (0,0) <label>to[short, o-] (1,0);".replace(
                "<label>",
                f"node[above]{{{working_label}}} " if working_label != "" else "",
            ),
        ]
        line: str
        pos: Tuple[float, float]
        dim: Tuple[float, float]
        symbols: Dict[Type[Element], str] = {
            Resistor: "R",
            Capacitor: "capacitor",
            Inductor: "L",
            ConstantPhaseElement: "cpe",
        }

        def replace_variables(
            line: str,
            start_x: float,
            start_y: float,
            end_x: float,
            end_y: float,
            element: str = "short",
        ) -> str:
            line = line.replace("<start_x>", str(start_x))
            line = line.replace("<start_y>", str(start_y))
            line = line.replace("<end_x>", str(end_x))
            line = line.replace("<end_y>", str(end_y))
            line = line.replace("<element>", element)
            return line

        def phase_2():
            for element_connection in positions:
                x, y = positions[element_connection]
                w, h = dimensions[element_connection]
                if type(element_connection) is Series:
                    continue
                elif type(element_connection) is Parallel:
                    start_x = x * (node_width - 1.0) + 1.0
                    start_y = 1.0
                    end_x = (x + w) * (node_width - 1.0) + 1.0
                    end_y = 1.0
                    for element in dimensions:
                        if not element_connection.contains(element, top_level=True):
                            continue
                        ey = positions[element][1]
                        if start_y > 0.0 or ey > start_y:
                            start_y = ey
                        if end_y > 0.0 or ey < end_y:
                            end_y = ey
                    assert start_y != end_y
                    start_y *= node_height
                    end_y *= node_height
                    line = r"\draw (<start_x>,<start_y>) to[<element>] (<start_x>,<end_y>);"
                    lines.append(
                        replace_variables(line, start_x, start_y, end_x, end_y)
                    )
                    line = r"\draw (<end_x>,<start_y>) to[<element>] (<end_x>,<end_y>);"
                    lines.append(
                        replace_variables(line, start_x, start_y, end_x, end_y)
                    )
                    if w == 1.0:
                        continue
                    for elem_con in filter(
                        lambda _: type(_) is not Parallel, dimensions
                    ):
                        if not element_connection.contains(elem_con, top_level=True):
                            continue
                        if w == dimensions[elem_con][0]:
                            continue
                        if type(elem_con) is Series:
                            ex, ey = positions[elem_con]
                            ew, eh = dimensions[elem_con]
                        else:
                            ex, ey = positions[elem_con]
                            ew, eh = dimensions[elem_con]
                        start_x = (ex + ew) * (node_width - 1.0) + 1.0
                        start_y = ey * node_height
                        # Use the same end_x as the RPar line
                        end_y = start_y
                        line = r"\draw (<start_x>,<start_y>) to[<element>] (<end_x>,<end_y>);"
                        lines.append(
                            replace_variables(line, start_x, start_y, end_x, end_y)
                        )
                elif isinstance(element_connection, Element):
                    start_x = x * (node_width - 1.0) + 1.0
                    start_y = y * node_height
                    end_x = (x + w) * (node_width - 1.0) + 1.0
                    end_y = start_y
                    line = (
                        r"\draw (<start_x>,<start_y>) to[<element>] (<end_x>,<end_y>);"
                    )
                    symbol: str
                    label: str = ""
                    if not hide_labels:
                        symbol, label = element_connection.get_label().split("_")
                        label = f"{symbol}_{{\\rm {label}}}"
                    symbol = symbols.get(type(element_connection), "generic")
                    lines.append(
                        replace_variables(
                            line,
                            max(1.0, start_x),
                            start_y,
                            end_x,
                            end_y,
                            f"{symbol}=${label}$",
                        )
                    )
                elif type(element_connection) is Short:
                    start_x = x * (node_width - 1.0) + 1.0
                    start_y = y * node_height
                    end_x = (x + w) * (node_width - 1.0) + 1.0
                    end_y = start_y
                    lines.append(
                        replace_variables(
                            r"\draw (<start_x>,<start_y>) to[short] (<end_x>,<end_y>);",
                            max(1.0, start_x),
                            start_y,
                            end_x,
                            end_y,
                        )
                    )

        phase_2()
        x, y = positions[self._elements]
        w, h = dimensions[self._elements]
        start_x = (x + w) * (node_width - 1) + 1
        end_x = start_x + 1
        line = (
            r"\draw (<start_x>,<start_y>) to[<element>, -o] (<end_x>,<end_y>)<label>;"
        )
        line = line.replace(
            "<label>", f" node[above]{{{counter_label}}}" if counter_label != "" else ""
        )
        lines.append(replace_variables(line, start_x, 0, end_x, 0))
        source: str = "\n\t".join(lines) + "\n\\end{circuitikz}"
        return source
