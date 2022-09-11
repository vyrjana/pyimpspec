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

from collections import OrderedDict
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)
from .base import (
    Connection,
    Element,
)
from .series import Series
from .parallel import Parallel
from .resistor import Resistor
from .capacitor import Capacitor
from .inductor import Inductor
from .constant_phase_element import ConstantPhaseElement
from numpy import (
    array,
    inf,
    ndarray,
)
from schemdraw import Drawing
import schemdraw.elements as elm
from sympy import (
    Expr,
    latex,
)


ElementParameters = Dict[str, Dict[str, str]]


class Circuit:
    """
    A class that represents an equivalent circuit.

    Parameters
    ----------
    elements: Series
        The elements of the circuit wrapped in a Series connection.

    label: str = ""
        The label assigned to the circuit.
    """

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
        """
        Get the label assigned to this circuit.

        Returns
        -------
        str
        """
        return self._label

    def set_label(self, label: str):
        """
        Set the label assigned to this circuit.

        Parameters
        ----------
        label: str
            The new label.
        """
        assert type(label) is str
        self._label = label.strip()

    def _assign_identifiers(self):
        return self._elements._assign_identifier(0)

    def to_stack(self) -> List[Tuple[str, Union[Element, Connection]]]:
        stack: List[Tuple[str, Union[Element, Connection]]] = []
        self._elements.to_stack(stack)
        return stack

    def to_string(self, decimals: int = -1) -> str:
        """
        Generate the circuit description code (CDC) that represents this circuit.

        Parameters
        ----------
        decimals: int = -1
            The number of decimals to include for the current element parameter values and limits.
            -1 means that the CDC is generated using the basic syntax, which omits element labels, parameter values, and parameter limits.

        Returns
        -------
        str
        """
        return self._elements.to_string(decimals=decimals)

    def impedance(self, f: float) -> complex:
        """
        Calculate the impedance of this circuit at a single frequency.

        Parameters
        ----------
        f: float
            The frequency in hertz.

        Returns
        -------
        complex
        """
        assert f > 0 and f < inf
        return self._elements.impedance(f)

    def impedances(self, f: Union[list, ndarray]) -> ndarray:
        """
        Calculate the impedance of this circuit at multiple frequencies.

        Parameters
        ----------
        f: Union[list, ndarray]

        Returns
        -------
        ndarray
        """
        assert type(f) is list or type(f) is ndarray, f
        assert min(f) > 0 and max(f) < inf, f
        return array(list(map(self.impedance, f)))

    def get_connections(self, flattened: bool = True) -> List[Connection]:
        """
        Get the connections in this circuit.

        Parameters
        ----------
        flattened: bool = True
            Whether or not the connections should be returned as a list of all connections or as a list connections that may also contain more connections.

        Returns
        -------
        List[Connection]
        """
        if flattened is True:
            return self._elements.get_connections(flattened=flattened)
        return [self._elements]

    def get_elements(self, flattened: bool = True) -> List[Union[Element, Connection]]:
        """
        Get the elements in this circuit.

        Parameters
        ----------
        flattened: bool = True
            Whether or not the elements should be returned as a list of only elements or as a list of connections containing elements.

        Returns
        -------
        List[Union[Element, Connection]]
        """
        if flattened is True:
            return self._elements.get_elements(flattened=flattened)
        return [self._elements]

    def get_parameters(self) -> "Dict[int, OrderedDict[str, float]]":
        """
        Get a mapping of each circuit element's integer identifier to an OrderedDict representing that element's parameters.

        Returns
        -------
        Dict[int, OrderedDict[str, float]]
        """
        return self._elements.get_parameters()

    def set_parameters(self, parameters: Dict[int, Dict[str, float]]):
        """
        Assign new parameters to the circuit elements.

        Parameters
        ----------
        parameters: Dict[int, Dict[str, float]]
            A mapping of circuit element integer identifiers to an OrderedDict mapping the parameter symbol to the new value.
        """
        self._elements.set_parameters(parameters)

    def get_element(self, ident: int) -> Optional[Element]:
        """
        Get the circuit element with a given integer identifier.

        Parameters
        ----------
        ident: int
            The integer identifier corresponding to an element in the circuit.

        Returns
        -------
        Optional[Element]
        """
        return self._elements.get_element(ident)

    def substitute_element(self, ident: int, element: Element):
        """
        Substitute the element with the given integer identifier in the circuit with another element.

        Parameters
        ----------
        ident: int
            The integer identifier corresponding to an element in the circuit.

        element: Element
            The new element that will substitute the old element.
        """
        assert (
            self._elements.substitute_element(ident, element) is True
        ), f"Failed to find an element with the identifier '{ident}'!"

    def to_sympy(self, substitute: bool = False) -> Expr:
        """
        Get the SymPy expression corresponding to this circuit's impedance.

        Parameters
        ----------
        substitute: bool = False
            Whether or not the variables should be substituted with the current values.

        Returns
        -------
        Expr
        """
        expr: Expr = self._elements.to_sympy(substitute=substitute)
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

    def to_drawing(
        self,
        node_height: float = 1.5,
        working_label: str = "WE",
        counter_label: str = "CE+RE",
        hide_labels: bool = False,
    ) -> Drawing:
        """
        Get a schemdraw.Drawing object to draw a circuit diagram using the matplotlib backend.

        Parameters
        ----------
        node_height: float = 1.5
            The height of each node.

        working_label: str = "WE"
            The label assigned to the terminal representing the working and working sense electrodes.

        counter_label: str = "CE+RE"
            The label assigned to the terminal representing the counter and reference electrodes.

        hide_labels: bool = False
            Whether or not to hide element and terminal labels.

        Returns
        -------
        Drawing
        """
        lookup: Dict[Type[Element], elm.Element] = {
            Resistor: elm.ResistorIEEE,
            Capacitor: elm.Capacitor,
            ConstantPhaseElement: elm.CPE,
            Inductor: elm.Inductor2,
        }

        def draw_element(elem: Element, drawing: Drawing):
            element: elm.Element = lookup.get(type(elem), elm.ResistorIEC)()
            symbol: str = elem.get_symbol()
            if not hide_labels:
                label: str = elem.get_label()[len(symbol) + 1 :]
                element.label(f"${symbol}_" + r"{\rm " + f"{label}}}$")
            drawing.add(element.right())

        def get_width(
            element_connection: Union[Element, Connection],
        ) -> int:
            widths: List[int] = []
            if isinstance(element_connection, Element):
                widths.append(2)
            elif type(element_connection) is Series:
                for elem_con in element_connection.get_elements(flattened=False):
                    widths.append(
                        get_width(elem_con) + (1 if type(elem_con) is Parallel else 0)
                    )
                widths = [sum(widths)]
            elif type(element_connection) is Parallel:
                for elem_con in element_connection.get_elements(flattened=False):
                    widths.append(get_width(elem_con))
            assert len(widths) > 0
            return max(widths)

        def get_height(element_connection: Union[Element, Connection]) -> int:
            heights: List[int] = []
            if isinstance(element_connection, Element):
                heights.append(node_height)
            elif type(element_connection) is Series:
                for elem_con in element_connection.get_elements(flattened=False):
                    heights.append(get_height(elem_con))
            elif type(element_connection) is Parallel:
                for elem_con in element_connection.get_elements(flattened=False):
                    heights.append(get_height(elem_con))
                heights = [sum(heights)]
            assert len(heights) > 0
            return max(heights)

        def draw_parallel(parallel: Parallel, drawing: Drawing):
            elements_connections: List[Union[Element, Connection]]
            elements_connections = parallel.get_elements(flattened=False)
            heights: List[int] = list(map(get_height, elements_connections))
            i: int
            height: int
            for i, height in enumerate(heights):
                if i < len(elements_connections) - 1:
                    drawing.push()
                    drawing.add(elm.Line(l=height).down())
            total_width: int = get_width(parallel)
            elem_con: Union[Element, Connection]
            for (i, elem_con) in reversed(list(enumerate(elements_connections))):
                width: int = get_width(elem_con)
                padding: int = total_width - width
                if isinstance(elem_con, Connection):
                    if type(elem_con) is Series:
                        draw_series(elem_con, drawing)
                    elif type(elem_con) is Parallel:
                        draw_parallel(elem_con, drawing)
                    else:
                        raise Exception("Unsupported connection: {type(elem_con)=}")
                else:
                    draw_element(elem_con, drawing)
                    if padding > 0:
                        padding += 1
                if padding > 0:
                    drawing.add(elm.Line(l=padding))
                if i > 0:
                    drawing.add(elm.Line(l=heights[i - 1]).up())
                    drawing.pop()

        def draw_series(series: Series, drawing: Drawing):
            i: int
            elem_con: Union[Element, Connection]
            for i, elem_con in enumerate(series.get_elements(flattened=False)):
                if isinstance(elem_con, Connection):
                    if type(elem_con) is Series:
                        draw_series(elem_con, drawing)
                    elif type(elem_con) is Parallel:
                        if i == 0:
                            drawing.add(elm.Line(l=1).right())
                        draw_parallel(elem_con, drawing)
                        if i > 0:
                            drawing.add(elm.Line(l=1))
                    else:
                        raise Exception("Unsupported connection: {type(elem_con)=}")
                else:
                    draw_element(elem_con, drawing)

        drawing: Drawing = Drawing()
        we_dot: elm.Dot = elm.Dot(open=True)
        if not hide_labels:
            we_dot.label(working_label)
        drawing.add(we_dot)
        connection: Connection
        for connection in self.get_connections(flattened=False):
            if type(connection) is Series:
                draw_series(connection, drawing)
            elif type(connection) is Parallel:
                draw_parallel(connection, drawing)
        ce_dot: elm.Dot = elm.Dot(open=True)
        if not hide_labels:
            ce_dot.label(counter_label)
        drawing.add(ce_dot)
        return drawing

    def to_circuitikz(
        self,
        node_width: float = 3.0,
        node_height: float = 1.5,
        working_label: str = "WE",
        counter_label: str = "CE+RE",
        hide_labels: bool = False,
    ) -> str:
        """
        Get the LaTeX source needed to draw a circuit diagram for this circuit using the circuitikz package.

        Parameters
        ----------
        node_width: float = 3.0
            The width of each node.

        node_height: float = 1.5
            The height of each node.

        working_label: str = "WE"
            The label assigned to the terminal representing the working and working sense electrodes.

        counter_label: str = "CE+RE"
            The label assigned to the terminal representing the counter and reference electrodes.

        hide_labels: bool = False
            Whether or not to hide element and terminal labels.

        Returns
        -------
        str
        """
        assert node_width > 0
        assert node_height > 0
        assert type(working_label) is str
        assert type(counter_label) is str
        assert type(hide_labels) is bool
        if hide_labels:
            working_label = ""
            counter_label = ""
        # Phase 1 - figure out the dimensions of the connections and the positions of elements.
        short_counter: int = 0
        dimensions: Dict[
            Union[Series, Parallel, Element, int], Tuple[float, float]
        ] = {}
        positions: Dict[Union[Series, Parallel, Element, int], Tuple[float, float]] = {}
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
                elif type(element_connection) is int:
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
