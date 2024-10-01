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

from pyimpspec.circuit.base import (
    Connection,
    Element,
)
from pyimpspec.circuit.series import Series
from pyimpspec.circuit.parallel import Parallel
from pyimpspec.circuit.resistor import Resistor
from pyimpspec.circuit.capacitor import Capacitor
from pyimpspec.circuit.inductor import (
    Inductor,
    ModifiedInductor,
)
from pyimpspec.circuit.constant_phase_element import ConstantPhaseElement
from pyimpspec.typing.helpers import (
    Dict,
    List,
    Optional,
    Type,
    Union,
    _is_floating,
)


def to_drawing(
    self,
    node_height: float = 1.5,
    left_terminal_label: str = "",
    right_terminal_label: str = "",
    hide_labels: bool = False,
    running: bool = False,
    custom_labels: Optional[Dict[Element, str]] = None,
    canvas: Optional[Union[str, "Axes"]] = None,  # noqa: F821
) -> "Drawing":  # noqa: F821
    """
    Get a |Drawing| object for drawing a circuit diagram using, e.g., the matplotlib_ backend.

    Parameters
    ----------
    node_height: float, optional
        The height of each node.

    left_terminal_label: str, optional
        The label assigned to the terminal representing the working and working sense electrodes.

    right_terminal_label: str, optional
        The label assigned to the terminal representing the counter and reference electrodes.

    hide_labels: bool, optional
        Whether or not to hide element and terminal labels.

    running: bool, optional
        Whether or not to use running counts as the lower indices of elements.

    custom_labels: Optional[Dict[|Element|, str]], optional
        A mapping of elements to their custom labels that are used instead of the automatically generated labels.
        The labels can make use of LaTeX's math mode.

    canvas: Optional[Union[str, Axes]], optional
        The canvas that the _schemdraw.Drawing class should use.

    Returns
    -------
    |Drawing|
    """
    from schemdraw import Drawing
    import schemdraw.elements as elm

    if not _is_floating(node_height):
        raise TypeError(f"Expected a float instead of {node_height=}")
    elif node_height <= 0.0:
        raise ValueError(f"Expected a value greater than 0.0 instead of {node_height=}")

    if not isinstance(left_terminal_label, str):
        raise TypeError(f"Expected a string instead of {left_terminal_label=}")

    if not isinstance(right_terminal_label, str):
        raise TypeError(f"Expected a string instead of {right_terminal_label=}")

    if not isinstance(hide_labels, bool):
        raise TypeError(f"Expected a boolean instead of {hide_labels=}")

    if not isinstance(running, bool):
        raise TypeError(f"Expected a boolean instead of {running=}")

    if custom_labels is None:
        pass
    elif not isinstance(custom_labels, dict):
        raise TypeError(f"Expected a dictionary or None instead of {custom_labels=}")
    elif not all(map(lambda key: isinstance(key, Element), custom_labels.keys())):
        raise TypeError(
            f"Expected all keys in {custom_labels=} to be Element instances"
        )
    elif not all(map(lambda value: isinstance(value, str), custom_labels.values())):
        raise TypeError(f"Expected all values in {custom_labels=} to be strings")

    identifiers: Dict[Element, int] = self.generate_element_identifiers(running=running)
    lookup: Dict[Type[Element], Type[elm.Element]] = {
        Resistor: elm.ResistorIEEE,
        Capacitor: elm.Capacitor,
        ConstantPhaseElement: elm.CPE,
        Inductor: elm.Inductor2,
        ModifiedInductor: elm.Inductor2,
    }
    unit_width: float = 2.0

    def draw_element(elem: Element, drawing: Drawing):
        element: elm.Element = lookup.get(type(elem), elm.ResistorIEC)()

        if not hide_labels:
            if custom_labels is not None and elem in custom_labels:
                element.label(custom_labels[elem])
            else:
                symbol: str = elem.get_symbol()
                label: str = elem.get_label() or str(identifiers[elem])
                element.label(f"${symbol}_" + r"{\rm " + f"{label}}}$")

        drawing.add(element.right())

    def get_width(element_connection: Union[Element, Connection]) -> float:
        if isinstance(element_connection, Element):
            return unit_width

        widths: List[float] = []
        if isinstance(element_connection, Series):
            for elem_con in element_connection:
                widths.append(
                    get_width(elem_con)
                    # Spacing around a parallel connection nested within a series connection
                    + (1.0 if isinstance(elem_con, Parallel) else 0.0)
                )

            if len(widths) < 1:
                raise ValueError(f"Expected at least one item instead of {widths=}")

            return sum(widths)

        elif isinstance(element_connection, Parallel):
            for elem_con in element_connection:
                widths.append(get_width(elem_con))

            if len(widths) < 1:
                raise ValueError(f"Expected at least one item instead of {widths=}")

            return max(widths)

        else:
            raise TypeError("Unsupported type: {type(element_connection)}")

    def get_height(element_connection: Union[Element, Connection]) -> float:
        if isinstance(element_connection, Element):
            return node_height

        heights: List[float] = []
        if isinstance(element_connection, Series):
            for elem_con in element_connection:
                heights.append(get_height(elem_con))

            if len(heights) < 1:
                raise ValueError(f"Expected at least one item instead of {heights=}")

            return max(heights)

        elif isinstance(element_connection, Parallel):
            for elem_con in element_connection:
                heights.append(get_height(elem_con))
            if len(heights) < 1:
                raise ValueError(f"Expected at least one item instead of {heights=}")

            return sum(heights)

        else:
            raise TypeError("Unsupported type: {type(element_connection)}")

    def draw_parallel(parallel: Parallel, drawing: Drawing):
        elements_connections: List[Union[Element, Connection]]
        elements_connections = list(iter(parallel))
        heights: List[float] = list(map(get_height, elements_connections))

        i: int
        height: float
        for i, height in enumerate(heights):
            if i < len(elements_connections) - 1:
                drawing.push()
                drawing.add(elm.Line(l=height).down())

        total_width: float = get_width(parallel)

        elem_con: Union[Element, Connection]
        for i, elem_con in reversed(list(enumerate(elements_connections))):
            width: float = get_width(elem_con)
            if width > total_width:
                raise ValueError(f"Expected {width=} <= {total_width=} for {elem_con=}")

            padding: float = total_width - width

            if isinstance(elem_con, Element):
                draw_element(elem_con, drawing)
            elif isinstance(elem_con, Series):
                draw_series(elem_con, drawing)
            elif isinstance(elem_con, Parallel):
                draw_parallel(elem_con, drawing)
            else:
                raise TypeError("Unsupported type: {type(elem_con)=}")

            if padding > 0:
                drawing.add(elm.Line(l=padding).right())

            if i > 0:
                drawing.add(elm.Line(l=heights[i - 1]).up())
                drawing.pop()

    def draw_series(series: Series, drawing: Drawing, outermost: bool = False):
        elements: List[Union[Element, Connection]] = list(iter(series))

        i: int
        elem_con: Union[Element, Connection]
        for i, elem_con in enumerate(elements):
            if isinstance(elem_con, Element):
                draw_element(elem_con, drawing)

            elif isinstance(elem_con, Series):
                draw_series(elem_con, drawing)

            elif isinstance(elem_con, Parallel):
                if not outermost:
                    drawing.add(elm.Line(l=0.5).right())
                draw_parallel(elem_con, drawing)
                if not outermost or (
                    i < len(elements) - 1 and isinstance(elements[i + 1], Parallel)
                ):
                    drawing.add(elm.Line(l=0.5).right())

            else:
                raise TypeError("Unsupported type: {type(elem_con)}")

    drawing: Drawing = Drawing(canvas=canvas)
    drawing.config(unit=unit_width)
    we_dot: elm.Dot = elm.Dot(open=True)

    if not hide_labels:
        we_dot.label(left_terminal_label)

    drawing.add(we_dot)
    drawing.add(elm.Line(l=1.0).right())

    connections: List[Connection]
    if isinstance(self, Connection):
        if isinstance(self, Series):
            connections = [self]
        else:
            connections = [Series([self])]
    else:
        # isinstance(self, Circuit) == True
        connections = self.get_connections(recursive=False)

    if not isinstance(connections, list):
        raise TypeError(f"Expected a list instead of {connections=}")
    elif len(connections) != 1:
        raise ValueError(f"Expected a list with one item instead of {connections=}")
    elif not isinstance(connections[0], Series):
        raise TypeError(f"Expected a Series instead of {connections[0]=}")

    draw_series(connections[0], drawing, outermost=True)
    drawing.add(elm.Line(l=1.0).right())
    ce_dot: elm.Dot = elm.Dot(open=True)

    if not hide_labels:
        ce_dot.label(right_terminal_label)

    drawing.add(ce_dot)

    return drawing
