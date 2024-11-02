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
    Tuple,
    Type,
    Union,
    _is_floating,
)


def to_circuitikz(
    self,
    node_width: float = 3.0,
    node_height: float = 1.5,
    left_terminal_label: str = "",
    right_terminal_label: str = "",
    hide_labels: bool = False,
    running: bool = False,
    custom_labels: Optional[Dict[Element, str]] = None,
) -> str:
    """
    Get the LaTeX source needed to draw a circuit diagram for this circuit using the CircuiTikZ_ package.

    Parameters
    ----------
    node_width: float, optional
        The width of each node.

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

    Returns
    -------
    str
    """
    if not _is_floating(node_width):
        raise TypeError(f"Expected a float instead of {node_width=}")
    elif node_width <= 0.0:
        raise ValueError(f"Expected a value greater than 0.0 instead of {node_width=}")

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

    if hide_labels:
        left_terminal_label = ""
        right_terminal_label = ""

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

    # Phase 1 - figure out the dimensions of the connections and the positions of elements.
    short_counter: int = 0
    dimensions: Dict[Union[Series, Parallel, Element, int], Tuple[float, float]] = {}
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

    def phase_1_element(element: Element, x: float, y: float) -> Tuple[float, float]:
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

        elements: List[Union[Element, Connection]] = list(iter(series))
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
                if not isinstance(element_connection, Element):
                    raise TypeError(
                        f"Expected an Element instead of {element_connection=}"
                    )

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

    def phase_1_parallel(parallel: Parallel, x: float, y: float) -> Tuple[float, float]:
        nonlocal num_nested_parallels
        num_nested_parallels += 1

        width: float = 0.0
        height: float = 0.0

        for element_connection in parallel:
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
                if not isinstance(element_connection, Element):
                    raise TypeError(
                        f"Expected an Element instead of {element_connection=}"
                    )

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

    main_connection: Series
    if not isinstance(self, Connection):
        main_connection = self._elements
    elif not isinstance(self, Series):
        main_connection = Series([self])
    else:
        main_connection = self

    if not isinstance(main_connection, Series):
        raise TypeError(f"Expected a Series instead of {main_connection=}")

    phase_1_series(main_connection, 0, 0)

    if set(dimensions.keys()) != set(positions.keys()):
        raise ValueError(
            f"Expected matching sets of keys for dimensions and positions instead of {set(dimensions.keys())=} and {set(positions.keys())=}"
        )

    # Phase 2 - generate the LaTeX source for drawing the circuit diagram.
    lines: List[str] = [
        r"\begin{circuitikz}",
        r"\draw (0,0) <label>to[short, o-] (1,0);".replace(
            "<label>",
            f"node[above]{{{left_terminal_label}}} "
            if left_terminal_label != ""
            else "",
        ),
    ]

    line: str
    symbols: Dict[Type[Element], str] = {
        Resistor: "R",
        Capacitor: "capacitor",
        Inductor: "L",
        ModifiedInductor: "L",
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

                if start_y == end_y:
                    raise ValueError(f"Expected {start_y=} != {end_y=}")

                start_y *= node_height
                end_y *= node_height

                line = r"\draw (<start_x>,<start_y>) to[<element>] (<start_x>,<end_y>);"
                lines.append(replace_variables(line, start_x, start_y, end_x, end_y))
                line = r"\draw (<end_x>,<start_y>) to[<element>] (<end_x>,<end_y>);"
                lines.append(replace_variables(line, start_x, start_y, end_x, end_y))

                if w == 1.0:
                    continue

                for elem_con in filter(lambda _: type(_) is not Parallel, dimensions):
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
                    line = (
                        r"\draw (<start_x>,<start_y>) to[<element>] (<end_x>,<end_y>);"
                    )
                    lines.append(
                        replace_variables(line, start_x, start_y, end_x, end_y)
                    )

            elif isinstance(element_connection, Element):
                start_x = x * (node_width - 1.0) + 1.0
                start_y = y * node_height
                end_x = (x + w) * (node_width - 1.0) + 1.0
                end_y = start_y
                line = r"\draw (<start_x>,<start_y>) to[<element>] (<end_x>,<end_y>);"

                symbol: str
                label: str = ""
                if not hide_labels:
                    if (
                        custom_labels is not None
                        and element_connection in custom_labels
                    ):
                        label = custom_labels[element_connection]
                    else:
                        symbol = element_connection.get_symbol()
                        label = element_connection.get_label() or str(
                            identifiers[element_connection]
                        )
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
    x, y = positions[main_connection]
    w, h = dimensions[main_connection]

    start_x = (x + w) * (node_width - 1) + 1
    end_x = start_x + 1

    line = r"\draw (<start_x>,<start_y>) to[<element>, -o] (<end_x>,<end_y>)<label>;"
    line = line.replace(
        "<label>",
        f" node[above]{{{right_terminal_label}}}" if right_terminal_label != "" else "",
    )
    lines.append(replace_variables(line, start_x, 0, end_x, 0))

    source: str = "\n  ".join(lines) + "\n\\end{circuitikz}"

    return source
