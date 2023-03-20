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

from dataclasses import dataclass
from string import (
    ascii_uppercase,
    ascii_lowercase,
    digits,
)
from typing import (
    Dict,
    List,
    Optional,
    Set,
    Type,
    Union,
)
from warnings import warn
from sympy import (
    Expr,
    latex,
    sympify,
)
from numpy import (
    allclose,
    array,
)
from .base import (
    Connection,
    Container,
    Element,
)
from pyimpspec.typing import (
    ComplexImpedance,
    ComplexImpedances,
    Frequencies,
    Frequency,
)


_VALIDATE_IMPEDANCES: bool = False
_ELEMENTS: Dict[str, Type[Element]] = {}


@dataclass(frozen=True)
class ParameterDefinition:
    """
    A circuit element's parameter.

    Parameters
    ----------
    symbol: str
        The symbol/name of the parameter that can be used when, e.g., obtaining the value for a specific parameter via `Element.get_value(symbol)`. Required to be a non-empty string.

    unit: str
        The parameter's unit (ohm*cm, F/cm, H, etc.). Can be an empty string.

    description: str
        A description of the parameter. Can be an empty string.

    value: float
        The default value for the parameter.

    lower_limit: float
        The lower limit of what this parameter's value can be during circuit fitting. A value of -numpy.inf means that there is no lower limit.

    upper_limit: float
        The upper limit of what this parameter's value can be during circuit fitting. A value of numpy.inf means that there is no upper limit.

    fixed: bool
        If true, then this parameter's value will not change during circuit fitting.
    """

    symbol: str
    unit: str
    description: str
    value: float
    lower_limit: float
    upper_limit: float
    fixed: bool

    def __post_init__(self):
        assert self.lower_limit <= self.value, "Invalid lower limit!"
        assert self.value <= self.upper_limit, "Invalid upper limit!"


@dataclass(frozen=True)
class SubcircuitDefinition:
    """
    A container element's subcircuit.

    Parameters
    ----------
    symbol: str
        The symbol/name of the subcircuit that can be used when, e.g., obtaining a specific subcircuit belonging to a container element. Required to be a non-empty string.

    unit: str
        The unit of the subcircuit's impedance.

    description: str
        A description of the subcircuit. Can be an empty string.

    value: Optional[Connection]
        The default value for the parameter. Can be a connection such as Series or Parallel, or None.
    """

    symbol: str
    unit: str
    description: str
    value: Optional[Connection]


@dataclass(frozen=True)
class ElementDefinition:
    r"""
    A class that defines a new circuit element and its parameters.
    The information contained in this object is also used to automatically generate a docstring for the circuit element's class.

    Parameters
    ----------
    Class: Type[Element]
        The class representing the circuit element.

    symbol: str
        The symbol that can be used to represent the circuit element in a circuit description code (CDC).

    name: str
        The name of the circuit element.

    description: str
        The description of the element.
        A placeholder, ``|equation|``, for the automatically formatted equation can be included in the description in case the intention is to have more text below the equation.
        Otherwise the formatted equation will be added to the end of the description.
        Another placeholder, ``|no-equation|``, can be included (e.g., at the end) in case the automatically formatted equation should be excluded for some reason (e.g., it would be too large to be properly displayed).

    equation: str
        The equation (assume an implicit "Z = " on the left-hand side) for the circuit element's impedance.
        The string must be compatible with SymPy's sympify_ function.
        For example, "(2*pi*f*C*I)^-1" for a capacitor is valid and will show up in the generated docstring as :math:`Z = \frac{1}{j 2 \pi f C}`.
        This is also used to verify that the circuit element's _impedance method outputs the same answer as the circuit element's impedance equation.

    parameters: List[ParameterDefinition]
        A list of objects that define the circuit element's parameter(s).
    """

    Class: Type[Element]
    symbol: str
    name: str
    description: str
    equation: str
    parameters: List[ParameterDefinition]


@dataclass(frozen=True)
class ContainerDefinition(ElementDefinition):
    Class: Type[Container]
    subcircuits: List[SubcircuitDefinition]


ContainerDefinition.__doc__ = ElementDefinition.__doc__ + """

    subcircuits: List[SubcircuitDefinition]
        A list of objects that define the circuit element's parameter(s) that are subcircuits.
    """


def _set_element_static_information(
    Class: Type[Element],
    symbol: str,
    name: str,
    description: str,
    equation: str,
    parameters: List[ParameterDefinition],
    subcircuits: List[SubcircuitDefinition],
):
    # General
    assertion_message: str = f"Incomplete definition for element '{Class}'!"
    assert symbol.strip() != "", f"{assertion_message} The symbol is missing!"
    assert name.strip() != "", f"{assertion_message} The name is missing!"
    assert description.strip() != "", f"{assertion_message} The description is missing!"
    if equation.strip().upper() != "TODO":
        assert equation.strip() != "", f"{assertion_message} The equation is missing!"
    else:
        warn(f"{assertion_message} The equation is missing!")
    assert (len(parameters) + len(subcircuits)) == len(
        set([_.symbol for _ in parameters] + [_.symbol for _ in subcircuits])
    ), (
        "Two or more parameters "
        + (" and/or subcircuits" if issubclass(Class, Container) else "")
        + " have non-unique symbols!"
    )
    Class._symbol = symbol.strip()
    Class._name = name.strip()
    Class._description = description.strip()
    Class._equation = equation.strip()
    # Parameters
    Class._parameter_unit = {}
    Class._parameter_description = {}
    Class._parameter_default_value = {}
    Class._parameter_default_lower_limit = {}
    Class._parameter_default_upper_limit = {}
    Class._parameter_default_fixed = {}
    p: ParameterDefinition
    for p in parameters:
        symbol = p.symbol.strip()
        assert symbol != ""
        assert symbol.isidentifier(), symbol
        Class._parameter_unit[symbol] = p.unit.strip()
        Class._parameter_description[symbol] = p.description.strip()
        Class._parameter_default_value[symbol] = p.value
        Class._parameter_default_lower_limit[symbol] = p.lower_limit
        Class._parameter_default_upper_limit[symbol] = p.upper_limit
        Class._parameter_default_fixed[symbol] = p.fixed
    Class._valid_kwargs_keys = set(Class._parameter_default_value.keys())
    # Subcircuits
    if not issubclass(Class, Container):
        return
    Class._subcircuit_unit = {}
    Class._subcircuit_description = {}
    Class._subcircuit_default_value = {}
    s: SubcircuitDefinition
    for s in subcircuits:
        symbol = s.symbol.strip()
        assert symbol != ""
        assert symbol.isidentifier(), symbol
        Class._subcircuit_unit[symbol] = s.unit
        Class._subcircuit_description[symbol] = s.description
        Class._subcircuit_default_value[symbol] = s.value
    Class._valid_kwargs_keys.update(set(Class._subcircuit_default_value.keys()))


def _process_description(Class: Type[Element]):
    description: str = Class._description
    prefix: str = ":math:`Z = "
    expr: Expr = sympify(Class._equation, evaluate=False)
    latex_expr: str = latex(expr, imaginary_unit='j')
    equation: str = f"{prefix}{latex_expr}`"
    if "|no-equation|" in description:
        description = description.replace("|no-equation|", "")
    elif "|equation|" in description:
        description = description.replace("|equation|", equation)
    elif prefix not in description:
        description += f"\n\n{equation}"
    return description.strip()


def _set_element_docstring(
    Class: Type[Element],
    parameters: List[ParameterDefinition],
    subcircuits: List[SubcircuitDefinition],
):
    description: str = _process_description(Class)
    # General
    lines: List[str] = [
        Class._name,
        "",
        f"Symbol: {Class._symbol}",
        "",
        description,
    ]
    # Parameters
    if len(parameters) > 0:
        lines.extend(
            [
                "",
                "Parameters",
                "----------",
            ]
        )
        unit: str
        value: Union[float, str]
        p: ParameterDefinition
        for p in parameters:
            symbol = p.symbol.strip()
            unit = p.unit.strip()
            description = p.description.strip()
            value = p.value
            fixed: bool = p.fixed
            assert isinstance(symbol, str), symbol
            assert isinstance(unit, str), unit
            assert isinstance(description, str), description
            float(value)  # Expected a numeric value
            assert isinstance(fixed, bool), fixed
            lines.append(
                f"{symbol}: float = {value:.3g}"
                + (f" {unit}" if unit != "" else "")
                + (", fixed" if fixed else "")
            )
            if description:
                lines.append(f"\t{description}\n")
    if len(subcircuits) > 0:
        c: SubcircuitDefinition
        for c in subcircuits:
            symbol = c.symbol.strip()
            unit = c.unit.strip()
            description = c.description.strip()
            if c.value is None:
                value = "None"
            else:
                value = c.value.to_string()
            assert isinstance(symbol, str), symbol
            assert isinstance(unit, str), unit
            assert isinstance(description, str), description
            lines.append(
                f"{symbol}: Optional[Connection] = {value}"
                + (f" {unit}" if unit != "" else "")
            )
            if description:
                lines.append(f"\t{description}\n")
    Class.__doc__ = "\n".join(lines)


def _validate_impedances(Class: Type[Element]):
    """
    Makes sure that the outputs of the _impedance method and the SymPy expression match.
    """
    element: Element = Class()
    # Check the impedance calculations of the function and the SymPy expression
    expr: Expr = element.to_sympy(substitute=True)
    f: Frequencies = array([1e-6, 1e-3, 1e0, 1e3, 1e6], dtype=Frequency)
    Z_func: ComplexImpedances = element.get_impedances(f)
    Z_sympy: ComplexImpedances = array(
        list(map(lambda _: complex(expr.subs("f", _)), f)),
        dtype=ComplexImpedance,
    )
    assert allclose(
        Z_func.real,
        Z_sympy.real,
    ), f"The real parts of the results of the _impedance method and SymPy expression do not match for '{Class}'!"
    assert allclose(
        Z_func.imag, Z_sympy.imag
    ), f"The imaginary parts of the results of the _impedance method and SymPy expression do not match for '{Class}'!"


def _validate_element_symbol(symbol: str):
    assert len(symbol) > 0
    # The first character must be a capital letter that acts as a sort of
    # delimiter so that, e.g., 'L' and 'Ls' can be identified correctly.
    try:
        assert symbol[0] in ascii_uppercase
        valid_chars: str = ascii_lowercase + digits + "_"
        char: str
        for char in symbol[1:]:
            assert char in valid_chars
    except AssertionError:
        raise Exception(
            "Element symbols must start with a capital letter and any "
            "additional characters can be lower-case letters, digits, "
            "and/or underscores."
        )


def register_element(definition: ElementDefinition, **kwargs):
    """
    Register a circuit element so that it is supported by the `parse_cdc` and `get_elements` functions. The class definition (symbol, name, etc.) must be provided as keyword arguments.

    Parameters
    ----------
    definition: ElementDefinition
        An ElementDefinition instance that defines a class, which inherits from the Element class and implements a method called `_impedance` that takes a float value (i.e., the excitation frequency in hertz) and returns a complex value.

    **kwargs
    """
    global _ELEMENTS
    assert isinstance(definition, ElementDefinition), type(definition)
    assert isinstance(definition, ContainerDefinition) == issubclass(
        definition.Class, Container
    )
    # Static information
    symbol: str = definition.symbol.strip()
    _validate_element_symbol(symbol)
    Class: Type[Element] = definition.Class
    parameters: List[ParameterDefinition] = definition.parameters
    subcircuits: List[SubcircuitDefinition] = (
        definition.subcircuits if isinstance(definition, ContainerDefinition) else []
    )
    _set_element_static_information(
        Class,
        symbol=symbol,
        name=definition.name.strip(),
        description=definition.description.strip(),
        equation=definition.equation.strip(),
        parameters=parameters,
        subcircuits=subcircuits,
    )
    _set_element_docstring(
        Class,
        parameters,
        subcircuits,
    )
    if kwargs.get("validate_impedances", _VALIDATE_IMPEDANCES):
        _validate_impedances(Class)
    # Make available via the get_elements function
    assert (
        symbol not in _ELEMENTS or _ELEMENTS[symbol] == Class
    ), f"An element with the symbol '{symbol}' ({_ELEMENTS[symbol]}) has already been registered before this attempt to register '{Class}'!"
    _ELEMENTS[symbol] = Class


def get_elements() -> Dict[str, Type[Element]]:
    """
    Returns a dictionary that maps element symbols to their corresponding classes.

    Returns
    -------
    Dict[str, Type[Element]]
    """
    elements: Dict[str, Type[Element]] = {
        _: _ELEMENTS[_] for _ in sorted(list(_ELEMENTS.keys()))
    }
    return elements
