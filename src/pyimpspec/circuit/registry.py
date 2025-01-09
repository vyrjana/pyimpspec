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

from dataclasses import dataclass
from string import (
    ascii_uppercase,
    ascii_lowercase,
    digits,
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
from pyimpspec.typing.helpers import (
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    Union,
    _is_boolean,
)


_VALIDATE_IMPEDANCES: bool = False
_ELEMENTS: Dict[str, Type[Element]] = {}
_DEFAULT_ELEMENTS: Dict[str, Type[Element]] = {}
_DEFAULT_ELEMENT_PARAMETERS: Dict[str, Dict[str, float]] = {}
_PRIVATE_ELEMENTS: Dict[str, Type[Element]] = {}


def _initialized():
    global _VALIDATE_IMPEDANCES

    _VALIDATE_IMPEDANCES = True
    _DEFAULT_ELEMENTS.update(_ELEMENTS)

    key: str
    element: Type[Element]
    for key, element in _DEFAULT_ELEMENTS.items():
        _DEFAULT_ELEMENT_PARAMETERS[key] = element.get_default_values().copy()


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
        if self.value < self.lower_limit:
            raise ValueError(
                f"Expected the current value {self.value=} to be greater than or equal to the lower limit {self.lower_limit=}"
            )

        if self.value > self.upper_limit:
            raise ValueError(
                f"Expected the current value {self.value=} to be less than or equal to the upper limit {self.upper_limit=}"
            )


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

    value: Optional[|Connection|]
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
    Class: Type[|Element|]
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
        This is also used to verify that the circuit element's ``_impedance`` method outputs the same answer as the circuit element's impedance equation.

    parameters: List[|ParameterDefinition|]
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


ContainerDefinition.__doc__ = (
    ElementDefinition.__doc__
    + """

    subcircuits: List[SubcircuitDefinition]
        A list of objects that define the circuit element's parameter(s) that are subcircuits.
    """
)


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

    if symbol.strip() == "":
        raise ValueError(f"{assertion_message} The symbol is missing!")

    if name.strip() == "":
        raise ValueError(f"{assertion_message} The name is missing!")

    if description.strip() == "":
        raise ValueError(f"{assertion_message} The description is missing!")

    if equation.strip().upper() != "TODO":
        if equation.strip() == "":
            raise ValueError(f"{assertion_message} The equation is missing!")
    elif equation.strip().upper() != "IGNORE":
        warn(f"{assertion_message} The equation is missing!")

    if (len(parameters) + len(subcircuits)) != len(
        set([_.symbol for _ in parameters] + [_.symbol for _ in subcircuits])
    ):
        raise KeyError(
            "Two or more parameters"
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

        if symbol == "":
            raise ValueError("Expected a non-empty string symbol")

        if not symbol.isidentifier():
            raise KeyError(
                "Expected a valid symbol (i.e., symbol.isidentifier() == True)"
            )

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

        if symbol == "":
            raise ValueError("Expected a non-empty string symbol")

        if not symbol.isidentifier():
            raise KeyError(
                "Expected a valid symbol (i.e., symbol.isidentifier() == True)"
            )

        Class._subcircuit_unit[symbol] = s.unit
        Class._subcircuit_description[symbol] = s.description
        Class._subcircuit_default_value[symbol] = s.value

    Class._valid_kwargs_keys.update(set(Class._subcircuit_default_value.keys()))


def _process_description(Class: Type[Element]):
    description: str = Class._description
    prefix: str = ":math:`Z = "
    expr: Expr = sympify(Class._equation, evaluate=False)
    latex_expr: str = latex(expr, imaginary_unit="j")
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
            if not isinstance(symbol, str):
                raise TypeError(f"Expected a string instead of {symbol=}")

            unit = p.unit.strip()
            if not isinstance(unit, str):
                raise TypeError(f"Expected a string instead of {unit=}")

            description = p.description.strip()
            if not isinstance(description, str):
                raise TypeError(f"Expected a string instead of {description=}")

            value = float(p.value)  # Expected a numeric value

            fixed: bool = p.fixed
            if not _is_boolean(fixed):
                raise TypeError(f"Expected a boolean instead of {fixed=}")

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
            if not isinstance(symbol, str):
                raise TypeError(f"Expected a string instead of {symbol=}")

            unit = c.unit.strip()
            if not isinstance(unit, str):
                raise TypeError(f"Expected a string instead of {unit=}")

            description = c.description.strip()
            if not isinstance(description, str):
                raise TypeError(f"Expected a string instead of {description=}")

            if c.value is None:
                value = "None"
            else:
                value = c.value.to_string()
            if not isinstance(value, str):
                raise TypeError(f"Expected a string instead of {value=}")

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
    f: Frequencies = array([1e6, 1e3, 1e0, 1e-3, 1e-6], dtype=Frequency)

    Z_func: ComplexImpedances = element.get_impedances(f)
    Z_sympy: ComplexImpedances = array(
        list(map(lambda _: complex(expr.subs("f", _)), f)),
        dtype=ComplexImpedance,
    )

    if not allclose(
        Z_func.real,
        Z_sympy.real,
    ):
        raise ValueError(
            f"The real parts of the results of the _impedance method and SymPy expression do not match for '{Class}'!"
        )

    if not allclose(Z_func.imag, Z_sympy.imag):
        raise ValueError(
            f"The imaginary parts of the results of the _impedance method and SymPy expression do not match for '{Class}'!"
        )


def _validate_element_symbol(symbol: str):
    if not isinstance(symbol, str):
        raise TypeError(f"Expected a string instead of {symbol=}")

    if symbol.strip() == "":
        raise ValueError(f"Expected a non-empty string instead of {symbol=}")

    # The first character must be a capital letter that acts as a sort of
    # delimiter so that, e.g., 'L' and 'Ls' can be identified correctly.
    if symbol[0] not in ascii_uppercase:
        raise ValueError(
            f"Expected the first character to be an upper-case ASCII letter instead of {symbol[0]=}"
        )

    valid_chars: str = ascii_lowercase + digits + "_"
    char: str
    for char in symbol[1:]:
        if char not in valid_chars:
            raise ValueError(
                f"Expected a character that exists in '{valid_chars}' instead of {char=}"
            )


def _initialize_element(
    definition: ElementDefinition,
    **kwargs,
) -> Tuple[str, Type[Element]]:
    if not isinstance(definition, ElementDefinition):
        raise TypeError(f"Expected an ElementDefinition instead of {definition=}")

    if isinstance(definition, ContainerDefinition) != issubclass(
        definition.Class, Container
    ):
        raise TypeError(
            f"Expected the element class of the ContainerDefinition to be a subclass of the Container class instead of {definition.Class=}"
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

    return (symbol, Class)


def get_elements(default_only: bool = False, private: bool = False) -> Dict[str, Type[Element]]:
    """
    Returns a dictionary that maps element symbols to their corresponding classes.

    Parameters
    ----------
    default_only: bool, optional
        Return only the elements that are included by default in pyimpspec (i.e., exclude user-defined elements).

    private: bool, optional
        Include elements that have been marked as private.

    Returns
    -------
    Dict[str, Type[|Element|]]
    """
    if not _is_boolean(default_only):
        raise TypeError(f"Expected a boolean instead of {default_only=}")

    keys: List[str] = sorted(list((_DEFAULT_ELEMENTS if default_only else _ELEMENTS).keys()))
    if not private:
        keys = [k for k in keys if k not in _PRIVATE_ELEMENTS]

    elements: Dict[str, Type[Element]] = {k: _ELEMENTS[k] for k in keys}

    return elements


def register_element(definition: ElementDefinition, **kwargs):
    """
    Register a circuit element so that it is supported by the `parse_cdc` and `get_elements` functions. The class definition (symbol, name, etc.) must be provided as keyword arguments.

    Parameters
    ----------
    definition: |ElementDefinition|
        An ElementDefinition instance that defines a class, which inherits from the Element class and implements a method called `_impedance` that takes a float value (i.e., the excitation frequency in hertz) and returns a complex value.

    **kwargs
    """
    global _ELEMENTS

    symbol: str
    Class: Type[Element]
    symbol, Class = _initialize_element(definition, **kwargs)

    # Make available via the get_elements function
    if not (symbol not in _ELEMENTS or _ELEMENTS[symbol] == Class):
        raise KeyError(
            f"An element with the symbol '{symbol}' ({_ELEMENTS[symbol]}) has already been registered before this attempt to register '{Class}'!"
        )

    _ELEMENTS[symbol] = Class
    if kwargs.get("private", False) is True:
        _PRIVATE_ELEMENTS[symbol] = Class


def reset_default_parameter_values(elements: Optional[Union[Type[Element], List[Type[Element]]]] = None):
    """
    Reset the default values for parameters of one or more elements.

    Parameters
    ----------
    elements: Optional[Union[Type[|Element|], List[Type[|Element|]]]], optional
        Specific element class(es) to reset. If none are provided, then all the elements that are included by default in pyimpspec are reset.
    """
    if elements is None:
        elements = list(get_elements(default_only=True).values())
    elif isinstance(elements,  list):
        if not all(map(lambda element: issubclass(element, Element), elements)):
            raise TypeError(f"Expected a list of Type[Element] instead of {elements=}")
        elif len(elements) < 1:
            raise ValueError(f"Expected a list with at least one item instead of {elements=}")
    elif issubclass(elements, Element):
        elements = [elements]
    else:
        raise TypeError(f"Expected either Type[Element] or List[Type[Element]] instead of {elements=}")

    assert isinstance(elements, list)

    key: str
    element: Type[Element]
    for key, element in _DEFAULT_ELEMENTS.items():
        if element in elements:
            element.set_default_values(**_DEFAULT_ELEMENT_PARAMETERS[key])


def reset(elements: bool = True, default_parameters: bool = True):
    """
    Remove all user-defined elements from the registry.

    Parameters
    ----------
    elements: bool, optional
        If true, then remove any user-defined elements.

    default_parameters: bool, optional
        If true, reset the default parameters of all elements.
    """
    if elements:
        _ELEMENTS.clear()
        _ELEMENTS.update(_DEFAULT_ELEMENTS)

    if default_parameters:
        reset_default_parameter_values()


def remove_elements(elements: Union[Type[Element], List[Type[Element]]]):
    """
    Remove a specific user-defined element from the registry.

    Parameters
    ----------
    elements: Union[Type[|Element|], List[Type[|Element|]]]
        The element(s) to remove.
    """
    if isinstance(elements, list):
        if not all(map(lambda element: issubclass(element, Element), elements)):
            raise TypeError(f"Expected a list of Type[Element] instead of {elements=}")
        elif len(elements) < 1:
            raise ValueError(f"Expected a list with at least one item instead of {elements=}")
    elif issubclass(elements, Element):
        elements = [elements]
    else:
        raise TypeError(f"Expected either Type[Element] or List[Type[Element]] instead of {elements=}")

    default_elements: List[Type[Element]] = list(_DEFAULT_ELEMENTS.values())

    element: Type[Element]
    for element in elements:
        if element in default_elements:
            raise ValueError(
                f"Expected a user-defined element instead of one of the default elements {element=}"
            )

    for element in elements:
        key: str
        for key in list(_ELEMENTS.keys()):
            if _ELEMENTS[key] is element:
                _ELEMENTS.pop(key)
                if key in _PRIVATE_ELEMENTS and _PRIVATE_ELEMENTS[key] is element:
                    _PRIVATE_ELEMENTS.pop(key)
                break
