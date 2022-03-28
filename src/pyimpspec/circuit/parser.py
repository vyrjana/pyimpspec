# Copyright 2022 pyimpspec developers
# pyimpspec is licensed under the GPLv3 or later (https://www.gnu.org/licenses/gpl-3.0.html).
# The licenses of pyimpspec's dependencies and/or sources of portions of code are included in
# the LICENSES folder.

from numpy import inf
from typing import Dict, List, Union, Type, Tuple, Optional
from .base import Element, Connection
from .parallel import Parallel
from .series import Series
from .kk import KramersKronigRC
from .resistor import Resistor
from .capacitor import Capacitor
from .inductor import Inductor
from .constant_phase_element import ConstantPhaseElement
from .gerischer import Gerischer
from .havriliak_negami import HavriliakNegami
from .warburg import Warburg, WarburgOpen, WarburgShort
from .de_levie import deLevieFiniteLength
from .circuit import Circuit
from .tokenizer import (
    Tokenizer,
    Token,
    Identifier,
    Number,
    FixedNumber,
    LBracket,
    RBracket,
    LCurly,
    RCurly,
    LParen,
    RParen,
    Comma,
    Percent,
    Equals,
    ForwardSlash,
    Colon,
    Label,
)


class ParsingError(Exception):
    def __init__(self, msg: str):
        super().__init__(msg)


class InsufficientTokens(ParsingError):
    def __init__(self):
        super().__init__("Ran out of tokens!")


class UnexpectedToken(ParsingError):
    def __init__(self, token: Token, Class: Optional[Type[Token]] = None):
        super().__init__(f"Unexpected token '{token.value}'!")
        self.token: Token = token
        self.expected: Optional[Type[Token]] = Class


class ExpectedParameterIdentifier(ParsingError):
    def __init__(self, Class: Type[Element]):
        super().__init__(f"Expected a parameter identifier for '{Class.get_symbol()}'!")
        self.element: Type[Element] = Class


class ExpectedNumericValue(ParsingError):
    def __init__(self, token: Optional[Token]):
        super().__init__("Expected numeric value!")
        self.token: Optional[Token] = token


class ConnectionWithoutElements(ParsingError):
    def __init__(self, Class: Type[Connection]):
        super().__init__(
            ("Series" if Class is Series else "Parallel")
            + " connection closed without any elements or connections!"
        )
        self.connection: Type[Connection] = Class


class InsufficientElementsInParallelConnection(ParsingError):
    def __init__(self):
        super().__init__(
            "Parallel connections must contain at least two elements and/or connections!"
        )


class InvalidElementSymbol(ParsingError):
    def __init__(self, identifier: Identifier):
        super().__init__(f"Invalid element symbol '{identifier.value}'!")
        self.identifier: Identifier = identifier


class DuplicateParameterDefinition(ParsingError):
    def __init__(self, key: str, remaining_keys: List[str], Class: Type[Element]):
        super().__init__(
            f"Duplicate parameter definition ('{key}') for '{Class.get_symbol()}'!"
        )
        self.key: str = key
        self.remaining_keys: List[str] = remaining_keys
        self.element: Type[Element] = Class


class InvalidParameterDefinition(ParsingError):
    def __init__(self, key: str, valid_keys: List[str], Class: Type[Element]):
        super().__init__(
            f"Invalid parameter definition ('{key}') for '{Class.get_symbol()}'!"
        )
        self.key: str = key
        self.valid_keys: List[str] = valid_keys
        self.element: Type[Element] = Class


class TooManyParameterDefinitions(ParsingError):
    def __init__(self, Class: Type[Element]):
        super().__init__(f"Too many parameter definitions for '{Class.get_symbol()}'!")
        self.element: Type[Element] = Class


class InvalidParameterLowerLimit(ParsingError):
    def __init__(self, identifier: Identifier, value: float, limit: float):
        super().__init__(f"Invalid lower limit '{limit:.3E}' for value '{value:.3E}'!")
        self.identifier: Identifier = identifier
        self.value: float = value
        self.limit: float = limit


class InvalidParameterUpperLimit(ParsingError):
    def __init__(self, identifier: Identifier, value: float, limit: float):
        super().__init__(f"Invalid upper limit '{limit:.3E}' for value '{value:.3E}'!")
        self.identifier: Identifier = identifier
        self.value: float = value
        self.limit: float = limit


Stackable = Union[Token, Element, Connection]


class Parser:
    def __init__(self):
        self._tokens: List[Token] = []
        self._stack: List[Stackable] = []
        Class: Type[Element]
        self._valid_elements: Dict[str, Type[Element]] = {
            Class.get_symbol(): Class
            for Class in [
                KramersKronigRC,
                Resistor,
                Capacitor,
                Inductor,
                ConstantPhaseElement,
                Gerischer,
                HavriliakNegami,
                Warburg,
                WarburgOpen,
                WarburgShort,
                deLevieFiniteLength,
            ]
        }
        aliases: Dict[str, Type[Element]] = {
            "T": WarburgOpen,
            "O": WarburgShort,
            "D": deLevieFiniteLength,
        }
        for symbol in aliases:
            assert (
                symbol not in self._valid_elements
            ), f"Symbol '{symbol}' already in use by {self._valid_elements[symbol]}"
        self._valid_elements.update(aliases)

    def process(self, string: str) -> Circuit:
        string = string.strip()
        if string == "" or string == "[]":
            self.push_stack(Series([]))
        else:
            tokenizer: Tokenizer = Tokenizer()
            tokens: List[Token] = tokenizer.process(string)
            if tokens:
                self._tokens = tokens
                while self._tokens:
                    self.main_loop()
        con: Union[Series, Parallel]
        if self.get_stack_length() > 1:
            elements: List[Union[Element, Connection]] = []
            while not self.is_stack_empty():
                elem: Union[Element, Connection] = self.pop_stack()  # type: ignore
                assert isinstance(elem, Element) or isinstance(elem, Connection)
                elements.append(elem)
            con = Series(elements)
        else:
            con = self.pop_stack()  # type: ignore
            if type(con) is not Series:
                con = Series([con])
        assert self.is_stack_empty() is True, self._stack
        return Circuit(con)

    def pop_token(self) -> Token:
        if len(self._tokens) == 0:
            raise InsufficientTokens()
        return self._tokens.pop(0)

    def pop_stack(self) -> Stackable:
        assert len(self._stack) > 0, "Ran out of items on the stack!"
        return self._stack.pop(0)

    def push_stack(self, item: Stackable):
        self._stack.insert(0, item)

    def is_stack_empty(self) -> bool:
        return len(self._stack) == 0

    def get_stack_length(self) -> int:
        return len(self._stack)

    def peek(self, n: int = 1) -> Optional[Token]:
        assert n >= 0, n
        if len(self._tokens) < n + 1:
            return None
        return self._tokens[n]

    def accept(self, Class: Type[Token]) -> bool:
        if not self._tokens:
            return False
        return type(self._tokens[0]) is Class

    def expect(self, Class: Type[Token]):
        if len(self._tokens) == 0:
            raise InsufficientTokens()
        token: Token = self._tokens[0]
        if type(token) is not Class:
            raise UnexpectedToken(token, Class)

    def expect_number(self):
        if len(self._tokens) == 0:
            raise ExpectedNumericValue(None)
        token: Token = self._tokens[0]
        if not (type(token) is Number or type(token) is FixedNumber):
            raise ExpectedNumericValue(token)

    def main_loop(self):
        if self.accept(LBracket):
            self.connection(LBracket, RBracket, Series)
        elif self.accept(LParen):
            self.connection(LParen, RParen, Parallel)
        elif self.accept(Identifier):
            self.element()
        elif self._tokens:
            raise UnexpectedToken(self.pop_token())
        else:
            raise InsufficientTokens()

    def connection(
        self, Opening: Type[Token], Closing: Type[Token], Class: Type[Connection]
    ):
        self.push_stack(self.pop_token())
        if self.accept(Closing):
            raise ConnectionWithoutElements(Class)
        while not self.accept(Closing):
            self.main_loop()
        self.expect(Closing)
        self.pop_token()
        items: List[Stackable] = []
        while self._stack:
            item: Stackable = self.pop_stack()
            if type(item) is Opening:
                break
            if type(item) is Class:
                items.extend(item._elements)
            else:
                items.append(item)
        assert len(items) > 0
        if Class is Parallel and len(items) < 2:
            raise InsufficientElementsInParallelConnection()
        assert all(
            map(lambda _: isinstance(_, Element) or isinstance(_, Connection), items)
        )
        if Class is Series and len(items) == 1:
            self.push_stack(items[0])
        else:
            self.push_stack(Class(items))  # type: ignore

    def element(self):
        identifier: Identifier = self.pop_token()
        if identifier.value not in self._valid_elements:
            raise InvalidElementSymbol(identifier)
        Class: Type[Element] = self._valid_elements[identifier.value]
        label: str
        parameters: Dict[str, float]
        lower_limits: Dict[str, float]
        upper_limits: Dict[str, float]
        fixed_parameters: Dict[str, bool]
        (
            label,
            parameters,
            lower_limits,
            upper_limits,
            fixed_parameters,
        ) = self.parameters(Class)
        element = Class(**parameters)
        element.set_label(label)
        key: str
        value: Union[float, bool]
        for key, value in lower_limits.items():
            element.set_lower_limit(key, value)
        for key, value in upper_limits.items():
            element.set_upper_limit(key, value)
        for key, value in fixed_parameters.items():
            element.set_fixed(key, value)
        self.push_stack(element)

    def parameters(
        self, Class: Type[Element]
    ) -> Tuple[
        str, Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, bool]
    ]:
        label: str = ""
        parameters: Dict[str, float] = {}
        lower_limits: Dict[str, float] = {}
        upper_limits: Dict[str, float] = {}
        fixed_parameters: Dict[str, bool] = {}
        defaults: Dict[str, float] = Class.get_defaults()
        if not self.accept(LCurly):
            return (
                label,
                defaults,
                lower_limits,
                upper_limits,
                fixed_parameters,
            )
        self.pop_token()
        if not self.accept(Colon):
            keys: List[str] = list(defaults.keys())
            key: str
            value: float
            while keys:
                key, value, lower, upper, fixed = self.param(Class)
                if key in parameters:
                    raise DuplicateParameterDefinition(key, keys, Class)
                if key not in keys:
                    raise InvalidParameterDefinition(key, keys, Class)
                keys.remove(key)
                del defaults[key]
                parameters[key] = value
                lower_limits[key] = lower
                upper_limits[key] = upper
                fixed_parameters[key] = fixed
                if self.accept(Comma):
                    if len(keys) == 0:
                        raise TooManyParameterDefinitions(Class)
                    self.pop_token()
                    continue
                break
        if self.accept(Colon):
            self.pop_token()
            self.expect(Label)
            label = self.pop_token().value
        self.expect(RCurly)
        self.pop_token()
        for key, value in defaults.items():
            if key not in parameters:
                parameters[key] = value
        return (
            label,
            parameters,
            lower_limits,
            upper_limits,
            fixed_parameters,
        )

    def param(self, Class: Type[Element]) -> Tuple[str, float, float, float, bool]:
        if not self.accept(Identifier):
            raise ExpectedParameterIdentifier(Class)
        key: Identifier
        key = self.pop_token()  # type: ignore
        self.expect(Equals)
        self.pop_token()
        self.expect_number()
        value: Union[Number, FixedNumber]
        value = self.pop_token()  # type: ignore
        fixed: bool = False
        lower: float = -inf
        upper: float = inf
        if type(value) is FixedNumber:
            fixed = True
        elif self.accept(ForwardSlash):
            self.pop_token()
            if self.accept(ForwardSlash):
                self.pop_token()
                upper = self.param_limit(value.value)
            else:
                lower = self.param_limit(value.value)
                if self.accept(ForwardSlash):
                    self.pop_token()
                    upper = self.param_limit(value.value)
        if lower > value.value:
            raise InvalidParameterLowerLimit(key, value.value, lower)
        if upper < value.value:
            raise InvalidParameterUpperLimit(key, value.value, upper)
        return (
            key.value,
            value.value,
            lower,
            upper,
            fixed,
        )

    def param_limit(self, value: float) -> float:
        self.expect(Number)
        limit: Number
        limit = self.pop_token()  # type: ignore
        if self.accept(Percent):
            self.pop_token()
            return value * limit.value / 100
        return limit.value
