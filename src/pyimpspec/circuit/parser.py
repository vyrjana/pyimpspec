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

from numpy import (
    inf,
    isnan,
    nan,
)
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)
from .base import (
    Connection,
    Container,
    Element,
)
from .registry import get_elements
from .parallel import Parallel
from .series import Series
from .circuit import (
    VERSION,
    Circuit,
)
from .tokenizer import (
    Colon,
    Comma,
    Equals,
    Exclamation,
    FixedNumber,
    ForwardSlash,
    Identifier,
    LBracket,
    LCurly,
    LParen,
    Label,
    Number,
    Percent,
    RBracket,
    RCurly,
    RParen,
    Token,
    Tokenizer,
)
from pyimpspec.exceptions import (
    InsufficientTokens,
    UnexpectedToken,
    UnexpectedIdentifier,
    ExpectedParameterIdentifier,
    ExpectedNumericValue,
    InvalidNumericValue,
    ConnectionWithoutElements,
    InsufficientElementsInParallelConnection,
    InvalidElementSymbol,
    DuplicateParameterDefinition,
    InvalidParameterDefinition,
    TooManyParameterDefinitions,
    InvalidParameterLowerLimit,
    InvalidParameterUpperLimit,
)


Stackable = Union[Token, Element, Connection]


class Parser:
    def __init__(self):
        self._tokens: List[Token] = []
        self._stack: List[Stackable] = []
        self._valid_elements: Dict[str, Type[Element]] = get_elements(private=True)

        if not isinstance(self._valid_elements, dict):
            raise TypeError(f"Expected a dictionary instead of {self._valid_elements=}")

        if len(self._valid_elements) < 1:
            raise ValueError(f"Expected at least one item in {self._valid_elements=}")

    def process(self, string: str, version: int = -1) -> Circuit:
        string = string.strip()

        if (
            string == ""
            or string == "[]"
            or (string.startswith("!") and string[string.find("!", 1) + 1:] == "[]")
        ):
            self.push_stack(Series([]))
        else:
            tokenizer: Tokenizer = Tokenizer()
            tokens: List[Token] = tokenizer.process(string)
            if tokens:
                self._tokens = tokens
                self.migrate(version=version)
                while self._tokens:
                    self.main_loop()

        con: Union[Series, Parallel]
        if self.get_stack_length() > 1:
            elements: List[Union[Element, Connection]] = []

            while not self.is_stack_empty():
                elem: Union[Element, Connection] = self.pop_stack()  # type: ignore
                if not (isinstance(elem, Element) or isinstance(elem, Connection)):
                    raise TypeError(
                        f"Expected a Connection or an Element instead of {elem=}"
                    )

                elements.append(elem)

            elements.reverse()
            con = Series(elements)
        else:
            con = self.pop_stack()  # type: ignore
            if type(con) is not Series:
                con = Series([con])

        if not self.is_stack_empty():
            raise ValueError(f"Expected an empty stack instead of {self._stack=}")

        return Circuit(con)

    def pop_token(self) -> Token:
        if len(self._tokens) == 0:
            raise InsufficientTokens()

        return self._tokens.pop(0)

    def pop_stack(self) -> Stackable:
        if len(self._stack) < 1:
            raise ValueError("Ran out of items on the stack!")

        return self._stack.pop(0)

    def push_stack(self, item: Stackable):
        self._stack.insert(0, item)

    def is_stack_empty(self) -> bool:
        return len(self._stack) == 0

    def get_stack_length(self) -> int:
        return len(self._stack)

    def peek(self, n: int = 1) -> Optional[Token]:
        if not (n >= 0):
            raise ValueError(f"Expected {n=} >= 0")

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
        self,
        Opening: Type[Token],
        Closing: Type[Token],
        Class: Type[Connection],
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
                items.extend(reversed(item._elements))
            else:
                items.append(item)

        if len(items) < 1:
            raise ValueError(f"Expected at least one item in {items=}")

        if Class is Parallel and len(items) < 2:
            raise InsufficientElementsInParallelConnection()

        if not all(
            map(lambda _: isinstance(_, Element) or isinstance(_, Connection), items)
        ):
            raise TypeError(f"Expected only Connections and Elements in {items=}")

        items.reverse()

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
        subcircuits: Dict[str, Optional[Connection]]
        (
            label,
            parameters,
            lower_limits,
            upper_limits,
            fixed_parameters,
            subcircuits,
        ) = self.parameters(Class)

        element = Class(**parameters, **subcircuits)
        element.set_label(label)
        element.set_lower_limits(
            **{k: v for k, v in lower_limits.items() if not isnan(v)}
        )
        element.set_upper_limits(
            **{k: v for k, v in upper_limits.items() if not isnan(v)}
        )
        element.set_fixed(**fixed_parameters)

        self.push_stack(element)

    def parameters(
        self,
        Class: Type[Element],
    ) -> Tuple[
        str,
        Dict[str, float],
        Dict[str, float],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, Optional[Connection]],
    ]:
        label: str = ""
        parameters: Dict[str, float] = {}
        lower_limits: Dict[str, float] = {}
        upper_limits: Dict[str, float] = {}
        fixed_parameters: Dict[str, bool] = {}
        subcircuits: Dict[str, Optional[Connection]] = {}

        if not self.accept(LCurly):
            return (
                label,
                parameters,
                lower_limits,
                upper_limits,
                fixed_parameters,
                subcircuits,
            )

        self.pop_token()

        if not self.accept(Colon):
            parameter_keys: List[str] = list(Class.get_default_values().keys())
            subcircuit_keys: List[str] = []

            if issubclass(Class, Container):
                subcircuit_keys.extend(Class.get_default_subcircuits().keys())

            key: str
            value: float
            while parameter_keys or subcircuit_keys:
                if not self.accept(Identifier):
                    raise ExpectedParameterIdentifier(Class)

                token: Token = self.pop_token()  # type: ignore
                key = token.value
                self.expect(Equals)
                self.pop_token()

                if (
                    self.accept(LBracket)
                    or self.accept(LParen)
                    or self.accept(Identifier)
                ):
                    if key in subcircuits:
                        raise DuplicateParameterDefinition(key, subcircuit_keys, Class)
                    elif key not in subcircuit_keys:
                        raise InvalidParameterDefinition(key, subcircuit_keys, Class)

                    subcircuits[key] = self.subcircuit(token)
                    subcircuit_keys.remove(key)

                else:
                    if key in parameters:
                        raise DuplicateParameterDefinition(key, parameter_keys, Class)
                    elif key not in parameter_keys:
                        raise InvalidParameterDefinition(key, parameter_keys, Class)

                    lower: float
                    upper: float
                    value, lower, upper, fixed = self.param(Class, token)
                    if not isnan(lower) and lower > value:
                        raise InvalidParameterLowerLimit(key, value, lower)
                    if not isnan(upper) and upper < value:
                        raise InvalidParameterUpperLimit(key, value, upper)

                    parameter_keys.remove(key)
                    parameters[key] = value
                    lower_limits[key] = lower
                    upper_limits[key] = upper
                    fixed_parameters[key] = fixed

                if self.accept(Comma):
                    if len(parameter_keys) == 0 and len(subcircuit_keys) == 0:
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

        return (
            label,
            parameters,
            lower_limits,
            upper_limits,
            fixed_parameters,
            subcircuits,
        )

    def subcircuit(self, key: Identifier) -> Optional[Connection]:
        con: Stackable
        if self.accept(Identifier):
            token: Optional[Token] = self.peek(0)
            if token is None:
                raise TypeError(f"Expected a Token instead of {token=}")

            if token.value == "zero" or token.value == "short":
                self.pop_token()
                return Series([])

            elif token.value == "inf" or token.value == "open":
                self.pop_token()
                return None

            while type(self.peek(0)) not in [Comma, Colon, RCurly]:
                if self.peek(0) is None:
                    raise InsufficientTokens()
                self.main_loop()

            elements: List[Element] = []

            while not self.is_stack_empty():
                con = self.pop_stack()
                if not isinstance(con, Element):
                    raise TypeError(f"Expected an Element instead of {con=}")

                elements.insert(0, con)

            elements.reverse()

            return Series(elements)

        opening: Type[Token]
        closing: Type[Token]
        Class: Type[Connection]

        if self.accept(LBracket):
            opening = LBracket
            closing = RBracket
            Class = Series
        else:
            opening = LParen
            closing = RParen
            Class = Parallel

        self.connection(opening, closing, Class)

        con = self.pop_stack()
        if isinstance(con, Token):
            raise TypeError(
                f"Expected a Connection instead of a Token ({key=}, {con=})"
            )

        if isinstance(con, Element):
            con = Series([con])

        if not isinstance(con, Connection):
            raise TypeError(f"Expected a Connection instead of {con=} ({key=})")

        return con

    def param(
        self,
        Class: Type[Element],
        key: Identifier,
    ) -> Tuple[float, float, float, bool]:
        self.expect_number()
        value: Union[Number, FixedNumber]
        value = self.pop_token()  # type: ignore
        fixed: bool = isinstance(value, FixedNumber)
        lower: float = nan
        upper: float = nan

        if self.accept(ForwardSlash):
            self.pop_token()

            if self.accept(ForwardSlash):
                self.pop_token()
                upper = self.param_limit(value.value, upper=True)

            else:
                lower = self.param_limit(value.value, upper=False)
                if self.accept(ForwardSlash):
                    self.pop_token()
                    upper = self.param_limit(value.value, upper=True)

        return (
            value.value,
            lower,
            upper,
            fixed,
        )

    def param_limit(self, value: float, upper: bool) -> float:
        limit: Union[Number, Identifier]
        if not self.accept(Number):
            self.expect(Identifier)
            limit = self.pop_token()
            if limit.value != "inf":
                raise ValueError(
                    f"Expected 'inf' or a number instead of {limit.value=}"
                )

            if upper:
                return inf

            return -inf

        limit = self.pop_token()

        if self.accept(Percent):
            self.pop_token()
            return value * limit.value / 100

        return limit.value

    def migrate(self, version: int = -1):
        if version == 0:
            self._v0_migrator()
            version = 1

        elif self.accept(Exclamation):
            self.pop_token()
            self.expect(Identifier)
            token: Token = self.pop_token()
            if token.value.upper() != "V":
                raise UnexpectedIdentifier(token, "V")

            self.expect(Equals)
            self.pop_token()
            self.expect_number()
            token = self.pop_token()
            version = int(token.value)
            if not (0 < version <= VERSION):
                raise InvalidNumericValue(token, f"Expected 0 < value <= {VERSION}!")

            self.expect(Exclamation)
            self.pop_token()
        else:
            return

        if not (version > 0):
            raise ValueError(f"Expected {version=} > 0")

        migrators: Dict[int, Callable] = {
            1: self._v1_migrator,
        }

        v: int
        migrator: Callable
        for v, migrator in migrators.items():
            if v < version:
                continue
            migrator()

    def _v0_migrator(self):
        new_tokens: List[Token] = []

        def replace_element_parameters(pairs: Dict[str, str]):
            token: Token = self._tokens[0]
            if not isinstance(token, LCurly):
                return

            nonlocal new_tokens
            token = self._tokens.pop(0)

            while not isinstance(token, RCurly):
                if isinstance(token, Identifier) and token.value in pairs:
                    token = Identifier(
                        start=token.start,
                        end=token.end,
                        value=pairs[token.value],
                    )

                elif isinstance(token, Colon):
                    while not isinstance(token, RCurly):
                        new_tokens.append(token)
                        token = self._tokens.pop(0)
                    break

                new_tokens.append(token)
                token = self._tokens.pop(0)

            new_tokens.append(token)
            if not isinstance(new_tokens[-1], RCurly):
                raise TypeError(f"Expected RCurly instead of {new_tokens[-1]=}")

        while self._tokens:
            token: Token = self._tokens.pop(0)
            new_tokens.append(token)

            if isinstance(token, Identifier):
                if token.value == "Ls":
                    replace_element_parameters(
                        {
                            "Ri": "R_i",
                            "Rr": "R_r",
                        }
                    )

                elif token.value == "H":
                    replace_element_parameters(
                        {
                            "t": "tau",
                        }
                    )

                elif token.value == "Ha":
                    replace_element_parameters(
                        {
                            "t": "tau",
                            "b": "a",
                            "g": "b",
                        }
                    )

                elif token.value == "K":
                    replace_element_parameters(
                        {
                            "t": "tau",
                        }
                    )

        if not (len(self._tokens) == 0):
            raise ValueError(f"Expected {self._tokens=} to be empty")

        self._tokens.extend(new_tokens)

    def _v1_migrator(self):
        if not (VERSION == 1):
            raise ValueError("Update the implementation since VERSION != 1")
