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
    ascii_letters,
    ascii_lowercase,
    digits,
    whitespace,
)
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Type,
    Union,
)
from pyimpspec.exceptions import UnexpectedCharacter


@dataclass(frozen=True)
class Token:
    start: int
    end: int
    value: Any


@dataclass(frozen=True)
class Identifier(Token):
    value: str

    def __post_init__(self):
        if not self.value.isidentifier():
            raise ValueError(f"Expected an identifier instead of {self.value=}")


@dataclass(frozen=True)
class Label(Token):
    value: str


@dataclass(frozen=True)
class Number(Token):
    value: float


@dataclass(frozen=True)
class FixedNumber(Token):
    value: float


@dataclass(frozen=True)
class LBracket(Token):
    value: str


@dataclass(frozen=True)
class RBracket(Token):
    value: str


@dataclass(frozen=True)
class LParen(Token):
    value: str


@dataclass(frozen=True)
class RParen(Token):
    value: str


@dataclass(frozen=True)
class LCurly(Token):
    value: str


@dataclass(frozen=True)
class RCurly(Token):
    value: str


@dataclass(frozen=True)
class Equals(Token):
    value: str


@dataclass(frozen=True)
class ForwardSlash(Token):
    value: str


@dataclass(frozen=True)
class Percent(Token):
    value: str


@dataclass(frozen=True)
class Comma(Token):
    value: str


@dataclass(frozen=True)
class Colon(Token):
    value: str


@dataclass(frozen=True)
class Exclamation(Token):
    value: str


class Tokenizer:
    def __init__(self):
        self._original: str = ""
        self._chars: List[str] = []
        self._tokens: List[Token] = []
        self._value: str = ""
        self._index: int = -1
        self._start: int = -1
        self._end: int = -1
        self._special_characters: Dict[str, Type[Token]] = {
            "[": LBracket,
            "]": RBracket,
            "(": LParen,
            ")": RParen,
            "{": LCurly,
            "}": RCurly,
            "=": Equals,
            "/": ForwardSlash,
            "%": Percent,
            ",": Comma,
            ":": Colon,
            "!": Exclamation,
        }

    def process(self, string: str) -> List[Token]:
        self._original = string
        self._chars = [_ for _ in string]
        self._tokens = []
        self._value = ""
        self._index = 0
        self._start = 0
        self._end = -1

        while self._chars:
            self.main_loop()

        return self._tokens

    def main_loop(self):
        char: Optional[str] = self.peek(0)

        if char in self._special_characters:
            self.consume(self.pop())
            self.push(self._special_characters[char])

        elif char in ascii_letters:
            self.identifier_or_label()

        elif char in digits or char == "-" and self.peek() in digits:
            self.number()

        elif char in whitespace:
            self.ignore()

        else:
            raise UnexpectedCharacter(f"Unexpected character: {char}")

    def identifier_or_label(self):
        self.consume(self.pop())
        char: Optional[str] = self.peek(0)

        if type(self.peek(-1)) is Colon:
            # Label for an element
            num_curly_scopes: int = 0

            while char is not None:
                if char == "{":
                    num_curly_scopes += 1
                elif char == "}":
                    if num_curly_scopes <= 0:
                        break
                    num_curly_scopes -= 1

                self.consume(self.pop())
                char = self.peek(0)

            self.push(Label)
            return

        prev_token: Optional[Token] = self.peek(-1)
        valid_chars: str

        if type(prev_token) is LCurly or type(prev_token) is Comma:
            # Identifier for a parameter
            valid_chars = ascii_letters + digits + "_"

            while char is not None and char in valid_chars:
                self.consume(self.pop())
                char = self.peek(0)

        else:
            # Identifier for an element
            valid_chars = ascii_lowercase + digits + "_"

            while char is not None and char in valid_chars:
                self.consume(self.pop())
                char = self.peek(0)

        self.push(Identifier)

    def number(self):
        self.consume(self.pop())

        while self.peek(0) is not None and self.peek(0) in digits:
            self.consume(self.pop())

        if self.accept("."):
            self.consume(self.pop())
            while self.peek(0) is not None and self.peek(0) in digits:
                self.consume(self.pop())

        if self.peek(0) is not None and self.peek(0) in "eE":
            self.consume(self.pop())
            if self.accept("-"):
                self.consume(self.pop())
            elif self.accept("+"):
                self.consume(self.pop())

            while self.peek(0) is not None and self.peek(0) in digits:
                self.consume(self.pop())

        if self.peek(0) is not None and self.peek(0) in "fF":
            self.consume("")
            self.pop()
            self.push(FixedNumber)
        else:
            self.push(Number)

    def look_ahead_for(self, char: str) -> int:
        for i, c in enumerate(self._chars):
            if c == char:
                return i

        return -1

    def peek(self, n: int = 1) -> Optional[Union[str, Token]]:
        if n >= 0:
            if len(self._chars) < n + 1:
                return None

            return self._chars[n]

        if len(self._tokens) < abs(n):
            return None

        return self._tokens[n]

    def accept(self, char: str) -> bool:
        if not self._chars:
            return False

        return self._chars[0] == char

    def pop(self) -> str:
        if len(self._chars) < 1:
            raise ValueError(f"Expected {self._chars=} to contain at least one item")

        self._index += 1

        return self._chars.pop(0)

    def consume(self, char: Optional[str] = None):
        if char is not None:
            self._value += char
        else:
            self._start = self._index

    def ignore(self):
        self.pop()
        self.consume()

    def push(self, Class: Type[Token]):
        self._end = self._index
        value: Union[str, float]

        if Class is Number or Class is FixedNumber:
            value = float(self._value)
        else:
            value = self._original[self._start:self._end]

        self._tokens.append(Class(self._start, self._end, value))
        self._start = self._index
        self._value = ""
