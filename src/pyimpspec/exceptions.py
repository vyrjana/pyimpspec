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

from pyimpspec.typing.helpers import List


class ImpedanceError(Exception):
    pass


class InfiniteImpedance(ImpedanceError):
    pass


class NotANumberImpedance(ImpedanceError):
    pass


class InfiniteLimit(InfiniteImpedance):
    pass


class InvalidEquation(ImpedanceError):
    pass


class InvalidParameterKey(Exception):
    pass


class KramersKronigError(Exception):
    pass


class FittingError(Exception):
    pass


class DRTError(Exception):
    pass


class ZHITError(Exception):
    pass


class TokenizingError(Exception):
    pass


class UnexpectedCharacter(TokenizingError):
    pass


class ParsingError(Exception):
    def __init__(self, msg: str):
        super().__init__(msg)


class InsufficientTokens(ParsingError):
    def __init__(self):
        super().__init__("Ran out of tokens!")


class UnexpectedToken(ParsingError):
    def __init__(self, token, Class=None):
        super().__init__(f"Unexpected token '{token.value}'!")
        self.token = token
        self.expected = Class


class UnexpectedIdentifier(ParsingError):
    def __init__(self, token, expected_value: str):
        super().__init__(
            f"Unexpected identifier '{token.value}'! Expected '{expected_value}'."
        )
        self.token = token


class ExpectedParameterIdentifier(ParsingError):
    def __init__(self, Class):
        super().__init__(f"Expected a parameter identifier for '{Class.get_symbol()}'!")
        self.element = Class


class ExpectedNumericValue(ParsingError):
    def __init__(self, token):
        super().__init__("Expected numeric value!")
        self.token = token


class InvalidNumericValue(ParsingError):
    def __init__(self, token, message: str):
        super().__init__("Invalid numeric value '{token.value:.2g}'! " + message)
        self.token = token


class ConnectionWithoutElements(ParsingError):
    def __init__(self, Class):
        super().__init__(
            f"{Class.__name__} connection closed without any elements or connections!"
        )
        self.connection = Class


class InsufficientElementsInParallelConnection(ParsingError):
    def __init__(self):
        super().__init__(
            "Parallel connections must contain at least two elements and/or connections!"
        )


class InvalidElementSymbol(ParsingError):
    def __init__(self, identifier):
        super().__init__(f"Invalid element symbol '{identifier.value}'!")
        self.identifier = identifier


class DuplicateParameterDefinition(ParsingError):
    def __init__(self, key: str, remaining_keys: List[str], Class):
        super().__init__(
            f"Duplicate parameter definition ('{key}') for '{Class.get_symbol()}'!"
        )
        self.key: str = key
        self.remaining_keys: List[str] = remaining_keys
        self.element = Class


class InvalidParameterDefinition(ParsingError):
    def __init__(self, key: str, valid_keys: List[str], Class):
        super().__init__(
            f"Invalid parameter definition ('{key}') for '{Class.get_symbol()}'!"
        )
        self.key: str = key
        self.valid_keys: List[str] = valid_keys
        self.element = Class


class TooManyParameterDefinitions(ParsingError):
    def __init__(self, Class):
        super().__init__(f"Too many parameter definitions for '{Class.get_symbol()}'!")
        self.element = Class


class InvalidParameterLowerLimit(ParsingError):
    def __init__(self, identifier, value: float, limit: float):
        super().__init__(f"Invalid lower limit '{limit:.3E}' for value '{value:.3E}'!")
        self.identifier = identifier
        self.value: float = value
        self.limit: float = limit


class InvalidParameterUpperLimit(ParsingError):
    def __init__(self, identifier, value: float, limit: float):
        super().__init__(f"Invalid upper limit '{limit:.3E}' for value '{value:.3E}'!")
        self.identifier = identifier
        self.value: float = value
        self.limit: float = limit


class UnsupportedFileFormat(Exception):
    pass
