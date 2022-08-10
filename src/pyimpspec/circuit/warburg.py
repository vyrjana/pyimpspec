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

from typing import (
    Dict,
    List,
)
from collections import OrderedDict
from math import pi
from numpy import (
    cosh,
    inf,
    sinh,
    tanh,
)
from .base import Element


def coth(x: complex) -> complex:
    return cosh(x) / sinh(x)


class Warburg(Element):
    """
    Warburg (semi-infinite diffusion)

        Symbol: W

        Z = 1/(Y*(2*pi*f*j)^(1/2))

        Variables
        ---------
        Y: float = 1.0 (S*s^(1/2))
    """

    def __init__(self, **kwargs):
        keys: List[str] = list(self.get_defaults().keys())
        super().__init__(keys=keys)
        self.reset_parameters(keys)
        self.set_parameters(kwargs)

    @staticmethod
    def get_symbol() -> str:
        return "W"

    @staticmethod
    def get_description() -> str:
        return "W: Warburg, semi-infinite"

    @staticmethod
    def get_defaults() -> Dict[str, float]:
        return {
            "Y": 1.0,
        }

    @staticmethod
    def get_default_fixed() -> Dict[str, bool]:
        return {
            "Y": False,
        }

    @staticmethod
    def get_default_lower_limits() -> Dict[str, float]:
        return {
            "Y": 0.0,
        }

    @staticmethod
    def get_default_upper_limits() -> Dict[str, float]:
        return {
            "Y": inf,
        }

    def impedance(self, f: float) -> complex:
        return 1 / (self._Y * (2 * pi * f * 1j) ** (1 / 2))

    def get_parameters(self) -> "OrderedDict[str, float]":
        return OrderedDict(
            {
                "Y": self._Y,
            }
        )

    def set_parameters(self, parameters: Dict[str, float]):
        if "Y" in parameters:
            self._Y = float(parameters["Y"])

    def _str_expr(self, substitute: bool = False) -> str:
        string: str = "1 / (Y * (2*pi*f*I)^(1/2))"
        return self._subs_str_expr(string, self.get_parameters(), not substitute)


class WarburgShort(Element):
    """
    Warburg, finite length or short (finite length diffusion with transmissive boundary)

        Symbol: Ws

        Z = tanh((B*j*2*pi*f)^n)/((Y*j*2*pi*f)^n)

        Variables
        ---------
        Y: float = 1.0 (S)
        B: float = 1.0 (s^n)
        n: float = 0.5
    """

    def __init__(self, **kwargs):
        keys: List[str] = list(self.get_defaults().keys())
        super().__init__(keys=keys)
        self.reset_parameters(keys)
        self.set_parameters(kwargs)

    @staticmethod
    def get_symbol() -> str:
        return "Ws"

    @staticmethod
    def get_description() -> str:
        return "Ws: Warburg, finite length or short"

    @staticmethod
    def get_defaults() -> Dict[str, float]:
        return {
            "Y": 1.0,
            "B": 1.0,
            "n": 0.5,
        }

    @staticmethod
    def get_default_fixed() -> Dict[str, bool]:
        return {
            "Y": False,
            "B": False,
            "n": True,
        }

    @staticmethod
    def get_default_lower_limits() -> Dict[str, float]:
        return {
            "Y": 0.0,
            "B": 0.0,
            "n": 0.0,
        }

    @staticmethod
    def get_default_upper_limits() -> Dict[str, float]:
        return {
            "Y": inf,
            "B": inf,
            "n": 1.0,
        }

    def impedance(self, f: float) -> complex:
        return (
            tanh((self._B * 2 * pi * f * 1j) ** self._n)
            / (self._Y * 2 * pi * f * 1j) ** self._n
        )

    def get_parameters(self) -> "OrderedDict[str, float]":
        return OrderedDict(
            {
                "Y": self._Y,
                "B": self._B,
                "n": self._n,
            }
        )

    def set_parameters(self, parameters: Dict[str, float]):
        if "Y" in parameters:
            self._Y = float(parameters["Y"])
        if "B" in parameters:
            self._B = float(parameters["B"])
        if "n" in parameters:
            self._n = float(parameters["n"])

    def _str_expr(self, substitute: bool = False) -> str:
        string: str = "tanh((B*I*2*pi*f)^n) / ((Y*I*2*pi*f)^n)"
        return self._subs_str_expr(string, self.get_parameters(), not substitute)


class WarburgOpen(Element):
    """
    Warburg, finite space or open (finite length diffusion with reflective boundary)

        Symbol: Wo

        Z = coth((B*j*2*pi*f)^n)/((Y*j*2*pi*f)^n)

        Variables
        ---------
        Y: float = 1.0 (S)
        B: float = 1.0 (s^n)
        n: float = 0.5
    """

    def __init__(self, **kwargs):
        keys: List[str] = list(self.get_defaults().keys())
        super().__init__(keys=keys)
        self.reset_parameters(keys)
        self.set_parameters(kwargs)

    @staticmethod
    def get_symbol() -> str:
        return "Wo"

    @staticmethod
    def get_description() -> str:
        return "Wo: Warburg, finite space or open"

    @staticmethod
    def get_defaults() -> Dict[str, float]:
        return {
            "Y": 1.0,
            "B": 1.0,
            "n": 0.5,
        }

    @staticmethod
    def get_default_fixed() -> Dict[str, bool]:
        return {
            "Y": False,
            "B": False,
            "n": True,
        }

    @staticmethod
    def get_default_lower_limits() -> Dict[str, float]:
        return {
            "Y": 0.0,
            "B": 0.0,
            "n": 0.0,
        }

    @staticmethod
    def get_default_upper_limits() -> Dict[str, float]:
        return {
            "Y": inf,
            "B": inf,
            "n": 1.0,
        }

    def impedance(self, f: float) -> complex:
        return (
            coth((self._B * 2 * pi * f * 1j) ** self._n)
            / (self._Y * 2 * pi * f * 1j) ** self._n
        )

    def get_parameters(self) -> "OrderedDict[str, float]":
        return OrderedDict(
            {
                "Y": self._Y,
                "B": self._B,
                "n": self._n,
            }
        )

    def set_parameters(self, parameters: Dict[str, float]):
        if "Y" in parameters:
            self._Y = float(parameters["Y"])
        if "B" in parameters:
            self._B = float(parameters["B"])
        if "n" in parameters:
            self._n = float(parameters["n"])

    def _str_expr(self, substitute: bool = False) -> str:
        string: str = "coth((B*I*2*pi*f)^n) / ((Y*I*2*pi*f)^n)"
        return self._subs_str_expr(string, self.get_parameters(), not substitute)
