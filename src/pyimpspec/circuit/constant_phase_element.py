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
from math import pi
from collections import OrderedDict
from .base import Element


class ConstantPhaseElement(Element):
    """
    Constant phase element

        Symbol: Q

        Z = 1/(Y*(j*2*pi*f)^n)

        Variables
        ---------
        Y: float = 1E-6 (F*s^(n-1))
        n: float = 0.95
    """

    def __init__(self, **kwargs):
        keys: List[str] = list(self.get_defaults().keys())
        super().__init__(keys=keys)
        self.reset_parameters(keys)
        self.set_parameters(kwargs)

    @staticmethod
    def get_symbol() -> str:
        return "Q"

    @staticmethod
    def get_description() -> str:
        return "Q: Constant phase element"

    @staticmethod
    def get_defaults() -> Dict[str, float]:
        return {
            "Y": 1e-6,
            "n": 0.95,
        }

    @staticmethod
    def get_default_fixed() -> Dict[str, bool]:
        return {
            "Y": False,
            "n": False,
        }

    @staticmethod
    def get_default_lower_limits() -> Dict[str, float]:
        return {
            "Y": 0.0,
            "n": 0.0,
        }

    @staticmethod
    def get_default_upper_limits() -> Dict[str, float]:
        return {
            "Y": 1e3,
            "n": 1.0,
        }

    def impedance(self, f: float) -> complex:
        return 1 / (self._Y * (2 * pi * f * 1j) ** self._n)

    def get_parameters(self) -> "OrderedDict[str, float]":
        return OrderedDict(
            {
                "Y": self._Y,
                "n": self._n,
            }
        )

    def set_parameters(self, parameters: Dict[str, float]):
        if "Y" in parameters:
            self._Y = float(parameters["Y"])
        if "n" in parameters:
            self._n = float(parameters["n"])

    def _str_expr(self, substitute: bool = False) -> str:
        string: str = "1 / (Y * (I*2*pi*f)^n)"
        return self._subs_str_expr(string, self.get_parameters(), not substitute)
