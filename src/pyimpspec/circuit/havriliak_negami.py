# Copyright 2022 pyimpspec developers
# pyimpspec is licensed under the GPLv3 or later (https://www.gnu.org/licenses/gpl-3.0.html).
# The licenses of pyimpspec's dependencies and/or sources of portions of code are included in
# the LICENSES folder.

from typing import Dict, List
from math import pi
from collections import OrderedDict
from .base import Element
from numpy import inf


class HavriliakNegami(Element):
    """
    H: Havriliak-Negami

        Z = (((1+(j*2*pi*f*t)^a)^b)/(j*2*pi*f*dC))

    Parameters
    ----------
    dC: float = 1E-6 (F)
    t: float = 1.0 (s)
    a: float = 0.9
    b: float = 0.9
    """

    def __init__(self, **kwargs):
        keys: List[str] = list(self.get_defaults().keys())
        super().__init__(keys=keys)
        self.reset_parameters(keys)
        self.set_parameters(kwargs)

    @staticmethod
    def get_symbol() -> str:
        return "H"

    @staticmethod
    def get_description() -> str:
        return "H: Havriliak-Negami"

    @staticmethod
    def get_defaults() -> Dict[str, float]:
        return {
            "dC": 1e-6,
            "t": 1.0,
            "a": 0.9,
            "b": 0.9,
        }

    @staticmethod
    def get_default_fixed() -> Dict[str, bool]:
        return {
            "dC": False,
            "t": False,
            "a": False,
            "b": False,
        }

    @staticmethod
    def get_default_lower_limits() -> Dict[str, float]:
        return {
            "dC": 0.0,
            "t": -inf,
            "a": -inf,
            "b": -inf,
        }

    @staticmethod
    def get_default_upper_limits() -> Dict[str, float]:
        return {
            "dC": inf,
            "t": inf,
            "a": inf,
            "b": inf,
        }

    def impedance(self, f: float) -> complex:
        return (1 + (2 * pi * f * 1j * self._t) ** self._a) ** self._b / (
            2 * pi * f * 1j * self._dC
        )

    def get_parameters(self) -> OrderedDict[str, float]:
        return OrderedDict(
            {
                "dC": self._dC,
                "t": self._t,
                "a": self._a,
                "b": self._b,
            }
        )

    def set_parameters(self, parameters: Dict[str, float]):
        if "dC" in parameters:
            self._dC = float(parameters["dC"])
        if "t" in parameters:
            self._t = float(parameters["t"])
        if "a" in parameters:
            self._a = float(parameters["a"])
        if "b" in parameters:
            self._b = float(parameters["b"])

    def _str_expr(self, substitute: bool = False) -> str:
        string: str = "(((1 + (I*2*pi*f*t)^a)^b) / (I*2*pi*f*dC))"
        return self._subs_str_expr(string, self.get_parameters(), not substitute)
