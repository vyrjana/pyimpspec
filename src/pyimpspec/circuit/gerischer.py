# Copyright 2022 pyimpspec developers
# pyimpspec is licensed under the GPLv3 or later (https://www.gnu.org/licenses/gpl-3.0.html).
# The licenses of pyimpspec's dependencies and/or sources of portions of code are included in
# the LICENSES folder.

from typing import Dict, List
from math import pi
from collections import OrderedDict
from .base import Element
from numpy import inf


class Gerischer(Element):
    """
    G: Gerischer

        Z = 1/(Y*(k+j*2*pi*f)^n)

    Parameters
    ----------
    Y: float = 1.0 (S*s^n)
    k: float = 1.0 (s^-1)
    n: float = 0.5
    """

    def __init__(self, **kwargs):
        keys: List[str] = list(self.get_defaults().keys())
        super().__init__(keys=keys)
        self.reset_parameters(keys)
        self.set_parameters(kwargs)

    @staticmethod
    def get_symbol() -> str:
        return "G"

    @staticmethod
    def get_description() -> str:
        return "G: Gerischer"

    @staticmethod
    def get_defaults() -> Dict[str, float]:
        return {
            "Y": 1.0,
            "k": 1.0,
            "n": 0.5,
        }

    @staticmethod
    def get_default_fixed() -> Dict[str, bool]:
        return {
            "Y": False,
            "k": False,
            "n": True,
        }

    @staticmethod
    def get_default_lower_limits() -> Dict[str, float]:
        return {
            "Y": 0.0,
            "k": 0.0,
            "n": 0.0,
        }

    @staticmethod
    def get_default_upper_limits() -> Dict[str, float]:
        return {
            "Y": inf,
            "k": inf,
            "n": 1.0,
        }

    def impedance(self, f: float) -> complex:
        return 1 / (self._Y * (self._k + 2 * pi * f * 1j) ** self._n)

    def get_parameters(self) -> OrderedDict[str, float]:
        return OrderedDict(
            {
                "Y": self._Y,
                "k": self._k,
                "n": self._n,
            }
        )

    def set_parameters(self, parameters: Dict[str, float]):
        if "Y" in parameters:
            self._Y = float(parameters["Y"])
        if "k" in parameters:
            self._k = float(parameters["k"])
        if "n" in parameters:
            self._n = float(parameters["n"])

    def _str_expr(self, substitute: bool = False) -> str:
        string: str = "1 / (Y * (k + I*2*pi*f)^n)"
        return self._subs_str_expr(string, self.get_parameters(), not substitute)
