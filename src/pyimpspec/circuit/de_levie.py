# Copyright 2022 pyimpspec developers
# pyimpspec is licensed under the GPLv3 or later (https://www.gnu.org/licenses/gpl-3.0.html).
# The licenses of pyimpspec's dependencies and/or sources of portions of code are included in
# the LICENSES folder.

from typing import Dict, List
from collections import OrderedDict
from math import pi
from numpy import cosh, sinh
from .base import Element


def coth(x: complex) -> complex:
    return cosh(x) / sinh(x)


class deLevieFiniteLength(Element):
    """
    Ls: de Levie pore
    (finite length)

        Z = (Ri*Rr)^(1/2)*coth(d*(Ri/Rr)^(1/2)*(1+Y*(2*pi*f*j)^n)^(1/2))/(1+Y*(2*pi*f*j)^n)^(1/2)

    Parameters
    ----------
    Ri: float = 10.0 (ohm/cm)
    Rr: float = 1.0 (ohm*cm)
    Y: float = 0.01 (F*s^(n-1)/cm)
    n: float = 0.8
    d: float = 0.2 (cm)
    """

    def __init__(self, **kwargs):
        keys: List[str] = list(self.get_defaults().keys())
        super().__init__(keys=keys)
        self.reset_parameters(keys)
        self.set_parameters(kwargs)

    @staticmethod
    def get_symbol() -> str:
        return "Ls"

    @staticmethod
    def get_description() -> str:
        return "Ls: de Levie, finite length"

    @staticmethod
    def get_defaults() -> Dict[str, float]:
        return {
            "Ri": 10.0,
            "Rr": 1.0,
            "Y": 0.01,
            "n": 0.8,
            "d": 0.2,
        }

    @staticmethod
    def get_default_fixed() -> Dict[str, bool]:
        return {
            "Ri": False,
            "Rr": False,
            "Y": False,
            "n": False,
            "d": False,
        }

    @staticmethod
    def get_default_lower_limits() -> Dict[str, float]:
        return {
            "Ri": 1e-12,
            "Rr": 1e-12,
            "Y": 1e-12,
            "n": 0.5,
            "d": 1e-6,
        }

    @staticmethod
    def get_default_upper_limits() -> Dict[str, float]:
        return {
            "Ri": 10.0,
            "Rr": 1.0,
            "Y": 10.0,
            "n": 1.0,
            "d": 5.0,
        }

    def impedance(self, f: float) -> complex:
        return (self._Ri * self._Rr) ** (1 / 2) * (
            coth(
                self._d
                * (self._Ri / self._Rr) ** (1 / 2)
                * (1 + self._Y * (2 * pi * f * 1j) ** self._n) ** (1 / 2)
            )
            / (1 + self._Y * (2 * pi * f * 1j) ** self._n) ** (1 / 2)
        )

    def get_parameters(self) -> OrderedDict[str, float]:
        return OrderedDict(
            {
                "Ri": self._Ri,
                "Rr": self._Rr,
                "Y": self._Y,
                "n": self._n,
                "d": self._d,
            }
        )

    def set_parameters(self, parameters: Dict[str, float]):
        if "Ri" in parameters:
            self._Ri = float(parameters["Ri"])
        if "Rr" in parameters:
            self._Rr = float(parameters["Rr"])
        if "Y" in parameters:
            self._Y = float(parameters["Y"])
        if "n" in parameters:
            self._n = float(parameters["n"])
        if "d" in parameters:
            self._d = float(parameters["d"])

    def _str_expr(self, substitute: bool = False) -> str:
        string: str = "sqrt(Ri*Rr)*(coth(d*sqrt(Ri/Rr)*sqrt(1+Y*(2*pi*f*I)^n))/sqrt(1+Y*(2*pi*f*I)^n))"
        return self._subs_str_expr(string, self.get_parameters(), not substitute)
