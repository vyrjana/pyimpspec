# Copyright 2022 pyimpspec developers
# pyimpspec is licensed under the GPLv3 or later (https://www.gnu.org/licenses/gpl-3.0.html).
# The licenses of pyimpspec's dependencies and/or sources of portions of code are included in
# the LICENSES folder.

from typing import Dict, List
from collections import OrderedDict
from math import pi
from numpy import cosh, inf, sinh, tanh
from .base import Element


def coth(x: complex) -> complex:
    return cosh(x) / sinh(x)


class Warburg(Element):
    """
    W: Warburg
    (semi-infinite diffusion)

        Z = 1/(Y*(2*pi*f*j)^(1/2))

    Parameters
    ----------
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

    def get_parameters(self) -> OrderedDict[str, float]:
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
    Ws: Warburg, finite length or short
    (finite length diffusion with transmissive boundary)

        Z = tanh((B*j*2*pi*f)^n)/((Y*j*2*pi*f)^n)

    Parameters
    ----------
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

    def get_parameters(self) -> OrderedDict[str, float]:
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
    Wo: Warburg, finite space or open
    (finite length diffusion with reflective boundary)

        Z = coth((B*j*2*pi*f)^n)/((Y*j*2*pi*f)^n)

    Parameters
    ----------
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

    def get_parameters(self) -> OrderedDict[str, float]:
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
