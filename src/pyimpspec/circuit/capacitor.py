# Copyright 2022 pyimpspec developers
# pyimpspec is licensed under the GPLv3 or later (https://www.gnu.org/licenses/gpl-3.0.html).
# The licenses of pyimpspec's dependencies and/or sources of portions of code are included in
# the LICENSES folder.

from typing import Dict, List
from math import pi
from collections import OrderedDict
from .base import Element


class Capacitor(Element):
    """
    C: Ideal capacitor

        Z = 1/(j*2*pi*f*C)

    Parameters
    ----------
    C: float = 1E-6 (F)
    """

    def __init__(self, **kwargs):
        keys: List[str] = list(self.get_defaults().keys())
        super().__init__(keys=keys)
        self.reset_parameters(keys)
        self.set_parameters(kwargs)

    @staticmethod
    def get_symbol() -> str:
        return "C"

    @staticmethod
    def get_description() -> str:
        return "C: Capacitor"

    @staticmethod
    def get_defaults() -> Dict[str, float]:
        return {
            "C": 1e-6,
        }

    @staticmethod
    def get_default_fixed() -> Dict[str, bool]:
        return {
            "C": False,
        }

    @staticmethod
    def get_default_lower_limits() -> Dict[str, float]:
        return {
            "C": 0.0,
        }

    @staticmethod
    def get_default_upper_limits() -> Dict[str, float]:
        return {
            "C": 1e3,
        }

    def impedance(self, f: float) -> complex:
        return 1 / (self._C * 2 * pi * f * 1j)

    def get_parameters(self) -> OrderedDict[str, float]:
        return OrderedDict(
            {
                "C": self._C,
            }
        )

    def set_parameters(self, parameters: Dict[str, float]):
        if "C" in parameters:
            self._C = float(parameters["C"])

    def _str_expr(self, substitute: bool = False) -> str:
        string: str = "1 / (I*2*pi*f*C)"
        return self._subs_str_expr(string, self.get_parameters(), not substitute)
