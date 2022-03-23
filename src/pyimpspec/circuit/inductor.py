# Copyright 2022 pyimpspec developers
# pyimpspec is licensed under the GPLv3 or later (https://www.gnu.org/licenses/gpl-3.0.html).
# The licenses of pyimpspec's dependencies and/or sources of portions of code are included in
# the LICENSES folder.

from typing import Dict, List
from math import pi
from collections import OrderedDict
from .base import Element


class Inductor(Element):
    """
    L: Ideal inductor

        Z = j*2*pi*f*L

    Parameters
    ----------
    L: float = 1E-6 (H)
    """

    def __init__(self, **kwargs):
        keys: List[str] = list(self.get_defaults().keys())
        super().__init__(keys=keys)
        self.reset_parameters(keys)
        self.set_parameters(kwargs)

    @staticmethod
    def get_symbol() -> str:
        return "L"

    @staticmethod
    def get_description() -> str:
        return "L: Inductor"

    @staticmethod
    def get_defaults() -> Dict[str, float]:
        return {
            "L": 1e-6,
        }

    @staticmethod
    def get_default_fixed() -> Dict[str, bool]:
        return {
            "L": False,
        }

    @staticmethod
    def get_default_lower_limits() -> Dict[str, float]:
        return {
            "L": 0.0,
        }

    @staticmethod
    def get_default_upper_limits() -> Dict[str, float]:
        return {
            "L": 1e3,
        }

    def impedance(self, f: float) -> complex:
        return complex(0, self._L * 2 * pi * f)

    def get_parameters(self) -> OrderedDict[str, float]:
        return OrderedDict(
            {
                "L": self._L,
            }
        )

    def set_parameters(self, parameters: Dict[str, float]):
        if "L" in parameters:
            self._L = float(parameters["L"])

    def _str_expr(self, substitute: bool = False) -> str:
        string: str = "I*2*pi*f*L"
        return self._subs_str_expr(string, self.get_parameters(), not substitute)
