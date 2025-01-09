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

from numpy import (
    inf,
    pi,
)
from .base import Element
from .registry import (
    ElementDefinition,
    ParameterDefinition,
    register_element,
)
from pyimpspec.typing import (
    ComplexImpedances,
    Frequencies,
)


class HavriliakNegami(Element):
    def _impedance(
        self,
        f: Frequencies,
        dC: float,
        tau: float,
        a: float,
        b: float,
    ) -> ComplexImpedances:
        return (1 + (2 * pi * f * 1j * tau) ** a) ** b / (2 * pi * f * 1j * dC)


register_element(
    ElementDefinition(
        Class=HavriliakNegami,
        symbol="H",
        name="Havriliak-Negami",
        description="Havriliak-Negami, Cole-Davidson (a = 1), Cole-Cole (b = 1), or Debye (a = b = 1) relaxation.",
        equation="((1+(I*2*pi*f*tau)^a)^b)/(I*2*pi*f*dC)",
        parameters=[
            ParameterDefinition(
                symbol="dC",
                unit="F",
                description="Difference in capacitance",
                value=1e-6,
                lower_limit=1e-24,
                upper_limit=inf,
                fixed=False,
            ),
            ParameterDefinition(
                symbol="tau",
                unit="s",
                description="Time constant",
                value=1.0,
                lower_limit=1e-24,
                upper_limit=inf,
                fixed=False,
            ),
            ParameterDefinition(
                symbol="a",
                unit="",
                description="Asymmetry exponent",
                value=0.9,
                lower_limit=0.0,
                upper_limit=1.0,
                fixed=False,
            ),
            ParameterDefinition(
                symbol="b",
                unit="",
                description="Broadness exponent",
                value=0.9,
                lower_limit=0.0,
                upper_limit=1.0,
                fixed=False,
            ),
        ],
    ),
)


class HavriliakNegamiAlternative(Element):
    def _impedance(
        self,
        f: Frequencies,
        R: float,
        tau: float,
        a: float,
        b: float,
    ) -> ComplexImpedances:
        return R / ((1 + (2 * pi * f * 1j * tau) ** a) ** b)


register_element(
    ElementDefinition(
        Class=HavriliakNegamiAlternative,
        symbol="Ha",
        name="Havriliak-Negami, alt.",
        description="Havriliak-Negami relaxation (alternative form).",
        equation="R/((1+(I*2*pi*f*tau)^a)^b)",
        parameters=[
            ParameterDefinition(
                symbol="R",
                unit="ohm",
                description="Resistance",
                value=1.0,
                lower_limit=0.0,
                upper_limit=inf,
                fixed=False,
            ),
            ParameterDefinition(
                symbol="tau",
                unit="s",
                description="Time constant",
                value=1.0,
                lower_limit=1e-24,
                upper_limit=inf,
                fixed=False,
            ),
            ParameterDefinition(
                symbol="a",
                unit="",
                description="Asymmetry exponent",
                value=0.7,
                lower_limit=0.0,
                upper_limit=1.0,
                fixed=False,
            ),
            ParameterDefinition(
                symbol="b",
                unit="",
                description="Broadness exponent",
                value=0.8,
                lower_limit=0.0,
                upper_limit=1.0,
                fixed=False,
            ),
        ],
    ),
)
