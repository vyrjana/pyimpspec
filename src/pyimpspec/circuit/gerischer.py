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


class Gerischer(Element):
    def _impedance(self, f: Frequencies, Y: float, k: float, n: float) -> ComplexImpedances:
        return 1 / (Y * (k + 2 * pi * f * 1j) ** n)


register_element(
    ElementDefinition(
        Class=Gerischer,
        symbol="G",
        name="Gerischer",
        description="The impedance associated with an electroactive species that is created by a reaction in the electrolyte solution.",
        equation="(Y*(k+I*2*pi*f)^n)^-1",
        parameters=[
            ParameterDefinition(
                symbol="Y",
                unit="S*s^n",
                description="'Admittance'",
                value=1.0,
                lower_limit=1e-24,
                upper_limit=inf,
                fixed=False,
            ),
            ParameterDefinition(
                symbol="k",
                unit="s^-1",
                description="'Rate constant'",
                value=1.0,
                lower_limit=1e-24,
                upper_limit=inf,
                fixed=False,
            ),
            ParameterDefinition(
                symbol="n",
                unit="",
                description="",
                value=0.5,
                lower_limit=0.0,
                upper_limit=1.0,
                fixed=True,
            ),
        ],
    ),
)


class GerischerAlternative(Element):
    def _impedance(self, f: Frequencies, R: float, tau: float, n: float) -> ComplexImpedances:
        return R / ((1 + 1j * 2 * pi * f * tau)**n)


register_element(
    ElementDefinition(
        Class=GerischerAlternative,
        symbol="Ga",
        name="Gerischer, alt.",
        description="The impedance associated with an electroactive species that is created by a reaction in the electrolyte solution (alternative form).",
        equation="R/((1+I*2*pi*f*tau)^n)",
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
                symbol="n",
                unit="",
                description="",
                value=0.5,
                lower_limit=0.0,
                upper_limit=1.0,
                fixed=True,
            ),
        ],
    ),
)
