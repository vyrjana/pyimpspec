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

from numpy import pi
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


class Inductor(Element):
    def _impedance(self, f: Frequencies, L: float) -> ComplexImpedances:
        return L * 1j * 2 * pi * f


register_element(
    ElementDefinition(
        Class=Inductor,
        symbol="L",
        name="Ideal inductor",
        description="An ideal inductor.",
        equation="I*2*pi*f*L",
        parameters=[
            ParameterDefinition(
                symbol="L",
                unit="H",
                description="Inductance",
                value=1e-6,
                lower_limit=0.0,
                upper_limit=1e3,
                fixed=False,
            ),
        ],
    ),
)


class ModifiedInductor(Element):
    def _impedance(self, f: Frequencies, L: float, n: float) -> ComplexImpedances:
        return L * (1j * 2 * pi * f) ** n


register_element(
    ElementDefinition(
        Class=ModifiedInductor,
        symbol="La",
        name="Modified inductor",
        description="Can be used to model non-ideal inductance.",
        equation="L*(I*2*pi*f)^n",
        parameters=[
            ParameterDefinition(
                symbol="L",
                unit="H*s^(n-1)",
                description="'Inductance'",
                value=1e-6,
                lower_limit=0.0,
                upper_limit=1e3,
                fixed=False,
            ),
            ParameterDefinition(
                symbol="n",
                unit="",
                description="Phase angle in pi/2 radians",
                value=0.95,
                lower_limit=0.0,
                upper_limit=1.0,
                fixed=False,
            ),
        ],
    ),
)
