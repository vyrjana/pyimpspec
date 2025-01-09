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


class ConstantPhaseElement(Element):
    def _impedance(self, f: Frequencies, Y: float, n: float) -> ComplexImpedances:
        return 1 / (Y * (1j * 2 * pi * f) ** n)


register_element(
    ElementDefinition(
        Class=ConstantPhaseElement,
        symbol="Q",
        name="Constant phase element",
        description="Can be used to model, e.g., non-ideal capacitance.",
        equation="(Y*(2*pi*f*I)^n)^-1",
        parameters=[
            ParameterDefinition(
                symbol="Y",
                unit="S*s^n",
                description="'Admittance'",
                value=1e-6,
                lower_limit=1e-24,
                upper_limit=1e6,
                fixed=False,
            ),
            ParameterDefinition(
                symbol="n",
                unit="",
                description="Phase angle in -pi/2 radians",
                value=0.95,
                lower_limit=0.0,
                upper_limit=1.0,
                fixed=False,
            ),
        ],
    ),
)
