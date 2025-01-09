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


class Capacitor(Element):
    def _impedance(self, f: Frequencies, C: float) -> ComplexImpedances:
        return 1 / (1j * 2 * pi * f * C)


register_element(
    ElementDefinition(
        Class=Capacitor,
        symbol="C",
        name="Capacitor",
        description="An ideal capacitor.",
        equation="(2*pi*f*C*I)^-1",
        parameters=[
            ParameterDefinition(
                symbol="C",
                unit="F",
                description="Capacitance",
                value=1e-6,
                lower_limit=1e-24,
                upper_limit=1e3,
                fixed=False,
            ),
        ],
    ),
)
