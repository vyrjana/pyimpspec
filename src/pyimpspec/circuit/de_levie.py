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
    float64,
    inf,
    pi,
)
from .functions import (
    coth,
    sqrt,
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


class DeLevieFiniteLength(Element):
    def _impedance(
        self,
        f: Frequencies,
        R_i: float,
        R_r: float,
        Y: float,
        n: float,
        d: float,
    ) -> ComplexImpedances:
        alpha: float64 = sqrt(R_r * R_i)
        beta: ComplexImpedances = sqrt(1 + Y * (1j * 2 * pi * f) ** n)
        return (alpha * (coth(d * alpha * beta)) / beta)


register_element(
    ElementDefinition(
        Class=DeLevieFiniteLength,
        symbol="Ls",
        name="de Levie",
        description="Can be used to model porous electrodes.",
        equation="sqrt(R_i*R_r) * (coth(d*sqrt(R_i/R_r) * sqrt(1+Y*(2*pi*f*I)^n)) / sqrt(1+Y*(2*pi*f*I)^n))",
        parameters=[
            ParameterDefinition(
                symbol="R_i",
                unit="ohm/cm",
                description="Ionic resistance",
                value=10.0,
                lower_limit=0.0,
                upper_limit=inf,
                fixed=False,
            ),
            ParameterDefinition(
                symbol="R_r",
                unit="ohm*cm",
                description="Faradaic reaction resistance",
                value=1.0,
                lower_limit=0.0,
                upper_limit=inf,
                fixed=False,
            ),
            ParameterDefinition(
                symbol="Y",
                unit="S*s^n/cm",
                description="'Admittance'",
                value=1e-2,
                lower_limit=1e-24,
                upper_limit=inf,
                fixed=False,
            ),
            ParameterDefinition(
                symbol="n",
                unit="",
                description="Phase angle in -pi/2 radians",
                value=0.8,
                lower_limit=0.0,
                upper_limit=1.0,
                fixed=False,
            ),
            ParameterDefinition(
                symbol="d",
                unit="cm",
                description="Electrode thickness",
                value=0.2,
                lower_limit=1e-24,
                upper_limit=inf,
                fixed=False,
            ),
        ],
    ),
)
