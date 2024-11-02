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
from .functions import (
    coth,
    tanh,
)
from .base import Element
from .registry import (
    ElementDefinition,
    ParameterDefinition,
    register_element,
)
from pyimpspec.typing import (
    ComplexImpedance,
    ComplexImpedances,
    Frequencies,
)


class Warburg(Element):
    def _impedance(self, f: Frequencies, Y: float, n: float) -> ComplexImpedances:
        return 1 / (Y * (1j * 2 * pi * f) ** n)


register_element(
    ElementDefinition(
        Class=Warburg,
        symbol="W",
        name="Warburg (semi-infinite)",
        description=r"""
Semi-infinite Warburg diffusion.

|equation|

where :math:`Y = \frac{1}{\sigma\sqrt{2}}` and :math:`\sigma` is the Warburg coefficient in ohm/s^(1/2) when :math:`n=0.5`.
""",
        equation="(Y*(2*pi*f*I)^n)^-1",
        parameters=[
            ParameterDefinition(
                symbol="Y",
                unit="S*s^n",
                description="'Admittance'",
                value=1e-3,
                lower_limit=1e-24,
                upper_limit=inf,
                fixed=False,
            ),
            ParameterDefinition(
                symbol="n",
                unit="",
                description="Phase angle in -pi/2 radians",
                value=0.5,
                lower_limit=0.0,
                upper_limit=1.0,
                fixed=True,
            ),
        ],
    ),
)


class WarburgOpen(Element):
    def _impedance(
        self,
        f: Frequencies,
        Y: float,
        B: float,
        n: float,
    ) -> ComplexImpedances:
        return (coth((B * 1j * 2 * pi * f) ** n) / ((Y * 1j * 2 * pi * f) ** n)).astype(
            ComplexImpedance
        )


register_element(
    ElementDefinition(
        Class=WarburgOpen,
        symbol="Wo",
        name="Warburg (finite length, reflective boundary)",
        description=r"""
Finite length diffusion with reflective boundary.

|equation|

where

- :math:`Y = \frac{1}{\sigma\sqrt{2}}` and :math:`\sigma` is the Warburg coefficient in ohm/s^(1/2)
- :math:`B = \frac{\delta}{\sqrt{D}}`, :math:`\delta` is the diffusion layer thickness, and :math:`D` is the diffusion coefficient

Re(Z) approaches :math:`\frac{(B/Y)^n}{3}` and Im(Z) exhibits capacitive-like behavior as the frequency approaches 0 Hz.
""",
        equation="coth((B*I*2*pi*f)^n)/((Y*I*2*pi*f)^n)",
        parameters=[
            ParameterDefinition(
                symbol="Y",
                unit="S",
                description="'Admittance'",
                value=1.0,
                lower_limit=1e-24,
                upper_limit=inf,
                fixed=False,
            ),
            ParameterDefinition(
                symbol="B",
                unit="s^n",
                description="'Time constant'",
                value=1.0,
                lower_limit=0.0,
                upper_limit=inf,
                fixed=False,
            ),
            ParameterDefinition(
                symbol="n",
                unit="",
                description="Phase angle in -pi/2 radians",
                value=0.5,
                lower_limit=0.0,
                upper_limit=1.0,
                fixed=True,
            ),
        ],
    ),
)


class WarburgShort(Element):
    def _impedance(
        self,
        f: Frequencies,
        Y: float,
        B: float,
        n: float,
    ) -> ComplexImpedances:
        return (tanh((B * 2 * pi * f * 1j) ** n) / (Y * 2 * pi * f * 1j) ** n).astype(
            ComplexImpedance,
        )


register_element(
    ElementDefinition(
        Class=WarburgShort,
        symbol="Ws",
        name="Warburg (finite length, transmissive boundary)",
        description=r"""
Finite length diffusion with transmissive boundary.

|equation|

where

- :math:`Y = \frac{1}{\sigma\sqrt{2}}` and :math:`\sigma` is the Warburg coefficient in ohm/s^(1/2)
- :math:`B = \frac{\delta}{\sqrt{D}}`, :math:`\delta` is the diffusion layer thickness, and :math:`D` is the diffusion coefficient

Re(Z) and Im(Z) approach :math:`(\frac{B}{Y})^n` and 0, respectively, as the frequency approaches 0 Hz.
""",
        equation="tanh((B*I*2*pi*f)^n)/((Y*I*2*pi*f)^n)",
        parameters=[
            ParameterDefinition(
                symbol="Y",
                unit="S",
                description="'Admittance'",
                value=1.0,
                lower_limit=1e-24,
                upper_limit=inf,
                fixed=False,
            ),
            ParameterDefinition(
                symbol="B",
                unit="s^n",
                description="'Time constant'",
                value=1.0,
                lower_limit=0.0,
                upper_limit=inf,
                fixed=False,
            ),
            ParameterDefinition(
                symbol="n",
                unit="",
                description="Phase angle in -pi/2 radians",
                value=0.5,
                lower_limit=0.0,
                upper_limit=1.0,
                fixed=True,
            ),
        ],
    ),
)
