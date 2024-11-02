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
from numpy.typing import NDArray
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


class KramersKronigRC(Element):
    def _impedance(self, f: Frequencies, R: float, tau: float) -> ComplexImpedances:
        return R / (1 + 1j * 2 * pi * f * tau)


register_element(
    ElementDefinition(
        Class=KramersKronigRC,
        symbol="K",
        name="'Parallel RC' element",
        description="Parallel RC element with a fixed time constant that is used in linear Kramers-Kronig tests on impedance data.",
        equation="R/(1+I*2*pi*f*tau)",
        parameters=[
            ParameterDefinition(
                symbol="R",
                unit="ohm",
                description="Resistance",
                value=1.0,
                lower_limit=-inf,
                upper_limit=inf,
                fixed=False,
            ),
            ParameterDefinition(
                symbol="tau",
                unit="s",
                description="Time constant",
                value=1.0,
                lower_limit=-inf,
                upper_limit=inf,
                fixed=True,
            ),
        ],
    ),
    private=True,
)


class KramersKronigAdmittanceRC(Element):
    def _impedance(self, f: Frequencies, C: float, tau: float) -> ComplexImpedances:
        w: NDArray[float64] = 2 * pi * f
        return 1 / ((C*w)/(w*tau-1j))


register_element(
    ElementDefinition(
        Class=KramersKronigAdmittanceRC,
        symbol="Ky",
        name="'Series RC' element",
        description="Series RC element with a fixed time constant that is used in linear Kramers-Kronig tests on admittance data.",
        equation="1/((C*2*pi*f)/(2*pi*f*tau-I))",
        parameters=[
            ParameterDefinition(
                symbol="C",
                unit="F",
                description="Capacitance",
                value=1.0,
                lower_limit=-inf,
                upper_limit=inf,
                fixed=False,
            ),
            ParameterDefinition(
                symbol="tau",
                unit="s",
                description="Time constant",
                value=1.0,
                lower_limit=-inf,
                upper_limit=inf,
                fixed=True,
            ),
        ],
    ),
    private=True,
)
