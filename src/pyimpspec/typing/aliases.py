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

from numpy.typing import NDArray
from numpy import (
    complex128,
    float64,
    int64,
)

# A 1-dimensional numpy array of integer indices
Indices = NDArray[int64]

# Frequency (unit is hertz)
Frequency = float64
Frequencies = NDArray[Frequency]

# Complex impedance (unit is ohm)
ComplexImpedance = complex128
ComplexImpedances = NDArray[ComplexImpedance]

# Real or imaginary impedance (unit is ohm)
Impedance = float64
Impedances = NDArray[Impedance]

# Phases (unit is degrees)
Phase = float64
Phases = NDArray[Phase]

# Time constants (unit is seconds)
TimeConstant = float64
TimeConstants = NDArray[TimeConstant]

# Gamma (unit is ohm)
Gamma = float64
Gammas = NDArray[Gamma]

# Residuals
Residual = float64
Residuals = NDArray[Residual]
ComplexResidual = complex128
ComplexResiduals = NDArray[ComplexResidual]
