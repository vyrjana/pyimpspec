# pyimpspec is licensed under the GPLv3 or later (https://www.gnu.org/licenses/gpl-3.0.html).
# Copyright 2023 pyimpspec developers
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

# Mock data for use in the documentation.
from numpy import (
    array,
    logspace,
    pi,
    sqrt,
)
from pyimpspec.circuit import (
    Circuit,
    parse_cdc,
    simulate_spectrum,
)
from pyimpspec.data import DataSet

# See DOI: 10.1149/1.2044210 for information about test circuit 1 (TC-1)
TC1: Circuit = parse_cdc("R{R=100}(R{R=200}C{C=0.8e-6})(R{R=500}W{Y=4e-4})")

EXAMPLE: DataSet = simulate_spectrum(
    TC1,
    frequencies=logspace(4, 0, num=29),
    label="TC-1",
)

_f = logspace(4, -5, num=46)
VALID_RANDLES: DataSet = DataSet(
    frequencies=_f,
    impedances=parse_cdc("R{R=100}(C{C=10e-6}R{R=1e4})").get_impedances(_f),
    label="Valid",
)

_t = []
for _ in _f:
    _t.append(1 + 0.0005 * 1 / _ if _ < 1 else 1)
DRIFTING_RANDLES: DataSet = DataSet(
    frequencies=_f,
    impedances=100
    + ((1 / (2j * pi * _f * 10e-6)) ** -1 + (10e3 / sqrt(array(_t))) ** -1) ** -1,
    label="Drifting",
)
