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

# Elements should be imported here to make them available.
from .capacitor import Capacitor
from .constant_phase_element import ConstantPhaseElement
from .de_levie import DeLevieFiniteLength
from .gerischer import (
    Gerischer,
    GerischerAlternative,
)
from .havriliak_negami import (
    HavriliakNegami,
    HavriliakNegamiAlternative,
)
from .inductor import (
    Inductor,
    ModifiedInductor,
)
from .kramers_kronig import (
    KramersKronigRC,
    KramersKronigAdmittanceRC,
)
from .resistor import Resistor
from .transmission_line_model import (
    TransmissionLineModel,
    TransmissionLineModelBlockingCPE,
    TransmissionLineModelBlockingOpen,
    TransmissionLineModelBlockingShort,
    TransmissionLineModelNonblockingCPE,
    TransmissionLineModelNonblockingOpen,
    TransmissionLineModelNonblockingShort,
)
from .warburg import (
    Warburg,
    WarburgOpen,
    WarburgShort,
)
from .zarc import ZARC

# Enable validation for user-defined elements that may be registered once
# pyimpspec has been imported. The non-user-defined elements are validated
# as part of the unit tests to reduce the amount of time it takes to import
# pyimpspec.
import pyimpspec.circuit.registry as _registry

_registry._initialized()
