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

from pyimpspec.exceptions import *
from pyimpspec.data import (
    DataSet,
    get_parsers,
    dataframe_to_data_sets,
    parse_data,
)
from pyimpspec.circuit import (
    parse_cdc,
    simulate_spectrum,
)
from pyimpspec.circuit.base import (
    Connection,
    Container,
    Element,
)
from pyimpspec.circuit.connections import *
from pyimpspec.circuit.registry import (
    ContainerDefinition,
    ElementDefinition,
    ParameterDefinition,
    SubcircuitDefinition,
    get_elements,
    register_element,
)
from pyimpspec.circuit.elements import *
from pyimpspec.circuit.circuit import Circuit
from pyimpspec.circuit.circuit_builder import CircuitBuilder
from pyimpspec.analysis.utility import (
    get_default_num_procs,
    set_default_num_procs,
)
from pyimpspec.analysis.drt import (
    DRTPeaks,
    DRTPeak,
    DRTResult,
    calculate_drt,
)
from pyimpspec.analysis.fitting import (
    FitIdentifiers,
    FitResult,
    FittedParameter,
    fit_circuit,
    generate_fit_identifiers,
)
from pyimpspec.analysis.kramers_kronig import (
    KramersKronigResult,
    perform_kramers_kronig_test,
    perform_exploratory_kramers_kronig_tests,
)
from pyimpspec.analysis.zhit import (
    ZHITResult,
    perform_zhit,
)
from pyimpspec.typing import *
import pyimpspec.plot.mpl as mpl
from pyimpspec.mock_data import (
    generate_mock_circuits,
    generate_mock_data,
)
from .version import PACKAGE_VERSION
