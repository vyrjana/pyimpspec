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
    _get_default_num_procs as get_default_num_procs,
    _set_default_num_procs as set_default_num_procs,
)
from pyimpspec.analysis.drt import (
    BHTResult,
    DRTResult,
    MRQFitResult,
    TRNNLSResult,
    TRRBFResult,
    calculate_drt,
    calculate_drt_bht,
    calculate_drt_mrq_fit,
    calculate_drt_tr_nnls,
    calculate_drt_tr_rbf,
)
from pyimpspec.analysis.fitting import (
    FitResult,
    FittedParameter,
    fit_circuit,
)
from pyimpspec.analysis.kramers_kronig import (
    TestResult,
    perform_exploratory_tests,
    perform_test,
)
from pyimpspec.analysis.zhit import (
    ZHITResult,
    perform_zhit,
)
from pyimpspec.typing import *
import pyimpspec.plot.mpl as mpl
