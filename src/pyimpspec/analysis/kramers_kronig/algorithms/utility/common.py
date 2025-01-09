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
    array,
    log10 as log,
    logspace,
)
from pyimpspec.analysis.kramers_kronig.result import KramersKronigResult
from pyimpspec.circuit.circuit import Circuit
from pyimpspec.circuit.kramers_kronig import KramersKronigAdmittanceRC
from pyimpspec.typing.aliases import (
    Frequencies,
    Frequency,
)
from pyimpspec.typing.helpers import (
    Dict,
    List,
)


def _truncate_circuits(
    circuits: Dict[int, Circuit],
    lower_limit: int,
    upper_limit: int,
) -> Dict[int, Circuit]:
    valid_keys: List[int]

    if lower_limit > 0 and upper_limit > 0:
        valid_keys = [
            num_RC for num_RC in circuits.keys() if lower_limit <= num_RC <= upper_limit
        ]
    elif upper_limit > 0:
        valid_keys = [num_RC for num_RC in circuits.keys() if num_RC <= upper_limit]
    elif lower_limit > 0:
        valid_keys = [num_RC for num_RC in circuits.keys() if lower_limit <= num_RC]
    else:
        valid_keys = list(circuits.keys())

    if len(valid_keys) == 0:
        raise ValueError(
            f"The specified limits mean that there are no valid number of RC elements to use: {lower_limit=}, {upper_limit=}"
        )

    return {num_RC: circuits[num_RC] for num_RC in valid_keys}


def _is_admittance_test_circuit(circuit: Circuit) -> bool:
    return any(
        map(
            lambda e: isinstance(e, KramersKronigAdmittanceRC),
            circuit.get_elements(recursive=True),
        )
    )


def subdivide_frequencies(
    frequencies: Frequencies,
    subdivision: int = 4,
) -> Frequencies:
    """
    Insert additional frequencies between each pair of frequencies.

    Parameters
    ----------
    frequencies: Frequencies
        The original frequencies that are to be subdivided.

    subdivision: int, optional
        The number of frequencies added between each of the original frequencies.

    Returns
    -------
    Frequencies
        
        A new set of frequencies with additional frequencies inserted between each of the original frequencies.
    """
    if subdivision < 1:
        raise ValueError(f"Expected {subdivision=} > 0")

    new_frequencies: List[float] = [frequencies[0]]

    f1: Frequency
    f2: Frequency
    for f1, f2 in zip(frequencies[:-1], frequencies[1:]):
        new_frequencies.extend(
            logspace(
                log(f1),
                log(f2),
                subdivision + 2,
            ).tolist()[1:]
        )

    m: int = len(new_frequencies)
    n: int = len(frequencies)
    if not (m == n + (n - 1) * subdivision):
        raise ValueError(f"Expected {m=} == {n + (n - 1) * subdivision=}")

    return array(new_frequencies)


def _generate_pseudo_chisqr_offsets(
    pseudo_chisqrs: Dict[int, float],
    factor: float,
) -> Dict[int, float]:
    if len(pseudo_chisqrs) == 1 or factor == 0.0:
        return {num_RC: 0.0 for num_RC, pseudo_chisqr in pseudo_chisqrs.items()}

    minimum: float = min(pseudo_chisqrs.values())
    maximum: float = max(pseudo_chisqrs.values())

    return {
        num_RC: ((pseudo_chisqr - minimum) / (maximum - minimum)) * factor
        for num_RC, pseudo_chisqr in pseudo_chisqrs.items()
    }
