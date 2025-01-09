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

# Mock data for use in the documentation.
from dataclasses import dataclass
from numpy import (
    float64,
    inf,
    logspace,
    zeros,
)
from numpy.random import RandomState
from numpy.typing import NDArray
from pyimpspec.circuit import (
    Circuit,
    parse_cdc,
    simulate_spectrum,
)
from pyimpspec.exceptions import ParsingError
from pyimpspec.data import DataSet
from pyimpspec.typing import (
    ComplexImpedances,
    ComplexImpedance,
)
from pyimpspec.typing.helpers import (
    Dict,
    List,
    Optional,
    _is_floating,
    _is_integer,
)
from pyimpspec.circuit.registry import (
    Frequencies,
    Connection,
    Element,
    ElementDefinition,
    ParameterDefinition,
    _initialize_element,
)
from pyimpspec.circuit.elements import Resistor


class DriftingResistor(Element):
    def _impedance(self, f: Frequencies, R: float, v: float) -> ComplexImpedances:
        Z: ComplexImpedances = R + 0j * f

        i: int
        f: float64
        for i, _f in enumerate(f):
            Z[i:] += v / _f

        return Z


_initialize_element(
    ElementDefinition(
        Class=DriftingResistor,
        symbol="Rdrifting",
        name="Drifting resistor",
        description="Element for generating invalid spectra.",
        equation="IGNORE",
        parameters=[
            ParameterDefinition(
                symbol="R",
                unit="ohm",
                description="Resistance",
                value=1000.0,
                lower_limit=0.0,
                upper_limit=inf,
                fixed=False,
            ),
            ParameterDefinition(
                symbol="v",
                unit="ohm/s",
                description="Drift",
                value=1.0,
                lower_limit=0.0,
                upper_limit=inf,
                fixed=False,
            ),
        ],
    ),
    validate_impedances=False,
)


@dataclass(frozen=True)
class MockDefinition:
    label: str
    cdc: str
    log_max_f: float
    log_min_f: float
    num_per_decade: int
    drift: float = 0.0

    def generate_circuit(self, **kwargs) -> Circuit:
        circuit: Circuit = parse_cdc(self.cdc)

        drift: float = kwargs.get("drift", 1.0)
        if not _is_floating(drift):
            try:
                drift = float(drift)
            except ValueError:
                raise TypeError(f"Expected a float instead of {drift=}")

        drift *= self.drift

        connection: Connection
        for connection in circuit.get_connections(recursive=True):
            element: Element
            for element in connection.get_elements(recursive=False):
                if isinstance(element, Resistor) and element.get_label() == "drift":
                    i: int = connection.index(element)
                    connection.pop(i)
                    connection.insert(
                        i,
                        DriftingResistor(**element.get_values()).set_values(v=drift),
                    )

        return circuit

    def get_identifier(self) -> str:
        label: str = self.label.upper().replace(" ", "_")
        if not label.isidentifier():
            raise ValueError(f"Expected the {label.isidentifier()=} to be True")

        return label


# Resistor instances with the label 'drift' are replaced by a DriftingResistor
_definitions: List[MockDefinition] = [
    MockDefinition(
        label="Circuit 1",
        cdc="R{R=100}(R{R=200}C{C=0.8e-6})(R{R=500}W{Y=4e-4})",
        log_max_f=4.0,
        log_min_f=0.0,
        num_per_decade=10,
    ),  # TC-1 from Boukamp, 1995
    MockDefinition(
        label="Circuit 1 invalid",
        cdc="R{R=100}(R{R=200:drift}C{C=0.8e-6})(R{R=500}W{Y=4e-4})",
        log_max_f=4.0,
        log_min_f=0.0,
        num_per_decade=10,
        drift=5.0,
    ),
    MockDefinition(
        label="Circuit 2",
        cdc="R{R=120}(R{R=430}Q{Y=6.2e-5,n=0.93})",
        log_max_f=2.25,
        log_min_f=-0.75,
        num_per_decade=10,
    ),
    MockDefinition(
        label="Circuit 2 invalid",
        cdc="R{R=120}(R{R=430:drift}Q{Y=6.2e-5,n=0.93})",
        log_max_f=2.25,
        log_min_f=-0.75,
        num_per_decade=10,
        drift=-1.0,
    ),
    MockDefinition(
        label="Circuit 3",
        cdc="R{R=20}(Q{Y=25e-6,n=0.89}[R{R=100}W{Y=2.357e-3}])",
        log_max_f=3.0,
        log_min_f=-0.5,
        num_per_decade=10,
    ),
    MockDefinition(
        label="Circuit 3 invalid",
        cdc="R{R=20}(Q{Y=25e-6,n=0.89}[R{R=100:drift}W{Y=2.357e-3}])",
        log_max_f=3.0,
        log_min_f=-0.5,
        num_per_decade=10,
        drift=1.0,
    ),
    MockDefinition(
        label="Circuit 4",
        cdc="La{L=2.000000E-06,n=8.80E-01}R{R=2.000000E-02}(R{R=1.000000E-01}Q{Y=5.3000E+02,n=7.8E-01})(R{R=1.000000E-02}Q{Y=4.1E+00,n=4.7E-01})",
        log_max_f=3.5,
        log_min_f=-2.0,
        num_per_decade=10,
    ),
    MockDefinition(
        label="Circuit 4 invalid",
        cdc="La{L=2.000000E-06,n=8.80E-01}R{R=2.000000E-02:drift}(R{R=1.000000E-01}Q{Y=5.3000E+02,n=7.8E-01})(R{R=1.000000E-02}Q{Y=4.1E+00,n=4.7E-01})",
        log_max_f=3.5,
        log_min_f=-2.0,
        num_per_decade=10,
        drift=5e-6,
    ),
    MockDefinition(
        label="Circuit 5",
        cdc="R{R=1.440000E+02}(R{R=2.407000E+02}Q{Y=1.792000E-07,n=0.91})(R{R=8.301000E+02}Q{Y=7.157000E-07,n=0.87})(R{R=4.903000E+02}Q{Y=1.629000E-05,n=0.94})",
        log_max_f=4.5,
        log_min_f=0.5,
        num_per_decade=10,
    ),
    MockDefinition(
        label="Circuit 5 invalid",
        cdc="R{R=1.440000E+02}(R{R=2.407000E+02}Q{Y=1.792000E-07,n=0.91})(R{R=8.301000E+02:drift}Q{Y=7.157000E-07,n=0.87})(R{R=4.903000E+02}Q{Y=1.629000E-05,n=0.94})",
        log_max_f=4.5,
        log_min_f=0.5,
        num_per_decade=10,
        drift=-50.0,
    ),
    MockDefinition(
        label="Circuit 6",
        cdc="[R{R=1.440000E+02}(R{R=2.210000E+02}Q{Y=8.570000E-08,n=0.93})(R{R=8.301000E+02}Q{Y=1.025000E-05,n=0.88})(R{R=4.903000E+02}Q{Y=4.862000E-03,n=0.98})]",
        log_max_f=4.0,
        log_min_f=-1.0,
        num_per_decade=10,
    ),
    MockDefinition(
        label="Circuit 6 invalid",
        cdc="[R{R=1.440000E+02}(R{R=2.210000E+02}Q{Y=8.570000E-08,n=0.93})(R{R=8.301000E+02:drift}Q{Y=1.025000E-05,n=0.88})(R{R=4.903000E+02}Q{Y=4.862000E-03,n=0.98})]",
        log_max_f=4.0,
        log_min_f=-1.0,
        num_per_decade=10,
        drift=-1.0,
    ),
    MockDefinition(
        label="Circuit 7",
        cdc="[R{R=2.000000E+00}Tlmbs{R_i=8.741000E-01,Y=6.877000E-03,n=7.274000E-01,L=1.000000E+00F}W{Y=1.000000E+01}]",
        log_max_f=5.0,
        log_min_f=-2.0,
        num_per_decade=10,
    ),
    MockDefinition(
        label="Circuit 7 invalid",
        cdc="[R{R=2.000000E+00:drift}Tlmbs{R_i=8.741000E-01,Y=6.877000E-03,n=7.274000E-01,L=1.000000E+00F}W{Y=1.000000E+01}]",
        log_max_f=5.0,
        log_min_f=-2.0,
        num_per_decade=10,
        drift=2.5e-4,
    ),
    MockDefinition(
        label="Circuit 8",
        cdc="R{R=25}(R{R=125}Q{Y=1e-6,n=0.97})(R{R=-250}Q{Y=5e-5,n=0.95})",
        log_max_f=4.0,
        log_min_f=-0.5,
        num_per_decade=10,
    ),
    MockDefinition(
        label="Circuit 8 invalid",
        cdc="R{R=25}(R{R=125}Q{Y=1e-6,n=0.97})(R{R=-250:drift}Q{Y=5e-5,n=0.95})",
        log_max_f=4.0,
        log_min_f=-0.5,
        num_per_decade=10,
        drift=0.75,
    ),
    MockDefinition(
        label="Circuit 9",
        cdc="R{R=9.272000E+00}(R{R=2.144000E+02}Q{Y=1.024000E-06,n=0.96}[L{L=2.213000E-04}R{R=1.844000E+01}])(R{R=-5.000000E+01}Q{Y=1.496000E-05,n=0.86})",
        log_max_f=5.0,
        log_min_f=1.5,
        num_per_decade=10,
    ),
    MockDefinition(
        label="Circuit 9 invalid",
        cdc="R{R=9.272000E+00:drift}(R{R=2.144000E+02}Q{Y=1.024000E-06,n=0.96}[L{L=2.213000E-04}R{R=1.844000E+01}])(R{R=-5.000000E+01}Q{Y=1.496000E-05,n=0.86})",
        log_max_f=5.0,
        log_min_f=1.5,
        num_per_decade=10,
        drift=-25.0,
    ),
    MockDefinition(
        label="Circuit 10",
        cdc="R{R=100}",
        log_max_f=5.0,
        log_min_f=-1.0,
        num_per_decade=10,
    ),
    MockDefinition(
        label="Circuit 10 invalid",
        cdc="R{R=100:drift}",
        log_max_f=5.0,
        log_min_f=-1.0,
        num_per_decade=10,
        drift=0.5,
    ),
    MockDefinition(
        label="Circuit 11",
        cdc="R{R=1}C{C=1e-1}",
        log_max_f=5.0,
        log_min_f=-1.0,
        num_per_decade=10,
    ),
    MockDefinition(
        label="Circuit 11 invalid",
        cdc="R{R=1:drift}C{C=1e-1}",
        log_max_f=2.0,
        log_min_f=-1.0,
        num_per_decade=10,
        drift=0.5,
    ),
    MockDefinition(
        label="Circuit 12",
        cdc="R{R=1}L{L=1e-2}",
        log_max_f=3.0,
        log_min_f=-1.0,
        num_per_decade=10,
    ),
    MockDefinition(
        label="Circuit 12 invalid",
        cdc="R{R=1:drift}L{L=1e-2}",
        log_max_f=3.0,
        log_min_f=-1.0,
        num_per_decade=10,
        drift=0.5,
    ),
    MockDefinition(
        label="Circuit 13",
        cdc="R{R=70}(R{R=200}C{C=2.5e-3})(R{R=100}C{C=1e-4})(R{R=50}L{L=3e2})",
        log_max_f=2.0,
        log_min_f=-2.0,
        num_per_decade=10,
    ),
    MockDefinition(
        label="Circuit 13 invalid",
        cdc="R{R=70}(R{R=200}C{C=2.5e-3})(R{R=100}C{C=1e-4})(R{R=50:drift}L{L=3e2})",
        log_max_f=2.0,
        log_min_f=-2.0,
        num_per_decade=10,
        drift=0.5,
    ),
    MockDefinition(
        label="Circuit 14",
        cdc="R{R=70}(R{R=200}Q{Y=2.5e-3,n=0.9})(R{R=100}Q{Y=1e-4,n=0.85})(R{R=50}La{L=3e2,n=0.95})",
        log_max_f=2.0,
        log_min_f=-2.0,
        num_per_decade=10,
    ),
    MockDefinition(
        label="Circuit 14 invalid",
        cdc="R{R=70}(R{R=200}Q{Y=2.5e-3,n=0.9})(R{R=100}Q{Y=1e-4,n=0.85})(R{R=50:drift}La{L=3e2,n=0.95})",
        log_max_f=2.0,
        log_min_f=-2.0,
        num_per_decade=10,
        drift=0.5,
    ),
    MockDefinition(
        label="Circuit 15",
        cdc="(C{C=2e-10}[R{R=1.5e3}(Q{Y=5e-5,n=0.8}[R{R=500}Ws{Y=0.00004,B=150}])])",
        log_max_f=6.0,
        log_min_f=-3.0,
        num_per_decade=10,
    ),
    MockDefinition(
        label="Circuit 15 invalid",
        cdc="(C{C=2e-10}[R{R=1.5e3}(Q{Y=5e-5,n=0.8}[R{R=500:drift}Ws{Y=0.00004,B=150}])])",
        log_max_f=6.0,
        log_min_f=-3.0,
        num_per_decade=10,
        drift=0.05,
    ),
    MockDefinition(
        label="Circuit 16",
        cdc="(R{R=1.2}Q{n=0.85,Y=0.011327051})(R{R=0.8}Q{n=0.95,Y=0.034339661})(R{R=2}Q{n=0.7,Y=0.138116033})(R{R=1.6}Q{n=1,Y=0.000009947})(R{R=0.8}Q{n=1,Y=0.000066315})(R{R=1.6}Q{n=1,Y=0.000099472})",
        log_max_f=6.0,
        log_min_f=-3.0,
        num_per_decade=10,
    ),
    MockDefinition(
        label="Circuit 16 invalid",
        cdc="(R{R=1.2}Q{n=0.85,Y=0.011327051})(R{R=0.8}Q{n=0.95,Y=0.034339661})(R{R=2}Q{n=0.7,Y=0.138116033})(R{R=1.6:drift}Q{n=1,Y=0.000009947})(R{R=0.8}Q{n=1,Y=0.000066315})(R{R=1.6}Q{n=1,Y=0.000099472})",
        log_max_f=6.0,
        log_min_f=-3.0,
        num_per_decade=10,
        drift=0.00005,
    ),
    MockDefinition(
        label="Circuit 17",
        cdc="Ha{R=1,tau=1,a=0.95,b=0.5}",
        log_max_f=3.0,
        log_min_f=-4.0,
        num_per_decade=10,
    ),
    MockDefinition(
        label="Circuit 18",
        cdc="R{R=10}Ga{R=50,tau=0.01}",
        log_max_f=5.0,
        log_min_f=-3.0,
        num_per_decade=10,
    ),
    MockDefinition(
        label="Circuit 19",
        cdc="Ws{Y=1.0,B=1.0,n=0.47}",
        log_max_f=4.0,
        log_min_f=-3.0,
        num_per_decade=10,
    ),
]

if len(set(d.get_identifier() for d in _definitions)) != len(_definitions):
    raise ValueError("Detected mock data definitions with identical identifiers")


def _add_noise(
    data: DataSet,
    noise: float = 0.1,
    seed: Optional[int] = 42,
    label: Optional[str] = None,
) -> DataSet:
    Z_ideal: ComplexImpedances = data.get_impedances()
    sd: NDArray[float64] = noise / 100 * abs(Z_ideal)

    if seed is not None:
        seed = seed & (2**32 - 1)  # Truncate to 32 bits just to be safe
    rs: RandomState = RandomState(seed=seed)

    f: Frequencies = data.get_frequencies()
    Z_noisy: ComplexImpedances = zeros(len(f), dtype=ComplexImpedance)
    Z_noisy.real = rs.normal(0, sd)
    Z_noisy.imag = rs.normal(0, sd)
    Z_noisy += Z_ideal

    if label is None:
        label = data.get_label()
        if label == "":
            label = "Noisy"
        else:
            label += " (noisy)"

    return DataSet(
        frequencies=data.get_frequencies(),
        impedances=Z_noisy,
        path=f"{label}.csv",
        label=label,
    )


def _simulate_spectrum(
    definition: Optional[MockDefinition],
    circuit: Optional[Circuit],
    **kwargs,
) -> DataSet:
    if isinstance(definition, MockDefinition):
        circuit = definition.generate_circuit(**kwargs)
    elif not isinstance(circuit, Circuit):
        raise TypeError(
            f"Expected a MockDefinition or a Circuit instead of {definition=} and {circuit=}"
        )

    log_max_f: float = kwargs.get(
        "log_max_f",
        5.0 if definition is None else definition.log_max_f,
    )
    if not _is_floating(log_max_f):
        try:
            log_max_f = float(log_max_f)
        except ValueError:
            raise TypeError(f"Expected a float instead of {log_max_f=}")
    
    log_min_f: float = kwargs.get(
        "log_min_f",
        -1.0 if definition is None else definition.log_min_f,
    )
    if not _is_floating(log_min_f):
        try:
            log_min_f = float(log_min_f)
        except ValueError:
            raise TypeError(f"Expected a float instead of {log_min_f=}")
    
    if log_max_f <= log_min_f:
        raise ValueError(f"Expected {log_min_f=} < {log_max_f=}")

    num_per_decade: int = kwargs.get(
        "num_per_decade",
        10 if definition is None else definition.num_per_decade,
    )
    if not _is_integer(num_per_decade):
        try:
            num_per_decade = int(num_per_decade)
        except ValueError:
            raise TypeError(f"Expected an integer instead of {num_per_decade=}")
    
    if num_per_decade < 1:
        raise ValueError(
            f"Expected a value greater than zero instead of {num_per_decade=}"
        )

    noise: float = kwargs.get("noise", 0.0)
    if not _is_floating(noise):
        try:
            noise = float(noise)
        except ValueError:
            raise TypeError(f"Expected a float instead of {noise=}")
    
    if noise < 0.0:
        raise ValueError(
            f"Expected a value greater than or equal to zero instead of {noise=}"
        )

    f: NDArray[float64] = logspace(
        log_max_f,
        log_min_f,
        num=int(round((log_max_f - log_min_f) * num_per_decade + 1)),
    )
    data: DataSet = simulate_spectrum(
        circuit=circuit,
        frequencies=f,
        label=circuit.to_string() if definition is None else definition.label,
    )

    if noise > 0.0:
        seed: int = kwargs.get("seed", None)
        data = _add_noise(data, noise=noise, seed=seed)

    return data


def _find_definitions(identifier: str) -> List[MockDefinition]:
    definition_lookup: Dict[str, MockDefinition] = {
        d.get_identifier(): d for d in _definitions
    }
    matches: List[MockDefinition] = []

    if identifier in definition_lookup:
        matches.append(definition_lookup[identifier])

    elif "*" in identifier:
        fragments: List[str] = list(
            filter(lambda f: f.strip() != "", identifier.split("*"))
        )

        key: str
        definition: MockDefinition
        for key, definition in definition_lookup.items():
            for f in fragments:
                if f not in key:
                    break
            else:
                matches.append(definition)

    return matches


def _try_to_parse(cdc: str) -> Circuit:
    try:
        return parse_cdc(cdc)
    except ParsingError:
        valid_values: str = "'\n- '".join(d.get_identifier() for d in _definitions)
        raise ValueError(
            f"Invalid identifier (or circuit description code): '{cdc}'! The valid identifiers are:\n- '{valid_values}'"
        )


def generate_mock_circuits(identifier: str, **kwargs) -> List[Circuit]:
    """
    Generate the circuits used by the ``generate_mock_data`` function.

    Parameters
    ----------
    identifier: str
        See |generate_mock_data| for details.

    **kwargs
        See |generate_mock_data| for details.

    Returns
    -------
    List[Circuit]
    """
    definitions: List[MockDefinition] = _find_definitions(identifier)
    if definitions:
        return [d.generate_circuit(**kwargs) for d in definitions]

    return [_try_to_parse(identifier)]


def generate_mock_data(identifier: str, **kwargs) -> List[DataSet]:
    """
    Generate impedance spectra either from a set of pre-defined equivalent circuits or a circuit description code.

    Parameters
    ----------
    identifier: str
        An identifier that corresponds to one of the predefined equivalent circuits.
        If the string contains the `*` wildcard, then the string is split into fragments and valid identifiers containing all the fragments are returned.
        For example, `CIRCUIT_7*` returns the immittance spectra corresponding to the predefined equivalent circuits with the identifiers `CIRCUIT_7` and `CIRCUIT_7_INVALID`.
        If no matches can be found (e.g., if an empty string is provided), then a list of valid identifiers will be printed.
        The identifiers are case-sensitive.
        If the identifier is not valid, then an attempt will be made to parse it as if it were a circuit description code.

    **kwargs
        Keyword arguments to use when simulating an immittance spectrum.
        Valid keys include:

        - `log_max_f`: the logarithm of the highest frequency.
        - `log_min_f`: the logarithm of the lowest frequency.
        - `num_per_decade`: the number of points per decade.
        - `noise`: the amount of noise to add (as a percentage of :math:`|Z|`).
        - `seed`: the seed for the pseudo-random number generator.
        - `drift`: if there is an element that exhibits drift, then it will be multiplied by this factor.

    Returns
    -------
    List[DataSet]
    """
    definitions: List[MockDefinition] = _find_definitions(identifier)
    if definitions:
        return [_simulate_spectrum(d, None, **kwargs) for d in definitions]

    return [_simulate_spectrum(None, _try_to_parse(identifier), **kwargs)]
