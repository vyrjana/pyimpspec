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

from pyimpspec.data.data_set import (
    DataSet,
    dataframe_to_data_sets,
)
from pyimpspec.exceptions import UnsupportedFileFormat
from pyimpspec.typing.helpers import (
    IO,
    List,
    Path,
    Union,
)
from .helpers import (
    _parse_string_as_float,
    _validate_path,
)


def parse_dfr(path: Union[str, Path]) -> List[DataSet]:
    """
    Parse an Eco Chemie .dfr file containing an impedance spectrum.

    Parameters
    ----------
    path: Union[str, pathlib.Path]
        The path to the file to process.

    Returns
    -------
    List[DataSet]
    """
    from pandas import DataFrame

    _validate_path(path)

    fp: IO
    with open(path, "r", encoding="latin1") as fp:
        lines: List[str] = list(
            filter(lambda _: _ != "", map(str.lower, map(str.strip, fp.readlines())))
        )

    line: str = lines.pop(0)
    # If the file fails to parse, then it might be an unsupported version.
    # This was developed under the expectation that 'line=="version8.0"'.
    if not line.startswith("version"):
        raise UnsupportedFileFormat(
            f"Expected a header line along the lines of 'VERSIONx.y' instead of {line=}"
        )

    line = lines.pop(0)
    if not line.isnumeric():
        raise UnsupportedFileFormat(
            f"Expected a line that specifies the number of data points instead of {line=}"
        )

    num_points: int = int(line)
    lines.pop(0)  # ?

    freq: List[float] = []
    real: List[float] = []
    imag: List[float] = []

    for _ in range(0, num_points):
        if not len(lines) >= 9:
            raise UnsupportedFileFormat(
                "Expected to have nine lines to parse for each measured frequency"
            )

        freq.append(_parse_string_as_float(lines.pop(0)))
        real.append(_parse_string_as_float(lines.pop(0)))
        imag.append(-_parse_string_as_float(lines.pop(0)))
        lines.pop(0)  # E_dc
        lines.pop(0)  # I_dc
        lines.pop(0)  # time
        lines.pop(0)  # ?
        lines.pop(0)  # ?
        lines.pop(0)  # ?

    if not (len(freq) == len(real) == len(imag) > 0):
        raise ValueError(
            "Expected at least one set of frequency, real, and imaginary values"
        )

    return dataframe_to_data_sets(
        DataFrame.from_dict(
            {
                "frequency": freq,
                "real": real,
                "imaginary": imag,
            }
        ),
        path=path,
    )
