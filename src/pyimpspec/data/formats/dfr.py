# pyimpspec is licensed under the GPLv3 or later (https://www.gnu.org/licenses/gpl-3.0.html).
# Copyright 2022 pyimpspec developers
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

from os.path import exists
from typing import (
    IO,
    List,
)
from pyimpspec.data.data_set import (
    DataFrame,
    DataSet,
    dataframe_to_dataset,
)


def parse_dfr(path: str) -> DataSet:
    """
    Parse an Eco Chemie .dfr file containing an impedance spectrum.

    Parameters
    ----------
    path: str
        The path to the file to process.

    Returns
    -------
    DataSet
    """
    assert type(path) is str and exists(path)
    fp: IO
    with open(path, "r", encoding="latin1") as fp:
        lines: List[str] = list(
            filter(lambda _: _ != "", map(str.lower, map(str.strip, fp.readlines())))
        )
    line: str = lines.pop(0)
    assert line == "version8.0", line
    num_points: int = int(lines.pop(0))
    lines.pop(0)  # ?
    freq: List[float] = []
    real: List[float] = []
    imag: List[float] = []
    for _ in range(0, num_points):
        assert len(lines) >= 10, lines
        freq.append(float(lines.pop(0)))
        real.append(float(lines.pop(0)))
        imag.append(-float(lines.pop(0)))
        lines.pop(0)  # E_dc
        lines.pop(0)  # I_dc
        lines.pop(0)  # time
        lines.pop(0)  # ?
        lines.pop(0)  # ?
        lines.pop(0)  # ?
    assert len(freq) == len(real) == len(imag) == num_points > 0, len(freq)
    return dataframe_to_dataset(
        DataFrame.from_dict(
            {
                "frequency": freq,
                "real": real,
                "imaginary": imag,
            }
        ),
        path,
        "",
    )
