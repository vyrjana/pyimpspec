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


def parse_p00(path: str) -> DataSet:
    """
    Parse a .P00 file containing an impedance spectrum.

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
    num_points: int = 0
    while lines:
        line: str = lines.pop(0)
        if line.startswith("f/hz"):
            num_points = int(lines.pop(0))
            break
    freq: List[float] = []
    real: List[float] = []
    imag: List[float] = []
    while lines:
        columns: List[str] = lines.pop(0).split("\t")
        assert len(columns) == 6, (
            len(columns),
            columns,
        )
        freq.append(float(columns[0]))
        real.append(float(columns[1]))
        imag.append(-float(columns[2]))
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
