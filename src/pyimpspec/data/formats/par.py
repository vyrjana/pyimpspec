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

from os.path import exists
from typing import (
    IO,
    List,
)
from pyimpspec.data.data_set import (
    DataSet,
    dataframe_to_data_sets,
)


def parse_par(path: str) -> List[DataSet]:
    """
    Parse a Parstat .par file containing an impedance spectrum.

    Parameters
    ----------
    path: str
        The path to the file to process.

    Returns
    -------
    List[DataSet]
    """
    from pandas import DataFrame
    assert isinstance(path, str) and exists(path), path
    fp: IO
    with open(path, "r", encoding="latin1") as fp:
        lines: List[str] = list(
            filter(lambda _: _ != "", map(str.lower, map(str.strip, fp.readlines())))
        )

    assert len(lines) > 0, f"Failed to find any data in '{path}'"  # check for blank file
    freq: List[float] = []
    real: List[float] = []
    imag: List[float] = []
    parse_data = False
    while lines:
        line = lines.pop(0).split(',')
        if parse_data:
            if line[0].startswith('</segment'):  # end of this data segment.
                parse_data = False
                continue
            if float(line[9]) != 0:  # found a frequency definition, assuming this is from an EIS sweep.
                try:
                    values: List[float] = list(map(float, line))
                except ValueError:
                    break
                freq.append(values[9])
                real.append(values[14])
                imag.append(values[15]) 
        if line[0].split('=')[0] == 'definition':  # Begin of data
            parse_data = True

    assert len(freq) == len(real) == len(imag) > 0, f"Failed to find any impedance data in '{path}'"
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
