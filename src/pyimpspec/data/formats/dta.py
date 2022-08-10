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


def parse_dta(path: str) -> DataSet:
    """
    Parse a Gamry .dta file containing an impedance spectrum.

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
    while lines:
        line: str = lines.pop(0)
        if line.startswith("zcurve"):
            break
    assert len(lines) > 0, f"Failed to find any impedance data in '{path}'"
    line = lines.pop(0)
    assert line.startswith("pt"), f"Expected a line containing column headers: {line}"
    line = lines.pop(0)
    assert line.startswith("#"), f"Expected a line containing column units: {line}"
    freq: List[float] = []
    real: List[float] = []
    imag: List[float] = []
    while lines:
        line = lines.pop(0).replace(",", ".")
        # Pt    Time    Freq	Zreal	Zimag	Zsig	Zmod	Zphz	Idc	Vdc	IERange
        #       s       Hz      ohm     ohm     V       ohm     °       A   V
        try:
            values: List[float] = list(map(float, line.split()))
        except Exception:
            break
        assert (
            len(values) >= 5
        ), f"Expected to parse at least five values from line: {line}"
        freq.append(values[2])
        real.append(values[3])
        imag.append(values[4])
    assert len(freq) == len(real) == len(imag) > 0, len(freq)
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
