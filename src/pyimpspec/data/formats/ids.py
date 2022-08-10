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

from os.path import (
    basename,
    exists,
    splitext,
)
from typing import (
    Dict,
    IO,
    List,
    Tuple,
)
from numpy import (
    integer,
    issubdtype,
)
from pyimpspec.data.data_set import (
    DataSet,
    dataframe_to_dataset,
)
from pandas import DataFrame
from string import printable


def _extract_primary_data(
    num_freq: int, lines: List[str]
) -> Tuple[List[float], List[float], List[float]]:
    assert issubdtype(type(num_freq), integer), num_freq
    assert type(lines) is list and all(map(lambda _: type(_) is str, lines))
    assert len(lines) > num_freq
    re: List[float] = []
    im: List[float] = []
    f: List[float] = []
    while lines:
        line: str = lines.pop(0)
        values: List[str] = line.replace(",", ".").split()
        assert len(values) == 3, values
        re.append(float(values.pop(0)))
        im.append(float(values.pop(0)))
        f.append(float(values.pop(0)))
        if len(re) == num_freq:
            break
    return (
        re,
        im,
        f,
    )


def parse_ids(path: str) -> List[DataSet]:
    """
    Parse an Ivium .ids or .idf file containing one or more impedance spectra.

    Parameters
    ----------
    path: str
        The path to the file to process.

    Returns
    -------
    DataSet
    """
    assert type(path) is str and exists(path)
    default_label: str = splitext(basename(path))[0]
    fp: IO
    with open(path, "r", encoding="latin1") as fp:
        content: str = "".join(map(lambda _: _ if _ in printable else "", fp.read()))
        lines: List[str] = list(
            filter(lambda _: _ != "", map(str.strip, content.split("\n")))
        )
    raw_data_sets: Dict[str, dict] = {}
    while lines:
        while lines:
            line: str = lines.pop(0)
            if line.startswith("Title="):
                break
        if len(lines) == 0:
            break
        label: str = line[6:].strip()
        while lines:
            line = lines.pop(0)
            if "primary_data" in line:
                break
        assert len(lines) > 0, f"Failed to find the primary data in '{path}'"
        assert int(lines.pop(0)) == 3  # Number of columns
        num_freq: int = int(lines.pop(0))
        re: List[float]
        im: List[float]
        f: List[float]
        re, im, f = _extract_primary_data(num_freq, lines)
        assert len(re) == len(im) == len(f) == num_freq
        if label == "":
            label = default_label
        if label in raw_data_sets:
            i: int = 2
            while f"{label} ({i})" in raw_data_sets:
                i += 1
            label = f"{label} ({i})"
        raw_data_sets[label] = {
            "freq": f,
            "real": re,
            "imag": im,
        }
    assert len(raw_data_sets) > 0, raw_data_sets
    assert len(lines) == 0, lines
    data_sets: List[DataSet] = []
    for label, values in raw_data_sets.items():
        data_sets.append(dataframe_to_dataset(DataFrame.from_dict(values), path, label))
    assert len(data_sets) == len(raw_data_sets), (
        len(data_sets),
        len(raw_data_sets),
    )
    return data_sets
