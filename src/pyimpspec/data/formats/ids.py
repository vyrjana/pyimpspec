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

from os.path import (
    basename,
    splitext,
)
from string import printable
from pyimpspec.data.data_set import (
    DataSet,
    dataframe_to_data_sets,
)
from pyimpspec.exceptions import UnsupportedFileFormat
from pyimpspec.typing.helpers import (
    Dict,
    IO,
    List,
    Path,
    Tuple,
    Union,
    _is_integer,
)
from .helpers import (
    _parse_string_as_float,
    _validate_path,
)


def _extract_primary_data(
    num_freq: int,
    lines: List[str],
) -> Tuple[List[float], List[float], List[float]]:
    if not _is_integer(num_freq):
        raise TypeError(f"Expected an integer instead of {num_freq=}")

    if not (isinstance(lines, list) and all(map(lambda _: type(_) is str, lines))):
        raise TypeError(f"Expected a list of strings instead of {lines=}")

    if not (len(lines) > num_freq):
        raise ValueError(f"Expected {len(lines)=} > {num_freq=}")

    re: List[float] = []
    im: List[float] = []
    f: List[float] = []

    while lines:
        line: str = lines.pop(0)
        values: List[str] = line.split()

        if len(values) != 3:
            raise UnsupportedFileFormat(
                f"Expected to parse three values instead of {values=}"
            )

        re.append(_parse_string_as_float(values.pop(0)))
        im.append(_parse_string_as_float(values.pop(0)))
        f.append(_parse_string_as_float(values.pop(0)))

        if len(re) == num_freq:
            break

    return (
        re,
        im,
        f,
    )


def parse_ids(path: Union[str, Path]) -> List[DataSet]:
    """
    Parse an Ivium .ids or .idf file containing one or more impedance spectra.

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

        if len(lines) == 0:
            raise UnsupportedFileFormat(f"Failed to find the primary data in '{path}'")

        num_columns: int = int(lines.pop(0))
        if num_columns != 3:
            raise UnsupportedFileFormat("Expected to find three columns")

        num_freq: int = int(lines.pop(0))

        re: List[float]
        im: List[float]
        f: List[float]
        re, im, f = _extract_primary_data(num_freq, lines)

        if not (len(re) == len(im) == len(f) == num_freq):
            raise UnsupportedFileFormat(
                f"Expected to parse {num_freq=} points instead of {len(re)}"
            )

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

    if len(raw_data_sets) < 1:
        raise UnsupportedFileFormat(
            f"Expected at least one data set instead of {len(raw_data_sets)}"
        )
    elif len(lines) > 0:
        raise UnsupportedFileFormat(
            f"Expected to consume all lines but there are {len(lines)} lines remaining"
        )

    data_sets: List[DataSet] = []

    for label, values in raw_data_sets.items():
        data_sets.extend(
            dataframe_to_data_sets(
                DataFrame.from_dict(values),
                path=path,
                label=label,
            )
        )

    if len(data_sets) != len(raw_data_sets):
        raise UnsupportedFileFormat(
            f"Expected to generate {len(raw_data_sets)} DataSet instances instead of {len(data_sets)}"
        )

    return data_sets
