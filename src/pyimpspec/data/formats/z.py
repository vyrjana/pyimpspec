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

from io import StringIO
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
from .helpers import _validate_path


def parse_z(path: Union[str, Path], **kwargs) -> List[DataSet]:
    """
    Parse a ZView/ZPlot .z file.

    Parameters
    ----------
    path: Union[str, pathlib.Path]
        The path to the file to process.

    **kwargs
        Keyword arguments are passed forward to `pandas.read_csv`_.

    Returns
    -------
    List[DataSet]
    """
    from pandas import (
        DataFrame,
        read_csv,
    )

    _validate_path(path)

    lines: List[str] = []

    fp: IO
    with open(path, "r", encoding="latin1") as fp:
        line: str
        for line in fp:
            line = line.lower().strip()
            if "freq(hz)" in line:
                lines.append(line)
                break

        if len(lines) != 1:
            raise UnsupportedFileFormat("Could not find the column headers")

        for line in fp:
            line = line.lower().strip()
            if "end comments" in line:
                continue
            else:
                lines.append(line)

        if len(lines) < 0:
            raise UnsupportedFileFormat("Expected at least one row of data")

    csv: str = "\n".join(lines)
    df: DataFrame = read_csv(StringIO(csv), engine="python", **kwargs)

    if len(df.columns) == 1:
        separators: List[str] = [
            "\t",
            " ",
            ";",
            ",",
        ]

        while len(df.columns) == 1:
            kwargs["sep"] = separators.pop(0)
            df = read_csv(StringIO(csv), engine="python", **kwargs)

    return dataframe_to_data_sets(df, path=path)
