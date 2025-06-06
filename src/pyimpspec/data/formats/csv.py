# pyimpspec is licensed under the GPLv3 or later (https://www.gnu.org/licenses/gpl-3.0.html).
# Copyright 2025 pyimpspec developers
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
from pyimpspec.typing.helpers import (
    List,
    Path,
    Union,
)
from .helpers import _validate_path


def parse_csv(path: Union[str, Path], **kwargs) -> List[DataSet]:
    """
    Parse a file containing data as character-separated values.

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

    df: DataFrame
    try:
        df = read_csv(path, engine="python", **kwargs)
    except UnicodeDecodeError:
        kwargs["encoding"] = "latin-1"
        df = read_csv(path, engine="python", **kwargs)

    if len(df.columns) == 1:
        separators: List[str] = [
            "\t",
            " ",
            ";",
            ",",
        ]

        while len(df.columns) == 1:
            kwargs["sep"] = separators.pop(0)
            df = read_csv(path, engine="python", **kwargs)

    return dataframe_to_data_sets(df, path=path)
