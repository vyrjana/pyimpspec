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
from pyimpspec.typing.helpers import (
    List,
    Path,
    Union,
)
from .helpers import _validate_path


def parse_spreadsheet(path: Union[str, Path], **kwargs) -> List[DataSet]:
    """
    Parse a spreadsheet (.xlsx or .ods) containing one or more impedance spectra.

    Parameters
    ----------
    path: Union[str, pathlib.Path]
        The path to the file to process.

    kwargs:
        Keyword arguments are passed forward to `pandas.read_excel`_.

    Returns
    -------
    List[DataSet]
    """
    from pandas import (
        DataFrame,
        read_excel,
    )

    _validate_path(path)

    data_sets: List[DataSet] = []

    if "sheet_name" not in kwargs:
        kwargs["sheet_name"] = None

    label: str
    df: DataFrame
    for label, df in read_excel(path, **kwargs).items():
        if len(df.columns) == 0:
            continue

        data_sets.extend(dataframe_to_data_sets(df, path=path, label=label))

    return data_sets
