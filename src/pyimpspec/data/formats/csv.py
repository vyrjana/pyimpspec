# Copyright 2022 pyimpspec developers
# pyimpspec is licensed under the GPLv3 or later (https://www.gnu.org/licenses/gpl-3.0.html).
# The licenses of pyimpspec's dependencies and/or sources of portions of code are included in
# the LICENSES folder.

from os.path import exists
from typing import Optional
from pyimpspec.data.dataset import DataSet, dataframe_to_dataset
from pandas import read_csv, DataFrame


def parse_csv(
    path: str, separator: Optional[str] = None, decimal_comma: bool = False
) -> DataSet:
    """
    Parse a file containing data as character-separated values.
    """
    assert type(path) is str and exists(path)
    assert type(separator) is str or separator is None
    assert type(decimal_comma) is bool
    df: DataFrame = read_csv(
        path, sep=separator, engine="python", decimal="," if decimal_comma else "."
    )
    return dataframe_to_dataset(df, path=path)
