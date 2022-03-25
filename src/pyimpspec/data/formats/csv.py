# Copyright 2022 pyimpspec developers
# pyimpspec is licensed under the GPLv3 or later (https://www.gnu.org/licenses/gpl-3.0.html).
# The licenses of pyimpspec's dependencies and/or sources of portions of code are included in
# the LICENSES folder.

from os.path import exists
from pyimpspec.data.dataset import DataSet, dataframe_to_dataset
from pandas import read_csv, DataFrame


def parse_csv(path: str, **kwargs) -> DataSet:
    """
    Parse a file containing data as character-separated values.
    """
    assert type(path) is str and exists(path)
    df: DataFrame = read_csv(path, engine="python", **kwargs)
    return dataframe_to_dataset(df, path=path)
