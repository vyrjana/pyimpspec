# Copyright 2022 pyimpspec developers
# pyimpspec is licensed under the GPLv3 or later (https://www.gnu.org/licenses/gpl-3.0.html).
# The licenses of pyimpspec's dependencies and/or sources of portions of code are included in
# the LICENSES folder.

from os.path import exists
from pyimpspec.data.dataset import DataSet, dataframe_to_dataset
from pandas import read_excel, DataFrame
from typing import List


def parse_spreadsheet(path: str, **kwargs) -> List[DataSet]:
    assert type(path) is str and exists(path)
    datasets: List[DataSet] = []
    if "sheet_name" not in kwargs:
        kwargs["sheet_name"] = None
    label: str
    df: DataFrame
    for label, df in read_excel(path, **kwargs).items():
        datasets.append(dataframe_to_dataset(df, path, label))
    assert type(datasets) is list
    assert all(map(lambda _: type(_) is DataSet, datasets))
    return datasets
