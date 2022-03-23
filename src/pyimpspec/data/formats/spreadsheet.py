# Copyright 2022 pyimpspec developers
# pyimpspec is licensed under the GPLv3 or later (https://www.gnu.org/licenses/gpl-3.0.html).
# The licenses of pyimpspec's dependencies and/or sources of portions of code are included in
# the LICENSES folder.

from os.path import exists
from pyimpspec.data.dataset import DataSet, dataframe_to_dataset
from pandas import read_excel, DataFrame
from typing import Dict, List


def parse_spreadsheet(path: str, sheets: List[str] = []) -> List[DataSet]:
    assert type(path) is str and exists(path)
    assert type(sheets) is list and all(map(lambda _: type(_) is str, sheets))
    dataframes: Dict[str, DataFrame] = {}
    datasets: List[DataSet] = []
    if sheets:
        assert all(map(lambda _: type(_) is str, sheets))
        dataframes = read_excel(path, sheet_name=sheets)
        assert len(dataframes) == len(sheets)
    else:
        dataframes = read_excel(path, sheet_name=None)
    label: str
    df: DataFrame
    for label, df in dataframes.items():
        datasets.append(dataframe_to_dataset(df, path, label))
    assert type(datasets) is list
    assert all(map(lambda _: type(_) is DataSet, datasets))
    return datasets
