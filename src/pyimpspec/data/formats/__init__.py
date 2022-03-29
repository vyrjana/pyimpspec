# Copyright 2022 pyimpspec developers
# pyimpspec is licensed under the GPLv3 or later (https://www.gnu.org/licenses/gpl-3.0.html).
# The licenses of pyimpspec's dependencies and/or sources of portions of code are included in
# the LICENSES folder.

from typing import Callable, List, Optional, Union
from pyimpspec.data.dataset import DataSet
from os.path import exists, splitext, basename
from .csv import parse_csv
from .spreadsheet import parse_spreadsheet
from .ids import parse_ids
from .dta import parse_dta


class UnsupportedFileFormat(Exception):
    pass


def is_spreadsheet(path: str = "", extension: str = "") -> bool:
    assert type(path) is str
    assert type(extension) is str
    assert path != "" or extension != ""
    if path == "":
        _, extension = splitext(path)
    return extension in [
        ".xls",
        ".xlsx",
        ".ods",
    ]


def parse_data(
    path: str,
    file_format: Optional[str] = None,
    **kwargs,
) -> List[DataSet]:
    """
    Parse experimental data and return a list of DataSet instances. One or more specific sheets
    can be specified by name when parsing spreadsheets (e.g. .xlsx or .ods) to only return
    DataSet instances for those sheets. If no sheets are specified, then all sheets will be
    processed and the data from successfully parsed sheets will be returned as DataSet instances.

    Parameters
    ----------
    path: str
        The path to a file containing experimental data that is to be parsed.
    file_format: Optional[str] = None
        The file format (or extension) that should be assumed when parsing the data. If no file
        format is specified, then the file format will be determined based on the file extension.
        If there is no file extension, then attempts will be made to parse the file as if it was
        one of the supported file formats.
    kwargs
        Keyword arguments are passed to the parser.

    Returns
    -------
    List[DataSet]
    """
    assert type(path) is str and exists(path)
    assert type(file_format) is str or file_format is None
    if file_format is not None:
        file_format = file_format.lower()
        if not file_format.startswith("."):
            file_format = f".{file_format}"
    datasets: List[DataSet] = []
    extension: str
    _, extension = splitext(basename(path))
    extension = extension.lower()
    data: Union[DataSet, List[DataSet]]
    if file_format or extension:
        if is_spreadsheet(extension=(file_format or extension)):
            datasets.extend(parse_spreadsheet(path, **kwargs))
        else:
            fmt: str = file_format or extension
            func: Optional[Callable] = {  # type: ignore
                ".csv": parse_csv,
                ".txt": parse_csv,
                ".ids": parse_ids,
                ".idf": parse_ids,
                ".xls": parse_spreadsheet,
                ".xlsx": parse_spreadsheet,
                ".ods": parse_spreadsheet,
                ".dta": parse_dta,
            }.get(fmt)
            if func is None:
                raise UnsupportedFileFormat(f"Unsupported file format: {fmt}")
            if fmt == ".csv":
                try:
                    data = func(path, **kwargs)
                except AssertionError:
                    data = func(path, sep=None, decimal=",", **kwargs)
            else:
                data = func(path, **kwargs)
            if type(data) is list:
                datasets.extend(data)
            else:
                datasets.append(data)  # type: ignore
    else:
        parsers: List[Callable] = [
            parse_csv,
            lambda _, **kwargs: parse_csv(_, sep=None, decimal=",", **kwargs),
            parse_ids,
            parse_spreadsheet,
            parse_dta,
        ]
        parsed_data: bool = False
        for parser in parsers:
            try:
                result = parser(path, **kwargs)
                if type(result) is list:
                    datasets.extend(result)
                else:
                    datasets.append(result)
                parsed_data = True
                break
            except Exception:
                pass
        if not parsed_data:
            raise UnsupportedFileFormat(f"Unknown/malformed file format: {path}")
    assert type(datasets) is list
    assert len(datasets) > 0
    assert all(map(lambda _: type(_) is DataSet, datasets))
    return datasets
