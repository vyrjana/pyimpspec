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

from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Union,
)
from pyimpspec.data.data_set import DataSet
from os.path import (
    basename,
    exists,
    splitext,
)
from .csv import parse_csv
from .dfr import parse_dfr
from .dta import parse_dta
from .i2b import parse_i2b
from .ids import parse_ids
from .mpt import parse_mpt
from .p00 import parse_p00
from .spreadsheet import parse_spreadsheet


class UnsupportedFileFormat(Exception):
    pass


def get_parsers() -> Dict[str, Callable]:
    return {
        ".P00": parse_p00,
        ".csv": parse_csv,
        ".dfr": parse_dfr,
        ".dta": parse_dta,
        ".i2b": parse_i2b,
        ".idf": parse_ids,
        ".ids": parse_ids,
        ".mpt": parse_mpt,
        ".ods": parse_spreadsheet,
        ".txt": parse_csv,
        ".xlsx": parse_spreadsheet,
    }


def is_spreadsheet(path: str = "", extension: str = "") -> bool:
    assert type(path) is str, path
    assert type(extension) is str, extension
    assert path != "" or extension != "", (
        path,
        extension,
    )
    if path == "":
        _, extension = splitext(path)
    return extension in [
        ".xlsx",
        ".ods",
    ]


def parse_data(
    path: str,
    file_format: Optional[str] = None,
    **kwargs,
) -> List[DataSet]:
    """
    Parse experimental data and return a list of DataSet instances.
    One or more specific sheets can be specified by name when parsing spreadsheets (e.g., .xlsx or .ods) to only return DataSet instances for those sheets.
    If no sheets are specified, then all sheets will be processed and the data from successfully parsed sheets will be returned as DataSet instances.

    Parameters
    ----------
    path: str
        The path to a file containing experimental data that is to be parsed.

    file_format: Optional[str] = None
        The file format (or extension) that should be assumed when parsing the data.
        If no file format is specified, then the file format will be determined based on the file extension.
        If there is no file extension, then attempts will be made to parse the file as if it was one of the supported file formats.

    kwargs:
        Keyword arguments are passed to the parser.

    Returns
    -------
    List[DataSet]
    """
    assert type(path) is str and exists(path), path
    assert type(file_format) is str or file_format is None, file_format
    if file_format is not None:
        file_format = file_format.lower()
        if not file_format.startswith("."):
            file_format = f".{file_format}"
    data_sets: List[DataSet] = []
    extension: str
    _, extension = splitext(basename(path))
    extension = extension.lower()
    data: Union[DataSet, List[DataSet]]
    if file_format or extension:
        if is_spreadsheet(extension=(file_format or extension)):
            data_sets.extend(parse_spreadsheet(path, **kwargs))
        else:
            fmt: str = file_format or extension
            func: Optional[Callable] = get_parsers().get(fmt)
            if func is None:
                func = {k.lower(): v for k, v in get_parsers().items()}.get(fmt)
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
                data_sets.extend(data)
            else:
                data_sets.append(data)  # type: ignore
    else:
        parsers: List[Callable] = [
            lambda _, **kwargs: parse_csv(_, sep=None, decimal=",", **kwargs),
        ] + list(set(get_parsers().values()))
        parsed_data: bool = False
        for parser in parsers:
            try:
                result = parser(path, **kwargs)
                if type(result) is list:
                    data_sets.extend(result)
                else:
                    data_sets.append(result)
                parsed_data = True
                break
            except Exception:
                pass
        if not parsed_data:
            raise UnsupportedFileFormat(f"Unknown/malformed file format: {path}")
    assert type(data_sets) is list, data_sets
    assert len(data_sets) > 0, data_sets
    assert all(map(lambda _: isinstance(_, DataSet), data_sets)), data_sets
    return data_sets
