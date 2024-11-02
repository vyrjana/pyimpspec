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
from pyimpspec.exceptions import UnsupportedFileFormat
from pyimpspec.typing.helpers import (
    Callable,
    Dict,
    List,
    Optional,
    Path,
    Union,
)
from .data_set import (
    DataSet,
    dataframe_to_data_sets,
)
from .formats import (
    parse_csv,
    parse_dfr,
    parse_dta,
    parse_i2b,
    parse_ids,
    parse_mpt,
    parse_p00,
    parse_spreadsheet,
    parse_z,
    parse_pssession,
)
from .formats.helpers import _validate_path


def get_parsers() -> Dict[str, Callable]:
    """
    Get a mapping of file extensions to their corresponding parser functions.

    Returns
    -------
    Dict[str, Callable]
    """
    return {
        ".P00": parse_p00,
        ".dfr": parse_dfr,
        ".dta": parse_dta,
        ".i2b": parse_i2b,
        ".idf": parse_ids,
        ".ids": parse_ids,
        ".mpt": parse_mpt,
        ".z": parse_z,
        ".pssession": parse_pssession,
        ".ods": parse_spreadsheet,
        ".xlsx": parse_spreadsheet,
        ".txt": parse_csv,
        ".csv": parse_csv,
    }


def _is_spreadsheet(path: str = "", extension: str = "") -> bool:
    if path != "":
        _, extension = splitext(path)

    return extension in [
        ".xlsx",
        ".ods",
    ]


def _brute_force(path: Union[str, Path], **kwargs) -> List[DataSet]:
    data_sets: List[DataSet] = []
    parsers: List[Callable] = list(set(get_parsers().values()))
    parsers.append(lambda _, **k: parse_csv(_, sep=None, decimal=",", **k))
    parsed_data: bool = False

    parser: Callable
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

    return data_sets


def parse_data(
    path: Union[str, Path],
    file_format: str = "",
    **kwargs,
) -> List[DataSet]:
    """
    Parse experimental data and return a list of DataSet instances.
    One or more specific sheets can be specified by name when parsing spreadsheets (e.g., .xlsx or .ods) to only return DataSet instances for those sheets.
    If no sheets are specified, then all sheets will be processed and the data from successfully parsed sheets will be returned as DataSet instances.

    Parameters
    ----------
    path: Union[str, pathlib.Path]
        The path to a file containing experimental data that is to be parsed.

    file_format: str, optional
        The file format (or extension) that should be assumed when parsing the data.
        If no file format is specified, then the file format will be determined based on the file extension.
        If there is no file extension, then attempts will be made to parse the file as if it was one of the supported file formats.

    **kwargs
        Keyword arguments are passed to the parser.

    Returns
    -------
    List[|DataSet|]
    """
    _validate_path(path)
    if not (isinstance(file_format, str) or file_format is None):
        raise TypeError(f"Expected a string or None instead of {file_format=}")
    elif file_format != "":
        file_format = file_format.lower()
        if not file_format.startswith("."):
            file_format = f".{file_format}"

    data_sets: List[DataSet] = []
    extension: str = splitext(basename(path))[1].lower()

    data: Union[DataSet, List[DataSet]]
    if file_format or extension:
        if _is_spreadsheet(extension=(file_format or extension)):
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
                except UnsupportedFileFormat:
                    data = func(path, sep=None, decimal=",", **kwargs)
            else:
                try:
                    data = func(path, **kwargs)
                except UnsupportedFileFormat:
                    data = _brute_force(path, **kwargs)

            if type(data) is list:
                data_sets.extend(data)
            else:
                data_sets.append(data)  # type: ignore
    else:
        data_sets.extend(_brute_force(path, **kwargs))

    if not isinstance(data_sets, list):
        raise TypeError(f"Expected a list instead of {data_sets=}")
    elif len(data_sets) == 0:
        raise ValueError("Expected a list with at least one element")
    elif not all(map(lambda _: isinstance(_, DataSet), data_sets)):
        raise TypeError(f"Expected a list of DataSet instead of {data_sets=}")
    elif not all(map(lambda _: _.get_label().strip() != "", data_sets)):
        raise ValueError(
            "Expected all DataSet instances to have a label that is not an empty string"
        )

    return data_sets
