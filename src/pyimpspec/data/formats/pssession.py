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

from json import loads
from pyimpspec.data.data_set import (
    DataSet,
    dataframe_to_data_sets,
)
from pyimpspec.exceptions import UnsupportedFileFormat
from pyimpspec.typing.helpers import (
    Dict,
    IO,
    List,
    Path,
    Union,
)
from .helpers import _validate_path


def parse_pssession(path: Union[str, Path]) -> List[DataSet]:
    """
    Parse a PalmSens .pssession file.

    Parameters
    ----------
    path: Union[str, pathlib.Path]
        The path to the file to process.

    Returns
    -------
    List[DataSet]
    """
    from pandas import DataFrame

    _validate_path(path)

    fp: IO
    with open(path, "r", encoding="latin1") as fp:
        contents: str = fp.read().replace("\x00", "")

    i: int = contents.find("{")
    j: int = contents.rfind("}") + 1
    json: dict = loads(contents[i:j])
    if "Measurements" not in json:
        raise UnsupportedFileFormat(
            "Expected to find a 'Measurements' key in the file"
        )

    dataframes: Dict[str, DataFrame] = {}
    measurement: dict
    for measurement in json["Measurements"]:
        if "DataSet" not in measurement:
            raise UnsupportedFileFormat(
                "Expected to find a 'DataSet' key in 'Measurements'"
            )
        elif "Type" not in measurement["DataSet"]:
            raise UnsupportedFileFormat("Expected to find a 'Type' key in 'DataSet'")
        elif "EIS" not in measurement["DataSet"]["Type"]:
            continue

        if "Values" not in measurement["DataSet"]:
            raise UnsupportedFileFormat("Expected to find 'Values' key in 'DataSet'")

        freq: List[float] = []
        real: List[float] = []
        imag: List[float] = []

        for values in measurement["DataSet"]["Values"]:
            if "DataValues" not in values:
                raise UnsupportedFileFormat(
                    "Expected to find 'DataValues' key in 'Values'"
                )
            elif "Unit" not in values:
                continue
            elif "Type" not in values["Unit"]:
                raise UnsupportedFileFormat("Expected to find 'Type' key in 'Unit'")

            if "Hertz" in values["Unit"]["Type"]:
                if len(freq) > 0:
                    raise UnsupportedFileFormat(
                        "Expected to not have parsed any frequencies yet"
                    )
                freq.extend([v for d in values["DataValues"] for v in d.values()])

            elif "ZRe" in values["Unit"]["Type"]:
                if len(real) > 0:
                    raise UnsupportedFileFormat(
                        "Expected to not have parsed any real values yet"
                    )
                real.extend([v for d in values["DataValues"] for v in d.values()])

            elif "ZIm" in values["Unit"]["Type"]:
                if len(imag) > 0:
                    raise UnsupportedFileFormat(
                        "Expected to not have parsed any imaginary values yet"
                    )
                imag.extend([-v for d in values["DataValues"] for v in d.values()])

            if len(freq) == len(real) == len(imag) > 0:
                break

        if not (len(freq) == len(real) == len(imag) > 0):
            raise ValueError(
                "Expected at least one set of frequency, real, and imaginary values"
            )

        label: str = measurement["Title"]
        i = 1
        while label in dataframes:
            label = f"{measurement['Title']} ({i})"
            i += 1

        dataframes[label] = DataFrame.from_dict(
            {
                "frequency": freq,
                "real": real,
                "imaginary": imag,
            }
        )

    data_sets: List[DataSet] = []
    for label, df in dataframes.items():
        data_sets.extend(
            dataframe_to_data_sets(
                df,
                path=path,
                label=label,
            )
        )

    return data_sets
