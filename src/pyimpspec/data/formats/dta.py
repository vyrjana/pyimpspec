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
from pyimpspec.data.data_set import (
    DataSet,
    dataframe_to_data_sets,
)
from pyimpspec.exceptions import UnsupportedFileFormat
from pyimpspec.typing.helpers import (
    IO,
    List,
    Path,
    Union,
)
from .helpers import (
    _parse_string_as_float,
    _validate_path,
)


def parse_dta(path: Union[str, Path]) -> List[DataSet]:
    """
    Parse a Gamry .dta file containing an impedance spectrum.

    If the file contains drift corrected data, then the returned list will contain the drift corrected spectrum as the first element and the uncorrected spectrum as the second element.

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
        lines: List[str] = list(
            filter(lambda _: _ != "", map(str.lower, map(str.strip, fp.readlines())))
        )

    drift_corrected: bool = False

    while lines:
        line: str = lines.pop(0)
        if line.startswith("driftcor") and "1" in line:
            drift_corrected = True
        elif line.startswith("zcurve"):
            break

    if len(lines) == 0:
        raise UnsupportedFileFormat(f"Failed to find any impedance data in '{path}'")

    line = lines.pop(0)
    if not line.startswith("pt"):
        raise UnsupportedFileFormat(
            f"Expected a line containing column headers: {line}"
        )
    elif drift_corrected and "zrealdrcor" not in line:
        raise UnsupportedFileFormat(
            f"Expected drift corrected data, but headers are incorrect: {line=}"
        )

    line = lines.pop(0)
    if not line.startswith("#"):
        raise UnsupportedFileFormat(
            f"Expected a line containing column units instead of {line=}"
        )

    freq: List[float] = []
    real: List[float] = []
    imag: List[float] = []
    drift_corrected_real: List[float] = []
    drift_corrected_imag: List[float] = []

    while lines:
        line = lines.pop(0).replace(",", ".")
        # Pt    Time    Freq	Zreal	Zimag	Zsig	Zmod	Zphz	Idc	Vdc	IERange
        #       s       Hz      ohm     ohm     V       ohm     °       A   V
        try:
            values: List[float] = list(map(_parse_string_as_float, line.split()))
        except ValueError:
            break

        if drift_corrected:
            if len(values) < 10:
                raise ValueError(f"Expected to parse at least ten values from {line=}")
        else:
            if len(values) < 5:
                raise ValueError(f"Expected to parse at least five values from {line=}")

        freq.append(values[2])
        if drift_corrected:
            drift_corrected_real.append(values[8])
            drift_corrected_imag.append(values[9])

        real.append(values[3])
        imag.append(values[4])

    if not (len(freq) == len(real) == len(imag) > 0):
        raise ValueError(
            "Expected at least one set of frequency, real, and imaginary values"
        )
    elif drift_corrected and not (
        len(freq) == len(drift_corrected_real) == len(drift_corrected_imag)
    ):
        raise ValueError(
            "Expected the set of drift corrected values to match the number of frequencies"
        )

    default_label: str = splitext(basename(path))[0]
    data_sets: List[DataSet] = []
    if drift_corrected:
        data_sets.extend(
            dataframe_to_data_sets(
                DataFrame.from_dict(
                    {
                        "frequency": freq,
                        "real": drift_corrected_real,
                        "imaginary": drift_corrected_imag,
                    }
                ),
                path=path,
                label=default_label + " (drift corrected)",
            )
        )

    data_sets.extend(
        dataframe_to_data_sets(
            DataFrame.from_dict(
                {
                    "frequency": freq,
                    "real": real,
                    "imaginary": imag,
                }
            ),
            path=path,
            label=default_label + (" (uncorrected)" if drift_corrected else ""),
        )
    )

    if drift_corrected and len(data_sets) != 2:
        raise UnsupportedFileFormat(
            f"Expected to parse two spectra: an uncorrected and a corrected spectrum ({len(data_sets)=})"
        )

    return data_sets
