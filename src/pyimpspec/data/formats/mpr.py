# pyimpspec is licensed under the GPLv3 or later (https://www.gnu.org/licenses/gpl-3.0.html).
# Copyright 2025 pyimpspec developers
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
from pyimpspec.exceptions import UnsupportedFileFormat
from pyimpspec.typing.helpers import (
    List,
    Path,
    Union,
)
from .helpers import _validate_path


def parse_mpr(path: Union[str, Path]) -> List[DataSet]:
    """
    Parse a BioLogic EC-Lab .mpr file (the raw binary format) containing one or
    more impedance spectra.

    The .mpr binary format is undocumented and version-dependent, so parsing is
    delegated to the optional `galvani` package. If a spectrum is available as an
    exported .mpt text file, that format (handled by :func:`parse_mpt`) does not
    require any additional dependencies and should be preferred.

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

    try:
        from galvani import BioLogic
    except ImportError:
        raise UnsupportedFileFormat(
            "Parsing BioLogic .mpr files requires the optional 'galvani' package "
            "(pip install galvani). Alternatively, export the data as an .mpt file "
            "from EC-Lab and parse that instead."
        )

    try:
        data = BioLogic.MPRfile(str(path)).data
    except Exception as e:
        raise UnsupportedFileFormat(f"Failed to parse '{path}' as a BioLogic .mpr file: {e}")

    names = data.dtype.names
    if "freq/Hz" not in names:
        raise UnsupportedFileFormat(
            f"'{path}' does not contain impedance data (no 'freq/Hz' column); "
            "only EIS/GEIS .mpr files are supported"
        )

    # EIS rows are the ones with a non-zero frequency (galvanostatic/rest rows
    # in a combined technique file have freq == 0).
    frequency = data["freq/Hz"]
    mask = frequency > 0
    if not mask.any():
        raise UnsupportedFileFormat(f"'{path}' contains no impedance data points")

    # .mpr stores impedance as magnitude and phase; let dataframe_to_data_sets
    # handle the conversion as well as splitting into multiple frequency sweeps.
    df: "DataFrame" = DataFrame.from_dict(
        {
            "frequency": frequency[mask],
            "magnitude": data["|Z|/Ohm"][mask],
            "phase": data["Phase(Z)/deg"][mask],
        }
    )

    return dataframe_to_data_sets(df, path=path, degrees=True)
