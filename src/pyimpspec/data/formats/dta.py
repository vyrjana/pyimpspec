# Copyright 2022 pyimpspec developers
# pyimpspec is licensed under the GPLv3 or later (https://www.gnu.org/licenses/gpl-3.0.html).
# The licenses of pyimpspec's dependencies and/or sources of portions of code are included in
# the LICENSES folder.

from os.path import exists
from typing import IO, List
from pyimpspec.data.dataset import DataSet, dataframe_to_dataset, DataFrame


def parse_dta(path: str) -> DataSet:
    assert type(path) is str and exists(path)
    fp: IO
    with open(path, "r", encoding="latin1") as fp:
        lines: List[str] = list(
            filter(lambda _: _ != "", map(str.lower, map(str.strip, fp.readlines())))
        )
    while lines:
        line: str = lines.pop(0)
        if line.startswith("zcurve"):
            break
    assert len(lines) > 0, f"Failed to find any impedance data in '{path}'"
    assert lines.pop(0).startswith("pt")
    assert lines.pop(0).startswith("#")
    freq: List[float] = []
    real: List[float] = []
    imag: List[float] = []
    while lines:
        line = lines.pop(0).replace(",", ".")
        # Pt	Time	Freq	Zreal	Zimag	Zsig	Zmod	Zphz	Idc	Vdc	IERange
        try:
            values: List[float] = list(map(float, line.split()))
        except Exception:
            break
        if len(values) != 11:
            break
        freq.append(values[2])
        real.append(values[3])
        imag.append(values[4])
    assert len(freq) == len(real) == len(imag) > 0, len(freq)
    return dataframe_to_dataset(
        DataFrame.from_dict(
            {
                "frequency": freq,
                "real": real,
                "imaginary": imag,
            }
        ),
        path,
        "",
    )
