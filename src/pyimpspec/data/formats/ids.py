# Copyright 2022 pyimpspec developers
# pyimpspec is licensed under the GPLv3 or later (https://www.gnu.org/licenses/gpl-3.0.html).
# The licenses of pyimpspec's dependencies and/or sources of portions of code are included in
# the LICENSES folder.

from os.path import basename, exists, splitext
from typing import Dict, IO, List, Tuple
from pyimpspec.data.dataset import DataSet, dataframe_to_dataset
from pandas import DataFrame
from string import printable


def _extract_primary_data(
    num_freq: int, lines: List[str]
) -> Tuple[List[float], List[float], List[float]]:
    assert type(num_freq) is int
    assert type(lines) is list and all(map(lambda _: type(_) is str, lines))
    assert len(lines) > num_freq
    re: List[float] = []
    im: List[float] = []
    f: List[float] = []
    while lines:
        line: str = lines.pop(0)
        values: List[str] = line.replace(",", ".").split()
        assert len(values) == 3, values
        re.append(float(values.pop(0)))
        im.append(float(values.pop(0)))
        f.append(float(values.pop(0)))
        if len(re) == num_freq:
            break
    return (
        re,
        im,
        f,
    )


def parse_ids(path: str) -> List[DataSet]:
    assert type(path) is str and exists(path)
    default_label: str = splitext(basename(path))[0]
    fp: IO
    with open(path, "r", encoding="latin1") as fp:
        content: str = "".join(map(lambda _: _ if _ in printable else "", fp.read()))
        lines: List[str] = list(
            filter(lambda _: _ != "", map(str.strip, content.split("\n")))
        )
    raw_datasets: Dict[str, dict] = {}
    while lines:
        while lines:
            line: str = lines.pop(0)
            if line.strip().startswith("Title="):
                break
        if len(lines) == 0:
            break
        label: str = line[6:].strip()
        while lines:
            line = lines.pop(0)
            if "primary_data" in line:
                break
        assert len(lines) > 0, f"Failed to find the primary data in '{path}'"
        assert int(lines.pop(0)) == 3  # Number of columns
        num_freq: int = int(lines.pop(0))
        re: List[float]
        im: List[float]
        f: List[float]
        re, im, f = _extract_primary_data(num_freq, lines)
        assert len(re) == len(im) == len(f) == num_freq
        if label == "":
            label = default_label
        if label in raw_datasets:
            i: int = 2
            while f"{label} ({i})" in raw_datasets:
                i += 1
            label = f"{label} ({i})"
        raw_datasets[label] = {
            "freq": f,
            "real": re,
            "imag": im,
        }
    assert len(raw_datasets) > 0
    assert len(lines) == 0
    datasets: List[DataSet] = []
    for label, values in raw_datasets.items():
        datasets.append(dataframe_to_dataset(DataFrame.from_dict(values), path, label))
    assert len(datasets) == len(raw_datasets)
    return datasets
