# Copyright 2022 pyimpspec developers
# pyimpspec is licensed under the GPLv3 or later (https://www.gnu.org/licenses/gpl-3.0.html).
# The licenses of pyimpspec's dependencies and/or sources of portions of code are included in
# the LICENSES folder.

from os.path import exists
from typing import Dict, IO, List, Optional, Tuple
from pyimpspec.data.dataset import DataSet, dataframe_to_dataset
from pandas import DataFrame


def find_value(key: str, lines: List[str]) -> Optional[str]:
    key += "="
    while lines:
        line: str = lines.pop(0)
        if not line.startswith(key):
            continue
        return line[len(key) :]
    return None


def _extract_primary_data(
    num_freq: int, lines: List[str]
) -> Tuple[List[float], List[float], List[float]]:
    assert len(lines) > num_freq
    re: List[float] = []
    im: List[float] = []
    f: List[float] = []
    while lines:
        line: str = lines.pop(0)
        values: List[str] = line.replace(",", ".").split()
        assert len(values) == 3
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


def _parse_measurement(lines: List[str]) -> Tuple[str, dict]:
    if find_value("starttime", lines) == "":
        return ("", {})
    if find_value("Method", lines) != "Impedance":
        return ("", {})
    # "Constant E" for potentiostatic EIS is the only one that has been tested
    find_value("Technique", lines)
    label: str = find_value("Title", lines) or ""
    assert len(lines) > 0
    while lines:
        line: str = lines.pop(0)
        if line.startswith("primary_data"):
            lines.pop(0)
            break
    num_freq: int = int(lines.pop(0))
    re: List[float]
    im: List[float]
    f: List[float]
    re, im, f = _extract_primary_data(num_freq, lines)
    return (
        label,
        {
            "freq": f,
            "real": re,
            "imag": im,
        },
    )


def parse_ids(path: str) -> List[DataSet]:
    assert type(path) is str and exists(path)
    fp: IO
    with open(path, "r", encoding="latin1") as fp:
        lines: List[str] = list(
            filter(lambda _: _ != "", map(str.strip, fp.read().split("\n")))
        )
    raw_datasets: Dict[str, dict] = {}
    key: str = "QR="
    while lines:
        line: str = lines.pop(0)
        if not line.startswith(key):
            continue
        if len(line) == len(key):
            continue
        lines.insert(0, line)
        label: str
        values: dict
        label, values = _parse_measurement(lines)
        if label == "" and len(values) == 0:
            continue
        if label in raw_datasets:
            i: int = 2
            while f"{label} ({i})" in raw_datasets:
                i += 1
            label = f"{label} ({i})"
        raw_datasets[label] = values
    assert len(raw_datasets) > 0
    assert len(lines) == 0
    datasets: List[DataSet] = []
    for label, values in raw_datasets.items():
        datasets.append(dataframe_to_dataset(DataFrame.from_dict(values), path, label))
    assert len(datasets) == len(raw_datasets)
    return datasets
