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

import cmath
from collections import OrderedDict
from math import pi
from os.path import (
    basename,
    splitext,
)
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)
from uuid import uuid4
from numpy import (
    allclose,
    angle,
    array,
    flip,
    integer,
    issubdtype,
    mean,
    ndarray,
    radians,
    bool_,
)
from pandas import DataFrame


VERSION: int = 1


def _parse_v1(dictionary: dict) -> dict:
    assert type(dictionary) is dict
    assert "frequency" in dictionary
    assert "real" in dictionary
    assert "imaginary" in dictionary
    return {
        "frequency": array(dictionary["frequency"]),
        "impedance": array(
            list(
                map(
                    lambda _: complex(*_),
                    zip(dictionary["real"], dictionary["imaginary"]),
                )
            )
        ),
        "mask": {int(k): v for k, v in dictionary.get("mask", {}).items()},
        "path": dictionary.get("path", ""),
        "label": dictionary.get("label", ""),
        "uuid": dictionary.get("uuid", ""),
    }


class DataSet:
    """
    A class that represents an impedance spectrum.
    The data points can be masked, which results in those data points being omitted from any analyses and visualization.

    Parameters
    ----------
    frequency: ndarray
        A 1-dimensional array of frequencies in hertz.

    impedance: ndarray
        A 1-dimensional array of complex impedances in ohms.

    mask: Dict[int, bool] = {}
        A mapping of integer indices to boolean values where a value of True means that the data point is to be omitted.

    path: str = ""
        The path to the file that has been parsed to generate this DataSet instance.

    label: str = ""
        The label assigned to this DataSet instance.

    uuid: str = ""
        The universally unique identifier assigned to this DataSet instance.
        If empty, then one will be automatically assigned.
    """

    def __init__(
        self,
        frequency: ndarray,
        impedance: ndarray,
        mask: Dict[int, bool] = {},
        path: str = "",
        label: str = "",
        uuid: str = "",
    ):
        assert type(frequency) is ndarray, type(frequency)
        assert type(impedance) is ndarray, type(impedance)
        assert frequency.shape == impedance.shape, (
            frequency.shape,
            impedance.shape,
        )
        assert (
            len(frequency) == len(impedance) > 0
        ), f"{path=} ({label=}): {len(frequency)=}, {len(impedance)=}"
        assert type(mask) is dict
        assert all(
            map(
                lambda _: issubdtype(type(_[0]), integer) and type(_[1]) is bool,
                mask.items(),
            )
        ), mask
        assert type(path) is str
        assert type(label) is str
        assert type(uuid) is str
        # Sort data points in descending order of frequency.
        if frequency[-1] > frequency[0]:
            frequency = flip(frequency)
            impedance = flip(impedance)
            if len(mask) > 0:
                i: int
                for i in range(0, frequency.size):
                    j: int = frequency.size - 1 - i
                    flag: bool = mask.get(i, False)
                    mask[i] = mask.get(j, False)
                    mask[j] = flag
        self.uuid: str = uuid or uuid4().hex
        self._path: str = path
        self._label: str = label or splitext(basename(path))[0]
        self._frequency: ndarray = frequency
        self._impedance: ndarray = impedance
        self._mask: Dict[int, bool] = mask or {
            i: False for i in range(0, len(frequency))
        }
        self._num_points: int = len(frequency)
        self.set_mask(mask)

    def __repr__(self) -> str:
        return f"DataSet ({self._label}, {hex(id(self))})"

    def subtract_impedance(self, Z: Union[complex, List[complex], ndarray]):
        """
        Subtract either the same complex value from all data points or a unique complex value for each data point in this DataSet.

        Parameters
        ----------
        Z: Union[complex, List[complex], ndarray]
            The complex value(s) to subtract from this DataSet's impedances.
        """
        if type(Z) is list or type(Z) is ndarray:
            assert all(map(lambda _: hasattr(_, "real") and hasattr(_, "imag"), Z)), (
                type(Z),
                type(Z[0]),
                Z[0],
            )
        else:
            assert hasattr(Z, "real") and hasattr(Z, "imag"), (
                type(Z),
                Z,
            )
        self._impedance = self._impedance - Z

    @staticmethod
    def _parse(dictionary: dict) -> dict:
        assert type(dictionary) is dict
        parsers: Dict[int, Callable] = {
            1: _parse_v1,
        }
        version: int = dictionary.get("version", VERSION)
        assert version <= VERSION, f"Unsupported version: {version=} > {VERSION=}"
        assert (
            version in parsers
        ), f"Unsupported version: {version=} not in {parsers.keys()=}"
        return parsers[version](dictionary)

    @classmethod
    def from_dict(Class, dictionary: dict) -> "DataSet":
        """
        Create a DataSet from a dictionary.

        Parameters
        ----------
        dictionary: dict
            A dictionary containing at least the frequencies, and the real and the imaginary parts of the impedances.

        Returns
        -------
        DataSet
        """
        return Class(**Class._parse(dictionary))

    @classmethod
    def copy(Class, data: "DataSet", label: Optional[str] = None) -> "DataSet":
        """
        Create a copy of an existing DataSet.

        Parameters
        ----------
        data: DataSet
            The existing DataSet to make a copy of.

        label: Optional[str] = None
            The label that the copy should have.

        Returns
        -------
        DataSet
        """
        assert type(data) is Class
        assert type(label) is str or label is None
        dictionary: dict = data.to_dict()
        if label is not None:
            dictionary["label"] = label
        del dictionary["uuid"]
        return Class.from_dict(dictionary)

    @classmethod
    def average(Class, data_sets: List["DataSet"], label: str = "Average") -> "DataSet":
        """
        Create a DataSet by averaging the impedances of multiple DataSet instances.

        Parameters
        ----------
        data_sets: List[DataSet]
            The DataSet instances to average.

        label: str = "Average"
            The label that the new DataSet should have.

        Returns
        -------
        DataSet
        """
        assert type(data_sets) is list and all(
            map(lambda _: type(_) is Class, data_sets)
        )
        assert type(label) is str
        freqs: List[ndarray] = list(
            map(lambda _: _.get_frequency(masked=None), data_sets)
        )
        imps: List[ndarray] = list(
            map(lambda _: _.get_impedance(masked=None), data_sets)
        )
        f: ndarray = freqs.pop(0)
        assert all(map(lambda _: allclose(f, _), freqs))
        Z: ndarray = mean(array(imps), axis=0)
        return Class.from_dict(
            {
                "frequency": f,
                "real": Z.real,
                "imaginary": Z.imag,
                "label": label,
            }
        )

    def get_path(self) -> str:
        """
        Get the path to the file that was parsed to generate this DataSet.

        Returns
        -------
        str
        """
        return self._path

    def set_path(self, path: str):
        """
        Set the path to the file that was parsed to generate this DataSet.

        Parameters
        ----------
        path: str
            The path.
        """
        assert type(path) is str
        self._path = path

    def get_label(self) -> str:
        """
        Get the label assigned to this DataSet.

        Returns
        -------
        str
        """
        return self._label

    def set_label(self, label: str):
        """
        Set the label assigned to this DataSet.

        Parameters
        ----------
        label: str
            The new label.
        """
        assert type(label) is str
        self._label = label

    def set_mask(self, mask: Dict[int, bool]):
        """
        Set the mask for this DataSet.

        Parameters
        ----------
        mask: Dict[int, bool]
            The new mask.
            The keys must be zero-based indices and the values must be boolean values.
            True means that the data point is to be omitted and False means that the data point is to be included.
        """
        assert (
            type(mask) is dict
            and all(map(lambda _: issubdtype(type(_), integer), mask.keys()))
            and all(map(lambda _: issubdtype(type(_), bool_), mask.values()))
        ), mask
        mask = mask.copy()
        i: int
        flag: bool
        for i, flag in mask.items():
            assert issubdtype(type(i), integer), type(i)
            assert issubdtype(type(flag), bool_), type(flag)
        for i in list(mask.keys()):
            if i < 0 or i >= self._num_points:
                del mask[i]
        self._mask = mask.copy()
        for i in range(0, self._num_points):
            if i not in self._mask:
                self._mask[i] = False

    def get_mask(self) -> Dict[int, bool]:
        """
        Get the mask for this DataSet.
        The keys are zero-based indices and the values are booleans.
        True means that the data point is to be omitted and False means that the data point is to be included.

        Returns
        -------
        Dict[int, bool]
        """
        return self._mask.copy()

    def get_frequency(self, masked: Optional[bool] = False) -> ndarray:
        """
        Get the frequencies in this DataSet.

        Parameters
        ----------
        masked: Optional[bool] = False
            None means that all frequencies are returned.
            True means that only frequencies that are to be omitted are returned.
            False means that only frequencies that are to be included are returned.

        Returns
        -------
        ndarray
        """
        assert type(masked) is bool or masked is None
        if masked is None:
            return array(self._frequency)
        i: int
        f: float
        return array(
            [
                f
                for i, f in enumerate(self._frequency)
                if self._mask.get(i, False) == masked
            ]
        )

    def get_impedance(self, masked: Optional[bool] = False) -> ndarray:
        """
        Get the complex impedances in this DataSet.

        Parameters
        ----------
        masked: Optional[bool] = False
            None means that all impedances are returned.
            True means that only impedances that are to be omitted are returned.
            False means that only impedances that are to be included are returned.

        Returns
        -------
        ndarray
        """
        assert type(masked) is bool or masked is None
        if masked is None:
            return self._impedance
        i: int
        c: complex
        return array(
            [
                c
                for i, c in enumerate(self._impedance)
                if self._mask.get(i, False) == masked
            ]
        )

    def get_real(self, masked: Optional[bool] = False) -> ndarray:
        """
        Get the real parts of the impedances in this DataSet.

        Parameters
        ----------
        masked: Optional[bool] = False
            None means that all impedances are returned.
            True means that only impedances that are to be omitted are returned.
            False means that only impedances that are to be included are returned.

        Returns
        -------
        ndarray
        """
        return self.get_impedance(masked=masked).real

    def get_imaginary(self, masked: Optional[bool] = False) -> ndarray:
        """
        Get the imaginary parts of the impedances in this DataSet.

        Parameters
        ----------
        masked: Optional[bool] = False
            None means that all impedances are returned.
            True means that only impedances that are to be omitted are returned.
            False means that only impedances that are to be included are returned.

        Returns
        -------
        ndarray
        """
        return self.get_impedance(masked=masked).imag

    def get_magnitude(self, masked: Optional[bool] = False) -> ndarray:
        """
        Get the absolute magnitudes of the impedances in this DataSet.

        Parameters
        ----------
        masked: Optional[bool] = False
            None means that all impedances are returned.
            True means that only impedances that are to be omitted are returned.
            False means that only impedances that are to be included are returned.

        Returns
        -------
        ndarray
        """
        return abs(self.get_impedance(masked=masked))

    def get_phase(self, masked: Optional[bool] = False) -> ndarray:
        """
        Get the phase angles/shifts of the impedances in this DataSet in degrees.

        Parameters
        ----------
        masked: Optional[bool] = False
            None means that all impedances are returned.
            True means that only impedances that are to be omitted are returned.
            False means that only impedances that are to be included are returned.

        Returns
        -------
        ndarray
        """
        return angle(self.get_impedance(masked=masked), deg=True)

    def get_num_points(self, masked: Optional[bool] = False) -> int:
        """
        Get the number of data points in this DataSet

        Parameters
        ----------
        masked: Optional[bool] = False
            None means that all impedances are returned.
            True means that only impedances that are to be omitted are returned.
            False means that only impedances that are to be included are returned.

        Returns
        -------
        int
        """
        return len(self.get_impedance(masked=masked))

    def get_nyquist_data(
        self, masked: Optional[bool] = False
    ) -> Tuple[ndarray, ndarray]:
        """
        Get the data necessary to plot this DataSet as a Nyquist plot: the real and the negative imaginary parts of the impedances.

        Parameters
        ----------
        masked: Optional[bool] = False
            None means that all impedances are returned.
            True means that only impedances that are to be omitted are returned.
            False means that only impedances that are to be included are returned.

        Returns
        -------
        Tuple[ndarray, ndarray]
        """
        Z: ndarray = self.get_impedance(masked=masked)
        return (
            Z.real,
            -Z.imag,
        )

    def get_bode_data(
        self, masked: Optional[bool] = False
    ) -> Tuple[ndarray, ndarray, ndarray]:
        """
        Get the data necessary to plot this DataSet as a Bode plot: the frequencies, the absolute magnitudes of the impedances, and the negative phase angles/shifts of the impedances in degrees.

        Parameters
        ----------
        masked: Optional[bool] = False
            None means that all impedances are returned.
            True means that only impedances that are to be omitted are returned.
            False means that only impedances that are to be included are returned.

        Returns
        -------
        Tuple[ndarray, ndarray, ndarray]
        """
        f: ndarray = self.get_frequency(masked=masked)
        Z: ndarray = self.get_impedance(masked=masked)
        return (
            f,
            abs(Z),
            -angle(Z, deg=True),
        )

    def to_dict(self) -> dict:
        """
        Get a dictionary that represents this DataSet, can be used to serialize the DataSet (e.g. as a JSON file), and then used to recreate this DataSet.

        Returns
        -------
        dict
        """
        return {
            "version": VERSION,
            "path": self._path,
            "label": self._label,
            "frequency": list(self._frequency),
            "real": list(self._impedance.real),
            "imaginary": list(self._impedance.imag),
            "mask": self.get_mask(),
            "uuid": self.uuid,
        }

    def to_dataframe(
        self,
        masked: Optional[bool] = False,
        frequency_label: str = "f (Hz)",
        real_label: Optional[str] = "Zre (ohm)",
        imaginary_label: Optional[str] = "Zim (ohm)",
        magnitude_label: Optional[str] = "|Z| (ohm)",
        phase_label: Optional[str] = "phase angle (deg.)",
        negative_imaginary: bool = False,
        negative_phase: bool = False,
    ) -> DataFrame:
        """
        Create a pandas.DataFrame instance from this DataSet.

        Parameters
        ----------
        masked: Optional[bool] = False
            None means that all impedances are returned.
            True means that only impedances that are to be omitted are returned.
            False means that only impedances that are to be included are returned.

        frequency_label: str = "f (Hz)"
            The label assigned to the frequency data.

        real_label: Optional[str] = "Zre (ohm)"
            The label assigned to the real part of the impedance data.

        imaginary_label: Optional[str] = "Zim (ohm)"
            The label assigned to the imaginary part of the impedance data.

        magnitude_label: Optional[str] = "|Z| (ohm)"
            The label assigned to the magnitude of the impedance data.

        phase_label: Optional[str] = "phase angle (deg.)"
            The label assigned to the phase of the imedance data.

        negative_imaginary: bool = False
            Whether or not the sign of the imaginary part of the impedance data should be inverted.

        negative_phase: bool = False
            Whether or not the sign of the phase of the impedance data should be inverted.

        Returns
        -------
        DataFrame
        """
        assert type(frequency_label) is str
        assert type(real_label) is str or real_label is None
        assert type(imaginary_label) is str or imaginary_label is None
        assert type(magnitude_label) is str or magnitude_label is None
        assert type(phase_label) is str or phase_label is None
        assert type(negative_imaginary) is bool
        assert type(negative_phase) is bool
        dictionary: Dict[str, ndarray] = {
            frequency_label: self._frequency,
        }
        Z: ndarray = self.get_impedance(masked=masked)
        if real_label is not None:
            assert imaginary_label is not None
            dictionary[real_label] = Z.real
            dictionary[imaginary_label] = Z.imag * (-1 if negative_imaginary else 1)
        if magnitude_label is not None:
            assert phase_label is not None
            dictionary[magnitude_label] = abs(Z)
            dictionary[phase_label] = angle(Z, deg=True) * (-1 if negative_phase else 1)
        return DataFrame(dictionary)


def dataframe_to_dataset(df: DataFrame, path: str, label: str = "") -> DataSet:
    """
    Convert a pandas.DataFrame into a DataSet.

    Parameters
    ----------
    df: DataFrame
        The DataFrame to be converted.

    path: str
        The path to the file that was used to create the DataFrame.

    label: str = ""
        The label assigned to the new DataSet.

    Returns
    -------
    DataSet
    """
    assert type(df) is DataFrame
    assert type(path) is str
    assert type(label) is str
    column_indices: Dict[str, int] = {}
    negative_columns: Dict[str, bool] = {}
    column_names: OrderedDict[str, List[str]] = OrderedDict(
        {
            "frequency": ["frequency", "freq", "f"],
            "imaginary": ['z"', "z''", "z_im", "zim", "imaginary", "imag", "im"],
            "real": ["z'", "z_re", "zre", "real", "re"],
            "magnitude": ["|z|", "z", "magnitude", "modulus", "mag", "mod"],
            "phase": ["phase", "phz", "phi"],
        }
    )
    i: int
    col: str
    for i, col in enumerate(df.columns):
        col = col.lower()
        key: str
        alternatives: List[str]
        for key, alternatives in column_names.items():
            if key in column_indices:
                # Skip columns that have already been identified
                continue
            alt: str
            for alt in alternatives:
                if col.startswith(alt) or col.startswith(f"-{alt}"):
                    column_indices[key] = i
                    negative_columns[key] = col.startswith("-")
                    break
            if key in column_indices:
                # The column has been identified
                break
    assert len(column_indices) >= 3, column_indices
    assert "frequency" in column_indices
    frequency: List[float] = []
    real: List[float] = []
    imaginary: List[float] = []
    magnitude: List[float] = []
    phase: Union[List[float], ndarray] = []
    for row in df.values:
        frequency.append(row[column_indices["frequency"]])
        if "real" in column_indices and "imaginary" in column_indices:
            re: float = row[column_indices["real"]]
            if negative_columns["real"]:
                re *= -1
            real.append(re)
            im: float = row[column_indices["imaginary"]]
            if negative_columns["imaginary"]:
                im *= -1
            imaginary.append(im)
        elif "magnitude" in column_indices and "phase" in column_indices:
            mag: float = row[column_indices["magnitude"]]
            magnitude.append(mag)
            phi: float = row[column_indices["phase"]]
            if negative_columns["phase"]:
                phi *= -1
            phase.append(phi)  # type: ignore
        else:
            raise Exception("Unsupported file format/structure: {path}")
    if len(phase) > 0:
        assert len(phase) == len(magnitude)
        phase = array(phase)
        if max(abs(phase)) > pi:
            phase = radians(phase)
        for mag, phi in zip(magnitude, phase):
            Z: complex = cmath.rect(mag, phi)
            real.append(Z.real)
            imaginary.append(Z.imag)
    assert len(frequency) == len(real) == len(imaginary) > 0, (
        len(frequency),
        len(real),
        len(imaginary),
    )
    return DataSet(
        array(frequency),
        array(list(map(lambda _: complex(*_), zip(real, imaginary)))),
        path=path,
        label=label,
    )
