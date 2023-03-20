# pyimpspec is licensed under the GPLv3 or later (https://www.gnu.org/licenses/gpl-3.0.html).
# Copyright 2023 pyimpspec developers
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
    bool_,
    flip,
    float64,
    integer,
    iscomplex,
    issubdtype,
    mean,
    ndarray,
    pi,
    radians as deg_to_rad,
)
from numpy.typing import NDArray
from pyimpspec.typing import (
    ComplexImpedance,
    ComplexImpedances,
    Frequencies,
    Frequency,
    Impedances,
    Phase,
    Phases,
)


VERSION: int = 2


def _parse_v2(dictionary: dict) -> dict:
    return dictionary


def _parse_v1(dictionary: dict) -> dict:
    assert "frequency" in dictionary
    assert "real" in dictionary
    assert "imaginary" in dictionary
    dictionary["frequencies"] = dictionary["frequency"]
    del dictionary["frequency"]
    dictionary["real_impedances"] = dictionary["real"]
    del dictionary["real"]
    dictionary["imaginary_impedances"] = dictionary["imaginary"]
    del dictionary["imaginary"]
    return dictionary


class DataSet:
    """
    A class that represents an impedance spectrum.
    The data points can be masked, which results in those data points being omitted from any analyses and visualization.

    Parameters
    ----------
    frequencies: |Frequencies|
        A 1-dimensional array of frequencies in hertz.

    impedances: |ComplexImpedances|
        A 1-dimensional array of complex impedances in ohms.

    mask: Optional[Dict[int, bool]], optional
        A mapping of integer indices to boolean values where a value of True means that the data point is to be omitted.

    path: str, optional
        The path to the file that has been parsed to generate this DataSet instance.

    label: str, optional
        The label assigned to this DataSet instance.

    uuid: str, optional
        The universally unique identifier assigned to this DataSet instance.
        If empty, then one will be automatically assigned.
    """

    def __init__(
        self,
        frequencies: Frequencies,
        impedances: ComplexImpedances,
        mask: Optional[Dict[int, bool]] = None,
        path: str = "",
        label: str = "",
        uuid: str = "",
    ):
        assert frequencies.shape == impedances.shape, (
            frequencies.shape,
            impedances.shape,
        )
        assert len(frequencies.shape) == 1
        assert (
            len(frequencies) == len(impedances) > 0
        ), f"{path=} ({label=}): {len(frequencies)=}, {len(impedances)=}"
        if mask is None:
            mask = {}
        assert isinstance(mask, dict), mask
        assert all(
            map(
                lambda _: issubdtype(type(_[0]), integer) and type(_[1]) is bool,
                mask.items(),
            )
        ), mask
        assert isinstance(path, str), path
        assert isinstance(label, str), label
        assert isinstance(uuid, str), uuid
        # Sort data points in descending order of frequency.
        if frequencies[-1] > frequencies[0]:
            frequencies = flip(frequencies)
            impedances = flip(impedances)
            if len(mask) > 0:
                i: int
                for i in range(0, frequencies.size):
                    j: int = frequencies.size - 1 - i
                    flag: bool = mask.get(i, False)
                    mask[i] = mask.get(j, False)
                    mask[j] = flag
        self.uuid: str = uuid or uuid4().hex
        self._path: str = path
        self._label: str = label or splitext(basename(path))[0]
        self._frequencies: Frequencies = frequencies
        self._impedances: ComplexImpedances = impedances
        self._num_points: int = len(frequencies)
        self._mask: Dict[int, bool] = {i: False for i in range(0, self._num_points)}
        self.set_mask(mask)

    def __repr__(self) -> str:
        return f"DataSet ({self._label}, {hex(id(self))})"

    def low_pass(self, cutoff: float):
        """
        Mask data points by applying a low-pass filter with the provided cutoff frequency.

        Parameters
        ----------
        cutoff: float
            The cutoff frequency to use. Data points with frequencies higher than this cutoff frequency are masked.
        """
        mask: Dict[int, bool] = self.get_mask()
        i: int
        f: float
        for i, f in enumerate(self.get_frequencies(masked=None)):
            if f > cutoff:
                mask[i] = True
        self.set_mask(mask)

    def high_pass(self, cutoff: float):
        """
        Mask data points by applying a high-pass filter with the provided cutoff frequency.

        Parameters
        ----------
        cutoff: float
            The cutoff frequency to use. Data points with frequencies lower than this cutoff frequency are masked.
        """
        mask: Dict[int, bool] = self.get_mask()
        i: int
        f: float
        for i, f in enumerate(self.get_frequencies(masked=None)):
            if f < cutoff:
                mask[i] = True
        self.set_mask(mask)

    def subtract_impedances(self, impedances: ComplexImpedances):
        """
        Subtract either the same complex value from all data points or a unique complex value for each data point in this DataSet.

        Parameters
        ----------
        impedances: |ComplexImpedances|
            The complex value(s) to subtract from this DataSet's impedances.
        """
        assert isinstance(impedances, ndarray)
        assert impedances.dtype == ComplexImpedance
        assert impedances.shape == self._impedances.shape or impedances.size == 1
        self._impedances = self._impedances - impedances

    @staticmethod
    def _parse(dictionary: dict) -> dict:
        assert isinstance(dictionary, dict), dictionary
        parsers: Dict[int, Callable] = {
            1: _parse_v1,
            2: _parse_v2,
        }
        version: int = dictionary.get("version", VERSION)
        del dictionary["version"]
        assert version <= VERSION, f"Unsupported version: {version=} > {VERSION=}"
        assert (
            version in parsers
        ), f"Unsupported version: {version=} not in {parsers.keys()=}"
        v: int
        p: Callable
        for v, p in parsers.items():
            if v < version:
                continue
            dictionary = p(dictionary)
        assert "frequencies" in dictionary
        assert "real_impedances" in dictionary
        assert "imaginary_impedances" in dictionary
        if "mask" not in dictionary:
            dictionary["mask"] = {}
        if "path" not in dictionary:
            dictionary["path"] = ""
        if "label" not in dictionary:
            dictionary["label"] = ""
        if "uuid" not in dictionary:
            dictionary["uuid"] = ""
        dictionary["frequencies"] = array(dictionary["frequencies"])
        dictionary["impedances"] = array(
            list(
                map(
                    lambda _: complex(*_),
                    zip(
                        dictionary["real_impedances"],
                        dictionary["imaginary_impedances"],
                    ),
                )
            )
        )
        del dictionary["real_impedances"]
        del dictionary["imaginary_impedances"]
        return dictionary

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
    def duplicate(Class, data: "DataSet", label: Optional[str] = None) -> "DataSet":
        """
        Create a duplicate of an existing DataSet but assign another universally unique identifier (UUID) to the copy.

        Parameters
        ----------
        data: DataSet
            The existing DataSet to duplicate.

        label: Optional[str], optional
            The label that the copy should have.

        Returns
        -------
        DataSet
        """
        assert isinstance(data, Class), data
        assert isinstance(label, str) or label is None, label
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

        label: str, optional
            The label that the new DataSet should have.

        Returns
        -------
        DataSet
        """
        assert isinstance(data_sets, list) and all(
            map(lambda _: type(_) is Class, data_sets)
        ), data_sets
        assert isinstance(label, str), label
        freqs: List[Frequencies] = list(
            map(lambda _: _.get_frequencies(masked=None), data_sets)
        )
        imps: List[ComplexImpedances] = list(
            map(lambda _: _.get_impedances(masked=None), data_sets)
        )
        f: Frequencies = freqs.pop(0)
        assert all(map(lambda _: allclose(f, _), freqs))
        Z: ComplexImpedances = mean(array(imps), axis=0)
        return Class(
            frequencies=f,
            impedances=Z,
            label=label,
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
        assert isinstance(path, str), path
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
        assert isinstance(label, str), label
        label = label.strip()
        assert label != ""
        self._label = label

    def set_mask(self, mask: Dict[int, bool]):
        """
        Set the mask for this DataSet.

        Parameters
        ----------
        mask: Dict[int, bool]
            The new mask that determines which data points are omitted.
            True means that the data point is to be omitted and False means that the data point is to be included.
            The keys must be zero-based integer indices and the values must be boolean values.
        """
        assert (
            type(mask) is dict
            and all(map(lambda _: issubdtype(type(_), integer), mask.keys()))
            and all(map(lambda _: issubdtype(type(_), bool_), mask.values()))
        ), mask
        i: int
        if len(mask) == 0:
            self._mask.update({i: False for i in range(0, self._num_points)})
            return
        mask = mask.copy()
        for i in list(mask.keys()):
            if i < 0 or i >= self._num_points:
                del mask[i]
        self._mask.update(mask)

    def get_mask(self) -> Dict[int, bool]:
        """
        Get the mask for this DataSet.
        True means that the data point is to be omitted and False means that the data point is to be included.
        The keys are zero-based integer indices and the values are booleans.

        Returns
        -------
        Dict[int, bool]
        """
        return self._mask.copy()

    def get_frequencies(self, masked: Optional[bool] = False) -> Frequencies:
        """
        Get the frequencies in this DataSet.

        Parameters
        ----------
        masked: Optional[bool], optional
            None means that all frequencies are returned.
            True means that only frequencies that are to be omitted are returned.
            False means that only frequencies that are to be included are returned.

        Returns
        -------
        |Frequencies|
        """
        assert isinstance(masked, bool) or masked is None, masked
        if masked is None:
            return array(self._frequencies, dtype=Frequency)
        i: int
        f: float
        return array(
            [
                f
                for i, f in enumerate(self._frequencies)
                if self._mask.get(i, False) == masked
            ],
            dtype=Frequency,
        )

    def get_impedances(self, masked: Optional[bool] = False) -> ComplexImpedances:
        """
        Get the complex impedances in this DataSet.

        Parameters
        ----------
        masked: Optional[bool], optional
            None means that all impedances are returned.
            True means that only impedances that are to be omitted are returned.
            False means that only impedances that are to be included are returned.

        Returns
        -------
        |ComplexImpedances|
        """
        assert isinstance(masked, bool) or masked is None, masked
        if masked is None:
            return self._impedances
        i: int
        c: complex
        return array(
            [
                c
                for i, c in enumerate(self._impedances)
                if self._mask.get(i, False) == masked
            ],
            dtype=ComplexImpedance,
        )

    def get_magnitudes(self, masked: Optional[bool] = False) -> Impedances:
        """
        Get the absolute magnitudes of the impedances in this DataSet.

        Parameters
        ----------
        masked: Optional[bool], optional
            None means that all impedances are returned.
            True means that only impedances that are to be omitted are returned.
            False means that only impedances that are to be included are returned.

        Returns
        -------
        |Impedances|
        """
        return abs(self.get_impedances(masked=masked))

    def get_phases(self, masked: Optional[bool] = False) -> Phases:
        """
        Get the phase angles/shifts of the impedances in this DataSet in degrees.

        Parameters
        ----------
        masked: Optional[bool], optional
            None means that all impedances are returned.
            True means that only impedances that are to be omitted are returned.
            False means that only impedances that are to be included are returned.

        Returns
        -------
        |Phases|
        """
        return angle(self.get_impedances(masked=masked), deg=True)

    def get_num_points(self, masked: Optional[bool] = False) -> int:
        """
        Get the number of data points in this DataSet

        Parameters
        ----------
        masked: Optional[bool], optional
            None means that all impedances are returned.
            True means that only impedances that are to be omitted are returned.
            False means that only impedances that are to be included are returned.

        Returns
        -------
        int
        """
        return len(self.get_impedances(masked=masked))

    def get_nyquist_data(
        self,
        masked: Optional[bool] = False,
    ) -> Tuple[Impedances, Impedances]:
        """
        Get the data necessary to plot this DataSet as a Nyquist plot: the real and the negative imaginary parts of the impedances.

        Parameters
        ----------
        masked: Optional[bool], optional
            None means that all impedances are returned.
            True means that only impedances that are to be omitted are returned.
            False means that only impedances that are to be included are returned.

        Returns
        -------
        Tuple[|Impedances|, |Impedances|]
        """
        Z: ComplexImpedances = self.get_impedances(masked=masked)
        return (
            Z.real,
            -Z.imag,
        )

    def get_bode_data(
        self,
        masked: Optional[bool] = False,
    ) -> Tuple[Frequencies, Impedances, Phases]:
        """
        Get the data necessary to plot this DataSet as a Bode plot: the frequencies, the absolute magnitudes of the impedances, and the negative phase angles/shifts of the impedances in degrees.

        Parameters
        ----------
        masked: Optional[bool], optional
            None means that all impedances are returned.
            True means that only impedances that are to be omitted are returned.
            False means that only impedances that are to be included are returned.

        Returns
        -------
        Tuple[|Frequencies|, |Impedances|, |Phases|]
        """
        f: Frequencies = self.get_frequencies(masked=masked)
        Z: ComplexImpedances = self.get_impedances(masked=masked)
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
            "frequencies": list(self._frequencies),
            "real_impedances": list(self._impedances.real),
            "imaginary_impedances": list(self._impedances.imag),
            "mask": self.get_mask(),
            "uuid": self.uuid,
        }

    def to_dataframe(
        self,
        masked: Optional[bool] = False,
        columns: Optional[List[str]] = None,
        negative_imaginary: bool = False,
        negative_phase: bool = False,
    ) -> "DataFrame":  # noqa: F821
        """
        Create a |DataFrame| instance from this DataSet.

        Parameters
        ----------
        masked: Optional[bool], optional
            None means that all impedances are returned.
            True means that only impedances that are to be omitted are returned.
            False means that only impedances that are to be included are returned.

        columns: Optional[List[str]], optional
            The column headers to use.

        negative_imaginary: bool, optional
            Whether or not the sign of the imaginary part of the impedance data should be inverted.
            Note that this does not automatically add a minus sign in front of the column label.

        negative_phase: bool, optional
            Whether or not the sign of the phase of the impedance data should be inverted.
            Note that this does not automatically add a minus sign in front of the column label.

        Returns
        -------
        |DataFrame|
        """
        from pandas import DataFrame

        if columns is None:
            columns = [
                "f (Hz)",
                "Re(Z) (ohm)",
                "Im(Z) (ohm)",
                "Mod(Z) (ohm)",
                "Phase(Z) (deg.)",
            ]
        assert isinstance(columns, list), columns
        assert len(columns) == 5, len(columns)
        assert all(map(lambda _: isinstance(_, str), columns)), columns
        dictionary: Dict[str, NDArray[float64]] = {
            columns[0]: self.get_frequencies(masked=masked),
        }
        Z: ComplexImpedances = self.get_impedances(masked=masked)
        if columns[1]:
            assert columns[2]
            dictionary[columns[1]] = Z.real
            dictionary[columns[2]] = Z.imag * (-1 if negative_imaginary else 1)
        if columns[3]:
            assert columns[4]
            dictionary[columns[3]] = abs(Z)
            dictionary[columns[4]] = angle(Z, deg=True) * (-1 if negative_phase else 1)
        return DataFrame(dictionary)


def _dataframe_to_data_set(
    df: "DataFrame",  # noqa: F821
    path: str,
    label: str = "",
    degrees: bool = True,
) -> DataSet:
    """
    Convert a |DataFrame| object into a DataSet object.

    Parameters
    ----------
    df: DataFrame
        The DataFrame to be converted.

    path: str
        The path to the file that was used to create the DataFrame.

    label: str, optional
        The label assigned to the new DataSet.

    degrees: bool, optional
        Whether or not the phase data (if provided) is in degrees instead of radians.

    Returns
    -------
    DataSet
    """
    from pandas import DataFrame

    assert isinstance(df, DataFrame), df
    assert isinstance(path, str), path
    assert isinstance(label, str), label
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
                if (
                    col.startswith(alt)
                    or col.startswith(f"-{alt}")
                    or col.startswith(f"−{alt}")
                ):
                    column_indices[key] = i
                    negative_columns[key] = col.startswith("-") or col.startswith("−")
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
    phase: Union[List[float], Phases] = []
    for row in df.values:
        f: float = row[column_indices["frequency"]]
        if type(f) is str:
            f = float(row[column_indices["frequency"]].replace(",", "."))
        frequency.append(f)
        if "real" in column_indices and "imaginary" in column_indices:
            re: float = row[column_indices["real"]]
            if type(re) is str:
                re = float(row[column_indices["real"]].replace(",", "."))
            if negative_columns["real"]:
                re *= -1
            real.append(re)
            im: float = row[column_indices["imaginary"]]
            if type(im) is str:
                im = float(row[column_indices["imaginary"]].replace(",", "."))
            if negative_columns["imaginary"]:
                im *= -1
            imaginary.append(im)
        elif "magnitude" in column_indices and "phase" in column_indices:
            mag: float = row[column_indices["magnitude"]]
            if type(mag) is str:
                mag = float(row[column_indices["magnitude"]].replace(",", "."))
            magnitude.append(mag)
            phi: float = row[column_indices["phase"]]
            if type(phi) is str:
                phi = float(row[column_indices["phase"]].replace(",", "."))
            if negative_columns["phase"]:
                phi *= -1
            phase.append(phi)  # type: ignore
        else:
            raise Exception(f"Unsupported file format/structure: {path}")
    if len(phase) > 0:
        assert len(phase) == len(magnitude)
        phase = array(phase, dtype=Phase)
        if degrees is True:
            phase = deg_to_rad(phase)
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


def dataframe_to_data_sets(
    df: "DataFrame",  # noqa: F821
    path: str,
    label: str = "",
    degrees: bool = True,
) -> List[DataSet]:
    """
    Takes a |DataFrame| object, checks if there are multiple frequency sweeps, and, if necessary, splits the data into multiple DataSet objects.

    Parameters
    ----------
    df: DataFrame
        The DataFrame to be converted.
        The object should contain columns for frequencies and either real and imaginary impedances, or the magnitudes and phases of impedances.
        Multiple labels are supported for the column headers and they are detected based on whether or not the header starts with one of the supported labels.
        The comparisons are case insensitive and inverted signs are detected based on if the column header starts with either a hyphen (-) or a minus sign (:math:`-`).

        - Frequencies: 'frequency', 'freq', and 'f'.
        - Real parts of the impedances: "z'", 'z_re', 'zre', 'real', and 're'.
        - Imaginary parts of the impedances: 'z"', "z'', 'z_im', 'zim', 'imaginary', 'imag', and 'im'.
        - Magnitudes of the impedances: '\|z\|', 'z', 'magnitude', 'modulus', 'mag', and 'mod'.
        - Phases of the impedances: 'phase', 'phz', and 'phi'.

    path: str
        The path to the file that was used to create the DataFrame.

    label: str, optional
        The label assigned to the new DataSet.

    degrees: bool, optional
        Whether or not the phase data (if provided) is in degrees instead of radians.

    Returns
    -------
    List[DataSet]
    """
    data: DataSet = _dataframe_to_data_set(
        df,
        path=path,
        label=label,
        degrees=degrees,
    )
    data_sets: List[DataSet] = []
    f: List[float] = list(data.get_frequencies())
    Z: List[complex] = list(data.get_impedances())
    decreasing_f: bool = f[0] >= f[1]
    while f:
        i: int = 0
        while i < len(f) - 1 and decreasing_f == (f[i] >= f[i + 1]):
            i += 1
        i += 1
        if i > 1:
            data_sets.append(
                DataSet(
                    array(f[:i]),
                    array(Z[:i]),
                    path=path,
                    label=label,
                )
            )
            f = f[i:]
            Z = Z[i:]
    assert len(f) == len(Z) == 0
    if len(data_sets) > 1:
        for i, data in enumerate(data_sets):
            data.set_label(f"{data.get_label()} ({i+1})")
        return data_sets
    return [data]
