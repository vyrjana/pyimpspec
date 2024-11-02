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

import cmath
from collections import OrderedDict
from os.path import (
    basename,
    splitext,
)
from uuid import uuid4
from numpy import (
    allclose,
    angle,
    array,
    flip,
    float64,
    mean,
    radians as deg_to_rad,
    unique,
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
from pyimpspec.typing.helpers import (
    Callable,
    Dict,
    List,
    Optional,
    Path,
    Tuple,
    Union,
    _cast_to_complex_array,
    _cast_to_floating_array,
    _is_boolean,
    _is_complex_array,
    _is_floating,
    _is_floating_array,
    _is_integer,
)
from pyimpspec.exceptions import UnsupportedFileFormat


VERSION: int = 2


def _parse_v2(dictionary: dict) -> dict:
    return dictionary


def _parse_v1(dictionary: dict) -> dict:
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
        if not _is_floating_array(frequencies):
            frequencies = _cast_to_floating_array(frequencies)

        if not _is_complex_array(impedances):
            impedances = _cast_to_complex_array(impedances)

        if frequencies.shape != impedances.shape:
            raise ValueError(
                f"Expected the frequency and impedances arrays to have the same shapes instead of {frequencies.shape=} and {impedances.shape=}"
            )
        elif len(frequencies.shape) != 1:
            raise ValueError(
                f"Expected the frequency array to be 1D instead of {frequencies.shape=}"
            )
        elif not (len(frequencies) == len(impedances) > 0):
            raise ValueError(
                "Expected the frequency and impedance arrays to be of equal length and have at least 1 element each"
            )
        elif len(unique(frequencies)) != len(frequencies):
            raise ValueError(
                f"Expected the frequencies in {label} ({path}) to be unique instead of {frequencies=}"
            )

        if mask is None:
            mask = {}
        elif not isinstance(mask, dict):
            raise TypeError(f"Expected a dictionary or None instead of {mask=}")
        elif not all(map(lambda key: _is_integer(key), mask.keys())):
            raise TypeError(f"Expected integer keys instead of {mask.keys()=}")
        elif not all(map(lambda value: _is_boolean(value), mask.values())):
            raise TypeError(f"Expected boolean values instead of {mask.values()=}")

        if not isinstance(path, str):
            raise TypeError(f"Expected a string instead of {path=}")

        if not isinstance(label, str):
            raise TypeError(f"Expected a string instead of {label=}")

        if not isinstance(uuid, str):
            raise TypeError(f"Expected a string instead of {uuid=}")

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
        if not (_is_floating(cutoff) or _is_integer(cutoff)):
            raise TypeError(f"Expected a float or an integer instead of {cutoff=}")

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
        if not (_is_floating(cutoff) or _is_integer(cutoff)):
            raise TypeError(f"Expected a float or an integer instead of {cutoff=}")

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
        if not _is_complex_array(impedances):
            raise TypeError(f"Expected an NDArray[complex128] instead of {impedances=}")

        self._impedances = self._impedances - impedances

    @staticmethod
    def _parse(dictionary: dict) -> dict:
        if not isinstance(dictionary, dict):
            raise TypeError(f"Expected a dictionary instead of {dictionary=}")

        parsers: Dict[int, Callable] = {
            1: _parse_v1,
            2: _parse_v2,
        }

        version: int = dictionary.get("version", VERSION)
        del dictionary["version"]

        if version > VERSION:
            raise ValueError(f"Unsupported version: {version=} > {VERSION=}")
        elif version not in parsers:
            raise ValueError(
                f"Unsupported version: {version=} not in {parsers.keys()=}"
            )

        v: int
        p: Callable
        for v, p in parsers.items():
            if v < version:
                continue

            dictionary = p(dictionary)

        if "mask" not in dictionary:
            dictionary["mask"] = {}
        else:
            dictionary["mask"] = {int(k): v for k, v in dictionary["mask"].items()}

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
    def from_dict(cls, dictionary: dict) -> "DataSet":
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
        return cls(**cls._parse(dictionary))

    @classmethod
    def duplicate(cls, data: "DataSet", label: Optional[str] = None) -> "DataSet":
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
        if not isinstance(data, cls):
            raise TypeError(f"Expected a {cls} instead of {data=}")

        if not (isinstance(label, str) or label is None):
            raise TypeError(f"Expected a string or None instead of {label=}")

        dictionary: dict = data.to_dict()
        if label is not None:
            dictionary["label"] = label

        del dictionary["uuid"]

        return cls.from_dict(dictionary)

    @classmethod
    def average(cls, data_sets: List["DataSet"], label: str = "Average") -> "DataSet":
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
        if not isinstance(data_sets, list):
            raise TypeError(f"Expected a list instead of {data_sets=}")
        elif not all(map(lambda data: isinstance(data, cls), data_sets)):
            raise TypeError(f"Expected all items to be {cls} instead of {data_sets=}")

        if not isinstance(label, str):
            raise TypeError(f"Expected a string instead of {label=}")

        frequencies: List[Frequencies] = list(
            map(
                lambda _: _.get_frequencies(masked=None),
                data_sets,
            )
        )

        impedances: List[ComplexImpedances] = list(
            map(
                lambda data: data.get_impedances(masked=None),
                data_sets,
            )
        )

        f: Frequencies = frequencies.pop(0)
        if not all(
            map(
                lambda other_f: allclose(f, other_f),
                frequencies,
            )
        ):
            raise ValueError("Expected all frequency arrays to have the same values")

        Z: ComplexImpedances = mean(array(impedances), axis=0)

        return cls(
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
        if not isinstance(path, str):
            raise TypeError(f"Expected a string instead of {path=}")

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
        if not isinstance(label, str):
            raise TypeError(f"Expected a string instead of {label=}")

        if label.strip() == "":
            raise ValueError(f"Expected a non-empty string instead of {label=}")

        self._label = label.strip()

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
        if not isinstance(mask, dict):
            raise TypeError(f"Expected a dictionary instead of {mask=}")
        elif not all(map(lambda key: _is_integer(key), mask.keys())):
            raise TypeError(f"Expected integer keys instead of {mask.keys()=}")
        elif not all(map(lambda value: _is_boolean(value), mask.values())):
            raise TypeError(f"Expected boolean values instead of {mask.values()=}")

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
        if not (_is_boolean(masked) or masked is None):
            raise TypeError(f"Expected a boolean or None instead of {masked=}")

        if masked is None:
            return array(self._frequencies, dtype=Frequency)

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
        if not (_is_boolean(masked) or masked is None):
            raise TypeError(f"Expected a boolean or None instead of {masked=}")

        if masked is None:
            return self._impedances

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
            "frequencies": self._frequencies.tolist(),
            "real_impedances": self._impedances.real.tolist(),
            "imaginary_impedances": self._impedances.imag.tolist(),
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
        Create a `pandas.DataFrame`_ instance from this DataSet.

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
        `pandas.DataFrame`_
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

        if not isinstance(columns, list):
            raise TypeError(f"Expected a list instead of {columns=}")
        elif not len(columns) == 5:
            raise ValueError(
                f"Expected a list with 5 elements instead of {len(columns)=}"
            )
        elif not all(map(lambda item: isinstance(item, str), columns)):
            raise TypeError(f"Expected a list of strings instead of {columns=}")
        elif not len(set([item.strip() for item in columns])) == 5:
            raise ValueError(
                f"Expected unique strings that are not just whitespace instead of {columns=}"
            )

        dictionary: Dict[str, NDArray[float64]] = {
            columns[0]: self.get_frequencies(masked=masked),
        }
        Z: ComplexImpedances = self.get_impedances(masked=masked)

        dictionary[columns[1]] = Z.real
        dictionary[columns[2]] = Z.imag * (-1 if negative_imaginary else 1)

        dictionary[columns[3]] = abs(Z)
        dictionary[columns[4]] = angle(Z, deg=True) * (-1 if negative_phase else 1)

        return DataFrame(dictionary)


def _detect_columns(
    df: "DataFrame",  # noqa: F821
) -> Tuple[Dict[str, int], Dict[str, bool]]:
    column_indices: Dict[str, int] = {}
    negative_columns: Dict[str, bool] = {}
    column_names: OrderedDict[str, List[str]] = OrderedDict(
        {
            "frequency": ["frequency", "freq", "f"],
            "imaginary": [
                'z"',
                "z''",
                "z im",
                "z_im",
                "zim",
                "imaginary",
                "imag",
                "im",
            ],
            "real": ["z'", "z re", "z_re", "zre", "real", "re"],
            "magnitude": ["|z|", "z", "magnitude", "modulus", "mag", "mod"],
            "phase": ["phase", "phz", "phi"],
        }
    )

    i: int
    col: str
    for i, col in enumerate(df.columns):
        col = col.lower().strip()

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
                    negative_columns[key] = col[0] in ("-", "−")
                    break

            if key in column_indices:
                # The column has been identified
                break

    if len(column_indices) < 3:
        raise ValueError(
            f"Expected to find at least 3 columns to process instead of {column_indices=}"
        )
    elif "frequency" not in column_indices:
        raise KeyError(
            f"Expected to find a 'frequency' column instead of {column_indices=}"
        )

    return (column_indices, negative_columns)


def _extract_data(
    df: "DataFrame",  # noqa: F821
    column_indices: Dict[str, int],
    negative_columns: Dict[str, bool],
    path: str,
    degrees: bool,
) -> Tuple[List[float], List[float], List[float]]:
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
            raise UnsupportedFileFormat(
                f"Unsupported data structure after parsing '{path}': {df=}"
            )

    if len(phase) > 0:
        if len(phase) != len(magnitude):
            raise ValueError(
                f"Expected to find equal numbers of phase and magnitude data points instead of {len(phase)=} and {len(magnitude)=}"
            )

        phase = array(phase, dtype=Phase)
        if degrees:
            phase = deg_to_rad(phase)

        for mag, phi in zip(magnitude, phase):
            Z: complex = cmath.rect(mag, phi)
            real.append(Z.real)
            imaginary.append(Z.imag)

    if not (len(frequency) == len(real) == len(imaginary) > 0):
        raise ValueError(
            f"Expected to find equal numbers of frequency, real, and imaginary data points instead of {len(frequency)=}, {len(real)=}, and {len(imaginary)=}"
        )

    return (frequency, real, imaginary)


def _split_sweeps(
    frequency: List[float],
    real: List[float],
    imaginary: List[float],
    path: str,
    label: str,
) -> List[DataSet]:
    data_sets: List[DataSet] = []
    decreasing_f: bool = frequency[0] > frequency[1]

    while frequency:
        i: int = 1
        for i in range(1, len(frequency)):
            if decreasing_f:
                if frequency[i - 1] > frequency[i]:
                    continue
                elif frequency[i - 1] < frequency[i]:
                    break
            elif not decreasing_f:
                if frequency[i - 1] < frequency[i]:
                    continue
                elif frequency[i - 1] > frequency[i]:
                    break

            raise ValueError(
                f"Expected frequencies to either decrease or increase instead of {frequency}"
            )
        else:
            i += 1

        data_sets.append(
            DataSet(
                array(frequency[:i]),
                array(
                    list(
                        map(
                            lambda _: complex(*_),
                            zip(
                                real[:i],
                                imaginary[:i],
                            ),
                        )
                    )
                ),
                path=path,
                label=label,
            )
        )

        frequency = frequency[i:]
        real = real[i:]
        imaginary = imaginary[i:]

    if len(frequency) > 0:
        raise ValueError(f"Not all data points were processed: {frequency=}")

    return data_sets


def dataframe_to_data_sets(
    df: "DataFrame",  # noqa: F821
    path: Union[str, Path],
    label: str = "",
    degrees: bool = True,
) -> List[DataSet]:
    """
    Takes a `pandas.DataFrame`_ object, checks if there are multiple frequency sweeps, and, if necessary, splits the data into multiple DataSet objects.

    Parameters
    ----------
    df: `pandas.DataFrame`_
        The `pandas.DataFrame`_ to be converted.
        The object should contain columns for frequencies and either real and imaginary impedances, or the magnitudes and phases of impedances.
        Multiple labels are supported for the column headers and they are detected based on whether or not the header starts with one of the supported labels.
        The comparisons are case insensitive and inverted signs are detected based on if the column header starts with either a hyphen (-) or a minus sign (:math:`-`).

        - Frequencies: 'frequency', 'freq', and 'f'.
        - Real parts of the impedances: "z'", 'z_re', 'zre', 'z re', 'real', and 're'.
        - Imaginary parts of the impedances: 'z"', "z'', 'z_im', 'zim', 'z im', 'imaginary', 'imag', and 'im'.
        - Magnitudes of the impedances: '\\|z\\|', 'z', 'magnitude', 'modulus', 'mag', and 'mod'.
        - Phases of the impedances: 'phase', 'phz', and 'phi'.

    path: Union[str, pathlib.Path]
        The path to the file that was used to create the `pandas.DataFrame`_.

    label: str, optional
        The label assigned to the new DataSet.

    degrees: bool, optional
        Whether or not the phase data (if provided) is in degrees instead of radians.

    Returns
    -------
    List[|DataSet|]
    """
    from pandas import DataFrame

    if not isinstance(df, DataFrame):
        raise TypeError(f"Expected a {DataFrame} instead of {df=}")

    if isinstance(path, Path):
        path = str(path)
    elif not isinstance(path, str):
        raise TypeError(f"Expected a string instead of {path=}")

    if not isinstance(label, str):
        raise TypeError(f"Expected a string instead of {label=}")

    if not _is_boolean(degrees):
        raise TypeError(f"Expected a boolean instead of {degrees=}")

    column_indices: Dict[str, int]
    negative_columns: Dict[str, bool]
    column_indices, negative_columns = _detect_columns(df)

    frequency: List[float]
    real: List[float]
    imaginary: List[float]
    frequency, real, imaginary = _extract_data(
        df,
        column_indices,
        negative_columns,
        path,
        degrees,
    )

    data_sets: List[DataSet] = _split_sweeps(
        frequency,
        real,
        imaginary,
        path,
        label,
    )
    if len(data_sets) > 1:
        for i, data in enumerate(data_sets):
            data.set_label(f"{data.get_label()} ({i+1})")

    return data_sets
