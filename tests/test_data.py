# Copyright 2022 pyimpspec developers
# pyimpspec is licensed under the GPLv3 or later (https://www.gnu.org/licenses/gpl-3.0.html).
# The licenses of pyimpspec's dependencies and/or sources of portions of code are included in
# the LICENSES folder.

from unittest import TestCase
from pyimpspec.data.dataset import DataSet, VERSION
from numpy import array, ndarray, log10 as log
from typing import Callable, Dict, List, Optional
from os.path import basename, splitext


class TestDataSet(TestCase):
    def test_01_constructors(self):
        freq: ndarray = array(list(range(1, 21)))
        Z: ndarray = array(list(map(lambda _: complex(_, -_), range(1, 21))))
        mask: Dict[int, bool] = {
            2: True,
            7: True,
            15: False,
        }
        path: str = "/test/path/file.ext"
        # __init__
        data: DataSet
        with self.assertRaises(AssertionError):
            DataSet(array([]), array([]))
            DataSet([1], [complex(1, -1)])
        data = DataSet(freq, Z, mask, path)
        self.assertEqual(data.get_path(), path)
        self.assertEqual(data.get_label(), splitext(basename(path))[0])
        self.assertEqual(data.get_num_points(), 18)
        self.assertEqual(data.get_num_points(masked=None), 20)
        self.assertEqual(data.get_num_points(masked=True), 2)
        self.assertEqual(data.get_num_points(masked=False), 18)
        methods: List[Callable] = [
            data.get_frequency,
            data.get_impedance,
            data.get_real,
            data.get_imaginary,
            data.get_magnitude,
            data.get_phase,
        ]
        method: Callable
        for method in methods:
            self.assertEqual(len(method()), 18)
            self.assertEqual(len(method(masked=None)), 20)
            self.assertEqual(len(method(masked=True)), 2)
            self.assertEqual(len(method(masked=False)), 18)

        self.assertEqual(len(data.get_nyquist_data()[0]), 18)
        self.assertEqual(len(data.get_nyquist_data(masked=None)[0]), 20)
        self.assertEqual(len(data.get_nyquist_data(masked=True)[0]), 2)
        self.assertEqual(len(data.get_nyquist_data(masked=False)[0]), 18)

        self.assertEqual(len(data.get_bode_data()[0]), 18)
        self.assertEqual(len(data.get_bode_data(masked=None)[0]), 20)
        self.assertEqual(len(data.get_bode_data(masked=True)[0]), 2)
        self.assertEqual(len(data.get_bode_data(masked=False)[0]), 18)

        # from_dict
        label: str = "label"
        data = DataSet(freq, Z, mask, path, label)
        self.assertEqual(data.get_label(), label)
        data = DataSet.from_dict({
            "frequency": freq,
            "real": Z.real,
            "imaginary": Z.imag,
            "mask": mask,
            "path": path,
            "label": label,
        })
        self.assertEqual(data.get_path(), path)
        self.assertEqual(data.get_label(), label)
        self.assertEqual(data.get_num_points(), 18)
        self.assertEqual(data.get_num_points(masked=None), 20)
        self.assertEqual(data.get_num_points(masked=True), 2)
        self.assertEqual(data.get_num_points(masked=False), 18)
        methods: List[Callable] = [
            data.get_frequency,
            data.get_impedance,
            data.get_real,
            data.get_imaginary,
            data.get_magnitude,
            data.get_phase,
        ]
        method: Callable
        for method in methods:
            self.assertEqual(len(method()), 18)
            self.assertEqual(len(method(masked=None)), 20)
            self.assertEqual(len(method(masked=True)), 2)
            self.assertEqual(len(method(masked=False)), 18)

        self.assertEqual(len(data.get_nyquist_data()[0]), 18)
        self.assertEqual(len(data.get_nyquist_data(masked=None)[0]), 20)
        self.assertEqual(len(data.get_nyquist_data(masked=True)[0]), 2)
        self.assertEqual(len(data.get_nyquist_data(masked=False)[0]), 18)

        self.assertEqual(len(data.get_bode_data()[0]), 18)
        self.assertEqual(len(data.get_bode_data(masked=None)[0]), 20)
        self.assertEqual(len(data.get_bode_data(masked=True)[0]), 2)
        self.assertEqual(len(data.get_bode_data(masked=False)[0]), 18)

    def test_02_subtract_impedance(self):
        freq: ndarray = array(list(range(1, 6)))
        Z: ndarray = array(list(map(lambda _: complex(_, -_), range(1, 6))))
        data: DataSet = DataSet(freq, Z)
        i: int
        z: complex
        for i, z in enumerate(data.get_impedance()):
            self.assertEqual(z, Z[i])
        data.subtract_impedance(complex(1, -1))
        for i, z in enumerate(data.get_impedance()):
            self.assertEqual(z, Z[i] - complex(1, -1))

    def test_03_get_set_label(self):
        data: DataSet = DataSet(array([1]), array([complex(1, -1)]))
        self.assertEqual(data.get_label(), "")
        label: str = "testing"
        data.set_label(label)
        self.assertEqual(data.get_label(), label)

    def test_04_get_set_mask(self):
        data: DataSet = DataSet(array([1]), array([complex(1, -1)]))
        data.set_mask({0: True})
        self.assertEqual(len(data.get_mask()), 1)
        self.assertTrue(data.get_mask()[0])

    def test_05_get_values(self):
        data: DataSet = DataSet(array([1]), array([complex(1, -1)]))
        self.assertEqual(data.get_frequency()[0], 1)
        self.assertEqual(data.get_impedance()[0].real, 1)
        self.assertEqual(data.get_impedance()[0].imag, -1)
        self.assertEqual(data.get_real()[0], 1)
        self.assertEqual(data.get_imaginary()[0], -1)
        self.assertEqual(data.get_magnitude()[0], abs(complex(1, -1)))
        self.assertEqual(data.get_phase()[0], -45)

        self.assertEqual(len(data.get_nyquist_data()), 2)
        self.assertEqual(data.get_nyquist_data()[0][0], 1)
        self.assertEqual(data.get_nyquist_data()[1][0], 1)

        self.assertEqual(len(data.get_bode_data()), 3)
        self.assertEqual(data.get_bode_data()[0][0], 0)
        self.assertEqual(data.get_bode_data()[1][0], log(abs(complex(1, -1))))
        self.assertEqual(data.get_bode_data()[2][0], 45)

    def test_06_to_dict(self):
        path: str = "/test/path/file.ext"
        label: str = "testing"
        data: DataSet = DataSet(array([1]), array([complex(1, -1)]), {0: True}, path, label)
        dictionary: dict = data.to_dict()
        self.assertTrue("version" in dictionary, True)
        self.assertTrue("path" in dictionary, True)
        self.assertTrue("label" in dictionary, True)
        self.assertTrue("frequency" in dictionary, True)
        self.assertTrue("real" in dictionary, True)
        self.assertTrue("imaginary" in dictionary, True)
        self.assertTrue("mask" in dictionary, True)

        self.assertEqual(type(dictionary["version"]), int)
        self.assertEqual(type(dictionary["path"]), str)
        self.assertEqual(type(dictionary["label"]), str)
        self.assertEqual(type(dictionary["frequency"]), list)
        self.assertEqual(type(dictionary["real"]), list)
        self.assertEqual(type(dictionary["imaginary"]), list)
        self.assertEqual(type(dictionary["mask"]), dict)

        self.assertEqual(dictionary["version"], VERSION)
        self.assertEqual(dictionary["path"], path)
        self.assertEqual(dictionary["label"], label)
        self.assertEqual(dictionary["frequency"][0], 1)
        self.assertEqual(dictionary["real"][0], 1)
        self.assertEqual(dictionary["imaginary"][0], -1)
        self.assertEqual(dictionary["mask"][0], True)

    def test_07_to_dataframe(self):
        # TODO: Implement
        pass

    def test_08_dataframe_to_dataset(self):
        # TODO: Implement
        pass

# TODO: Create tests for the parsers for the various file formats
