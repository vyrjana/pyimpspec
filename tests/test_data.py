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

from unittest import TestCase
from pyimpspec import (
    DataSet,
    parse_data,
)
from pyimpspec.data.data_set import VERSION
from numpy import (
    allclose,
    array,
    ndarray,
)
from typing import (
    Callable,
    Dict,
    List,
    Optional,
)
from os import walk
from os.path import (
    basename,
    dirname,
    join,
    splitext,
)


def get_control_data() -> DataSet:
    return DataSet.from_dict(
        {
            "label": "Control",
            "frequency": [
                10000,
                7196.85673001152,
                5179.47467923121,
                3727.59372031494,
                2682.69579527973,
                1930.69772888325,
                1389.49549437314,
                1000,
                719.685673001152,
                517.947467923121,
                372.759372031494,
                268.269579527973,
                193.069772888325,
                138.949549437314,
                100,
                71.9685673001152,
                51.7947467923121,
                37.2759372031494,
                26.8269579527973,
                19.3069772888325,
                13.8949549437314,
                10,
                7.19685673001152,
                5.17947467923121,
                3.72759372031494,
                2.68269579527973,
                1.93069772888325,
                1.38949549437314,
                1,
            ],
            "real": [
                109.00918219439,
                112.057759954682,
                116.906245842316,
                124.834566504841,
                137.770479052544,
                157.971701100428,
                186.636916072821,
                221.690825019137,
                257.437427532301,
                288.118363568086,
                311.563115366785,
                328.958937177027,
                342.639052460841,
                354.649295730333,
                366.399861210884,
                378.777952592123,
                392.326418700802,
                407.370451984566,
                424.085899178091,
                442.527901931489,
                462.638253653398,
                484.245515868873,
                507.068153598917,
                530.727486829165,
                554.773334457305,
                578.720842454073,
                602.093062736189,
                624.461682820694,
                645.478700150494,
            ],
            "imaginary": [
                -26.5556798765152,
                -35.1662256016599,
                -46.4663772865298,
                -60.8522924167586,
                -78.0893530523511,
                -96.480585382064,
                -112.204629862651,
                -120.39912459346,
                -118.650600986126,
                -109.310321223647,
                -97.2956995983817,
                -86.5539533982431,
                -78.8886755242402,
                -74.587868105523,
                -73.2559400473505,
                -74.2956945797458,
                -77.1022034740841,
                -81.1148201939911,
                -85.8172962476514,
                -90.7274808764653,
                -95.3931737367316,
                -99.3992549174071,
                -102.385330669065,
                -104.069381709955,
                -104.270595391674,
                -102.92415906369,
                -100.082675458352,
                -95.9025788682872,
                -90.618128307383,
            ],
        }
    )


def get_test_files(extension: Optional[str] = None) -> List[str]:
    root: str
    files: List[str]
    for root, _, files in walk(dirname(__file__)):
        break
    assert len(files) > 0
    files.sort()
    files = list(map(lambda _: join(root, _), files))
    if extension is None:
        return files
    files = list(filter(lambda _: splitext(_)[1] == extension, files))
    assert len(files) > 0
    return files


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
        self.assertGreater(data.get_frequency()[0], data.get_frequency()[-1])
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
        #
        self.assertEqual(len(data.get_nyquist_data()[0]), 18)
        self.assertEqual(len(data.get_nyquist_data(masked=None)[0]), 20)
        self.assertEqual(len(data.get_nyquist_data(masked=True)[0]), 2)
        self.assertEqual(len(data.get_nyquist_data(masked=False)[0]), 18)
        #
        self.assertEqual(len(data.get_bode_data()[0]), 18)
        self.assertEqual(len(data.get_bode_data(masked=None)[0]), 20)
        self.assertEqual(len(data.get_bode_data(masked=True)[0]), 2)
        self.assertEqual(len(data.get_bode_data(masked=False)[0]), 18)
        # from_dict
        label: str = "label"
        data = DataSet(freq, Z, mask, path, label)
        self.assertEqual(data.get_label(), label)
        data = DataSet.from_dict(
            {
                "frequency": freq,
                "real": Z.real,
                "imaginary": Z.imag,
                "mask": mask,
                "path": path,
                "label": label,
            }
        )
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
        #
        self.assertEqual(len(data.get_nyquist_data()[0]), 18)
        self.assertEqual(len(data.get_nyquist_data(masked=None)[0]), 20)
        self.assertEqual(len(data.get_nyquist_data(masked=True)[0]), 2)
        self.assertEqual(len(data.get_nyquist_data(masked=False)[0]), 18)
        #
        self.assertEqual(len(data.get_bode_data()[0]), 18)
        self.assertEqual(len(data.get_bode_data(masked=None)[0]), 20)
        self.assertEqual(len(data.get_bode_data(masked=True)[0]), 2)
        self.assertEqual(len(data.get_bode_data(masked=False)[0]), 18)

    def test_02_subtract_impedance(self):
        freq: ndarray = array(list(reversed(range(1, 6))))
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
        #
        self.assertEqual(len(data.get_nyquist_data()), 2)
        self.assertEqual(data.get_nyquist_data()[0][0], 1)
        self.assertEqual(data.get_nyquist_data()[1][0], 1)
        #
        self.assertEqual(len(data.get_bode_data()), 3)
        self.assertEqual(data.get_bode_data()[0][0], 1)
        self.assertEqual(data.get_bode_data()[1][0], abs(complex(1, -1)))
        self.assertEqual(data.get_bode_data()[2][0], 45)

    def test_06_to_dict(self):
        path: str = "/test/path/file.ext"
        label: str = "testing"
        data: DataSet = DataSet(
            array([1]), array([complex(1, -1)]), {0: True}, path, label
        )
        dictionary: dict = data.to_dict()
        self.assertTrue("version" in dictionary, True)
        self.assertTrue("path" in dictionary, True)
        self.assertTrue("label" in dictionary, True)
        self.assertTrue("frequency" in dictionary, True)
        self.assertTrue("real" in dictionary, True)
        self.assertTrue("imaginary" in dictionary, True)
        self.assertTrue("mask" in dictionary, True)
        #
        self.assertEqual(type(dictionary["version"]), int)
        self.assertEqual(type(dictionary["path"]), str)
        self.assertEqual(type(dictionary["label"]), str)
        self.assertEqual(type(dictionary["frequency"]), list)
        self.assertEqual(type(dictionary["real"]), list)
        self.assertEqual(type(dictionary["imaginary"]), list)
        self.assertEqual(type(dictionary["mask"]), dict)
        #
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


class TestFormatParsers(TestCase):
    def validate(self, data: DataSet, control: DataSet, atol: float = 1e-8):
        self.assertEqual(control.get_num_points(), data.get_num_points())
        self.assertTrue(
            allclose(control.get_frequency(), data.get_frequency(), atol=atol)
        )
        self.assertTrue(
            allclose(control.get_impedance(), data.get_impedance(), atol=atol)
        )

    def test_01_csv(self):
        control: DataSet = get_control_data()
        path: str
        for path in get_test_files(".csv"):
            data: DataSet
            for data in parse_data(path):
                self.validate(data, control)

    def test_02_i2b(self):
        control: DataSet = get_control_data()
        paths: List[str] = get_test_files(".i2b")
        self.assertTrue(len(paths) > 0)
        path: str
        for path in paths:
            data: DataSet
            for data in parse_data(path):
                self.validate(data, control)

    def test_03_xlsx(self):
        control: DataSet = get_control_data()
        paths: List[str] = get_test_files(".xlsx")
        self.assertTrue(len(paths) > 0)
        path: str
        for path in paths:
            data: DataSet
            for data in parse_data(path):
                self.validate(data, control)

    def test_04_ods(self):
        control: DataSet = get_control_data()
        paths: List[str] = get_test_files(".ods")
        self.assertTrue(len(paths) > 0)
        path: str
        for path in paths:
            data: DataSet
            for data in parse_data(path):
                self.validate(data, control)

    def test_05_dta(self):
        control: DataSet = get_control_data()
        paths: List[str] = get_test_files(".dta")
        self.assertTrue(len(paths) > 0)
        path: str
        for path in paths:
            data: DataSet
            for data in parse_data(path):
                self.validate(data, control)

    def test_06_idf(self):
        control: DataSet = get_control_data()
        paths: List[str] = get_test_files(".idf")
        self.assertTrue(len(paths) > 0)
        path: str
        for path in paths:
            data: DataSet
            for data in parse_data(path):
                self.validate(data, control)

    def test_07_ids(self):
        control: DataSet = get_control_data()
        paths: List[str] = get_test_files(".ids")
        self.assertTrue(len(paths) > 0)
        path: str
        for path in paths:
            data: DataSet
            for data in parse_data(path):
                self.validate(data, control)

    def test_08_no_extension(self):
        control: DataSet = get_control_data()
        paths: List[str] = get_test_files("")
        self.assertTrue(len(paths) > 0)
        path: str
        for path in paths:
            data: DataSet
            for data in parse_data(path):
                self.validate(data, control, atol=1e-1)

    def test_09_dfr(self):
        control: DataSet = get_control_data()
        paths: List[str] = get_test_files(".dfr")
        self.assertTrue(len(paths) > 0)
        path: str
        for path in paths:
            data: DataSet
            for data in parse_data(path):
                self.validate(data, control)

    def test_10_p00(self):
        control: DataSet = get_control_data()
        paths: List[str] = get_test_files(".P00")
        self.assertTrue(len(paths) > 0)
        path: str
        for path in paths:
            data: DataSet
            for data in parse_data(path):
                self.validate(data, control, atol=1e-1)

    def test_11_mpt(self):
        control: DataSet = get_control_data()
        paths: List[str] = get_test_files(".mpt")
        self.assertTrue(len(paths) > 0)
        path: str
        for path in paths:
            data: DataSet
            for data in parse_data(path):
                self.validate(data, control, atol=1e-1)
