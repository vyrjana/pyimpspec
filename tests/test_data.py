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

from pathlib import Path
from unittest import TestCase
from json import dumps, loads
from pyimpspec import (
    DataSet,
    parse_data,
)
from pyimpspec.data.data_set import VERSION
from pyimpspec.typing import (
    ComplexImpedance,
    ComplexImpedances,
    Frequency,
    Frequencies,
)
from numpy import (
    allclose,
    array,
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
from pandas import DataFrame
from test_matplotlib import (
    UNFILLED_MARKERS,
    check_mpl_return_values,
    mpl,
    primitive_mpl_plotters,
)


def get_control_data() -> DataSet:
    return DataSet.from_dict(
        {
            "version": 1,
            "label": "Control",
            "frequency": [
                10000.0,
                7196.85673001152,
                5179.47467923121,
                3727.59372031494,
                2682.69579527973,
                1930.69772888325,
                1389.49549437314,
                1000.0,
                719.685673001152,
                517.947467923121,
                372.759372031494,
                268.269579527973,
                193.069772888325,
                138.949549437314,
                100.0,
                71.9685673001152,
                51.7947467923121,
                37.2759372031494,
                26.8269579527973,
                19.3069772888325,
                13.8949549437314,
                10.0,
                7.19685673001152,
                5.17947467923121,
                3.72759372031494,
                2.68269579527973,
                1.93069772888325,
                1.38949549437314,
                1.0,
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

    files = list(filter(lambda _: _.startswith("data"), files))
    assert len(files) > 0

    files.sort()
    files = list(map(lambda _: join(root, _), files))
    if extension is None:
        return files

    files = list(filter(lambda _: splitext(_)[1] == extension, files))
    assert len(files) > 0

    return files


class TestDataSet(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.f: Frequencies = array(
            list(range(1, 21)),
            dtype=Frequency,
        )
        cls.Z: ComplexImpedances = array(
            list(
                map(
                    lambda _: complex(_, -_),
                    range(1, 21),
                )
            ),
            dtype=ComplexImpedance,
        )
        cls.mask: Dict[int, bool] = {
            2: True,
            7: True,
            15: False,
        }
        cls.path: str = "/test/path/file.ext"
        cls.data: DataSet = DataSet(cls.f, cls.Z, cls.mask, cls.path)

    def test_constructor(self):
        with self.assertRaises(ValueError):
            DataSet(array([]), array([]))

        DataSet([1], [complex(1, -1)])

    def test_get_set_path(self):
        self.assertEqual(self.data.get_path(), self.path)
        self.data.set_path("test")
        self.assertEqual(self.data.get_path(), "test")
        self.data.set_path(self.path)

    def test_unique_frequencies(self):
        with self.assertRaises(ValueError):
            DataSet([1, 1], [complex(1, -1)] * 2)

    def test_frequency_order(self):
        f: Frequencies = self.data.get_frequencies()
        self.assertGreater(f[0], f[-1])
        self.assertEqual(f[0], self.f[-1])
        self.assertEqual(f[-1], self.f[0])

    def test_get_label(self):
        self.assertEqual(self.data.get_label(), splitext(basename(self.path))[0])

    def test_get_num_points(self):
        self.assertEqual(self.data.get_num_points(), 18)
        self.assertEqual(self.data.get_num_points(masked=None), 20)
        self.assertEqual(self.data.get_num_points(masked=True), 2)
        self.assertEqual(self.data.get_num_points(masked=False), 18)

    def test_getters(self):
        methods: List[Callable] = [
            self.data.get_frequencies,
            self.data.get_impedances,
            self.data.get_magnitudes,
            self.data.get_phases,
        ]
        method: Callable
        for method in methods:
            self.assertEqual(len(method()), 18)
            self.assertEqual(len(method(masked=None)), 20)
            self.assertEqual(len(method(masked=True)), 2)
            self.assertEqual(len(method(masked=False)), 18)
        # Nyquist data
        self.assertEqual(len(self.data.get_nyquist_data()[0]), 18)
        self.assertEqual(len(self.data.get_nyquist_data(masked=None)[0]), 20)
        self.assertEqual(len(self.data.get_nyquist_data(masked=True)[0]), 2)
        self.assertEqual(len(self.data.get_nyquist_data(masked=False)[0]), 18)
        # Bode data
        self.assertEqual(len(self.data.get_bode_data()[0]), 18)
        self.assertEqual(len(self.data.get_bode_data(masked=None)[0]), 20)
        self.assertEqual(len(self.data.get_bode_data(masked=True)[0]), 2)
        self.assertEqual(len(self.data.get_bode_data(masked=False)[0]), 18)

    def test_from_dict(self):
        label: str = "label"
        dictionaries: List[dict] = [
            {
                "version": 1,
                "frequency": self.f,
                "real": self.Z.real,
                "imaginary": self.Z.imag,
                "mask": self.mask,
                "path": self.path,
                "label": label,
            },
            {
                "version": 2,
                "frequencies": self.f,
                "real_impedances": self.Z.real,
                "imaginary_impedances": self.Z.imag,
                "mask": self.mask,
                "path": self.path,
                "label": label,
            },
        ]
        d: dict
        for d in dictionaries:
            data: DataSet = DataSet.from_dict(d)
            self.assertEqual(data.get_path(), self.path)
            self.assertEqual(data.get_label(), label)
            self.assertEqual(data.get_num_points(), 18)
            self.assertEqual(data.get_num_points(masked=None), 20)
            self.assertEqual(data.get_num_points(masked=True), 2)
            self.assertEqual(data.get_num_points(masked=False), 18)
            json: str = dumps(data.to_dict())
            d = loads(json)
            data = DataSet.from_dict(d)
            self.assertEqual(data.get_path(), self.path)
            self.assertEqual(data.get_label(), label)
            self.assertEqual(data.get_num_points(), 18)
            self.assertEqual(data.get_num_points(masked=None), 20)
            self.assertEqual(data.get_num_points(masked=True), 2)
            self.assertEqual(data.get_num_points(masked=False), 18)

    def test_subtract_impedances(self):
        freq: Frequencies = array(list(map(float, reversed(range(1, 6)))))
        Z: ComplexImpedances = array(list(map(lambda _: complex(_, -_), range(1, 6))))
        data: DataSet = DataSet(freq, Z)

        i: int
        z: complex
        for i, z in enumerate(data.get_impedances()):
            self.assertEqual(z, Z[i])

        data.subtract_impedances(array(complex(1, -1)))
        for i, z in enumerate(data.get_impedances()):
            self.assertEqual(z, Z[i] - complex(1, -1))

    def test_get_set_label(self):
        data: DataSet = DataSet(array([1.0]), array([complex(1, -1)]))
        self.assertEqual(data.get_label(), "")
        label: str = "testing"
        data.set_label(label)
        self.assertEqual(data.get_label(), label)

    def test_get_set_mask(self):
        data: DataSet = DataSet(array([1.0]), array([complex(1, -1)]))

        data.set_mask({0: True})
        self.assertEqual(len(data.get_mask()), 1)
        self.assertTrue(any(data.get_mask().values()))
        self.assertTrue(data.get_mask()[0])

        data = DataSet.duplicate(get_control_data())
        self.assertEqual(len(data.get_mask()), 29)
        self.assertTrue(not any(data.get_mask().values()))

        data = DataSet(
            data.get_frequencies(),
            data.get_impedances(),
            mask={-1: True, 30: True},
        )
        self.assertEqual(len(data.get_mask()), 29)
        with self.assertRaises(KeyError):
            data.get_mask()[-1]
        with self.assertRaises(KeyError):
            data.get_mask()[30]
        self.assertTrue(not any(data.get_mask().values()))

        data = DataSet(
            data.get_frequencies(),
            data.get_impedances(),
            mask={5: True, 6: False},
        )
        self.assertEqual(len(data.get_mask()), 29)
        self.assertTrue(any(data.get_mask().values()))
        self.assertEqual(data.get_mask()[5], True)
        self.assertEqual(data.get_mask()[6], False)

    def test_get_values(self):
        data: DataSet = DataSet(array([1.0]), array([complex(1, -1)]))
        self.assertEqual(data.get_frequencies()[0], 1)

        self.assertEqual(data.get_impedances()[0].real, 1)
        self.assertEqual(data.get_impedances()[0].imag, -1)

        self.assertEqual(data.get_magnitudes()[0], abs(complex(1, -1)))
        self.assertEqual(data.get_phases()[0], -45)

        self.assertEqual(len(data.get_nyquist_data()), 2)
        self.assertEqual(data.get_nyquist_data()[0][0], 1)
        self.assertEqual(data.get_nyquist_data()[1][0], 1)

        self.assertEqual(len(data.get_bode_data()), 3)
        self.assertEqual(data.get_bode_data()[0][0], 1)
        self.assertEqual(data.get_bode_data()[1][0], abs(complex(1, -1)))
        self.assertEqual(data.get_bode_data()[2][0], 45)

    def test_to_dict(self):
        path: str = "/test/path/file.ext"
        label: str = "testing"
        data: DataSet = DataSet(
            array([1.0]),
            array([complex(1, -1)]),
            {0: True},
            path,
            label,
        )
        dictionary: dict = data.to_dict()
        self.assertTrue("version" in dictionary, True)
        self.assertTrue("path" in dictionary, True)
        self.assertTrue("label" in dictionary, True)
        self.assertTrue("frequencies" in dictionary, True)
        self.assertTrue("real_impedances" in dictionary, True)
        self.assertTrue("imaginary_impedances" in dictionary, True)
        self.assertTrue("mask" in dictionary, True)

        self.assertIsInstance(dictionary["version"], int)
        self.assertIsInstance(dictionary["path"], str)
        self.assertIsInstance(dictionary["label"], str)
        self.assertIsInstance(dictionary["frequencies"], list)
        self.assertIsInstance(dictionary["real_impedances"], list)
        self.assertIsInstance(dictionary["imaginary_impedances"], list)
        self.assertIsInstance(dictionary["mask"], dict)

        self.assertEqual(dictionary["version"], VERSION)
        self.assertEqual(dictionary["path"], path)
        self.assertEqual(dictionary["label"], label)
        self.assertEqual(dictionary["frequencies"][0], 1)
        self.assertEqual(dictionary["real_impedances"][0], 1)
        self.assertEqual(dictionary["imaginary_impedances"][0], -1)
        self.assertEqual(dictionary["mask"][0], True)

        json: str = dumps(dictionary)

    def test_duplicate(self):
        data: DataSet = DataSet.duplicate(get_control_data())
        self.assertEqual(data.get_label(), "Control")
        data = DataSet.duplicate(get_control_data(), label="test")
        self.assertEqual(data.get_label(), "test")

    def test_average(self):
        control: DataSet = get_control_data()

        data_1: DataSet = get_control_data()
        data_1.subtract_impedances(array(complex(-1, -2)))
        self.assertFalse(allclose(control.get_impedances(), data_1.get_impedances()))

        data_2: DataSet = get_control_data()
        data_2.subtract_impedances(array(complex(1, 2)))
        self.assertFalse(allclose(control.get_impedances(), data_2.get_impedances()))

        average: DataSet = DataSet.average([data_1, data_2])
        self.assertTrue(allclose(control.get_impedances(), average.get_impedances()))

    def test_to_dataframe(self):
        data: DataSet = get_control_data()

        df: DataFrame = data.to_dataframe()
        self.assertIsInstance(df, DataFrame)
        self.assertEqual(len(df), 29)

        data.set_mask({5: True, 7: True})
        df = data.to_dataframe(masked=None)
        self.assertEqual(len(df), 29)

        df = data.to_dataframe(masked=True)
        self.assertEqual(len(df), 2)

        df = data.to_dataframe(masked=False)
        self.assertEqual(len(df), 27)

        df = data.to_dataframe(
            columns=[
                "test1",
                "test2",
                "test3",
                "test4",
                "test5",
            ]
        )

        i: int
        label: str
        for i, label in enumerate(df.columns, start=1):
            self.assertEqual(label, f"test{i}")

        self.assertTrue(df[df.columns[-3]][0] < 0.0)
        self.assertTrue(df[df.columns[-1]][0] < 0.0)

        df = data.to_dataframe(negative_imaginary=True, negative_phase=True)
        self.assertTrue(df[df.columns[-3]][0] > 0.0)
        self.assertTrue(df[df.columns[-1]][0] > 0.0)

    def test_repr(self):
        data: DataSet = DataSet.duplicate(get_control_data())
        self.assertEqual(repr(data), f"DataSet ({data.get_label()}, {hex(id(data))})")

    def test_low_pass(self):
        data: DataSet = get_control_data()
        data.set_mask({15: True})
        data.low_pass(1000.0)
        self.assertEqual(data.get_num_points(), 21)

    def test_high_pass(self):
        data: DataSet = get_control_data()
        data.set_mask({5: True})
        data.high_pass(1000.0)
        self.assertEqual(data.get_num_points(), 7)

    def test_matplotlib(self):
        data: DataSet = self.data
        plotter: Callable
        for plotter in primitive_mpl_plotters:
            check_mpl_return_values(self, *plotter(data=None))
            check_mpl_return_values(self, *plotter(data=data))
            check_mpl_return_values(self, *plotter(data=data, colored_axes=True))

        check_mpl_return_values(self, *mpl.plot_data(data=data))
        check_mpl_return_values(
            self,
            *mpl.plot_magnitude(
                data=data,
                marker=next(
                    iter(UNFILLED_MARKERS),
                ),
            ),
        )


class TestFormatParsers(TestCase):
    def validate(self, data: DataSet, control: DataSet, atol: float = 1e-8):
        self.assertEqual(control.get_num_points(), data.get_num_points())
        self.assertTrue(
            allclose(control.get_frequencies(), data.get_frequencies(), atol=atol)
        )
        self.assertTrue(
            allclose(control.get_impedances(), data.get_impedances(), atol=atol)
        )

    def test_pathlib_path(self):
        control: DataSet = get_control_data()

        path: Path = Path(__file__).parent
        path = path.joinpath("data.mpt")

        data: DataSet
        for data in parse_data(path):
            self.validate(data, control)

    def test_csv(self):
        control: DataSet = get_control_data()

        path: str
        data: DataSet
        for path in get_test_files(".csv"):
            for data in parse_data(path):
                self.validate(data, control)

        for data in parse_data(Path(path)):
            self.validate(data, control)

        self.validate(parse_data(path, file_format="CSV")[0], control)
        data_sets: List[DataSet] = parse_data("./case-multiple-spectra.csv")
        self.assertEqual(len(data_sets), 4)
        self.assertTrue(all(map(lambda d: len(d.get_frequencies()) == 29, data_sets)))

        for data in parse_data("./case-multiple-spectra.csv"):
            self.validate(data, control, atol=1e-1)

        self.assertEqual(len(parse_data("./case-multiple-spectra-reverse.csv")), 2)

        for data in parse_data("./case-multiple-spectra-reverse.csv"):
            self.validate(data, control, atol=1e-1)

        self.assertEqual(len(parse_data("./case-multiple-spectra-reverse.csv")), 2)

        for data in parse_data("./case-negative-real-imaginary.csv"):
            self.validate(data, control, atol=1e-1)

    def test_i2b(self):
        control: DataSet = get_control_data()

        paths: List[str] = get_test_files(".i2b")
        self.assertTrue(len(paths) > 0)

        path: str
        for path in paths:
            data: DataSet
            for data in parse_data(path):
                self.validate(data, control)

        for data in parse_data(Path(path)):
            self.validate(data, control)

    def test_xlsx(self):
        control: DataSet = get_control_data()

        paths: List[str] = get_test_files(".xlsx")
        self.assertTrue(len(paths) > 0)

        path: str
        for path in paths:
            data: DataSet
            for data in parse_data(path):
                self.validate(data, control)

        for data in parse_data(Path(path)):
            self.validate(data, control)

    def test_ods(self):
        control: DataSet = get_control_data()

        paths: List[str] = get_test_files(".ods")
        self.assertTrue(len(paths) > 0)

        path: str
        for path in paths:
            data: DataSet
            for data in parse_data(path):
                self.validate(data, control)

        for data in parse_data(Path(path)):
            self.validate(data, control)

    def test_dta(self):
        control: DataSet = get_control_data()

        paths: List[str] = get_test_files(".dta")
        self.assertTrue(len(paths) > 0)

        path: str
        for path in paths:
            data: DataSet
            for data in parse_data(path):
                self.validate(data, control)

        for data in parse_data(Path(path)):
            self.validate(data, control)

        path = join(dirname(__file__), "drift-corrected-data.dta")
        data_sets: List[DataSet] = parse_data(path)

        self.assertIsInstance(data_sets, list)
        self.assertEqual(len(data_sets), 2)
        self.assertTrue(all(map(lambda data: isinstance(data, DataSet), data_sets)))

        self.assertTrue("corrected" in data_sets[0].get_label())
        self.assertTrue("uncorrected" in data_sets[1].get_label())

        self.assertTrue(
            allclose(
                data_sets[0].get_frequencies(),
                data_sets[1].get_frequencies(),
            )
        )

        Z_corrected: ComplexImpedances = data_sets[0].get_impedances()
        Z_uncorrected: ComplexImpedances = data_sets[1].get_impedances()
        self.assertFalse(allclose(Z_corrected.real, Z_uncorrected.real))
        self.assertFalse(allclose(Z_corrected.imag, Z_uncorrected.imag))

    def test_idf(self):
        control: DataSet = get_control_data()

        paths: List[str] = get_test_files(".idf")
        self.assertTrue(len(paths) > 0)

        path: str
        for path in paths:
            data: DataSet
            for data in parse_data(path):
                self.validate(data, control)

        for data in parse_data(Path(path)):
            self.validate(data, control)

    def test_ids(self):
        control: DataSet = get_control_data()

        paths: List[str] = get_test_files(".ids")
        self.assertTrue(len(paths) > 0)

        path: str
        for path in paths:
            data: DataSet
            for data in parse_data(path):
                self.validate(data, control)

        for data in parse_data(Path(path)):
            self.validate(data, control)

    def test_no_extension(self):
        control: DataSet = get_control_data()

        paths: List[str] = get_test_files("")
        self.assertTrue(len(paths) > 0)

        path: str
        for path in paths:
            data: DataSet
            for data in parse_data(path):
                self.validate(data, control, atol=1e-1)

        for data in parse_data(Path(path)):
            self.validate(data, control, atol=1e-1)

    def test_dfr(self):
        control: DataSet = get_control_data()

        paths: List[str] = get_test_files(".dfr")
        self.assertTrue(len(paths) > 0)

        path: str
        for path in paths:
            data: DataSet
            for data in parse_data(path):
                self.validate(data, control)

        for data in parse_data(Path(path)):
            self.validate(data, control)

    def test_p00(self):
        control: DataSet = get_control_data()

        paths: List[str] = get_test_files(".P00")
        self.assertTrue(len(paths) > 0)

        path: str
        for path in paths:
            data: DataSet
            for data in parse_data(path):
                self.validate(data, control, atol=1e-1)

        for data in parse_data(Path(path)):
            self.validate(data, control, atol=1e-1)

    def test_mpt(self):
        control: DataSet = get_control_data()

        paths: List[str] = get_test_files(".mpt")
        self.assertTrue(len(paths) > 0)

        data: DataSet
        path: str
        for path in paths:
            for data in parse_data(path):
                self.validate(data, control, atol=1e-1)

        for data in parse_data(Path(path)):
            self.validate(data, control, atol=1e-1)

        self.assertEqual(len(parse_data("./case-multiple-spectra.mpt")), 3)

        for data in parse_data("./case-multiple-spectra.mpt"):
            self.validate(data, control, atol=1e-1)

    def test_z(self):
        control: DataSet = get_control_data()

        paths: List[str] = get_test_files(".z")
        self.assertTrue(len(paths) > 0)

        data: DataSet
        path: str
        for path in paths:
            for data in parse_data(path):
                self.validate(data, control)

        for data in parse_data(Path(path)):
            self.validate(data, control)

    def test_pssession(self):
        control: DataSet = get_control_data()

        paths: List[str] = get_test_files(".pssession")
        self.assertTrue(len(paths) > 0)

        data: DataSet
        path: str
        for path in paths:
            for data in parse_data(path):
                self.validate(data, control)

        for data in parse_data(Path(path)):
            self.validate(data, control)
