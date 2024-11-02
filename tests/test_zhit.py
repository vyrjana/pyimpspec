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

from unittest import TestCase
from numpy import allclose
from pyimpspec import (
    ComplexImpedances,
    ComplexResiduals,
    Frequencies,
    ZHITResult,
    perform_zhit,
    generate_mock_data,
)
from pyimpspec.analysis.utility import _calculate_residuals


DATA = generate_mock_data("CIRCUIT_2_INVALID", seed=42, noise=5e-2)[0]


class ZHIT(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result: ZHITResult = perform_zhit(
            DATA,
            smoothing="lowess",
            interpolation="akima",
            window="hann",
        )

    def test_frequencies(self):
        f_exp: Frequencies = DATA.get_frequencies()
        f: Frequencies = self.result.get_frequencies()
        self.assertEqual(f_exp.shape, f.shape)
        self.assertTrue(allclose(f_exp, f))

    def test_impedances(self):
        Z_exp: ComplexImpedances = DATA.get_impedances()
        Z: ComplexImpedances = self.result.get_impedances()
        self.assertEqual(Z_exp.shape, Z.shape)

    def test_residuals(self):
        Z_exp: ComplexImpedances = DATA.get_impedances()
        Z: ComplexImpedances = self.result.get_impedances()
        residuals: ComplexResiduals = self.result.residuals
        self.assertEqual(Z_exp.shape, residuals.shape)
        self.assertTrue(allclose(residuals, _calculate_residuals(Z_exp=Z_exp, Z_fit=Z)))

    def test_pseudo_chisqr(self):
        self.assertAlmostEqual(self.result.pseudo_chisqr, 0.012, delta=1e-3)

    def test_default(self):
        zhit: ZHITResult = perform_zhit(DATA)
        self.assertIsInstance(zhit, ZHITResult)
        self.assertEqual(zhit.smoothing, "modsinc")
        self.assertEqual(zhit.interpolation, "makima")
        self.assertNotEqual(zhit.window, "auto")

    def test_no_smoothing(self):
        zhit: ZHITResult = perform_zhit(
            DATA,
            smoothing="none",
            interpolation="cubic",
            window="boxcar",
        )
        self.assertIsInstance(zhit, ZHITResult)
        self.assertEqual(zhit.smoothing, "none")
        self.assertEqual(zhit.interpolation, "cubic")
        self.assertEqual(zhit.window, "boxcar")

    def test_auto_smoothing(self):
        zhit: ZHITResult = perform_zhit(
            DATA,
            smoothing="auto",
            interpolation="cubic",
            window="boxcar",
        )
        self.assertIsInstance(zhit, ZHITResult)
        self.assertNotEqual(zhit.smoothing, "auto")
        self.assertEqual(zhit.interpolation, "cubic")
        self.assertEqual(zhit.window, "boxcar")

    def test_auto_interpolation(self):
        zhit: ZHITResult = perform_zhit(
            DATA,
            smoothing="savgol",
            interpolation="auto",
            window="triang",
        )
        self.assertIsInstance(zhit, ZHITResult)
        self.assertEqual(zhit.smoothing, "savgol")
        self.assertNotEqual(zhit.interpolation, "auto")
        self.assertEqual(zhit.window, "triang")

    def test_auto_window(self):
        zhit: ZHITResult = perform_zhit(
            DATA,
            smoothing="savgol",
            interpolation="pchip",
            window="auto",
        )
        self.assertIsInstance(zhit, ZHITResult)
        self.assertEqual(zhit.smoothing, "savgol")
        self.assertEqual(zhit.interpolation, "pchip")
        self.assertNotEqual(zhit.window, "auto")

    def test_auto_smoothing_interpolation(self):
        zhit: ZHITResult = perform_zhit(
            DATA,
            smoothing="auto",
            interpolation="auto",
            window="hamming",
        )
        self.assertIsInstance(zhit, ZHITResult)
        self.assertNotEqual(zhit.smoothing, "auto")
        self.assertNotEqual(zhit.interpolation, "auto")
        self.assertEqual(zhit.window, "hamming")

    def test_auto_smoothing_window(self):
        zhit: ZHITResult = perform_zhit(
            DATA,
            smoothing="auto",
            interpolation="akima",
            window="auto",
        )
        self.assertIsInstance(zhit, ZHITResult)
        self.assertNotEqual(zhit.smoothing, "auto")
        self.assertEqual(zhit.interpolation, "akima")
        self.assertNotEqual(zhit.window, "auto")

    def test_auto_interpolation_window(self):
        zhit: ZHITResult = perform_zhit(
            DATA,
            smoothing="savgol",
            interpolation="auto",
            window="auto",
        )
        self.assertIsInstance(zhit, ZHITResult)
        self.assertEqual(zhit.smoothing, "savgol")
        self.assertNotEqual(zhit.interpolation, "auto")
        self.assertNotEqual(zhit.window, "auto")

    def test_auto_smoothing_interpolation_window(self):
        zhit: ZHITResult = perform_zhit(
            DATA,
            smoothing="auto",
            interpolation="auto",
            window="auto",
        )
        self.assertIsInstance(zhit, ZHITResult)
        self.assertNotEqual(zhit.smoothing, "auto")
        self.assertNotEqual(zhit.interpolation, "auto")
        self.assertNotEqual(zhit.window, "auto")
