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

from typing import (
    Callable,
    Dict,
    Optional,
)
from unittest import TestCase
from numpy import (
    allclose,
    angle,
    ndarray,
)
from numpy.random import (
    seed,
)
from pyimpspec import (
    Circuit,
    DRTError,
    DRTResult,
    DataSet,
    calculate_drt,
    fit_circuit,
    parse_cdc,
    parse_data,
)
from pyimpspec.analysis.drt import _METHODS as METHODS
from pyimpspec.analysis.drt.tr_rbf import (
    MODES,
    RBF_SHAPES,
    RBF_TYPES,
)
import pyimpspec


def progress_callback(*args, **kwargs):
    message: Optional[str] = kwargs.get("message")
    assert message is not None
    assert type(message) is str, message
    assert message.strip() != ""
    progress: Optional[float] = kwargs.get("progress")
    assert type(progress) is float or progress is None, progress
    if type(progress) is float:
        assert 0.0 <= progress <= 1.0, progress
    # print(args, kwargs)


pyimpspec.progress.register(progress_callback)


seed(28041948)  # GNU STP
DATA: DataSet = parse_data("data.dta")[0]


def drt_func(*args, **kwargs) -> DRTResult:
    kwargs["num_procs"] = 1
    return calculate_drt(DATA, *args, **kwargs)


class TestDRT(TestCase):
    @classmethod
    def setUpClass(cls):
        cls._method_results: Dict[str, DRTResult] = {}
        cls._f_exp: ndarray = DATA.get_frequency()
        cls._Z_exp: ndarray = DATA.get_impedance()
        cls._circuit: Circuit = fit_circuit(parse_cdc("R(RQ)(RQ)"), DATA).circuit

    def test_01_data(self):
        self.assertRaises(AssertionError, calculate_drt, {})

    def test_02_methods(self):
        wrapper: Callable = drt_func
        self.assertRaises(AssertionError, wrapper, method="crivens")
        method: str
        for method in METHODS:
            drt: DRTResult = wrapper(
                method=method,
                circuit=self._circuit if method == "m(RQ)fit" else None,
            )
            self._method_results[method] = drt

    def test_03_result_label(self):
        method: str
        drt: DRTResult
        for method, drt in self._method_results.items():
            self.assertTrue(drt.get_label().strip() != "")
            if method == "m(RQ)fit":
                self.assertEqual(drt.get_label(), "R-2(RQ)")
            else:
                self.assertTrue(method.upper() in drt.get_label())

    def test_04_result_get_frequency(self):
        method: str
        drt: DRTResult
        for method, drt in self._method_results.items():
            self.assertEqual(len(drt.get_frequency()), 29)
            f_mod: ndarray = drt.get_frequency()
            self.assertEqual(type(f_mod), ndarray)
            self.assertEqual(self._f_exp.shape, f_mod.shape)
            self.assertTrue(allclose(self._f_exp, f_mod))

    def test_05_result_get_impedance(self):
        method: str
        drt: DRTResult
        for method, drt in self._method_results.items():
            self.assertEqual(len(drt.get_impedance()), 29)
            Z_mod: ndarray = drt.get_impedance()
            self.assertEqual(type(Z_mod), ndarray)
            self.assertEqual(self._Z_exp.shape, Z_mod.shape)

    def test_06_result_get_nyquist_data(self):
        method: str
        drt: DRTResult
        for method, drt in self._method_results.items():
            Z_mod: ndarray = drt.get_impedance()
            self.assertTrue(allclose(Z_mod.real, drt.get_nyquist_data()[0]))
            self.assertTrue(allclose(-Z_mod.imag, drt.get_nyquist_data()[1]))

    def test_07_result_get_bode_data(self):
        method: str
        drt: DRTResult
        for method, drt in self._method_results.items():
            f_mod: ndarray = drt.get_frequency()
            Z_mod: ndarray = drt.get_impedance()
            self.assertTrue(allclose(f_mod, drt.get_bode_data()[0]))
            self.assertTrue(allclose(abs(Z_mod), drt.get_bode_data()[1]))
            self.assertTrue(allclose(-angle(Z_mod, deg=True), drt.get_bode_data()[2]))

    def test_08_result_get_tau(self):
        method: str
        drt: DRTResult
        for method, drt in self._method_results.items():
            tau: ndarray = drt.get_tau()
            self.assertEqual(type(tau), ndarray)

    def test_09_result_get_gamma(self):
        method: str
        drt: DRTResult
        for method, drt in self._method_results.items():
            gamma: ndarray = drt.get_gamma()
            self.assertEqual(type(gamma), ndarray)

    def test_10_result_get_drt_data(self):
        method: str
        drt: DRTResult
        for method, drt in self._method_results.items():
            self.assertEqual(len(drt.get_drt_data()), 2)
            tau: ndarray
            gamma: ndarray
            tau, gamma = drt.get_drt_data()
            self.assertEqual(type(tau), ndarray)
            self.assertEqual(type(gamma), ndarray)
            self.assertEqual(tau.shape, gamma.shape)
            self.assertTrue(tau.size >= self._f_exp.size)
            self.assertTrue(gamma.size >= self._f_exp.size)
            self.assertTrue(allclose(tau, drt.get_tau()))
            self.assertTrue(allclose(gamma, drt.get_gamma()))
            if method == "bht":
                self.assertEqual(gamma.size, drt.get_gamma(imaginary=True).size)
                self.assertEqual(tau.size, drt.get_drt_data(imaginary=True)[0].size)
                self.assertEqual(gamma.size, drt.get_drt_data(imaginary=True)[1].size)
            else:
                self.assertEqual(drt.get_gamma(imaginary=True).size, 0)
                self.assertEqual(drt.get_drt_data(imaginary=True)[0].size, 0)
                self.assertEqual(drt.get_drt_data(imaginary=True)[1].size, 0)

    def test_11_result_scores(self):
        method: str
        drt: DRTResult
        for method, drt in self._method_results.items():
            scores: Dict[str, complex] = drt.get_scores()
            self.assertEqual(type(scores), dict)
            if method == "bht":
                self.assertEqual(len(scores), 6)
                self.assertTrue(all(map(lambda _: type(_) is str, scores.keys())))
                self.assertTrue(all(map(lambda _: type(_) is complex, scores.values())))
                self.assertTrue("mean" in scores)
                self.assertTrue("residuals_1sigma" in scores)
                self.assertTrue("residuals_2sigma" in scores)
                self.assertTrue("residuals_3sigma" in scores)
                self.assertTrue("hellinger_distance" in scores)
                self.assertTrue("jensen_shannon_distance" in scores)
                # Range - real
                self.assertTrue(0.0 <= scores["mean"].real <= 1.0)
                self.assertTrue(0.0 <= scores["residuals_1sigma"].real <= 1.0)
                self.assertTrue(0.0 <= scores["residuals_2sigma"].real <= 1.0)
                self.assertTrue(0.0 <= scores["residuals_3sigma"].real <= 1.0)
                self.assertTrue(0.0 <= scores["hellinger_distance"].real <= 1.0)
                self.assertTrue(0.0 <= scores["jensen_shannon_distance"].real <= 1.0)
                # Range - imaginary
                self.assertTrue(0.0 <= scores["mean"].imag <= 1.0)
                self.assertTrue(0.0 <= scores["residuals_1sigma"].imag <= 1.0)
                self.assertTrue(0.0 <= scores["residuals_2sigma"].imag <= 1.0)
                self.assertTrue(0.0 <= scores["residuals_3sigma"].imag <= 1.0)
                self.assertTrue(0.0 <= scores["hellinger_distance"].imag <= 1.0)
                self.assertTrue(0.0 <= scores["jensen_shannon_distance"].imag <= 1.0)
            else:
                self.assertEqual(len(scores), 0)

    def test_12_result_get_drt_credible_intervals(self):
        method: str
        drt: DRTResult
        for method, drt in self._method_results.items():
            tau: ndarray
            mean_gamma: ndarray
            lower_bound: ndarray
            upper_bound: ndarray
            tau, mean_gamma, lower_bound, upper_bound = drt.get_drt_credible_intervals()
            self.assertEqual(tau.size, 0)
            self.assertEqual(tau.size, mean_gamma.size)
            self.assertEqual(tau.size, lower_bound.size)
            self.assertEqual(tau.size, upper_bound.size)

    def test_13_get_peaks(self):
        drt: DRTResult
        tau: ndarray
        gamma: ndarray
        drt = self._method_results["tr-nnls"]
        tau, gamma = drt.get_peaks()
        self.assertGreater(tau.size, 0)
        self.assertEqual(tau.size, gamma.size)
        tau, gamma = drt.get_peaks(imaginary=True)
        self.assertEqual(tau.size, 0)
        self.assertEqual(tau.size, gamma.size)
        drt = self._method_results["tr-rbf"]
        tau, gamma = drt.get_peaks()
        self.assertGreater(tau.size, 0)
        self.assertEqual(tau.size, gamma.size)
        tau, gamma = drt.get_peaks(imaginary=True)
        self.assertEqual(tau.size, 0)
        self.assertEqual(tau.size, gamma.size)
        drt = self._method_results["bht"]
        tau, gamma = drt.get_peaks()
        self.assertGreater(tau.size, 0)
        self.assertEqual(tau.size, gamma.size)
        tau, gamma = drt.get_peaks(imaginary=True)
        self.assertGreater(tau.size, 0)
        self.assertEqual(tau.size, gamma.size)

    def test_14_method_tr_nnls(self):
        wrapper: Callable = drt_func
        method: str = "tr-nnls"
        self.assertRaises(
            AssertionError,
            wrapper,
            method=method,
            mode=0,
        )
        self.assertRaises(
            AssertionError,
            wrapper,
            method=method,
            mode="crivens",
        )
        wrapper(method="tr-nnls", mode="real")
        wrapper(method="tr-nnls", mode="imaginary")

    def test_15_method_tr_rbf(self):
        wrapper: Callable = drt_func
        method: str = "tr-rbf"
        self.assertRaises(
            AssertionError,
            wrapper,
            method=method,
            rbf_type="crivens",
        )
        self.assertRaises(
            AssertionError,
            wrapper,
            method=method,
            derivative_order=0,
        )
        self.assertRaises(
            AssertionError,
            wrapper,
            method=method,
            derivative_order=3,
        )
        self.assertRaises(
            AssertionError,
            wrapper,
            method=method,
            derivative_order=2.5,
        )
        self.assertRaises(
            AssertionError,
            wrapper,
            method=method,
            rbf_shape="crivens",
        )
        self.assertRaises(
            AssertionError,
            wrapper,
            method=method,
            inductance="crivens",
        )
        self.assertRaises(
            AssertionError,
            wrapper,
            method=method,
            credible_intervals="crivens",
        )
        self.assertRaises(
            AssertionError,
            wrapper,
            method=method,
            num_samples=0,
        )
        self.assertRaises(
            AssertionError,
            wrapper,
            method=method,
            num_samples="crivens",
        )
        self.assertRaises(
            AssertionError,
            wrapper,
            method=method,
            maximum_symmetry=-0.1,
        )
        self.assertRaises(
            AssertionError,
            wrapper,
            method=method,
            maximum_symmetry=1.1,
        )
        self.assertRaises(
            AssertionError,
            wrapper,
            method=method,
            maximum_symmetry="crivens",
        )
        mode: str
        for mode in MODES:
            wrapper(method=method, mode=mode)
        rbf_type: str
        for rbf_type in RBF_TYPES:
            wrapper(method=method, rbf_type=rbf_type)
        self.assertTrue("factor" in RBF_SHAPES)
        rbf_shape: str
        for rbf_shape in RBF_SHAPES:
            if rbf_shape == "factor":
                self.assertRaises(DRTError, wrapper, method=method, rbf_shape=rbf_shape)
            else:
                wrapper(method=method, rbf_shape=rbf_shape)

    def test_16_credible_intervals(self):
        wrapper: Callable = drt_func
        drt: DRTResult = wrapper(method="tr-rbf", credible_intervals=True)
        tau: ndarray
        mean_gamma: ndarray
        lower_bound: ndarray
        upper_bound: ndarray
        tau, mean_gamma, lower_bound, upper_bound = drt.get_drt_credible_intervals()
        self.assertEqual(type(tau), ndarray)
        self.assertEqual(type(mean_gamma), ndarray)
        self.assertEqual(type(lower_bound), ndarray)
        self.assertEqual(type(upper_bound), ndarray)
        self.assertEqual(tau.size, drt.get_tau().size)
        self.assertEqual(tau.size, mean_gamma.size)
        self.assertEqual(tau.size, lower_bound.size)
        self.assertEqual(tau.size, upper_bound.size)

    def test_17_method_bht(self):
        wrapper: Callable = drt_func
        method: str = "bht"
        self.assertRaises(
            AssertionError,
            wrapper,
            method=method,
            rbf_type="crivens",
        )
        self.assertRaises(
            AssertionError,
            wrapper,
            method=method,
            derivative_order=0,
        )
        self.assertRaises(
            AssertionError,
            wrapper,
            method=method,
            derivative_order=3,
        )
        self.assertRaises(
            AssertionError,
            wrapper,
            method=method,
            derivative_order=2.5,
        )
        self.assertRaises(
            AssertionError,
            wrapper,
            method=method,
            rbf_shape="crivens",
        )
        self.assertRaises(
            AssertionError,
            wrapper,
            method=method,
            num_samples=0,
        )
        self.assertRaises(
            AssertionError,
            wrapper,
            method=method,
            num_samples="crivens",
        )
        self.assertRaises(
            AssertionError,
            wrapper,
            method=method,
            num_attempts=0,
        )
        self.assertRaises(
            AssertionError,
            wrapper,
            method=method,
            num_attempts="crivens",
        )
        self.assertRaises(
            AssertionError,
            wrapper,
            method=method,
            maximum_symmetry=-0.1,
        )
        self.assertRaises(
            AssertionError,
            wrapper,
            method=method,
            maximum_symmetry=1.1,
        )
        self.assertRaises(
            AssertionError,
            wrapper,
            method=method,
            maximum_symmetry="crivens",
        )
        rbf_type: str
        for rbf_type in RBF_TYPES:
            wrapper(
                method=method,
                rbf_type=rbf_type,
                num_attempts=1,
                maximum_symmetry=1.0,
            )
        rbf_shape: str
        for rbf_shape in RBF_SHAPES:
            wrapper(
                method=method,
                rbf_shape=rbf_shape,
                num_attempts=1,
                maximum_symmetry=1.0,
            )

    def test_18_method_mRQfit(self):
        # TODO: Implement
        wrapper: Callable = drt_func
        method: str = "m(RQ)fit"
        self.assertRaises(
            AssertionError,
            wrapper,
            method=method,
            circuit=5,
        )
        self.assertRaises(
            AssertionError,
            wrapper,
            method=method,
            circuit=self._circuit.to_string(),
        )
        self.assertRaises(
            DRTError,
            wrapper,
            method=method,
            circuit=parse_cdc("L"),
        )
        self.assertRaises(
            DRTError,
            wrapper,
            method=method,
            circuit=parse_cdc("R"),
        )
        self.assertRaises(
            DRTError,
            wrapper,
            method=method,
            circuit=parse_cdc("RR"),
        )
        self.assertRaises(
            DRTError,
            wrapper,
            method=method,
            circuit=parse_cdc("R(RQ)R"),
        )
        self.assertRaises(
            DRTError,
            wrapper,
            method=method,
            circuit=parse_cdc("R([RC]C)"),
        )
        self.assertRaises(
            DRTError,
            wrapper,
            method=method,
            circuit=parse_cdc("R(C(R[RC]))"),
        )
        self.assertRaises(
            DRTError,
            wrapper,
            method=method,
            circuit=parse_cdc("R(RL)"),
        )
        self.assertRaises(
            DRTError,
            wrapper,
            method=method,
            circuit=parse_cdc("R(CL)"),
        )
        self.assertRaises(
            DRTError,
            wrapper,
            method=method,
            circuit=parse_cdc("R(RCQ)"),
        )
        self.assertRaises(
            AssertionError,
            wrapper,
            method=method,
            circuit=self._circuit,
            W=0,
        )
        self.assertRaises(
            AssertionError,
            wrapper,
            method=method,
            circuit=self._circuit,
            W="crivens",
        )
        self.assertRaises(
            AssertionError,
            wrapper,
            method=method,
            circuit=self._circuit,
            num_per_decade=1.6,
        )
        self.assertRaises(
            AssertionError,
            wrapper,
            method=method,
            circuit=self._circuit,
            num_per_decade="crivens",
        )
        wrapper(
            method=method,
            circuit=self._circuit,
        )
        wrapper(
            method=method,
            circuit=parse_cdc(self._circuit.to_string()),
        )
