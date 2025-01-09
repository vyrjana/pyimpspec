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

from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)
from unittest import TestCase
from numpy import (
    allclose,
    angle,
    concatenate,
    exp,
    float64,
    isclose,
    log10 as log,
    ndarray,
    sort,
)
from numpy.random import (
    seed,
)
from scipy.integrate import quad
from pandas import DataFrame
from pyimpspec import (
    Circuit,
    DRTPeak,
    DRTPeaks,
    DRTResult,
    DataSet,
    calculate_drt,
    fit_circuit,
    generate_mock_data,
    parse_cdc,
    parse_data,
)
from pyimpspec.analysis.drt import (
    BHTResult,
    LMResult,
    MRQFitResult,
    TRNNLSResult,
    TRRBFResult,
)
from pyimpspec.exceptions import DRTError
from pyimpspec.analysis.drt import _METHODS
from pyimpspec.analysis.drt.tr_rbf import (
    _MODES,
    _RBF_SHAPES,
    _RBF_TYPES,
)
from pyimpspec import progress as PROGRESS
from pyimpspec.typing import (
    ComplexImpedances,
    Frequencies,
    Gamma,
    Gammas,
    TimeConstant,
    TimeConstants,
)
from test_matplotlib import (
    check_mpl_return_values,
    mpl,
    primitive_mpl_plotters,
)


# TODO: test get_residuals_data


def progress_callback(*args, **kwargs):
    message: Optional[str] = kwargs.get("message")
    assert message is not None
    assert isinstance(message, str), message
    assert message.strip() != ""
    progress: Optional[float] = kwargs.get("progress")
    assert isinstance(progress, float) or progress is None, progress
    if type(progress) is float:
        assert 0.0 <= progress <= 1.0, progress
    # print(args, kwargs)


PROGRESS.register(progress_callback)


# Note: The results may vary depending on the platforms that is being used.
# Similar results should be returned if a little bit of noise is included
# in the impedance spectrum. However, this will result in a lot peaks.
class TestLoewnerMethod(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data: DataSet = generate_mock_data(
            "R{R=140}L{L=1e-4}(R{R=230}C{C=1e-6})(R{R=576}C{C=1e-4})(R{R=150}L{L=4e1})",
        )[0]
        cls.expected_time_constants = list(map(TimeConstant, (
            230.0 * 1e-6,
            576.0 * 1e-4,
            4e1 / 150.0,
        )))
        cls.expected_gammas_RC = list(map(Gamma, (
            230.0,
            576.0,
        )))
        cls.expected_gammas_RL = list(map(Gamma, (
            150.0,
        )))
        cls.result: LMResult = calculate_drt(cls.data, method="lm")

    def calculate_drt(self, *args, **kwargs):
        return calculate_drt(
            self.data,
            method="lm",
            *args,
            **kwargs,
        )

    def test_calculate_drt_lm(self):
        with self.assertRaises(TypeError):
            self.calculate_drt(model_order=1.0)

        with self.assertRaises(TypeError):
            self.calculate_drt(model_order_method=True)

        with self.assertRaises(TypeError):
            self.calculate_drt(model_order_method=1)

        with self.assertRaises(TypeError):
            self.calculate_drt(model_order_method=0.5)

    def test_result_get_peaks(self):
        peaks: Tuple[TimeConstants, Gammas, TimeConstants, Gammas]
        peaks = self.result.get_peaks()
        self.assertIsInstance(peaks, tuple)

        time_constants_RC, gammas_RC, time_constants_RL, gammas_RL = peaks
        self.assertEqual(time_constants_RC.shape, gammas_RC.shape)
        self.assertEqual(time_constants_RL.shape, gammas_RL.shape)
        self.assertEqual(time_constants_RC.dtype, TimeConstant)
        self.assertEqual(time_constants_RL.dtype, TimeConstant)
        self.assertEqual(gammas_RC.dtype, Gamma)
        self.assertEqual(gammas_RL.dtype, Gamma)
        self.assertTrue((gammas_RC >= 0.0).all())
        self.assertTrue((gammas_RL >= 0.0).all())

    def test_result_to_statistics_dataframe(self):
        df: DataFrame = self.result.to_statistics_dataframe()
        self.assertIsInstance(df, DataFrame)
        self.assertEqual(len(df.keys()), 2)
        self.assertTrue("Label" in df.keys())
        self.assertTrue("Value" in df.keys())

        md: str = df.to_markdown(index=False)
        self.assertTrue("Log pseudo chi-squared" in md)
        self.assertTrue("Model order (k)" in md)

    def test_result_to_peaks_dataframe(self):
        df: DataFrame = self.result.to_peaks_dataframe()
        self.assertIsInstance(df, DataFrame)
        self.assertEqual(len(df.keys()), 4, msg=df.keys())
        self.assertTrue("tau, RC (s)" in df.keys())
        self.assertTrue("gamma, RC (ohm)" in df.keys())
        self.assertTrue("tau, RL (s)" in df.keys())
        self.assertTrue("gamma, RL (ohm)" in df.keys())

        md: str = df.to_markdown(index=False)
        lines: List[str] = [line for line in md.split("\n") if line.strip() != ""]
        self.assertGreaterEqual(len(lines), 4)

    def test_result_get_drt_data(self):
        time_constants_RC: TimeConstants
        gammas_RC: Gammas
        time_constants_RL: TimeConstants
        gammas_RL: Gammas
        time_constants_RC, gammas_RC, time_constants_RL, gammas_RL = self.result.get_drt_data()

        time_constants: TimeConstants
        gammas: Gammas
        for time_constants, gammas in (
            (time_constants_RC, gammas_RC),
            (time_constants_RL, gammas_RL),
        ):
            self.assertIsInstance(time_constants, ndarray)
            self.assertEqual(time_constants.dtype, TimeConstant)
            self.assertIsInstance(gammas, ndarray)
            self.assertEqual(gammas.dtype, Gamma)
            self.assertEqual(time_constants.shape, gammas.shape)

    def test_result_get_time_constants(self):
        time_constants: TimeConstants = self.result.get_time_constants()
        self.assertGreaterEqual(len(time_constants), 3)
        self.assertEqual(time_constants.dtype, TimeConstant)

        t: TimeConstant
        for t in self.expected_time_constants:
            self.assertTrue(isclose(time_constants, t).any())

    def test_result_get_gammas(self):
        gammas: Tuple[Gammas, Gammas] = self.result.get_gammas()
        self.assertIsInstance(gammas, tuple)
        self.assertEqual(len(gammas), 2)

        gammas_RC, gammas_RL = gammas
        self.assertGreaterEqual(gammas_RC.shape, (2,))
        self.assertGreaterEqual(gammas_RL.shape, (1,))
        self.assertEqual(gammas_RC.dtype, Gamma)
        self.assertEqual(gammas_RL.dtype, Gamma)

        g: float64
        for g in self.expected_gammas_RC:
            self.assertTrue(isclose(gammas_RC, g).any())

        for g in self.expected_gammas_RL:
            self.assertTrue(isclose(gammas_RL, g).any())


seed(28041948)  # GNU STP
DATA: DataSet = parse_data("data.dta")[0]


def drt_func(*args, **kwargs) -> DRTResult:
    kwargs["num_procs"] = 1
    return calculate_drt(DATA, **kwargs)


# TODO: Refactor
class TestDRT(TestCase):
    @classmethod
    def setUpClass(cls):
        cls._f_exp: Frequencies = DATA.get_frequencies()
        cls._Z_exp: ComplexImpedances = DATA.get_impedances()
        cls._circuit: Circuit = fit_circuit(parse_cdc("R(RQ)(RQ)"), DATA).circuit
        cls.wrapper: Callable = drt_func
        cls._method_results: Dict[str, DRTResult] = {}
        method: str
        for method in _METHODS:
            drt: DRTResult = cls.wrapper(
                method=method,
                circuit=cls._circuit if method == "mrq-fit" else None,
            )
            cls._method_results[method] = drt

    def test_data(self):
        self.assertRaises(
            AttributeError,
            calculate_drt,
            {},
        )

    def test_methods(self):
        self.assertRaises(
            ValueError,
            self.wrapper,
            method="crivens",
        )
        self.assertRaises(
            ValueError,
            self.wrapper,
            method="invalid method",
        )

    def test_result_label(self):
        method: str
        drt: DRTResult
        for method, drt in self._method_results.items():
            self.assertTrue(drt.get_label().strip() != "")
            if method == "mrq-fit":
                self.assertEqual(drt.get_label(), "R-2(RQ)")
            else:
                self.assertTrue(method.upper() in drt.get_label(), msg=method)

    def test_result_get_frequencies(self):
        method: str
        drt: DRTResult
        for method, drt in self._method_results.items():
            self.assertEqual(len(drt.get_frequencies()), 29, msg=method)
            f_mod: Frequencies = drt.get_frequencies()
            self.assertIsInstance(f_mod, ndarray, msg=method)
            self.assertEqual(self._f_exp.shape, f_mod.shape, msg=method)
            self.assertTrue(allclose(self._f_exp, f_mod), msg=method)

    def test_result_get_impedances(self):
        method: str
        drt: DRTResult
        for method, drt in self._method_results.items():
            self.assertEqual(len(drt.get_impedances()), 29, msg=method)
            Z_mod: ComplexImpedances = drt.get_impedances()
            self.assertIsInstance(Z_mod, ndarray, msg=method)
            self.assertEqual(self._Z_exp.shape, Z_mod.shape, msg=method)

    def test_result_get_nyquist_data(self):
        method: str
        drt: DRTResult
        for method, drt in self._method_results.items():
            Z_mod: ComplexImpedances = drt.get_impedances()
            self.assertTrue(allclose(Z_mod.real, drt.get_nyquist_data()[0]), msg=method)
            self.assertTrue(allclose(-Z_mod.imag, drt.get_nyquist_data()[1]), msg=method)

    def test_result_get_bode_data(self):
        method: str
        drt: DRTResult
        for method, drt in self._method_results.items():
            f_mod: Frequencies = drt.get_frequencies()
            Z_mod: ComplexImpedances = drt.get_impedances()
            self.assertTrue(allclose(f_mod, drt.get_bode_data()[0]), msg=method)
            self.assertTrue(allclose(abs(Z_mod), drt.get_bode_data()[1]), msg=method)
            self.assertTrue(allclose(-angle(Z_mod, deg=True), drt.get_bode_data()[2]), msg=method)

    def test_result_get_time_constants(self):
        method: str
        drt: DRTResult
        for method, drt in self._method_results.items():
            time_constants: TimeConstants = drt.get_time_constants()
            self.assertIsInstance(time_constants, ndarray)

    def test_result_get_gammas(self):
        method: str
        drt: DRTResult
        for method, drt in self._method_results.items():
            if isinstance(drt, BHTResult):
                real_gammas: Gammas
                imag_gammas: Gammas
                real_gammas, imag_gammas = drt.get_gammas()
                self.assertIsInstance(real_gammas, ndarray)
                self.assertIsInstance(imag_gammas, ndarray)
            elif isinstance(drt, LMResult):
                gammas_RC: Gammas
                gammas_RL: Gammas
                gammas_RC, gammas_RL = drt.get_gammas()
                self.assertIsInstance(gammas_RC, ndarray)
                self.assertIsInstance(gammas_RL, ndarray)
            else:
                gamma: Gammas = drt.get_gammas()
                self.assertIsInstance(gamma, ndarray, msg=method)

    def test_result_get_drt_data(self):
        method: str
        drt: DRTResult
        for method, drt in self._method_results.items():
            time_constants: TimeConstants
            if isinstance(drt, BHTResult):
                self.assertEqual(len(drt.get_drt_data()), 3)
                real_gammas: Gammas
                imag_gammas: Gammas
                time_constants, real_gammas, imag_gammas = drt.get_drt_data()
                self.assertIsInstance(time_constants, ndarray)
                self.assertIsInstance(real_gammas, ndarray)
                self.assertIsInstance(imag_gammas, ndarray)
                self.assertEqual(time_constants.shape, real_gammas.shape)
                self.assertEqual(time_constants.shape, imag_gammas.shape)
                self.assertTrue(time_constants.size >= self._f_exp.size)
                self.assertTrue(real_gammas.size >= self._f_exp.size)
                self.assertTrue(imag_gammas.size >= self._f_exp.size)
                self.assertTrue(allclose(time_constants, drt.get_time_constants()))
                self.assertEqual(real_gammas.size, drt.get_gammas()[0].size)
                self.assertEqual(imag_gammas.size, drt.get_gammas()[1].size)
                self.assertTrue(allclose(real_gammas, drt.get_gammas()[0]))
                self.assertTrue(allclose(imag_gammas, drt.get_gammas()[1]))
            elif isinstance(drt, LMResult):
                self.assertEqual(len(drt.get_drt_data()), 4)
                time_constants_RC: TimeConstants
                gammas_RC: Gammas
                time_constants_RL: TimeConstants
                gammas_RL: Gammas
                time_constants_RC, gammas_RC, time_constants_RL, gammas_RL = drt.get_drt_data()
                self.assertIsInstance(time_constants_RC, ndarray)
                self.assertIsInstance(gammas_RC, ndarray)
                self.assertIsInstance(time_constants_RL, ndarray)
                self.assertIsInstance(gammas_RL, ndarray)
                self.assertEqual(time_constants_RC.shape, gammas_RC.shape)
                self.assertEqual(time_constants_RL.shape, gammas_RL.shape)
                self.assertTrue(0 <= time_constants_RC.size <= self._f_exp.size)
                self.assertTrue(0 <= time_constants_RL.size <= self._f_exp.size)
                self.assertTrue(allclose(sort(concatenate((time_constants_RC, time_constants_RL))), drt.get_time_constants()))
                self.assertTrue(allclose(gammas_RC, drt.get_gammas()[0]))
                self.assertTrue(allclose(gammas_RL, drt.get_gammas()[1]))
            else:
                self.assertEqual(len(drt.get_drt_data()), 2, msg=method)
                gamma: Gammas
                time_constants, gamma = drt.get_drt_data()
                self.assertIsInstance(time_constants, ndarray, msg=method)
                self.assertIsInstance(gamma, ndarray, msg=method)
                self.assertEqual(time_constants.shape, gamma.shape, msg=method)
                self.assertTrue(time_constants.size >= self._f_exp.size, msg=method)
                self.assertTrue(gamma.size >= self._f_exp.size, msg=method)
                self.assertTrue(allclose(time_constants, drt.get_time_constants()), msg=method)
                self.assertTrue(allclose(gamma, drt.get_gammas()), msg=method)
                self.assertEqual(gamma.size, drt.get_gammas().size, msg=method)

    def test_result_scores(self):
        method: str
        drt: DRTResult
        for method, drt in self._method_results.items():
            if isinstance(drt, BHTResult):
                scores: Dict[str, complex] = drt.get_scores()
                self.assertIsInstance(scores, dict)
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
                # DataFrame
                self.assertIsInstance(drt.to_scores_dataframe(), DataFrame)
            else:
                with self.assertRaises(AttributeError):
                    drt.get_scores()

    def test_result_get_drt_credible_intervals_data(self):
        method: str
        drt: DRTResult
        for method, drt in self._method_results.items():
            time_constants: TimeConstants
            mean_gamma: Gammas
            lower_bound: Gammas
            upper_bound: Gammas
            if isinstance(drt, TRRBFResult):
                (
                    time_constants,
                    mean_gamma,
                    lower_bound,
                    upper_bound,
                ) = drt.get_drt_credible_intervals_data()
                self.assertEqual(time_constants.size, 0)
                self.assertEqual(time_constants.size, mean_gamma.size)
                self.assertEqual(time_constants.size, lower_bound.size)
                self.assertEqual(time_constants.size, upper_bound.size)
            else:
                with self.assertRaises(AttributeError):
                    drt.get_drt_credible_intervals_data()

    def test_get_peaks(self):
        drt: DRTResult
        time_constants: TimeConstants
        gamma: Gammas
        time_constants_re: TimeConstants
        time_constants_im: TimeConstants
        gammas_re: Gammas
        gammas_im: Gammas

        drt = self._method_results["tr-nnls"]
        time_constants, gamma = drt.get_peaks()
        self.assertGreater(time_constants.size, 0)
        self.assertEqual(time_constants.size, gamma.size)
        self.assertIsInstance(drt.to_peaks_dataframe(), DataFrame)

        drt = self._method_results["tr-rbf"]
        time_constants, gamma = drt.get_peaks()
        self.assertGreater(time_constants.size, 0)
        self.assertEqual(time_constants.size, gamma.size)
        self.assertIsInstance(drt.to_peaks_dataframe(), DataFrame)

        drt = self._method_results["bht"]
        time_constants_re, gammas_re, time_constants_im, gammas_im = drt.get_peaks()
        self.assertGreater(time_constants_re.size, 0)
        self.assertEqual(time_constants_re.size, gammas_re.size)
        self.assertGreater(time_constants_im.size, 0)
        self.assertEqual(time_constants_im.size, gammas_im.size)
        self.assertIsInstance(drt.to_peaks_dataframe(), DataFrame)

        drt = self._method_results["mrq-fit"]
        time_constants, gamma = drt.get_peaks()
        self.assertGreater(time_constants.size, 0)
        self.assertEqual(time_constants.size, gamma.size)
        self.assertIsInstance(drt.to_peaks_dataframe(), DataFrame)

        drt = self._method_results["lm"]
        time_constants_RC, gammas_RC, time_constants_RL, gammas_RL = drt.get_peaks()
        self.assertGreater(time_constants_RC.size, 0)
        self.assertEqual(time_constants_RC.size, gammas_RC.size)
        self.assertEqual(time_constants_RL.size, 0)
        self.assertEqual(time_constants_RL.size, gammas_RL.size)
        self.assertIsInstance(drt.to_peaks_dataframe(), DataFrame)

    def test_method_tr_nnls(self):
        method: str = "tr-nnls"
        self.assertRaises(
            TypeError,
            self.wrapper,
            method=method,
            mode=0,
        )
        self.assertRaises(
            ValueError,
            self.wrapper,
            method=method,
            mode="crivens",
        )

        drt: TRNNLSResult = self.wrapper(method=method, mode="real")
        self.wrapper(method=method, mode="imaginary")
        self.wrapper(method=method, mode="real", lambda_value=1e-3)

    def test_method_tr_rbf(self):
        method: str = "tr-rbf"
        self.assertRaises(
            ValueError,
            self.wrapper,
            method=method,
            rbf_type="crivens",
        )
        self.assertRaises(
            ValueError,
            self.wrapper,
            method=method,
            derivative_order=0,
        )
        self.assertRaises(
            ValueError,
            self.wrapper,
            method=method,
            derivative_order=3,
        )
        self.assertRaises(
            TypeError,
            self.wrapper,
            method=method,
            derivative_order=2.5,
        )
        self.assertRaises(
            ValueError,
            self.wrapper,
            method=method,
            rbf_shape="crivens",
        )
        self.assertRaises(
            TypeError,
            self.wrapper,
            method=method,
            inductance="crivens",
        )
        self.assertRaises(
            TypeError,
            self.wrapper,
            method=method,
            credible_intervals="crivens",
        )
        self.assertRaises(
            ValueError,
            self.wrapper,
            method=method,
            num_samples=0,
            credible_intervals=True,
        )
        self.assertRaises(
            TypeError,
            self.wrapper,
            method=method,
            num_samples="crivens",
        )

        mode: str
        for mode in _MODES:
            self.wrapper(method=method, mode=mode)

        derivative_order: int
        for derivative_order in range(1, 3):
            rbf_type: str
            for rbf_type in _RBF_TYPES:
                self.wrapper(
                    method=method,
                    rbf_type=rbf_type,
                    derivative_order=derivative_order,
                )

        self.assertTrue("factor" in _RBF_SHAPES)
        rbf_shape: str
        for rbf_shape in _RBF_SHAPES:
            self.wrapper(method=method, rbf_shape=rbf_shape)

    def test_credible_intervals(self):
        # TODO: Switch to an impedance spectrum with some noise? 
        # Calculating credible intervals seems to be faster and thus
        # the timeout would not need to be raised. Other test cases might
        # need to be updated in terms of expected values.
        drt: DRTResult = self.wrapper(method="tr-rbf", credible_intervals=True, timeout=300)

        time_constants: TimeConstants
        mean_gamma: Gammas
        lower_bound: Gammas
        upper_bound: Gammas
        (
            time_constants,
            mean_gamma,
            lower_bound,
            upper_bound,
        ) = drt.get_drt_credible_intervals_data()

        self.assertIsInstance(time_constants, ndarray)
        self.assertIsInstance(mean_gamma, ndarray)
        self.assertIsInstance(lower_bound, ndarray)
        self.assertIsInstance(upper_bound, ndarray)
        self.assertEqual(time_constants.size, drt.get_time_constants().size)
        self.assertEqual(time_constants.size, mean_gamma.size)
        self.assertEqual(time_constants.size, lower_bound.size)
        self.assertEqual(time_constants.size, upper_bound.size)

    def test_method_bht(self):
        method: str = "bht"
        self.assertRaises(
            ValueError,
            self.wrapper,
            method=method,
            rbf_type="crivens",
        )
        self.assertRaises(
            ValueError,
            self.wrapper,
            method=method,
            derivative_order=0,
        )
        self.assertRaises(
            ValueError,
            self.wrapper,
            method=method,
            derivative_order=3,
        )
        self.assertRaises(
            TypeError,
            self.wrapper,
            method=method,
            derivative_order=2.5,
        )
        self.assertRaises(
            ValueError,
            self.wrapper,
            method=method,
            rbf_shape="crivens",
        )
        self.assertRaises(
            ValueError,
            self.wrapper,
            method=method,
            num_samples=0,
        )
        self.assertRaises(
            TypeError,
            self.wrapper,
            method=method,
            num_samples="crivens",
        )
        self.assertRaises(
            ValueError,
            self.wrapper,
            method=method,
            num_attempts=0,
        )
        self.assertRaises(
            TypeError,
            self.wrapper,
            method=method,
            num_attempts="crivens",
        )
        self.assertRaises(
            ValueError,
            self.wrapper,
            method=method,
            maximum_symmetry=-0.1,
        )
        self.assertRaises(
            ValueError,
            self.wrapper,
            method=method,
            maximum_symmetry=1.1,
        )
        self.assertRaises(
            TypeError,
            self.wrapper,
            method=method,
            maximum_symmetry="crivens",
        )

        derivative_order: int
        for derivative_order in range(1, 3):
            rbf_type: str
            for rbf_type in _RBF_TYPES:
                self.wrapper(
                    method=method,
                    rbf_type=rbf_type,
                    num_attempts=1,
                    maximum_symmetry=1.0,
                    derivative_order=derivative_order,
                )

        rbf_shape: str
        for rbf_shape in _RBF_SHAPES:
            self.wrapper(
                method=method,
                rbf_shape=rbf_shape,
                num_attempts=1,
                maximum_symmetry=1.0,
            )

    def test_method_mrq_fit(self):
        self.assertEqual(
            MRQFitResult._generate_label(parse_cdc("R(RC)(RC)(RC)")),
            "R-3(RC)",
        )
        self.assertEqual(
            MRQFitResult._generate_label(parse_cdc("R(RC)(RQ)(RC)")),
            "R(RQ)-2(RC)",
        )
        self.assertEqual(
            MRQFitResult._generate_label(parse_cdc("R(RQ)(RQ)(RC)")),
            "R-2(RQ)-(RC)",
        )
        self.assertEqual(
            MRQFitResult._generate_label(parse_cdc("R(RQ)(RQ)(RC)(RC)")),
            "R-2(RQ)-2(RC)",
        )

        method: str = "mrq-fit"
        self.assertRaises(
            TypeError,
            self.wrapper,
            method=method,
            circuit=5,
        )
        self.assertRaises(
            TypeError,
            self.wrapper,
            method=method,
            circuit=self._circuit.to_string(),
        )
        self.assertRaises(
            DRTError,
            self.wrapper,
            method=method,
            circuit=parse_cdc("L"),
        )
        self.assertRaises(
            DRTError,
            self.wrapper,
            method=method,
            circuit=parse_cdc("R"),
        )
        self.assertRaises(
            DRTError,
            self.wrapper,
            method=method,
            circuit=parse_cdc("RR"),
        )
        self.assertRaises(
            DRTError,
            self.wrapper,
            method=method,
            circuit=parse_cdc("R(RQ)R"),
        )
        self.assertRaises(
            DRTError,
            self.wrapper,
            method=method,
            circuit=parse_cdc("R([RC]C)"),
        )
        self.assertRaises(
            DRTError,
            self.wrapper,
            method=method,
            circuit=parse_cdc("R(C(R[RC]))"),
        )
        self.assertRaises(
            DRTError,
            self.wrapper,
            method=method,
            circuit=parse_cdc("R(RL)"),
        )
        self.assertRaises(
            DRTError,
            self.wrapper,
            method=method,
            circuit=parse_cdc("R(CL)"),
        )
        self.assertRaises(
            DRTError,
            self.wrapper,
            method=method,
            circuit=parse_cdc("R(RCQ)"),
        )
        self.assertRaises(
            TypeError,
            self.wrapper,
            method=method,
            circuit=self._circuit,
            gaussian_width=0,
        )
        self.assertRaises(
            TypeError,
            self.wrapper,
            method=method,
            circuit=self._circuit,
            gaussian_width="crivens",
        )
        self.assertRaises(
            TypeError,
            self.wrapper,
            method=method,
            circuit=self._circuit,
            num_per_decade=1.6,
        )
        self.assertRaises(
            TypeError,
            self.wrapper,
            method=method,
            circuit=self._circuit,
            num_per_decade="crivens",
        )
        self.assertRaises(
            DRTError,
            self.wrapper,
            method=method,
            circuit=parse_cdc("RL"),
        )
        self.assertRaises(
            DRTError,
            self.wrapper,
            method=method,
            circuit=parse_cdc("R(R[CQ])"),
        )
        self.assertRaises(
            DRTError,
            self.wrapper,
            method=method,
            circuit=parse_cdc("R(R(CQ))"),
        )
        self.assertRaises(
            DRTError,
            self.wrapper,
            method=method,
            circuit=parse_cdc("R(CQ)"),
        )
        self.assertRaises(
            DRTError,
            self.wrapper,
            method=method,
            circuit=parse_cdc("R(RL)"),
        )
        self.wrapper(
            method=method,
            circuit=self._circuit,
        )
        self.wrapper(
            method=method,
            circuit=parse_cdc(self._circuit.to_string()),
        )

    def test_method_lm(self):
        method: str = "lm"
        self.assertRaises(
            TypeError,
            self.wrapper,
            method=method,
            model_order=0.2,
        )
        self.assertRaises(
            TypeError,
            self.wrapper,
            method=method,
            model_order="crivens",
        )
        self.assertRaises(
            TypeError,
            self.wrapper,
            method=method,
            model_order_method=2,
        )
        self.assertRaises(
            ValueError,
            self.wrapper,
            method=method,
            model_order_method="crivens",
        )

        drt: LMResult = self.wrapper(method=method)
        self.wrapper(method=method, model_order=2)
        self.wrapper(method=method, model_order_method="pseudo_chisqr")

    def test_analyze_peaks(self):
        result: DRTResult
        for result in self._method_results.values():
            peaks: Union[DRTPeaks, Tuple[DRTPeaks, ...]] = result.analyze_peaks()
            if isinstance(result, BHTResult):
                self.assertIsInstance(peaks, tuple)
                self.assertEqual(len(peaks), 2)
                self.assertIsInstance(peaks[0], DRTPeaks)
                self.assertIsInstance(peaks[1], DRTPeaks)
            elif isinstance(result, LMResult):
                self.assertIsInstance(peaks, tuple)
                self.assertEqual(len(peaks), 2)
                self.assertIsInstance(peaks[0], DRTPeaks)
                self.assertIsInstance(peaks[1], DRTPeaks)
            else:
                self.assertIsInstance(peaks, DRTPeaks)

        data: DataSet = generate_mock_data(
            "R{R=140}(R{R=230}C{C=1e-6})(R{R=576}C{C=1e-4})",
        )[0]
        expected_values: List[float] = [230.0, 576.0]
        result = calculate_drt(data, method="tr-rbf")

        with self.assertRaises(TypeError):
            result.analyze_peaks(num_peaks=2.3)

        with self.assertRaises(TypeError):
            result.analyze_peaks(peak_positions=(4, 2))

        with self.assertRaises(TypeError):
            result.analyze_peaks(disallow_skew=2)

        method: str
        for method in _METHODS:
            if method in ("bht", "lm", "mrq-fit"):
                continue

            result = calculate_drt(data, method=method)
            peaks = result.analyze_peaks(num_peaks=2)
            self.assertIsInstance(peaks, DRTPeaks)
            self.assertEqual(peaks.get_num_peaks(), 2)

            time_constants: TimeConstants = peaks.get_time_constants()

            i: int
            peak: DRTPeak
            for i, peak in enumerate(peaks):
                area: float = quad(
                    func=lambda x: peak.get_gammas(10**x),
                    a=log(min(time_constants)),
                    b=log(max(time_constants)),
                )[0] / log(exp(1))

                self.assertAlmostEqual(peaks.get_peak_area(i), area)
                self.assertAlmostEqual(
                    area,
                    expected_values[i],
                    delta=0.1*expected_values[i],
                )

    def test_matplotlib(self):
        drt: DRTResult
        for drt in self._method_results.values():
            plotter: Callable
            for plotter in primitive_mpl_plotters:
                check_mpl_return_values(self, *plotter(data=drt))

            check_mpl_return_values(self, *mpl.plot_residuals(drt))
            check_mpl_return_values(self, *mpl.plot_gamma(None))
            check_mpl_return_values(self, *mpl.plot_gamma(drt))
            check_mpl_return_values(self, *mpl.plot_gamma(drt, colored_axes=True))
            check_mpl_return_values(self, *mpl.plot_gamma(drt, peak_threshold=0.3))
            check_mpl_return_values(self, *mpl.plot_drt(drt, DATA, colored_axes=True))
            check_mpl_return_values(self, *mpl.plot_drt(drt, data=DATA))
            check_mpl_return_values(self, *mpl.plot_fit(drt, data=DATA))
            if isinstance(drt, BHTResult):
                check_mpl_return_values(self, *mpl.plot_bht_scores(drt))
