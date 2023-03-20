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

from typing import (
    Callable,
    List,
    Optional,
)
from unittest import TestCase
from numpy import (
    allclose,
    angle,
    array,
    isnan,
    ndarray,
)
from numpy.random import (
    seed,
    normal,
)
from pyimpspec import (
    Circuit,
    DataSet,
    TestResult,
    perform_exploratory_tests,
    perform_test,
    parse_data,
)
from pyimpspec import progress as PROGRESS
from pyimpspec.typing import (
    Frequencies,
    Impedances,
    Phases,
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


seed(42)
DATA: DataSet = parse_data("data-comma.csv")[0]
sd: float = 0.005
DATA.subtract_impedances(
    -array(
        list(
            map(
                lambda _: complex(
                    abs(_) * normal(0, sd, 1),
                    abs(_) * normal(0, sd, 1),
                ),
                DATA.get_impedances(),
            )
        )
    )
)


TEST_PLOTS: bool = True


# TODO: Implement tests for invalid arguments
class KramersKronigComplex(TestCase):
    # kwargs
    arg_test: str = "complex"
    arg_num_RC: int = 0
    arg_mu_criterion: float = 0.85
    arg_add_capacitance: bool = False
    arg_add_inductance: bool = False
    arg_method: str = "leastsq"
    arg_max_nfev: int = -1
    arg_num_procs: int = -1
    # comparison values
    cmp_cdc: str = "[RKKL]"
    cmp_mu: float = 0.8465317822199895
    cmp_num_RC: int = 2
    cmp_pseudo_chisqr: float = 5.170991769675877

    @classmethod
    def setUpClass(cls):
        cls.result: TestResult = perform_test(
            DATA,
            test=cls.arg_test,
            num_RC=cls.arg_num_RC,
            mu_criterion=cls.arg_mu_criterion,
            add_capacitance=cls.arg_add_capacitance,
            add_inductance=cls.arg_add_inductance,
            method=cls.arg_method,
            max_nfev=cls.arg_max_nfev,
            num_procs=cls.arg_num_procs,
        )

    def test_single_process(self):
        result: TestResult = perform_test(
            DATA,
            test=self.arg_test,
            num_RC=self.arg_num_RC,
            mu_criterion=self.arg_mu_criterion,
            add_capacitance=self.arg_add_capacitance,
            add_inductance=self.arg_add_inductance,
            method=self.arg_method,
            max_nfev=self.arg_max_nfev,
            num_procs=1,
        )
        self.assertEqual(self.result.circuit.to_string(), result.circuit.to_string())
        self.assertEqual(self.result.num_RC, result.num_RC)
        self.assertEqual(self.result.mu, result.mu)
        self.assertEqual(self.result.pseudo_chisqr, result.pseudo_chisqr)
        self.assertTrue(allclose(self.result.frequencies, result.frequencies))
        self.assertTrue(allclose(self.result.impedances, result.impedances))
        self.assertTrue(allclose(self.result.residuals, result.residuals))

    def test_multi_process(self):
        result: TestResult = perform_test(
            DATA,
            test=self.arg_test,
            num_RC=self.arg_num_RC,
            mu_criterion=self.arg_mu_criterion,
            add_capacitance=self.arg_add_capacitance,
            add_inductance=self.arg_add_inductance,
            method=self.arg_method,
            max_nfev=self.arg_max_nfev,
            num_procs=2,
        )
        self.assertEqual(self.result.circuit.to_string(), result.circuit.to_string())
        self.assertEqual(self.result.num_RC, result.num_RC)
        self.assertEqual(self.result.mu, result.mu)
        self.assertEqual(self.result.pseudo_chisqr, result.pseudo_chisqr)
        self.assertTrue(allclose(self.result.frequencies, result.frequencies))
        self.assertTrue(allclose(self.result.impedances, result.impedances))
        self.assertTrue(allclose(self.result.residuals, result.residuals))

    def test_return_type(self):
        self.assertIsInstance(self.result, TestResult)

    def test_repr(self):
        self.assertEqual(
            repr(self.result),
            f"TestResult (num_RC={self.result.num_RC}, {hex(id(self.result))})",
        )

    def test_get_frequencies(self):
        self.assertIsInstance(self.result.get_frequencies(), ndarray)
        self.assertEqual(
            len(self.result.get_frequencies()), len(DATA.get_frequencies())
        )
        self.assertTrue(allclose(self.result.get_frequencies(), DATA.get_frequencies()))

    def test_get_impedances(self):
        self.assertIsInstance(self.result.get_impedances(), ndarray)
        self.assertEqual(len(self.result.get_impedances()), len(DATA.get_impedances()))
        if self.cmp_num_RC >= 0:
            self.assertTrue(
                allclose(
                    self.result.get_impedances(),
                    DATA.get_impedances(),
                    rtol=1,
                )
            )

    def test_get_nyquist_data(self):
        self.assertIsInstance(self.result.get_nyquist_data(), tuple)
        self.assertEqual(len(self.result.get_nyquist_data()), 2)
        Zre: Impedances
        Zim: Impedances
        Zre, Zim = self.result.get_nyquist_data(num_per_decade=2)
        self.assertTrue(len(Zre) == len(Zim) == 9)
        Zre, Zim = self.result.get_nyquist_data()
        self.assertTrue(allclose(self.result.get_impedances().real, Zre))
        self.assertTrue(allclose(self.result.get_impedances().imag, -Zim))

    def test_get_bode_data(self):
        self.assertIsInstance(self.result.get_bode_data(), tuple)
        self.assertEqual(len(self.result.get_bode_data()), 3)
        f: Frequencies
        mag: Impedances
        phi: Phases
        f, mag, phi = self.result.get_bode_data(num_per_decade=2)
        self.assertTrue(len(f) == len(mag) == len(phi) == 9)
        f, mag, phi = self.result.get_bode_data()
        self.assertTrue(allclose(self.result.get_frequencies(), f))
        self.assertTrue(allclose(abs(self.result.get_impedances()), mag))
        self.assertTrue(allclose(angle(self.result.get_impedances(), deg=True), -phi))

    def test_num_RC(self):
        self.assertIsInstance(self.result.num_RC, int)
        self.assertGreater(self.result.num_RC, 0)
        if self.cmp_num_RC >= 0:
            self.assertEqual(self.result.num_RC, self.cmp_num_RC)

    def test_mu(self):
        self.assertIsInstance(self.result.mu, float)
        self.assertTrue(0.0 <= self.result.mu <= 1.0)
        if self.cmp_mu >= 0.0:
            self.assertAlmostEqual(
                self.result.mu,
                self.cmp_mu,
                delta=5e-4,
            )
            self.assertTrue(self.result.mu <= self.arg_mu_criterion)

    def test_pseudo_chisqr(self):
        self.assertIsInstance(self.result.pseudo_chisqr, float)
        self.assertGreater(self.result.pseudo_chisqr, 0.0)
        if self.cmp_pseudo_chisqr >= 0.0:
            self.assertAlmostEqual(
                self.result.pseudo_chisqr,
                self.cmp_pseudo_chisqr,
                delta=5e-4,
            )

    def test_cdc(self):
        self.assertIsInstance(self.result.circuit, Circuit)
        cdc: str = self.result.circuit.to_string()
        if self.cmp_num_RC >= 0:
            self.assertEqual(cdc, self.cmp_cdc)
        else:
            self.assertTrue(cdc.startswith("[RK"))
            self.assertTrue(cdc.endswith(self.cmp_cdc), msg=cdc)

    def test_matplotlib(self):
        global TEST_PLOTS
        if not TEST_PLOTS:
            return
        TEST_PLOTS = False
        plotter: Callable
        for plotter in primitive_mpl_plotters:
            check_mpl_return_values(
                self,
                *plotter(
                    data=self.result,
                ),
            )
        check_mpl_return_values(
            self,
            *mpl.plot_residuals(
                self.result,
            ),
        )
        check_mpl_return_values(
            self,
            *mpl.plot_residuals(
                self.result,
                colored_axes=True,
            ),
        )
        check_mpl_return_values(
            self,
            *mpl.plot_fit(
                self.result,
                data=DATA,
            ),
        )
        check_mpl_return_values(
            self,
            *mpl.plot_fit(
                self.result,
                data=DATA,
                colored_axes=True,
            ),
        )

    def test_get_series_resistance(self):
        R: float = self.result.get_series_resistance()
        self.assertFalse(isnan(R))

    def test_get_series_capacitance(self):
        C: float = self.result.get_series_capacitance()
        self.assertNotEqual(isnan(C), self.arg_add_capacitance)

    def test_get_series_inductance(self):
        L: float = self.result.get_series_inductance()
        if self.arg_test == "cnls":
            self.assertNotEqual(isnan(L), self.arg_add_inductance)
        else:
            self.assertFalse(isnan(L))


class KramersKronigComplexWithCapacitance(KramersKronigComplex):
    arg_add_capacitance: bool = True
    cmp_cdc: str = "[RKKKKKKKKKKKKKKKKCL]"
    cmp_mu: float = 0.5662037409812818
    cmp_num_RC: int = 16
    cmp_pseudo_chisqr: float = 0.0009746704484412114


class KramersKronigComplexWithInductance(KramersKronigComplex):
    arg_add_inductance: bool = True
    cmp_cdc: str = "[RKKL]"


class KramersKronigComplexWithCapacitanceInductance(KramersKronigComplex):
    arg_add_capacitance: bool = True
    arg_add_inductance: bool = True
    cmp_cdc: str = "[RKKKKKKKKKKKKKKKKCL]"
    cmp_mu: float = 0.5662037409812818
    cmp_num_RC: int = 16
    cmp_pseudo_chisqr: float = 0.0009746704484412114


class KramersKronigComplexManual(KramersKronigComplex):
    arg_num_RC: int = 16
    cmp_cdc: str = "[RKKKKKKKKKKKKKKKKL]"
    cmp_num_RC: int = 16
    cmp_mu: float = 0.4642977683437579
    cmp_pseudo_chisqr: float = 0.001003393927007882


class KramersKronigComplexManualWithCapacitance(KramersKronigComplexManual):
    arg_add_capacitance: bool = True
    cmp_cdc: str = "[RKKKKKKKKKKKKKKKKCL]"
    cmp_mu: float = 0.5662037409812818


class KramersKronigComplexManualWithInductance(KramersKronigComplexManual):
    arg_add_inductance: bool = True


class KramersKronigComplexManualWithCapacitanceInductance(KramersKronigComplexManual):
    arg_add_capacitance: bool = True
    arg_add_inductance: bool = True
    cmp_cdc: str = "[RKKKKKKKKKKKKKKKKCL]"
    cmp_mu: float = 0.5662037409812818


# ==============================================================================


class KramersKronigReal(KramersKronigComplex):
    arg_test: str = "real"
    cmp_cdc: str = "[RKKKKKKKKKKKKKKKL]"
    cmp_mu: float = 0.8351972078416112
    cmp_num_RC: int = 15
    cmp_pseudo_chisqr: float = 0.0022502074349343235


class KramersKronigRealWithCapacitance(KramersKronigReal):
    arg_add_capacitance: bool = True
    cmp_cdc: str = "[RKKKKKKKKKKKKKKKCL]"
    cmp_mu: float = 0.8351972078416252
    cmp_pseudo_chisqr: float = 0.0012584749704440204


class KramersKronigRealWithInductance(KramersKronigReal):
    arg_add_inductance: bool = True


class KramersKronigRealWithCapacitanceInductance(KramersKronigReal):
    arg_add_capacitance: bool = True
    arg_add_inductance: bool = True
    cmp_cdc: str = "[RKKKKKKKKKKKKKKKCL]"
    cmp_mu: float = 0.8351972078416252
    cmp_pseudo_chisqr: float = 0.0012584749704440204


class KramersKronigRealManual(KramersKronigReal):
    arg_test: str = "real"
    arg_num_RC: int = 16
    cmp_cdc: str = "[RKKKKKKKKKKKKKKKKL]"
    cmp_mu: float = 0.752262545112554
    cmp_num_RC: int = 16
    cmp_pseudo_chisqr: float = 0.0022502074349343235


class KramersKronigRealManualWithCapacitance(KramersKronigRealManual):
    arg_add_capacitance: bool = True
    cmp_cdc: str = "[RKKKKKKKKKKKKKKKKCL]"
    cmp_pseudo_chisqr: float = 0.001269308657064358


class KramersKronigRealManualWithInductance(KramersKronigRealManual):
    arg_add_inductance: bool = True


class KramersKronigRealManualWithCapacitanceInductance(KramersKronigRealManual):
    arg_add_capacitance: bool = True
    arg_add_inductance: bool = True
    cmp_cdc: str = "[RKKKKKKKKKKKKKKKKCL]"
    cmp_pseudo_chisqr: float = 0.001269308657064358


# ==============================================================================


class KramersKronigImaginary(KramersKronigComplex):
    arg_test: str = "imaginary"
    cmp_cdc: str = "[RKKL]"
    cmp_mu: float = 0.7634133056991227
    cmp_num_RC: int = 2
    cmp_pseudo_chisqr: float = 5.397326608733074


class KramersKronigImaginaryWithCapacitance(KramersKronigImaginary):
    arg_add_capacitance: bool = True
    cmp_cdc: str = "[RKKKKKKKKKKKCL]"
    cmp_mu: float = 0.8463746482381072
    cmp_num_RC: int = 11
    cmp_pseudo_chisqr: float = 0.004059147148708858


class KramersKronigImaginaryWithInductance(KramersKronigImaginary):
    arg_add_inductance: bool = True


class KramersKronigImaginaryWithCapacitanceInductance(KramersKronigImaginary):
    arg_add_capacitance: bool = True
    arg_add_inductance: bool = True
    cmp_cdc: str = "[RKKKKKKKKKKKCL]"
    cmp_mu: float = 0.8463746482381072
    cmp_num_RC: int = 11
    cmp_pseudo_chisqr: float = 0.004059147148708858


class KramersKronigImaginaryManual(KramersKronigImaginary):
    arg_num_RC: int = 17
    cmp_cdc: str = "[RKKKKKKKKKKKKKKKKKL]"
    cmp_mu: float = 0.18854545209673368
    cmp_num_RC: int = 17
    cmp_pseudo_chisqr: float = 0.01497161839945128


class KramersKronigImaginaryManualWithCapacitance(KramersKronigImaginaryManual):
    arg_add_capacitance: bool = True
    cmp_cdc: str = "[RKKKKKKKKKKKKKKKKKCL]"
    cmp_mu: float = 0.1863497549045704


class KramersKronigImaginaryManualWithInductance(KramersKronigImaginaryManual):
    arg_add_inductance: bool = True


class KramersKronigImaginaryManualWithCapacitanceInductance(
    KramersKronigImaginaryManual
):
    arg_add_capacitance: bool = True
    arg_add_inductance: bool = True
    cmp_cdc: str = "[RKKKKKKKKKKKKKKKKKCL]"
    cmp_mu: float = 0.1863497549045704


# ==============================================================================


class KramersKronigCNLS(KramersKronigComplex):
    arg_test: str = "cnls"
    cmp_cdc: str = "[RKKKKKKKKKKKKKK]"
    cmp_mu: float = 0.8309531383631978
    cmp_num_RC: int = 14
    cmp_pseudo_chisqr: float = 0.0013382040618625823


class KramersKronigCNLSWithCapacitance(KramersKronigCNLS):
    arg_max_nfev: int = 5
    arg_add_capacitance: bool = True
    cmp_cdc: str = "KC]"
    cmp_mu: float = -1.0
    cmp_num_RC: int = -1
    cmp_pseudo_chisqr: float = -1.0


class KramersKronigCNLSWithInductance(KramersKronigCNLS):
    arg_max_nfev: int = 5
    arg_add_inductance: bool = True
    cmp_cdc: str = "KL]"
    cmp_mu: float = -1.0
    cmp_num_RC: int = -1
    cmp_pseudo_chisqr: float = -1.0


class KramersKronigCNLSWithCapacitanceInductance(KramersKronigCNLS):
    arg_max_nfev: int = 5
    arg_add_capacitance: bool = True
    arg_add_inductance: bool = True
    cmp_cdc: str = "KCL]"
    cmp_mu: float = -1.0
    cmp_num_RC: int = -1
    cmp_pseudo_chisqr: float = -1.0


class KramersKronigCNLSManual(KramersKronigCNLS):
    arg_num_RC: int = 16
    cmp_cdc: str = "[RKKKKKKKKKKKKKKKK]"
    cmp_mu: float = 0.5071129711073297
    cmp_num_RC: int = 16
    cmp_pseudo_chisqr: float = 0.00147272884582866392


class KramersKronigCNLSManualWithCapacitance(KramersKronigCNLSManual):
    arg_max_nfev: int = 5
    arg_add_capacitance: bool = True
    cmp_cdc: str = "KC]"
    cmp_mu: float = -1.0
    cmp_num_RC: int = -1
    cmp_pseudo_chisqr: float = -1.0


class KramersKronigCNLSManualWithInductance(KramersKronigCNLSManual):
    arg_max_nfev: int = 5
    arg_add_inductance: bool = True
    cmp_cdc: str = "KL]"
    cmp_mu: float = -1.0
    cmp_num_RC: int = -1
    cmp_pseudo_chisqr: float = -1.0


class KramersKronigCNLSManualWithCapacitanceInductance(KramersKronigCNLSManual):
    arg_max_nfev: int = 5
    arg_add_capacitance: bool = True
    arg_add_inductance: bool = True
    cmp_cdc: str = "KCL]"
    cmp_mu: float = -1.0
    cmp_num_RC: int = -1
    cmp_pseudo_chisqr: float = -1.0


# ==============================================================================

TEST_EXPLORATORY_PLOTS: bool = True


class ExploratoryComplex(KramersKronigComplex):
    # kwargs
    arg_test: str = "complex"
    arg_num_RCs: List[int] = []
    arg_mu_criterion: float = 0.85
    arg_add_capacitance: bool = False
    arg_add_inductance: bool = False
    arg_method: str = "leastsq"
    arg_max_nfev: int = -1
    arg_num_procs: int = -1
    # comparison values
    cmp_cdc: str = "[RKKKKKKKKKKKKKKL]"
    cmp_mu: float = 0.828611766498053
    cmp_num_RC: int = 14
    cmp_pseudo_chisqr: float = 0.0010644917191345309

    @classmethod
    def setUpClass(cls):
        cls.results: List[TestResult] = perform_exploratory_tests(
            DATA,
            test=cls.arg_test,
            num_RCs=cls.arg_num_RCs,
            mu_criterion=cls.arg_mu_criterion,
            add_capacitance=cls.arg_add_capacitance,
            add_inductance=cls.arg_add_inductance,
            method=cls.arg_method,
            max_nfev=cls.arg_max_nfev,
            num_procs=cls.arg_num_procs,
        )
        cls.result: TestResult = cls.results[0]

    def test_single_process(self):
        results: List[TestResult] = perform_exploratory_tests(
            DATA,
            test=self.arg_test,
            num_RCs=self.arg_num_RCs,
            mu_criterion=self.arg_mu_criterion,
            add_capacitance=self.arg_add_capacitance,
            add_inductance=self.arg_add_inductance,
            method=self.arg_method,
            max_nfev=self.arg_max_nfev,
            num_procs=1,
        )
        self.assertIsInstance(results, list)
        if self.cmp_num_RC < 0:
            return
        self.assertTrue(all(map(lambda _: type(_) is TestResult, results)))
        self.assertTrue(
            results[0].calculate_score(self.arg_mu_criterion)
            > results[1].calculate_score(self.arg_mu_criterion)
            > results[-1].calculate_score(self.arg_mu_criterion)
        )
        result: TestResult = results[0]
        self.assertEqual(self.result.circuit.to_string(), result.circuit.to_string())
        self.assertEqual(self.result.num_RC, result.num_RC)
        self.assertEqual(self.result.mu, result.mu)
        self.assertEqual(self.result.pseudo_chisqr, result.pseudo_chisqr)
        self.assertTrue(allclose(self.result.frequencies, result.frequencies))
        self.assertTrue(allclose(self.result.impedances, result.impedances))
        self.assertTrue(allclose(self.result.residuals, result.residuals))

    def test_multi_process(self):
        results: List[TestResult] = perform_exploratory_tests(
            DATA,
            test=self.arg_test,
            num_RCs=self.arg_num_RCs,
            mu_criterion=self.arg_mu_criterion,
            add_capacitance=self.arg_add_capacitance,
            add_inductance=self.arg_add_inductance,
            method=self.arg_method,
            max_nfev=self.arg_max_nfev,
            num_procs=2,
        )
        self.assertIsInstance(results, list)
        if self.cmp_num_RC < 0:
            return
        self.assertTrue(all(map(lambda _: type(_) is TestResult, results)))
        self.assertTrue(
            results[0].calculate_score(self.arg_mu_criterion)
            > results[1].calculate_score(self.arg_mu_criterion)
            > results[-1].calculate_score(self.arg_mu_criterion)
        )
        result: TestResult = results[0]
        self.assertEqual(self.result.circuit.to_string(), result.circuit.to_string())
        self.assertEqual(self.result.num_RC, result.num_RC)
        self.assertEqual(self.result.mu, result.mu)
        self.assertEqual(self.result.pseudo_chisqr, result.pseudo_chisqr)
        self.assertTrue(allclose(self.result.frequencies, result.frequencies))
        self.assertTrue(allclose(self.result.impedances, result.impedances))
        self.assertTrue(allclose(self.result.residuals, result.residuals))

    def test_matplotlib(self):
        global TEST_EXPLORATORY_PLOTS
        if not TEST_EXPLORATORY_PLOTS:
            return
        TEST_EXPLORATORY_PLOTS = False
        check_mpl_return_values(
            self,
            *mpl.plot_mu_xps(
                self.results,
                mu_criterion=self.arg_mu_criterion,
            ),
        )
        check_mpl_return_values(
            self,
            *mpl.plot_mu_xps(
                self.results,
                mu_criterion=self.arg_mu_criterion,
                colored_axes=True,
            ),
        )
        check_mpl_return_values(
            self,
            *mpl.plot_tests(
                self.results,
                data=DATA,
                mu_criterion=self.arg_mu_criterion,
            ),
        )
        check_mpl_return_values(
            self,
            *mpl.plot_tests(
                self.results,
                data=DATA,
                mu_criterion=self.arg_mu_criterion,
                colored_axes=True,
            ),
        )


class ExploratoryReal(ExploratoryComplex):
    arg_test: str = "real"
    cmp_cdc: str = "[RKKKKKKKKKKKKKKKL]"
    cmp_mu: float = 0.8351972078416252
    cmp_num_RC: int = 15
    cmp_pseudo_chisqr: float = 0.0022502074349335334


class ExploratoryRealWithCapacitance(ExploratoryReal):
    arg_add_capacitance: bool = True
    cmp_cdc: str = "[RKKKKKKKKKKKKKKKCL]"
    cmp_mu: float = 0.8351972078416252
    cmp_num_RC: int = 15
    cmp_pseudo_chisqr: float = 0.0012584749704440204


class ExploratoryRealWithInductance(ExploratoryReal):
    arg_add_inductance: bool = True


class ExploratoryRealWithCapacitanceInductance(ExploratoryReal):
    arg_add_capacitance: bool = True
    arg_add_inductance: bool = True
    cmp_cdc: str = "[RKKKKKKKKKKKKKKKCL]"
    cmp_mu: float = 0.8351972078416252
    cmp_num_RC: int = 15
    cmp_pseudo_chisqr: float = 0.0012584749704440204


class ExploratoryRealManual(ExploratoryReal):
    arg_num_RCs: List[int] = list(range(2, DATA.get_num_points()))
    arg_add_capacitance: bool = True
    cmp_cdc: str = "[RKKKKKKKKKKKKKKKCL]"
    cmp_mu: float = 0.8351972078416252
    cmp_num_RC: int = 15
    cmp_pseudo_chisqr: float = 0.0012584749704440204


class ExploratoryRealManualWithCapacitance(ExploratoryRealManual):
    arg_add_capacitance: bool = True
    cmp_cdc: str = "[RKKKKKKKKKKKKKKKCL]"
    cmp_mu: float = 0.8351972078416252
    cmp_num_RC: int = 15
    cmp_pseudo_chisqr: float = 0.0012584749704440204


class ExploratoryRealManualWithInductance(ExploratoryRealManual):
    arg_add_inductance: bool = True


class ExploratoryRealManualWithCapacitanceInductance(ExploratoryRealManual):
    arg_add_capacitance: bool = True
    arg_add_inductance: bool = True
    cmp_cdc: str = "[RKKKKKKKKKKKKKKKCL]"
    cmp_mu: float = 0.8351972078416252
    cmp_num_RC: int = 15
    cmp_pseudo_chisqr: float = 0.0012584749704440204


# ==============================================================================


class ExploratoryImaginary(ExploratoryComplex):
    arg_test: str = "imaginary"
    cmp_cdc: str = "[RKKKKKKKKKKKKKL]"
    cmp_mu: float = 0.8454783508069239
    cmp_num_RC: int = 13
    cmp_pseudo_chisqr: float = 0.002190045082573739


class ExploratoryImaginaryWithCapacitance(ExploratoryImaginary):
    arg_add_capacitance: bool = True
    cmp_cdc: str = "[RKKKKKKKKKKKCL]"
    cmp_mu: float = 0.8463746482381072
    cmp_num_RC: int = 11
    cmp_pseudo_chisqr: float = 0.004059147148708858


class ExploratoryImaginaryWithInductance(ExploratoryImaginary):
    arg_add_inductance: bool = True


class ExploratoryImaginaryWithCapacitanceInductance(ExploratoryImaginary):
    arg_add_capacitance: bool = True
    arg_add_inductance: bool = True
    cmp_cdc: str = "[RKKKKKKKKKKKCL]"
    cmp_mu: float = 0.8463746482381072
    cmp_num_RC: int = 11
    cmp_pseudo_chisqr: float = 0.004059147148708858


class ExploratoryImaginaryManual(ExploratoryImaginary):
    arg_num_RCs: List[int] = list(range(2, DATA.get_num_points()))
    cmp_cdc: str = "[RKKKKKKKKKKKKKL]"
    cmp_mu: float = 0.8454783508069239
    cmp_num_RC: int = 13
    cmp_pseudo_chisqr: float = 0.002190045082573739


class ExploratoryImaginaryManualWithCapacitance(ExploratoryImaginaryManual):
    arg_add_capacitance: bool = True
    arg_add_capacitance: bool = True
    cmp_cdc: str = "[RKKKKKKKKKKKCL]"
    cmp_mu: float = 0.8463746482381072
    cmp_num_RC: int = 11
    cmp_pseudo_chisqr: float = 0.004059147148708858


class ExploratoryImaginaryManualWithInductance(ExploratoryImaginaryManual):
    arg_add_inductance: bool = True


class ExploratoryImaginaryManualWithCapacitanceInductance(ExploratoryImaginaryManual):
    arg_add_capacitance: bool = True
    arg_add_inductance: bool = True
    cmp_cdc: str = "[RKKKKKKKKKKKCL]"
    cmp_mu: float = 0.8463746482381072
    cmp_num_RC: int = 11
    cmp_pseudo_chisqr: float = 0.004059147148708858


# ==============================================================================


class ExploratoryCNLS(ExploratoryComplex):
    arg_test: str = "cnls"
    cmp_cdc: str = "[RKKKKKKKKKKKKKK]"
    cmp_mu: float = 0.830916902923411
    cmp_num_RC: int = 14
    cmp_pseudo_chisqr: float = 0.0013382191152589255


class ExploratoryCNLSWithCapacitance(ExploratoryCNLS):
    arg_max_nfev: int = 5
    arg_add_capacitance: bool = True
    cmp_cdc: str = "KC]"
    cmp_mu: float = -1.0
    cmp_num_RC: int = -1
    cmp_pseudo_chisqr: float = -1.0


class ExploratoryCNLSWithInductance(ExploratoryCNLS):
    arg_max_nfev: int = 5
    arg_add_inductance: bool = True
    cmp_cdc: str = "KL]"
    cmp_mu: float = -1.0
    cmp_num_RC: int = -1
    cmp_pseudo_chisqr: float = -1.0


class ExploratoryCNLSWithCapacitanceInductance(ExploratoryCNLS):
    arg_max_nfev: int = 5
    arg_add_capacitance: bool = True
    arg_add_inductance: bool = True
    cmp_cdc: str = "KCL]"
    cmp_mu: float = -1.0
    cmp_num_RC: int = -1
    cmp_pseudo_chisqr: float = -1.0


class ExploratoryCNLSManual(ExploratoryCNLS):
    arg_num_RCs: List[int] = list(range(2, DATA.get_num_points()))
    cmp_cdc: str = "[RKKKKKKKKKKKKKK]"
    cmp_mu: float = 0.830916902923411
    cmp_num_RC: int = 14
    cmp_pseudo_chisqr: float = 0.0013382191152589255


class ExploratoryCNLSManualWithCapacitance(ExploratoryCNLSManual):
    arg_max_nfev: int = 5
    arg_add_capacitance: bool = True
    cmp_cdc: str = "KC]"
    cmp_mu: float = -1.0
    cmp_num_RC: int = -1
    cmp_pseudo_chisqr: float = -1.0


class ExploratoryCNLSManualWithInductance(ExploratoryCNLSManual):
    arg_max_nfev: int = 5
    arg_add_inductance: bool = True
    cmp_cdc: str = "KL]"
    cmp_mu: float = -1.0
    cmp_num_RC: int = -1
    cmp_pseudo_chisqr: float = -1.0


class ExploratoryCNLSManualWithCapacitanceInductance(ExploratoryCNLSManual):
    arg_max_nfev: int = 5
    arg_add_capacitance: bool = True
    arg_add_inductance: bool = True
    cmp_cdc: str = "KCL]"
    cmp_mu: float = -1.0
    cmp_num_RC: int = -1
    cmp_pseudo_chisqr: float = -1.0
