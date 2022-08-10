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
    Circuit,
    DataSet,
    KramersKronigResult,
    FittingResult,
    FittedParameter,
    fit_circuit_to_data,
    perform_exploratory_tests,
    perform_test,
    score_test_results,
    string_to_circuit,
)
from pyimpspec.analysis.fitting import validate_circuit
from numpy import array
from numpy.random import seed, normal
from typing import List


seed(42)
DATA: DataSet = DataSet.from_dict(
    {
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
sd: float = 0.005
DATA.subtract_impedance(
    -array(
        list(
            map(
                lambda _: complex(
                    abs(_) * normal(0, sd, 1),
                    abs(_) * normal(0, sd, 1),
                ),
                DATA.get_impedance(),
            )
        )
    )
)


class TestKramersKronig(TestCase):
    def test_01_test(self):
        # complex
        test: KramersKronigResult = perform_test(DATA)
        self.assertTrue(type(test) is KramersKronigResult)
        self.assertEqual(test.num_RC, 2)
        self.assertAlmostEqual(test.mu, 0.8465317822199895)
        self.assertAlmostEqual(test.pseudo_chisqr, 5.170991769675877)
        self.assertEqual(test.circuit.to_string(), "[RKKL]")
        # real
        test = perform_test(DATA, test="real")
        self.assertEqual(test.num_RC, 15)
        self.assertAlmostEqual(test.mu, 0.8351972078416112)
        self.assertAlmostEqual(test.pseudo_chisqr, 0.0022502074349343235)
        self.assertEqual(test.circuit.to_string(), "[RKKKKKKKKKKKKKKKL]")
        # imaginary
        test = perform_test(DATA, test="imaginary")
        self.assertEqual(test.num_RC, 2)
        self.assertAlmostEqual(test.mu, 0.7634133056991227)
        self.assertAlmostEqual(test.pseudo_chisqr, 5.397326608733074)
        self.assertEqual(test.circuit.to_string(), "[RKKL]")
        # cnls
        test = perform_test(DATA, test="cnls", max_nfev=1000)
        self.assertEqual(test.num_RC, 14)
        self.assertAlmostEqual(test.mu, 0.8309531383631978)
        self.assertAlmostEqual(test.pseudo_chisqr, 0.0013382040618625823)
        self.assertEqual(test.circuit.to_string(), "[RKKKKKKKKKKKKKK]")

    def test_02_add_capacitance(self):
        # complex
        test: KramersKronigResult = perform_test(DATA, add_capacitance=True)
        self.assertTrue("C" in test.circuit.to_string())
        # real
        test = perform_test(DATA, test="real", add_capacitance=True)
        self.assertTrue("C" in test.circuit.to_string())
        # imaginary
        test = perform_test(DATA, test="imaginary", add_capacitance=True)
        self.assertTrue("C" in test.circuit.to_string())
        # cnls
        test = perform_test(DATA, test="cnls", add_capacitance=True, max_nfev=1000)
        self.assertTrue("C" in test.circuit.to_string())

    def test_03_add_inductance(self):
        # complex
        test: KramersKronigResult = perform_test(DATA, add_inductance=True)
        self.assertTrue("L" in test.circuit.to_string())
        # real
        test = perform_test(DATA, test="real", add_inductance=True)
        self.assertTrue("L" in test.circuit.to_string())
        # imaginary
        test = perform_test(DATA, test="imaginary", add_inductance=True)
        self.assertTrue("L" in test.circuit.to_string())
        # cnls
        test = perform_test(DATA, test="cnls", add_inductance=True, max_nfev=1000)
        self.assertTrue("L" in test.circuit.to_string())

    def test_04_add_capacitance_inductance(self):
        # complex
        test: KramersKronigResult = perform_test(
            DATA, add_capacitance=True, add_inductance=True
        )
        self.assertTrue("CL" in test.circuit.to_string())
        # real
        test = perform_test(
            DATA, test="real", add_capacitance=True, add_inductance=True
        )
        self.assertTrue("CL" in test.circuit.to_string())
        # imaginary
        test = perform_test(
            DATA, test="imaginary", add_capacitance=True, add_inductance=True
        )
        self.assertTrue("CL" in test.circuit.to_string())
        # cnls
        test = perform_test(
            DATA, test="cnls", add_capacitance=True, add_inductance=True, max_nfev=1000
        )
        self.assertTrue("CL" in test.circuit.to_string())

    def test_05_num_RC(self):
        num_RC: int = 10
        # complex
        test: KramersKronigResult = perform_test(DATA, num_RC=num_RC)
        self.assertEqual(test.num_RC, num_RC)
        self.assertEqual(test.circuit.to_string().count("K"), num_RC)
        # real
        test = perform_test(DATA, test="real", num_RC=num_RC)
        self.assertEqual(test.num_RC, num_RC)
        self.assertEqual(test.circuit.to_string().count("K"), num_RC)
        # imaginary
        test = perform_test(DATA, test="imaginary", num_RC=num_RC)
        self.assertEqual(test.num_RC, num_RC)
        self.assertEqual(test.circuit.to_string().count("K"), num_RC)
        # cnls
        test = perform_test(DATA, test="cnls", num_RC=num_RC, max_nfev=1000)
        self.assertEqual(test.num_RC, num_RC)
        self.assertEqual(test.circuit.to_string().count("K"), num_RC)

    def test_06_mu_criterion(self):
        mu_criterion: float = 0.74
        # complex
        test: KramersKronigResult = perform_test(DATA, mu_criterion=mu_criterion)
        self.assertTrue(test.mu <= mu_criterion)
        # real
        test = perform_test(DATA, test="real", mu_criterion=mu_criterion)
        self.assertTrue(test.mu <= mu_criterion)
        # imaginary
        test = perform_test(DATA, test="imaginary", mu_criterion=mu_criterion)
        self.assertTrue(test.mu <= mu_criterion)
        # cnls
        test = perform_test(DATA, test="cnls", mu_criterion=mu_criterion, max_nfev=1000)
        self.assertTrue(test.mu <= mu_criterion)

    def test_07_threading(self):
        # complex
        test_single_thread: KramersKronigResult = perform_test(DATA, num_procs=1)
        test_multithreaded: KramersKronigResult = perform_test(DATA, num_procs=2)
        self.assertEqual(test_single_thread.num_RC, 2)
        self.assertAlmostEqual(test_single_thread.mu, 0.8465317822199895)
        self.assertAlmostEqual(test_single_thread.pseudo_chisqr, 5.170991769675877)
        self.assertEqual(test_single_thread.circuit.to_string(), "[RKKL]")
        self.assertEqual(test_multithreaded.num_RC, 2)
        self.assertAlmostEqual(test_multithreaded.mu, 0.8465317822199895)
        self.assertAlmostEqual(test_multithreaded.pseudo_chisqr, 5.170991769675877)
        self.assertEqual(test_multithreaded.circuit.to_string(), "[RKKL]")
        # cnls
        test_single_thread: KramersKronigResult = perform_test(
            DATA, test="cnls", num_procs=1, max_nfev=1000
        )
        test_multithreaded: KramersKronigResult = perform_test(
            DATA, test="cnls", num_procs=2, max_nfev=1000
        )
        self.assertEqual(test_single_thread.num_RC, 14)
        self.assertAlmostEqual(test_single_thread.mu, 0.8309531383631978)
        self.assertAlmostEqual(test_single_thread.pseudo_chisqr, 0.0013382040618625823)
        self.assertEqual(test_single_thread.circuit.to_string(), "[RKKKKKKKKKKKKKK]")
        self.assertEqual(test_multithreaded.num_RC, 14)
        self.assertAlmostEqual(test_multithreaded.mu, 0.8309531383631978)
        self.assertAlmostEqual(test_multithreaded.pseudo_chisqr, 0.0013382040618625823)
        self.assertEqual(test_single_thread.circuit.to_string(), "[RKKKKKKKKKKKKKK]")

    def test_08_exploratory(self):
        mu_criterion: float = 0.58
        tests: List[KramersKronigResult]
        test: KramersKronigResult
        # complex
        tests = perform_exploratory_tests(DATA, mu_criterion=mu_criterion)
        self.assertTrue(
            type(tests) is list
            and all(map(lambda _: type(_) is KramersKronigResult, tests))
        )
        test = score_test_results(tests, mu_criterion)[0][1]
        self.assertTrue(test.mu <= mu_criterion)
        # real
        tests = perform_exploratory_tests(DATA, test="real")
        # imaginary
        tests = perform_exploratory_tests(DATA, test="imaginary")
        # cnls
        tests = perform_exploratory_tests(DATA, test="cnls", max_nfev=1000)


class TestFitting(TestCase):
    def test_01_default(self):
        circuit: Circuit = string_to_circuit("R(RC)(RW)")
        fit: FittingResult = fit_circuit_to_data(circuit, DATA)
        self.assertEqual(
            fit.circuit.to_string(0),
            "[R{R=1E+02/0E+00}(R{R=2E+02/0E+00}C{C=8E-07/0E+00/1E+03})(R{R=5E+02/0E+00}W{Y=4E-04/0E+00})]",
        )
        param: FittedParameter = fit.parameters["R_0"]["R"]
        self.assertAlmostEqual(param.value, 1.00E2, delta=2E0)
        param: FittedParameter = fit.parameters["R_1"]["R"]
        self.assertAlmostEqual(param.value, 2.02E2, delta=2E0)
        param: FittedParameter = fit.parameters["C_2"]["C"]
        self.assertAlmostEqual(param.value, 8.00E-7, delta=1E-8)
        param: FittedParameter = fit.parameters["R_3"]["R"]
        self.assertAlmostEqual(param.value, 5.03E2, delta=2E0)
        param: FittedParameter = fit.parameters["W_4"]["Y"]
        self.assertAlmostEqual(param.value, 4.00E-4, delta=1E-5)
        # Markdown table
        lines: List[str] = fit.to_dataframe().to_markdown().split("\n")
        self.assertEqual(len(lines), 7)
        line: str = lines.pop(0)
        self.assertTrue(
            0
            < line.index("Element")
            < line.index("Parameter")
            < line.index("Value")
            < line.index("Std. err. (%)")
            < line.index("Fixed")
        )
        lines.pop(0)
        i: int = 0
        while lines:
            line = lines.pop(0)
            columns: List[str] = list(
                filter(lambda _: _ != "", map(str.strip, line.split("|")))
            )
            self.assertEqual(len(columns), 6)
            self.assertEqual(int(columns[0]), i)
            self.assertTrue(columns[1].endswith(f"_{i}"))
            self.assertTrue(columns[1] in fit.parameters)
            self.assertAlmostEqual(
                float(columns[3]),
                fit.parameters[columns[1]][columns[2]].value,
                delta=0.1 * fit.parameters[columns[1]][columns[2]].value,
            )
            self.assertEqual(
                columns[5],
                "Yes"
                if fit.parameters[columns[1]][columns[2]].fixed is True
                else "No",
            )
            i += 1
        # LaTeX table
        lines = fit.to_dataframe().to_latex().split("\n")
        self.assertEqual(lines.pop(0), r"\begin{tabular}{lllrrl}")
        self.assertEqual(lines.pop(0), r"\toprule")
        line = lines.pop(0)
        self.assertTrue(
            0
            < line.index("Element")
            < line.index("Parameter")
            < line.index("Value")
            < line.index(r"Std. err. (\%)")
            < line.index("Fixed")
        )
        self.assertEqual(lines.pop(0), r"\midrule")
        self.assertEqual(lines.pop(), "")
        self.assertEqual(lines.pop(), r"\end{tabular}")
        self.assertEqual(lines.pop(), r"\bottomrule")
        i = 0
        while lines:
            line = lines.pop(0).replace(r"\\", "").strip()
            if line == "":
                continue
            columns: List[str] = list(
                filter(lambda _: _ != "", map(str.strip, line.split("&")))
            )
            self.assertEqual(len(columns), 6)
            self.assertEqual(int(columns[0]), i)
            self.assertTrue(columns[1].endswith(f"_{i}"))
            self.assertTrue(columns[1].replace(r"\_", "_") in fit.parameters)
            self.assertAlmostEqual(
                float(columns[3]),
                fit.parameters[columns[1].replace(r"\_", "_")][columns[2]].value,
                delta=0.1
                * fit.parameters[columns[1].replace(r"\_", "_")][columns[2]].value,
            )
            self.assertEqual(
                columns[5],
                "Yes"
                if fit.parameters[columns[1].replace(r"\_", "_")][columns[2]].fixed
                is True
                else "No",
            )
            i += 1

    def test_02_threading(self):
        circuit: Circuit = string_to_circuit("R(RC)(RW)")
        fit_single_thread: FittingResult = fit_circuit_to_data(
            circuit, DATA, max_nfev=100, num_procs=1
        )
        fit_multithreaded: FittingResult = fit_circuit_to_data(
            circuit, DATA, max_nfev=100, num_procs=2
        )
        self.assertEqual(
            fit_single_thread.circuit.to_string(2),
            fit_multithreaded.circuit.to_string(2),
        )

    def test_03_circuit_validation(self):
        self.assertEqual(validate_circuit(string_to_circuit("RR")), None)
        self.assertEqual(validate_circuit(string_to_circuit("R{:a}R{:b}")), None)
        with self.assertRaises(AssertionError):
            validate_circuit(string_to_circuit("R{:a}R{:a}"))
