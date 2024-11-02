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
from lmfit.minimizer import MinimizerResult
from numpy import (
    allclose,
    angle,
    array,
    float64,
)
from numpy.random import (
    seed,
    normal,
)
from numpy.typing import NDArray
from pandas import DataFrame
from pyimpspec import (
    Circuit,
    DataSet,
    Element,
    FitIdentifiers,
    FitResult,
    FittedParameter,
    fit_circuit,
    generate_mock_data,
    generate_fit_identifiers,
    parse_cdc,
    parse_data,
)
from pyimpspec.analysis.fitting import validate_circuit
from pyimpspec.typing import (
    ComplexImpedance,
    ComplexImpedances,
    Frequencies,
    Impedances,
    Phases,
    Residuals,
)
from pyimpspec.typing.helpers import (
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
)
from pyimpspec import progress as PROGRESS
from test_matplotlib import (
    check_mpl_return_values,
    mpl,
    primitive_mpl_plotters,
)


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
        ),
        dtype=ComplexImpedance,
    )
)


# TODO: Implement tests for invalid arguments
class Fitting(TestCase):
    # kwargs
    arg_method: str = "least_squares"
    arg_weight: str = "proportional"
    arg_max_nfev: int = -1
    arg_num_procs: int = -1
    # comparison values
    cmp_pseudo_chisqr: float = 1.3369801618060505e-3
    cmp_chisqr: float = 1.959433221995857e-05
    cmp_redchi: float = 3.697043815086523e-07
    cmp_aic: float = -854
    cmp_bic: float = -844
    cmp_units: List[str] = [
        "ohm",
        "ohm",
        "F",
        "ohm",
        "S*s^n",
        "",
        "F",
    ]

    @classmethod
    def setUpClass(cls):
        cls.circuit: Circuit = parse_cdc("R(RC)(RW)C{C=5e2f}")
        cls.result: FitResult = fit_circuit(
            circuit=cls.circuit,
            data=DATA,
            method=cls.arg_method,
            weight=cls.arg_weight,
            max_nfev=cls.arg_max_nfev,
            num_procs=cls.arg_num_procs,
        )

    def test_auto(self):
        result: FitResult = fit_circuit(
            circuit=self.circuit,
            data=DATA,
            method="auto",
            weight="auto",
            max_nfev=self.arg_max_nfev,
            num_procs=1,
        )
        self.assertNotEqual(result.method, "")
        self.assertNotEqual(result.weight, "")
        self.assertNotEqual(result.method, "auto")
        self.assertNotEqual(result.weight, "auto")

    def test_methods_list(self):
        methods: List[str] = ["powell"]
        result: FitResult = fit_circuit(
            circuit=self.circuit,
            data=DATA,
            method=methods,
            weight="boukamp",
            max_nfev=self.arg_max_nfev,
            num_procs=1,
        )
        self.assertTrue(result.method in methods)
        self.assertEqual(result.weight, "boukamp")

    def test_weights_list(self):
        weights: List[str] = ["unity"]
        result: FitResult = fit_circuit(
            circuit=self.circuit,
            data=DATA,
            method="leastsq",
            weight=weights,
            max_nfev=self.arg_max_nfev,
            num_procs=1,
        )
        self.assertEqual(result.method, "leastsq")
        self.assertTrue(result.weight in weights)

    def test_single_process(self):
        result: FitResult = fit_circuit(
            circuit=self.circuit,
            data=DATA,
            method=self.arg_method,
            weight=self.arg_weight,
            max_nfev=self.arg_max_nfev,
            num_procs=1,
        )
        self.assertEqual(self.result.circuit.to_string(), result.circuit.to_string())
        self.assertEqual(
            self.result.minimizer_result.chisqr, result.minimizer_result.chisqr
        )
        self.assertEqual(
            self.result.minimizer_result.redchi, result.minimizer_result.redchi
        )
        self.assertTrue(allclose(self.result.frequencies, result.frequencies))
        self.assertTrue(allclose(self.result.impedances, result.impedances))
        self.assertTrue(allclose(self.result.residuals, result.residuals))

    def test_multi_process(self):
        result: FitResult = fit_circuit(
            circuit=self.circuit,
            data=DATA,
            method=self.arg_method,
            weight=self.arg_weight,
            max_nfev=self.arg_max_nfev,
            num_procs=2,
        )
        self.assertEqual(self.result.circuit.to_string(), result.circuit.to_string())
        self.assertEqual(
            self.result.minimizer_result.chisqr, result.minimizer_result.chisqr
        )
        self.assertEqual(
            self.result.minimizer_result.redchi, result.minimizer_result.redchi
        )
        self.assertTrue(allclose(self.result.frequencies, result.frequencies))
        self.assertTrue(allclose(self.result.impedances, result.impedances))
        self.assertTrue(allclose(self.result.residuals, result.residuals))

    def test_return_type(self):
        self.assertIsInstance(self.result, FitResult)

    def test_repr(self):
        self.assertEqual(
            repr(self.result),
            f"FitResult ({self.circuit.to_string()}, {hex(id(self.result))})",
        )

    def test_cdc(self):
        self.assertEqual(
            self.result.circuit.to_string(0),
            "[R{R=1E+02/0E+00/inf}(R{R=2E+02/0E+00/inf}C{C=8E-07/1E-24/1E+03})(R{R=5E+02/0E+00/inf}W{Y=4E-04/1E-24/inf,n=5E-01F/0E+00/1E+00})C{C=5E+02F/1E-24/1E+03}]",
        )

    def test_get_frequencies(self):
        self.assertTrue(allclose(self.result.get_frequencies(), DATA.get_frequencies()))

    def test_get_impedances(self):
        self.assertTrue(
            allclose(self.result.get_impedances(), DATA.get_impedances(), rtol=1)
        )

    def test_get_nyquist_data(self):
        Zre: Impedances
        Zim: Impedances
        Zre, Zim = self.result.get_nyquist_data(num_per_decade=2)
        self.assertTrue(len(Zre) == len(Zim) == 9)
        Zre, Zim = self.result.get_nyquist_data()
        self.assertTrue(allclose(self.result.get_impedances().real, Zre))
        self.assertTrue(allclose(self.result.get_impedances().imag, -Zim))

    def test_get_bode_data(self):
        f: Frequencies
        mag: Impedances
        phi: Phases
        f, mag, phi = self.result.get_bode_data(num_per_decade=2)
        self.assertTrue(len(f) == len(mag) == len(phi) == 9)
        f, mag, phi = self.result.get_bode_data()
        self.assertTrue(allclose(self.result.get_frequencies(), f))
        self.assertTrue(allclose(abs(self.result.get_impedances()), mag))
        self.assertTrue(allclose(angle(self.result.get_impedances(), deg=True), -phi))

    def test_minimizer_result(self):
        self.assertIsInstance(self.result.minimizer_result, MinimizerResult)

    def test_pseudo_chisqr(self):
        self.assertAlmostEqual(
            self.result.pseudo_chisqr,
            self.cmp_pseudo_chisqr,
            delta=1e-4,
        )

    def test_chisqr(self):
        self.assertAlmostEqual(
            self.result.minimizer_result.chisqr,
            self.cmp_chisqr,
            delta=1e-6,
        )

    def test_redchi(self):
        self.assertAlmostEqual(
            self.result.minimizer_result.redchi,
            self.cmp_redchi,
            delta=1e-8,
        )

    def test_aic(self):
        self.assertAlmostEqual(
            self.result.minimizer_result.aic,
            self.cmp_aic,
            delta=1e0,
        )

    def test_bic(self):
        self.assertAlmostEqual(
            self.result.minimizer_result.bic,
            self.cmp_bic,
            delta=1e0,
        )

    def test_get_residuals_data(self):
        residual_data: Tuple[
            Frequencies,
            Residuals,
            Residuals,
        ] = self.result.get_residuals_data()
        self.assertIsInstance(residual_data, tuple)
        self.assertEqual(len(residual_data), 3)
        f: Frequencies = self.result.get_frequencies()
        self.assertTrue(allclose(residual_data[0], f))
        Z_exp: ComplexImpedances = DATA.get_impedances()
        Z_fit: ComplexImpedances = self.result.get_impedances()
        real_residual: NDArray[float64] = (Z_exp.real - Z_fit.real) / abs(Z_exp) * 100
        self.assertTrue(allclose(residual_data[1], real_residual))
        imaginary_residual: NDArray[float64] = (
            (Z_exp.imag - Z_fit.imag) / abs(Z_exp) * 100
        )
        self.assertTrue(allclose(residual_data[2], imaginary_residual))

    def test_method(self):
        self.assertIsInstance(self.result.method, str)
        self.assertNotEqual(self.result.method.strip(), "")
        self.assertEqual(self.result.method, self.arg_method)

    def test_weight(self):
        self.assertIsInstance(self.result.weight, str)
        self.assertNotEqual(self.result.weight.strip(), "")
        self.assertEqual(self.result.weight, self.arg_weight)

    def test_to_parameters_dataframe_to_markdown(self):
        df: DataFrame = self.result.to_parameters_dataframe()
        self.assertIsInstance(df, DataFrame)
        lines: List[str] = df.to_markdown().split("\n")
        self.assertEqual(len(lines), 9)
        line: str = lines.pop(0)
        self.assertTrue(
            0
            < line.index("Element")
            < line.index("Parameter")
            < line.index("Value")
            < line.index("Std. err. (%)")
            < line.index("Unit")
            < line.index("Fixed")
        )
        lines.pop(0)

        i: int = 0
        while lines:
            line = lines.pop(0)
            columns: List[str] = list(map(str.strip, line.split("|")))[1:-1]
            self.assertEqual(len(columns), 7, msg=f"{line=}")
            self.assertEqual(int(columns[0]), i)
            self.assertTrue(columns[1] in self.result.parameters)
            self.assertAlmostEqual(
                float(columns[3]),
                self.result.parameters[columns[1]][columns[2]].value,
                delta=0.1 * self.result.parameters[columns[1]][columns[2]].value,
            )
            self.assertEqual(
                columns[5],
                self.cmp_units[i],
            )
            self.assertEqual(
                columns[6],
                "Yes" if self.result.parameters[columns[1]][columns[2]].fixed else "No",
            )
            i += 1
        markdown: str = self.result.to_parameters_dataframe(running=True).to_markdown()
        self.assertTrue("R_0" in markdown)
        self.assertTrue("R_1" in markdown)
        self.assertTrue("C_2" in markdown)
        self.assertTrue("R_3" in markdown)
        self.assertTrue("W_4" in markdown)
        self.assertTrue("C_5" in markdown)

    def test_to_parameters_dataframe_to_latex(self):
        df: DataFrame = self.result.to_parameters_dataframe()
        self.assertIsInstance(df, DataFrame)
        lines = (
            df.style.format(precision=8)
            .format_index(axis="columns", escape="latex")
            .to_latex(hrules=True)
            .split("\n")
        )
        self.assertEqual(lines.pop(0), r"\begin{tabular}{lllrrll}")
        self.assertEqual(lines.pop(0), r"\toprule")
        line = lines.pop(0)
        self.assertTrue(
            0
            < line.index("Element")
            < line.index("Parameter")
            < line.index("Value")
            < line.index(r"Std. err. (\%)")
            < line.index("Unit")
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

            columns: List[str] = list(map(str.strip, line.split("&")))
            self.assertEqual(len(columns), 7, msg=f"{line=}")
            self.assertEqual(int(columns[0]), i)
            self.assertTrue(columns[1].replace(r"\_", "_") in self.result.parameters)
            self.assertAlmostEqual(
                float(columns[3]),
                self.result.parameters[columns[1].replace(r"\_", "_")][
                    columns[2]
                ].value,
                delta=0.1
                * self.result.parameters[columns[1].replace(r"\_", "_")][
                    columns[2]
                ].value,
            )
            self.assertEqual(
                columns[5],
                self.cmp_units[i],
            )
            self.assertEqual(
                columns[6],
                "Yes"
                if self.result.parameters[columns[1].replace(r"\_", "_")][
                    columns[2]
                ].fixed
                else "No",
            )

            i += 1

    def test_to_statistics_dataframe(self):
        df: DataFrame = self.result.to_statistics_dataframe()
        self.assertIsInstance(df, DataFrame)

    def test_circuit_validation(self):
        self.assertEqual(validate_circuit(parse_cdc("RR")), None)
        self.assertEqual(validate_circuit(parse_cdc("R{:a}R{:b}")), None)
        with self.assertRaises(ValueError):
            validate_circuit(parse_cdc("R{:a}R{:a}"))

    def test_labeled_elements(self):
        circuit: Circuit = parse_cdc("R{:a}(R{:b}C{:c})(R{:d}W{:e})C{C=5e2f:f}")
        fit_circuit(
            circuit=circuit,
            data=DATA,
            method=self.arg_method,
            weight=self.arg_weight,
            max_nfev=self.arg_max_nfev,
            num_procs=self.arg_num_procs,
        )

    def test_matplotlib(self):
        plotter: Callable
        for plotter in primitive_mpl_plotters:
            check_mpl_return_values(self, *plotter(data=self.result))

        check_mpl_return_values(self, *mpl.plot_residuals(self.result))
        check_mpl_return_values(self, *mpl.plot_fit(self.result, data=DATA))
        check_mpl_return_values(self, *mpl.plot_fit(self.result, data=DATA))
        with self.assertRaises(AttributeError):
            mpl.plot_fit(DATA, data=DATA)

    def test_fit_identifiers(self):
        data: DataSet = generate_mock_data("CIRCUIT_5", noise=5e-2, seed=42)[0]

        circuit: Circuit = parse_cdc("R(RQ)(RQ)(RQ)")
        elements: List[Element] = circuit.get_elements()
        R1, R2, Q1, R3, Q2, R4, Q3 = elements
        identifiers: Dict[Element, FitIdentifiers] = generate_fit_identifiers(circuit)

        # Make sure the types are as expected
        self.assertIsInstance(identifiers, dict)
        self.assertTrue(all(map(lambda k: isinstance(k, Element), identifiers.keys())))
        self.assertTrue(all(map(lambda v: isinstance(v, FitIdentifiers), identifiers.values())))
        self.assertTrue(all(map(lambda v: all(map(lambda kv: isinstance(kv[0], str) and isinstance(kv[1], str), v.items())), identifiers.values())))

        # Make sure that each element was processed
        self.assertTrue(all(map(lambda e: e in identifiers, elements)))

        # Make sure that all parameter symbols are valid
        for element in elements:
            for symbol in identifiers[element]:
                element.get_value(symbol)

        # Make sure all identifiers are unique
        self.assertEqual(len(set([identifiers[R].R for R in (R1, R2, R3, R4)])), 4)
        self.assertEqual(len(set([identifiers[Q].Y for Q in (Q1, Q2, Q3)])), 3)
        self.assertEqual(len(set([identifiers[Q].n for Q in (Q1, Q2, Q3)])), 3)

        # Make sure the identifiers have the expected values
        self.assertEqual(identifiers[R1].R, "R_0")

        self.assertEqual(identifiers[R2].R, "R_1")
        self.assertEqual(identifiers[Q1].Y, "Y_2")
        self.assertEqual(identifiers[Q1].n, "n_2")

        self.assertEqual(identifiers[R3].R, "R_3")
        self.assertEqual(identifiers[Q2].Y, "Y_4")
        self.assertEqual(identifiers[Q2].n, "n_4")

        self.assertEqual(identifiers[R4].R, "R_5")
        self.assertEqual(identifiers[Q3].Y, "Y_6")
        self.assertEqual(identifiers[Q3].n, "n_6")

        # Make sure the identifiers can be accessed
        self.assertEqual(identifiers[R1].R, identifiers[R1]["R"])
        self.assertEqual(identifiers[Q1].Y, identifiers[Q1]["Y"])
        self.assertEqual(identifiers[Q1].n, identifiers[Q1]["n"])

        with self.assertRaises(AttributeError):
            identifiers[R1].Y

        with self.assertRaises(AttributeError):
            identifiers[R1]["Y"]

        with self.assertRaises(AttributeError):
            identifiers[Q1].N

        with self.assertRaises(AttributeError):
            identifiers[Q1]["N"]

        with self.assertRaises(AttributeError):
            identifiers[Q1].R

        with self.assertRaises(AttributeError):
            identifiers[Q1]["R"]

        fit: FitResult = fit_circuit(
          circuit,
          data,
          method="least_squares",
          weight="boukamp",
          constraint_expressions={
            identifiers[R3].R: f"{identifiers[R2].R} + alpha",
            identifiers[R4].R: f"{identifiers[R3].R} - beta",
            identifiers[Q2].Y: f"{identifiers[Q1].Y} + gamma",
            identifiers[Q3].Y: f"{identifiers[Q2].Y} + delta",
          },
          constraint_variables=dict(
            alpha=dict(
              value=500,
              min=1,
            ),
            beta=dict(
              value=300,
              min=1,
            ),
            gamma=dict(
              value=1e-8,
              min=1e-12,
            ),
            delta=dict(
              value=2e-7,
              min=1e-12,
            ),
          ),
        )

        R1, R2, Q1, R3, Q2, R4, Q3 = fit.circuit.get_elements()

        # Make sure the values are in the expected order
        self.assertLess(R2.get_value("R"), R4.get_value("R"))
        self.assertLess(R4.get_value("R"), R3.get_value("R"))
        self.assertLess(Q1.get_value("Y"), Q2.get_value("Y"))
        self.assertLess(Q2.get_value("Y"), Q3.get_value("Y"))
