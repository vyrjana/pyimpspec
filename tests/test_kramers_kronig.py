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
from numpy import (
    array,
    complex128,
    diff,
    float64,
    isclose,
    isnan,
    isneginf,
    isposinf,
    ndarray,
    pi,
    sqrt,
    zeros,
)
from pandas import DataFrame
import pyimpspec
from pyimpspec import (
    Circuit,
    Element,
    Capacitor,
    Inductor,
    KramersKronigRC,
    KramersKronigAdmittanceRC,
    Resistor,
    DataSet,
    KramersKronigResult,
    parse_cdc,
)
from pyimpspec.exceptions import KramersKronigError
from pyimpspec import progress as PROGRESS
from pyimpspec.typing import (
    ComplexImpedances,
    Frequencies,
    Impedances,
    Phases,
    Residuals,
    TimeConstants,
)
from pyimpspec.typing.helpers import (
    Callable,
    Dict,
    List,
    NDArray,
    Optional,
    Tuple,
    _is_floating,
)
from test_matplotlib import (
    check_mpl_return_values,
    mpl,
    primitive_mpl_plotters,
)
from pyimpspec.analysis.kramers_kronig.algorithms.utility.osculating_circle import (
    _fit_osculating_circle,
    calculate_curvatures,
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


VALID_DATA: DataSet
INVALID_DATA: DataSet
VALID_DATA, INVALID_DATA, *_ = pyimpspec.generate_mock_data(
    "CIRCUIT_1*",
    noise=5e-2,
    seed=42,
)


class KramersKronigUtility(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.w: NDArray[float64] = 2 * pi * VALID_DATA.get_frequencies()
        cls.Z_exp: ComplexImpedances = VALID_DATA.get_impedances()

    def test_generate_time_constants(self):
        num_RC: int
        for num_RC in (2, 5, 10):
            log_F_ext: float
            for log_F_ext in (0.0, 0.5, -0.5):
                taus = (
                    pyimpspec.analysis.kramers_kronig.utility._generate_time_constants(
                        w=self.w,
                        num_RC=num_RC,
                        log_F_ext=log_F_ext,
                    )
                )
                F_ext: float64 = 10**log_F_ext

                self.assertIsInstance(taus, ndarray)
                self.assertEqual(taus.dtype, float64)
                self.assertEqual(len(taus.shape), 1)
                self.assertEqual(len(taus), num_RC)
                self.assertAlmostEqual(min(taus), 1 / (max(self.w) * F_ext))
                self.assertAlmostEqual(max(taus), F_ext / min(self.w))
                self.assertEqual(min(taus), taus[0])
                self.assertEqual(max(taus), taus[-1])
                self.assertTrue((diff(taus) > 0.0).all())

        with self.assertRaises(ValueError):
            pyimpspec.analysis.kramers_kronig.utility._generate_time_constants(
                w=self.w,
                num_RC=1,
                log_F_ext=0.0,
            )

        with self.assertRaises(ValueError):
            pyimpspec.analysis.kramers_kronig.utility._generate_time_constants(
                w=self.w,
                num_RC=0,
                log_F_ext=0.0,
            )

        with self.assertRaises(ValueError):
            pyimpspec.analysis.kramers_kronig.utility._generate_time_constants(
                w=self.w,
                num_RC=-1,
                log_F_ext=0.0,
            )

    def test_generate_circuit(self):
        num_RC: int
        for num_RC in (2, 5, 10):
            taus: NDArray[float64]
            taus = pyimpspec.analysis.kramers_kronig.utility._generate_time_constants(
                w=self.w,
                num_RC=num_RC,
                log_F_ext=0.0,
            )

            admittance: bool
            for admittance in (False, True):
                add_capacitance: bool
                add_inductance: bool
                for add_capacitance, add_inductance in (
                    (False, False),
                    (True, False),
                    (False, True),
                    (True, True),
                ):
                    circuit: Circuit
                    circuit = (
                        pyimpspec.analysis.kramers_kronig.utility._generate_circuit(
                            taus=taus,
                            add_capacitance=add_capacitance,
                            add_inductance=add_inductance,
                            admittance=admittance,
                        )
                    )

                    self.assertIsInstance(circuit, Circuit)

                    elements: List[Element] = circuit.get_elements(recursive=True)
                    n: int = (
                        num_RC
                        + 1  # Resistance
                        + (1 if add_capacitance else 0)
                        + (1 if add_inductance else 0)
                    )
                    self.assertEqual(len(elements), n)
                    self.assertEqual(
                        len([e for e in elements if isinstance(e, Resistor)]), 1
                    )
                    self.assertEqual(
                        len([e for e in elements if isinstance(e, Capacitor)]),
                        1 if add_capacitance else 0,
                    )
                    self.assertEqual(
                        len([e for e in elements if isinstance(e, Inductor)]),
                        1 if add_inductance else 0,
                    )
                    self.assertEqual(
                        len(
                            [
                                e
                                for e in elements
                                if isinstance(
                                    e,
                                    (
                                        KramersKronigAdmittanceRC
                                        if admittance
                                        else KramersKronigRC
                                    ),
                                )
                            ]
                        ),
                        num_RC,
                    )

    def test_boukamp_weight(self):
        admittance: bool
        for admittance in (False, True):
            weight: NDArray[float64]
            weight = pyimpspec.analysis.kramers_kronig.utility._boukamp_weight(
                Z=self.Z_exp,
                admittance=admittance,
            )

            self.assertIsInstance(weight, ndarray)
            self.assertEqual(weight.dtype, float64)
            self.assertEqual(len(weight.shape), 1)
            self.assertEqual(len(weight), len(self.Z_exp))
            self.assertAlmostEqual(
                (
                    weight - (1 / abs(self.Z_exp ** (-1 if admittance else 1))) ** 2
                ).sum(),
                0,
            )

    def test_estimate_pseudo_chisqr(self):
        q: float
        for q in (0.0, 1e-4, 1e-2, 1e0):
            pseudo_chisqr: float
            pseudo_chisqr = (
                pyimpspec.analysis.kramers_kronig.utility._estimate_pseudo_chisqr(
                    Z=self.Z_exp,
                    pct_noise=q,
                )
            )

            self.assertIsInstance(pseudo_chisqr, float)
            self.assertAlmostEqual(
                pseudo_chisqr,
                len(self.Z_exp) * q**2 / 5000,
            )

    def test_estimate_pct_noise(self):
        pseudo_chisqr: float
        for pseudo_chisqr in (0.0, 1e-8, 1e-6, 1e-4):
            pct_noise: float
            pct_noise = pyimpspec.analysis.kramers_kronig.utility._estimate_pct_noise(
                Z=self.Z_exp,
                pseudo_chisqr=pseudo_chisqr,
            )

            self.assertIsInstance(pct_noise, float)
            self.assertAlmostEqual(
                pct_noise,
                sqrt(5000 * pseudo_chisqr / len(self.Z_exp)),
            )

    def test_format_log_F_ext_for_latex(self):
        log_F_ext: float
        expected: str
        for log_F_ext, expected in (
            (0.0, "0"),
            (-0.52362, "-0.524"),
            (1.3575923, "1.36"),
            (0.2, "0.2"),
            (1e-6, "1 \\times 10^{-6}"),
            (-1.000000, "-1"),
        ):
            formatted_string: str = (
                pyimpspec.analysis.kramers_kronig.utility._format_log_F_ext_for_latex(
                    log_F_ext
                )
            )

            self.assertEqual(formatted_string, expected)


class KramersKronigLeastSquaresFitting(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.f: NDArray[float64] = VALID_DATA.get_frequencies()
        cls.w: NDArray[float64] = 2 * pi * cls.f
        cls.taus: NDArray[float64] = (
            pyimpspec.analysis.kramers_kronig.utility._generate_time_constants(
                w=cls.w,
                num_RC=13,
                log_F_ext=0.0,
            )
        )
        cls.Z_exp: ComplexImpedances = VALID_DATA.get_impedances()

    def test_initialize_A_matrix(self):
        test: str
        add_capacitance: bool
        add_inductance: bool
        m: int
        n: int
        for test, add_capacitance, add_inductance, m, n in (
            ("real", False, False, len(self.w), len(self.taus) + 1),
            ("real", True, False, len(self.w), len(self.taus) + 2),
            ("real", False, True, len(self.w), len(self.taus) + 2),
            ("real", True, True, len(self.w), len(self.taus) + 3),
            ("imaginary", False, False, len(self.w), len(self.taus) + 1),
            ("imaginary", True, False, len(self.w), len(self.taus) + 2),
            ("imaginary", False, True, len(self.w), len(self.taus) + 2),
            ("imaginary", True, True, len(self.w), len(self.taus) + 3),
            ("complex", False, False, len(self.w) * 2, len(self.taus) + 1),
            ("complex", True, False, len(self.w) * 2, len(self.taus) + 2),
            ("complex", False, True, len(self.w) * 2, len(self.taus) + 2),
            ("complex", True, True, len(self.w) * 2, len(self.taus) + 3),
        ):
            A: NDArray[float64]
            A = pyimpspec.analysis.kramers_kronig.least_squares._initialize_A_matrix(
                test=test,
                w=self.w,
                taus=self.taus,
                add_capacitance=add_capacitance,
                add_inductance=add_inductance,
            )

            self.assertEqual(A.dtype, float64)
            self.assertEqual(len(A.shape), 2)
            self.assertEqual(A.shape[0], m)
            self.assertEqual(A.shape[1], n)
            self.assertEqual(A.sum(), 0.0)

    def test_add_resistance_to_A_matrix(self):
        test: str = "complex"
        A: NDArray[float64]
        A = pyimpspec.analysis.kramers_kronig.least_squares._initialize_A_matrix(
            test=test,
            w=self.w,
            taus=self.taus,
            add_capacitance=True,
            add_inductance=True,
        )
        m: int
        m = A.shape[0]

        pyimpspec.analysis.kramers_kronig.least_squares._add_resistance_to_A_matrix(
            A=A,
            test=test,
        )
        self.assertEqual(A.sum(), m // 2)
        self.assertEqual(A[0:m // 2].sum(), m // 2)
        self.assertEqual(A[0:m // 2, 0].sum(), m // 2)
        self.assertEqual(A[0:m // 2, 1:].sum(), 0)
        self.assertEqual(A[m // 2:].sum(), 0)

        test = "real"
        A = pyimpspec.analysis.kramers_kronig.least_squares._initialize_A_matrix(
            test=test,
            w=self.w,
            taus=self.taus,
            add_capacitance=True,
            add_inductance=True,
        )
        m = A.shape[0]

        pyimpspec.analysis.kramers_kronig.least_squares._add_resistance_to_A_matrix(
            A=A,
            test=test,
        )
        self.assertEqual(A.sum(), m)
        self.assertEqual(A[0:m].sum(), m)
        self.assertEqual(A[0:m, 0].sum(), m)
        self.assertEqual(A[0:m, 1:].sum(), 0)

        test = "imaginary"
        A = pyimpspec.analysis.kramers_kronig.least_squares._initialize_A_matrix(
            test=test,
            w=self.w,
            taus=self.taus,
            add_capacitance=True,
            add_inductance=True,
        )
        m = A.shape[0]

        pyimpspec.analysis.kramers_kronig.least_squares._add_resistance_to_A_matrix(
            A=A,
            test=test,
        )
        self.assertEqual(A.sum(), 0)

    def test_calculate_kth_A_matrix_variable(self):
        i: int = 0

        admittance: bool
        expected: NDArray[complex128]
        for admittance, expected in (
            (False, 1 / (1 + 1j * self.w * self.taus[i])),
            (True, self.w / (self.w * self.taus[i] - 1j)),
        ):
            c: NDArray[complex128]
            c = pyimpspec.analysis.kramers_kronig.least_squares._calculate_kth_A_matrix_variables(
                w=self.w,
                tau=self.taus[i],
                admittance=admittance,
            )
            self.assertEqual(c.dtype, complex128)
            self.assertEqual(c.shape[0], len(self.w))
            self.assertEqual(len(c.shape), 1)
            self.assertTrue((c == expected).all())

    def test_add_kth_variable_to_A_matrix(self):
        test: str = "complex"
        A: NDArray[float64]
        A = pyimpspec.analysis.kramers_kronig.least_squares._initialize_A_matrix(
            test=test,
            w=self.w,
            taus=self.taus,
            add_capacitance=True,
            add_inductance=True,
        )
        m: int
        n: int
        m, n = A.shape

        admittance: bool
        for admittance in (False, True):
            expected: NDArray[complex128] = zeros((m // 2, n), dtype=complex128)

            i: int
            tau: float64
            for i, tau in enumerate(self.taus, start=1):
                expected[:, i] = (
                    pyimpspec.analysis.kramers_kronig.least_squares._calculate_kth_A_matrix_variables(
                        w=self.w,
                        tau=tau,
                        admittance=admittance,
                    )
                )

            for i, tau in enumerate(self.taus, start=1):
                A *= 0.0
                self.assertEqual(A.sum(), 0)
                pyimpspec.analysis.kramers_kronig.least_squares._add_kth_variables_to_A_matrix(
                    A=A,
                    test=test,
                    w=self.w,
                    tau=tau,
                    i=i,
                    admittance=admittance,
                )
                self.assertNotEqual(A.sum(), 0.0)
                self.assertEqual(A[0:m // 2, i - 1].sum(), 0)
                self.assertEqual(A[0:m // 2, i + 1:].sum(), 0)
                self.assertTrue((A[0:m // 2, i] == expected[:, i].real).all())
                self.assertTrue((A[m // 2:, i] == expected[:, i].imag).all())

        test = "real"
        A = pyimpspec.analysis.kramers_kronig.least_squares._initialize_A_matrix(
            test=test,
            w=self.w,
            taus=self.taus,
            add_capacitance=True,
            add_inductance=True,
        )
        m, n = A.shape

        for admittance in (False, True):
            expected = zeros((m, n), dtype=complex128)

            for i, tau in enumerate(self.taus, start=1):
                expected[:, i] = (
                    pyimpspec.analysis.kramers_kronig.least_squares._calculate_kth_A_matrix_variables(
                        w=self.w,
                        tau=tau,
                        admittance=admittance,
                    )
                )

            for i, tau in enumerate(self.taus, start=1):
                A *= 0.0
                self.assertEqual(A.sum(), 0)
                pyimpspec.analysis.kramers_kronig.least_squares._add_kth_variables_to_A_matrix(
                    A=A,
                    test=test,
                    w=self.w,
                    tau=tau,
                    i=i,
                    admittance=admittance,
                )
                self.assertNotEqual(A.sum(), 0.0)
                self.assertEqual(A[:, i - 1].sum(), 0)
                self.assertEqual(A[:, i + 1:].sum(), 0)
                if test == "real":
                    self.assertTrue((A[:, i] == expected[:, i].real).all())
                elif test == "imaginary":
                    self.assertTrue((A[:, i] == expected[:, i].imag).all())

    def test_add_capacitance_to_A_matrix(self):
        admittance: bool
        for admittance in (False, True):
            test: str = "complex"
            A: NDArray[float64]
            A = pyimpspec.analysis.kramers_kronig.least_squares._initialize_A_matrix(
                test=test,
                w=self.w,
                taus=self.taus,
                add_capacitance=True,
                add_inductance=True,
            )
            m: int
            n: int
            m, n = A.shape

            pyimpspec.analysis.kramers_kronig.least_squares._add_capacitance_to_A_matrix(
                A=A,
                test=test,
                w=self.w,
                i=n - 2,
                admittance=admittance,
            )
            self.assertNotEqual(A.sum(), 0)
            self.assertTrue(
                (A[m // 2:, n - 2] == (self.w if admittance else -1 / self.w)).all()
            )
            self.assertEqual(A[: m // 2, n - 2].sum(), 0)
            self.assertEqual(A[:, :n - 2].sum(), 0)
            self.assertEqual(A[:, n - 1:].sum(), 0)

            test = "real"
            A = pyimpspec.analysis.kramers_kronig.least_squares._initialize_A_matrix(
                test=test,
                w=self.w,
                taus=self.taus,
                add_capacitance=True,
                add_inductance=True,
            )
            m, n = A.shape

            pyimpspec.analysis.kramers_kronig.least_squares._add_capacitance_to_A_matrix(
                A=A,
                test=test,
                w=self.w,
                i=n - 2,
                admittance=admittance,
            )
            self.assertEqual(A.sum(), 0)

            test = "imaginary"
            A = pyimpspec.analysis.kramers_kronig.least_squares._initialize_A_matrix(
                test=test,
                w=self.w,
                taus=self.taus,
                add_capacitance=True,
                add_inductance=True,
            )
            m, n = A.shape

            pyimpspec.analysis.kramers_kronig.least_squares._add_capacitance_to_A_matrix(
                A=A,
                test=test,
                w=self.w,
                i=n - 2,
                admittance=admittance,
            )
            self.assertNotEqual(A.sum(), 0)
            self.assertTrue(
                (A[:, n - 2] == (self.w if admittance else -1 / self.w)).all()
            )
            self.assertEqual(A[:, :n - 2].sum(), 0)
            self.assertEqual(A[:, n - 1:].sum(), 0)

    def test_add_inductance_to_A_matrix(self):
        admittance: bool
        for admittance in (False, True):
            test: str = "complex"
            A: NDArray[float64]
            A = pyimpspec.analysis.kramers_kronig.least_squares._initialize_A_matrix(
                test=test,
                w=self.w,
                taus=self.taus,
                add_capacitance=True,
                add_inductance=True,
            )
            m: int
            n: int
            m, n = A.shape

            pyimpspec.analysis.kramers_kronig.least_squares._add_inductance_to_A_matrix(
                A=A,
                test=test,
                w=self.w,
                i=n - 1,
                admittance=admittance,
            )
            self.assertNotEqual(A.sum(), 0)
            self.assertTrue(
                (A[m // 2:, n - 1] == (1 / self.w) if admittance else self.w).all()
            )
            self.assertEqual(A[:m // 2, n - 1].sum(), 0)
            self.assertEqual(A[:, :n - 1].sum(), 0)

            test = "real"
            A = pyimpspec.analysis.kramers_kronig.least_squares._initialize_A_matrix(
                test=test,
                w=self.w,
                taus=self.taus,
                add_capacitance=True,
                add_inductance=True,
            )
            m, n = A.shape

            pyimpspec.analysis.kramers_kronig.least_squares._add_inductance_to_A_matrix(
                A=A,
                test=test,
                w=self.w,
                i=n - 2,
                admittance=admittance,
            )
            self.assertEqual(A.sum(), 0)

            test = "imaginary"
            A = pyimpspec.analysis.kramers_kronig.least_squares._initialize_A_matrix(
                test=test,
                w=self.w,
                taus=self.taus,
                add_capacitance=True,
                add_inductance=True,
            )
            m, n = A.shape

            pyimpspec.analysis.kramers_kronig.least_squares._add_inductance_to_A_matrix(
                A=A,
                test=test,
                w=self.w,
                i=n - 2,
                admittance=admittance,
            )
            self.assertNotEqual(A.sum(), 0)
            self.assertTrue(
                (A[:, n - 2] == (1 / self.w) if admittance else self.w).all()
            )
            self.assertEqual(A[:, :n - 2].sum(), 0)
            self.assertEqual(A[:, n - 1:].sum(), 0)

    def test_generate_A_matrix(self):
        test: str
        add_capacitance: bool
        add_inductance: bool
        m: int
        n: int
        for test, add_capacitance, add_inductance, m, n in (
            ("real", False, False, len(self.w), len(self.taus) + 1),
            ("real", True, False, len(self.w), len(self.taus) + 2),
            ("real", False, True, len(self.w), len(self.taus) + 2),
            ("real", True, True, len(self.w), len(self.taus) + 3),
            ("imaginary", False, False, len(self.w), len(self.taus) + 1),
            ("imaginary", True, False, len(self.w), len(self.taus) + 2),
            ("imaginary", False, True, len(self.w), len(self.taus) + 2),
            ("imaginary", True, True, len(self.w), len(self.taus) + 3),
            ("complex", False, False, len(self.w) * 2, len(self.taus) + 1),
            ("complex", True, False, len(self.w) * 2, len(self.taus) + 2),
            ("complex", False, True, len(self.w) * 2, len(self.taus) + 2),
            ("complex", True, True, len(self.w) * 2, len(self.taus) + 3),
        ):
            for admittance in (False, True):
                A: NDArray[float64]
                A = pyimpspec.analysis.kramers_kronig.least_squares._generate_A_matrix(
                    test=test,
                    w=self.w,
                    taus=self.taus,
                    add_capacitance=add_capacitance,
                    add_inductance=add_inductance,
                    admittance=admittance,
                )

                self.assertEqual(A.dtype, float64)
                self.assertEqual(len(A.shape), 2)
                self.assertEqual(A.shape[0], m)
                self.assertEqual(A.shape[1], n)
                self.assertNotEqual(A.sum(), 0.0)

    def test_initialize_b_vector(self):
        test: str
        for test in ("complex", "real", "imaginary"):
            b: NDArray[float64]
            b = pyimpspec.analysis.kramers_kronig.least_squares._initialize_b_vector(
                test=test,
                Z_exp=self.Z_exp,
            )

            self.assertEqual(b.dtype, float64)
            self.assertEqual(len(b.shape), 1)
            self.assertEqual(
                b.shape[0],
                len(self.Z_exp) * (2 if test == "complex" else 1),
            )

    def test_add_values_to_b_vector(self):
        test: str
        for test in ("complex", "real", "imaginary"):
            admittance: bool
            for admittance in (False, True):
                X_exp: NDArray[complex128] = self.Z_exp ** (-1 if admittance else 1)

                b: NDArray[float64]
                b = pyimpspec.analysis.kramers_kronig.least_squares._initialize_b_vector(
                    test=test,
                    Z_exp=self.Z_exp,
                )
                m: int = b.shape[0]

                pyimpspec.analysis.kramers_kronig.least_squares._add_values_to_b_vector(
                    b=b,
                    test=test,
                    Z_exp=self.Z_exp,
                    admittance=admittance,
                )

                if test == "complex":
                    self.assertTrue((b[: m // 2] == X_exp.real).all())
                    self.assertTrue((b[m // 2:] == X_exp.imag).all())
                elif test == "real":
                    self.assertTrue((b == X_exp.real).all())
                elif test == "imaginary":
                    self.assertTrue((b == X_exp.imag).all())

    def test_generate_b_vector(self):
        test: str
        for test in ("complex", "real", "imaginary"):
            admittance: bool
            for admittance in (False, True):
                b: NDArray[float64]
                b = pyimpspec.analysis.kramers_kronig.least_squares._generate_b_vector(
                    test=test,
                    Z_exp=self.Z_exp,
                    admittance=admittance,
                )

                self.assertEqual(b.dtype, float64)
                self.assertEqual(len(b.shape), 1)
                self.assertEqual(
                    b.shape[0],
                    len(self.Z_exp) * (2 if test == "complex" else 1),
                )
                self.assertNotEqual(b.sum(), 0)

    def test_update_circuit(self):
        admittance: bool
        for admittance in (False, True):
            cdc: str = "RKKK"
            if admittance:
                cdc = cdc.replace("K", "Ky")

            variables = array([2.0, 1.1, 1.2, 1.3])
            circuit: Circuit = pyimpspec.parse_cdc(cdc)
            pyimpspec.analysis.kramers_kronig.least_squares._update_circuit(
                circuit,
                variables,
                add_capacitance=False,
                add_inductance=False,
                admittance=admittance,
            )
            elements: List[Element] = circuit.get_elements(recursive=True)

            self.assertEqual(len(elements), 4)
            self.assertEqual(
                elements[0].get_value("R"),
                (1 / variables[0]) if admittance else variables[0],
            )

            i: int
            for i in range(1, len(variables)):
                self.assertEqual(
                    elements[i].get_value("C" if admittance else "R"),
                    variables[i],
                )

            with self.assertRaises(ValueError):
                pyimpspec.analysis.kramers_kronig.least_squares._update_circuit(
                    circuit,
                    array(variables.tolist() + [1.0]),
                    add_capacitance=False,
                    add_inductance=False,
                    admittance=admittance,
                )

            with self.assertRaises(KramersKronigError):
                pyimpspec.analysis.kramers_kronig.least_squares._update_circuit(
                    circuit,
                    variables,
                    add_capacitance=True,
                    add_inductance=False,
                    admittance=admittance,
                )

            with self.assertRaises(KramersKronigError):
                pyimpspec.analysis.kramers_kronig.least_squares._update_circuit(
                    circuit,
                    variables,
                    add_capacitance=False,
                    add_inductance=True,
                    admittance=admittance,
                )

            with self.assertRaises(KramersKronigError):
                pyimpspec.analysis.kramers_kronig.least_squares._update_circuit(
                    circuit,
                    variables,
                    add_capacitance=True,
                    add_inductance=True,
                    admittance=admittance,
                )

            variables = array([2.0, 1.1, 1.2, 1.3, 3.0, 4.0])
            circuit = pyimpspec.parse_cdc(cdc + "CL")
            pyimpspec.analysis.kramers_kronig.least_squares._update_circuit(
                circuit,
                variables,
                add_capacitance=True,
                add_inductance=True,
                admittance=admittance,
            )
            elements = circuit.get_elements(recursive=True)

            self.assertEqual(len(elements), 6)
            self.assertEqual(
                elements[0].get_value("R"),
                (1 / variables[0]) if admittance else variables[0],
            )

            for i in range(1, len(variables[:-2])):
                self.assertEqual(
                    elements[i].get_value("C" if admittance else "R"),
                    variables[i],
                )

            self.assertEqual(
                elements[-2].get_value("C"),
                (variables[-2]) if admittance else 1 / variables[-2],
            )
            self.assertEqual(
                elements[-1].get_value("L"),
                (-1 / variables[-1]) if admittance else variables[-1],
            )

    def test_real_test(self):
        admittance: bool
        for admittance in (False, True):
            add_capacitance: bool
            add_inductance: bool
            for add_capacitance, add_inductance in (
                (False, False),
                (True, False),
                (False, True),
                (True, True),
            ):
                circuit: Circuit
                circuit = pyimpspec.analysis.kramers_kronig.least_squares._real_test(
                    Z_exp=self.Z_exp,
                    f=self.f,
                    w=self.w,
                    taus=self.taus,
                    add_capacitance=add_capacitance,
                    add_inductance=add_inductance,
                    admittance=admittance,
                )

                self.assertIsInstance(circuit, Circuit)

    def test_imaginary_test(self):
        admittance: bool
        for admittance in (False, True):
            weight: NDArray[float64] = (
                pyimpspec.analysis.kramers_kronig.utility._boukamp_weight(
                    Z=self.Z_exp,
                    admittance=admittance,
                )
            )

            add_capacitance: bool
            add_inductance: bool
            for add_capacitance, add_inductance in (
                (False, False),
                (True, False),
                (False, True),
                (True, True),
            ):
                circuit: Circuit
                circuit = (
                    pyimpspec.analysis.kramers_kronig.least_squares._imaginary_test(
                        Z_exp=self.Z_exp,
                        f=self.f,
                        w=self.w,
                        taus=self.taus,
                        weight=weight,
                        add_capacitance=add_capacitance,
                        add_inductance=add_inductance,
                        admittance=admittance,
                    )
                )

                self.assertIsInstance(circuit, Circuit)

    def test_complex_test(self):
        admittance: bool
        for admittance in (False, True):
            add_capacitance: bool
            add_inductance: bool
            for add_capacitance, add_inductance in (
                (False, False),
                (True, False),
                (False, True),
                (True, True),
            ):
                circuit: Circuit
                circuit = pyimpspec.analysis.kramers_kronig.least_squares._real_test(
                    Z_exp=self.Z_exp,
                    f=self.f,
                    w=self.w,
                    taus=self.taus,
                    add_capacitance=add_capacitance,
                    add_inductance=add_inductance,
                    admittance=admittance,
                )

                self.assertIsInstance(circuit, Circuit)

    def test_test_wrapper(self):
        test: str
        for test in ("complex", "real", "imaginary"):
            admittance: bool
            for admittance in (False, True):
                weight: NDArray[float64]
                weight = pyimpspec.analysis.kramers_kronig.utility._boukamp_weight(
                    Z=self.Z_exp,
                    admittance=admittance,
                )

                add_capacitance: bool
                add_inductance: bool
                for add_capacitance, add_inductance in (
                    (False, False),
                    (True, False),
                    (False, True),
                    (True, True),
                ):
                    num_RC: int
                    for num_RC in (2, 5, 10):
                        log_F_ext: float
                        for log_F_ext in (0.0, -0.5, 0.5):
                            n: int
                            circuit: Circuit
                            (
                                n,
                                circuit,
                            ) = pyimpspec.analysis.kramers_kronig.least_squares._test_wrapper(
                                (
                                    test,
                                    self.f,
                                    self.Z_exp,
                                    weight,
                                    num_RC,
                                    add_capacitance,
                                    add_inductance,
                                    admittance,
                                    log_F_ext,
                                )
                            )
                            elements: List[Element] = circuit.get_elements(
                                recursive=True
                            )

                            self.assertIsInstance(num_RC, int)
                            self.assertEqual(n, num_RC)
                            self.assertIsInstance(circuit, Circuit)
                            self.assertEqual(
                                len(elements),
                                num_RC
                                + (
                                    3
                                    if add_capacitance and add_inductance
                                    else (2 if add_capacitance or add_inductance else 1)
                                ),
                            )
                            self.assertEqual(
                                len([e for e in elements if isinstance(e, Resistor)]),
                                1,
                            )
                            self.assertEqual(
                                len(
                                    [
                                        e
                                        for e in elements
                                        if isinstance(
                                            e,
                                            (
                                                KramersKronigAdmittanceRC
                                                if admittance
                                                else KramersKronigRC
                                            ),
                                        )
                                    ]
                                ),
                                num_RC,
                            )
                            self.assertEqual(
                                len([e for e in elements if isinstance(e, Capacitor)]),
                                1 if add_capacitance else 0,
                            )
                            self.assertEqual(
                                len([e for e in elements if isinstance(e, Inductor)]),
                                1 if add_inductance else 0,
                            )


class KramersKronigMatrixInversion(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.f: Frequencies = VALID_DATA.get_frequencies()
        cls.w: NDArray[float64] = 2 * pi * cls.f
        cls.taus: NDArray[float64] = (
            pyimpspec.analysis.kramers_kronig.utility._generate_time_constants(
                w=cls.w,
                num_RC=13,
                log_F_ext=0.0,
            )
        )
        cls.Z_exp: ComplexImpedances = VALID_DATA.get_impedances()

    def test_initialize_A_matrices(self):
        add_capacitance: bool
        m: int
        n: int
        for add_capacitance, m, n in (
            (False, self.w.size, len(self.taus) + 2),
            (True, self.w.size, len(self.taus) + 3),
        ):
            A_re: NDArray[float64]
            A_im: NDArray[float64]
            (
                A_re,
                A_im,
            ) = pyimpspec.analysis.kramers_kronig.matrix_inversion._initialize_A_matrices(
                w=self.w,
                taus=self.taus,
                add_capacitance=add_capacitance,
            )

            A: NDArray[float64]
            for A in (A_re, A_im):
                self.assertEqual(A.dtype, float64)
                self.assertEqual(len(A.shape), 2)
                self.assertEqual(A.shape[0], m)
                self.assertEqual(A.shape[1], n)
                self.assertEqual(A.sum(), 0.0)

    def test_add_resistance_to_A_matrix(self):
        A_re: NDArray[float64]
        A_im: NDArray[float64]
        (
            A_re,
            A_im,
        ) = pyimpspec.analysis.kramers_kronig.matrix_inversion._initialize_A_matrices(
            w=self.w,
            taus=self.taus,
            add_capacitance=False,
        )

        pyimpspec.analysis.kramers_kronig.matrix_inversion._add_resistance_to_A_matrix(
            A_re
        )

        self.assertEqual(A_re[:, 0].sum(), self.w.size)
        self.assertEqual(A_re[:, 1:].sum(), 0)

    def test_add_capacitance_to_A_matrix(self):
        A_re: NDArray[float64]
        A_im: NDArray[float64]
        (
            A_re,
            A_im,
        ) = pyimpspec.analysis.kramers_kronig.matrix_inversion._initialize_A_matrices(
            w=self.w,
            taus=self.taus,
            add_capacitance=True,
        )

        admittance: bool
        for admittance in (False, True):
            A_im *= 0
            pyimpspec.analysis.kramers_kronig.matrix_inversion._add_capacitance_to_A_matrix(
                A_im=A_im,
                w=self.w,
                admittance=admittance,
            )

            self.assertEqual(
                (A_im[:, -2] - (self.w if admittance else (-1 / self.w))).sum(),
                0,
            )
            self.assertEqual(A_im[:, :-2].sum(), 0)
            self.assertEqual(A_im[:, -1].sum(), 0)

    def test_add_inductance_to_A_matrix(self):
        A_re: NDArray[float64]
        A_im: NDArray[float64]
        (
            A_re,
            A_im,
        ) = pyimpspec.analysis.kramers_kronig.matrix_inversion._initialize_A_matrices(
            w=self.w,
            taus=self.taus,
            add_capacitance=True,
        )

        admittance: bool
        for admittance in (False, True):
            A_im *= 0
            pyimpspec.analysis.kramers_kronig.matrix_inversion._add_inductance_to_A_matrix(
                A_im=A_im,
                w=self.w,
                admittance=admittance,
            )

            self.assertEqual(
                (A_im[:, -1] - (1 / self.w if admittance else (self.w))).sum(),
                0,
            )
            self.assertEqual(A_im[:, :-1].sum(), 0)

    def test_add_kth_variables_to_A_matrices(self):
        A_re: NDArray[float64]
        A_im: NDArray[float64]
        (
            A_re,
            A_im,
        ) = pyimpspec.analysis.kramers_kronig.matrix_inversion._initialize_A_matrices(
            w=self.w,
            taus=self.taus,
            add_capacitance=False,
        )

        admittance: bool
        for admittance in (False, True):
            A_im *= 0
            pyimpspec.analysis.kramers_kronig.matrix_inversion._add_kth_variables_to_A_matrices(
                A_re=A_re,
                A_im=A_im,
                w=self.w,
                taus=self.taus,
                admittance=admittance,
            )

            self.assertEqual(A_re[:, 0].sum(), 0)
            self.assertEqual(A_im[:, 0].sum(), 0)
            if admittance:
                for i, tau in enumerate(self.taus):
                    self.assertEqual(
                        (
                            A_re[:, i + 1] - self.w**2 * tau / (1 + (self.w * tau) ** 2)
                        ).sum(),
                        0,
                    )
                    self.assertEqual(
                        (A_im[:, i + 1] - self.w / (1 + (self.w * tau) ** 2)).sum(),
                        0,
                    )
            else:
                for i, tau in enumerate(self.taus):
                    self.assertEqual(
                        (A_re[:, i + 1] - (1 / (1 + 1j * self.w * tau)).real).sum(),
                        0,
                    )
                    self.assertEqual(
                        (A_im[:, i + 1] - (1 / (1 + 1j * self.w * tau)).imag).sum(),
                        0,
                    )

    def test_scale_A_matrices(self):
        admittance: bool
        for admittance in (False, True):
            A_re: NDArray[float64]
            A_im: NDArray[float64]
            (
                A_re,
                A_im,
            ) = pyimpspec.analysis.kramers_kronig.matrix_inversion._initialize_A_matrices(
                w=self.w,
                taus=self.taus,
                add_capacitance=True,
            )
            pyimpspec.analysis.kramers_kronig.matrix_inversion._add_resistance_to_A_matrix(
                A_re
            )
            pyimpspec.analysis.kramers_kronig.matrix_inversion._add_capacitance_to_A_matrix(
                A_im=A_im,
                w=self.w,
                admittance=admittance,
            )
            pyimpspec.analysis.kramers_kronig.matrix_inversion._add_inductance_to_A_matrix(
                A_im=A_im,
                w=self.w,
                admittance=admittance,
            )

            abs_X_exp: NDArray[complex128] = abs(
                self.Z_exp ** (-1 if admittance else 1)
            )
            scaled_A_re: NDArray[complex128] = zeros(A_re.shape, dtype=A_re.dtype)
            scaled_A_im: NDArray[complex128] = zeros(A_im.shape, dtype=A_im.dtype)

            i: int
            for i in range(A_re.shape[1]):
                scaled_A_re[:, i] += A_re[:, i] / abs_X_exp
                scaled_A_im[:, i] += A_im[:, i] / abs_X_exp

            pyimpspec.analysis.kramers_kronig.matrix_inversion._scale_A_matrices(
                A_re=A_re,
                A_im=A_im,
                abs_X_exp=abs_X_exp,
            )

            self.assertEqual((A_re - scaled_A_re).sum(), 0)
            self.assertEqual((A_im - scaled_A_im).sum(), 0)

    def test_generate_A_matrices(self):
        admittance: bool
        for admittance in (False, True):
            abs_X_exp: NDArray[complex128] = abs(
                self.Z_exp ** (-1 if admittance else 1)
            )

            add_capacitance: bool
            m: int
            n: int
            for add_capacitance, m, n in (
                (False, self.w.size, len(self.taus) + 2),
                (True, self.w.size, len(self.taus) + 3),
            ):
                A_re: NDArray[float64]
                A_im: NDArray[float64]
                (
                    A_re,
                    A_im,
                ) = pyimpspec.analysis.kramers_kronig.matrix_inversion._generate_A_matrices(
                    w=self.w,
                    taus=self.taus,
                    add_capacitance=add_capacitance,
                    admittance=admittance,
                    abs_X_exp=abs_X_exp,
                )

                A: NDArray[float64]
                for A in (A_re, A_im):
                    self.assertEqual(A.dtype, float64)
                    self.assertEqual(len(A.shape), 2)
                    self.assertEqual(A.shape[0], m)
                    self.assertEqual(A.shape[1], n)
                    self.assertNotEqual(A.sum(), 0.0)

    def test_update_circuit(self):
        admittance: bool
        for admittance in (False, True):
            cdc: str = "RKKKL"
            if admittance:
                cdc = cdc.replace("K", "Ky")

            variables = array([2.0, 1.1, 1.2, 1.3, 4.0])
            circuit: Circuit = pyimpspec.parse_cdc(cdc)
            pyimpspec.analysis.kramers_kronig.matrix_inversion._update_circuit(
                circuit,
                variables,
                add_capacitance=False,
                admittance=admittance,
            )
            elements: List[Element] = circuit.get_elements(recursive=True)

            self.assertEqual(len(elements), 5)
            self.assertEqual(
                elements[0].get_value("R"),
                (1 / variables[0]) if admittance else variables[0],
            )

            i: int
            for i in range(1, len(variables) - 1):
                self.assertEqual(
                    elements[i].get_value("C" if admittance else "R"),
                    variables[i],
                )

            with self.assertRaises(ValueError):
                pyimpspec.analysis.kramers_kronig.matrix_inversion._update_circuit(
                    circuit,
                    array(variables.tolist() + [1.0]),
                    add_capacitance=False,
                    admittance=admittance,
                )

            with self.assertRaises(KramersKronigError):
                pyimpspec.analysis.kramers_kronig.matrix_inversion._update_circuit(
                    circuit,
                    variables,
                    add_capacitance=True,
                    admittance=admittance,
                )

            variables = array([2.0, 1.1, 1.2, 1.3, 3.0, 4.0])
            circuit = pyimpspec.parse_cdc(cdc[:-1] + "CL")
            pyimpspec.analysis.kramers_kronig.matrix_inversion._update_circuit(
                circuit,
                variables,
                add_capacitance=True,
                admittance=admittance,
            )
            elements = circuit.get_elements(recursive=True)

            self.assertEqual(len(elements), 6)
            self.assertEqual(
                elements[0].get_value("R"),
                (1 / variables[0]) if admittance else variables[0],
            )

            for i in range(1, len(variables[:-2])):
                self.assertEqual(
                    elements[i].get_value("C" if admittance else "R"),
                    variables[i],
                )

            self.assertEqual(
                elements[-2].get_value("C"),
                (variables[-2]) if admittance else 1 / variables[-2],
            )
            self.assertEqual(
                elements[-1].get_value("L"),
                (-1 / variables[-1]) if admittance else variables[-1],
            )

    def test_real_test(self):
        admittance: bool
        for admittance in (False, True):
            add_capacitance: bool
            for add_capacitance in (False, True):
                A_re: NDArray[float64]
                A_im: NDArray[float64]
                (
                    A_re,
                    A_im,
                ) = pyimpspec.analysis.kramers_kronig.matrix_inversion._generate_A_matrices(
                    w=self.w,
                    taus=self.taus,
                    add_capacitance=add_capacitance,
                    admittance=admittance,
                    abs_X_exp=abs(self.Z_exp ** (-1 if admittance else 1)),
                )

                circuit: Circuit
                circuit = pyimpspec.analysis.kramers_kronig.matrix_inversion._generate_circuit(
                    taus=self.taus,
                    add_capacitance=add_capacitance,
                    add_inductance=True,
                    admittance=admittance,
                )
                elements: List[Element] = circuit.get_elements(recursive=True)

                variables: NDArray[float64]
                variables = (
                    pyimpspec.analysis.kramers_kronig.matrix_inversion._real_test(
                        A_re=A_re,
                        X_exp=self.Z_exp ** (-1 if admittance else 1),
                        w=self.w,
                        f=self.f,
                        taus=self.taus,
                        add_capacitance=add_capacitance,
                        admittance=admittance,
                        circuit=circuit,
                    )
                )

                self.assertIsInstance(variables, ndarray)
                self.assertEqual(variables.dtype, float64)
                self.assertEqual(len(variables.shape), 1)
                self.assertEqual(variables.shape[0], len(elements))

    def test_imaginary_test(self):
        admittance: bool
        for admittance in (False, True):
            weight: NDArray[float64] = (
                pyimpspec.analysis.kramers_kronig.utility._boukamp_weight(
                    Z=self.Z_exp,
                    admittance=admittance,
                )
            )

            add_capacitance: bool
            for add_capacitance in (False, True):
                A_re: NDArray[float64]
                A_im: NDArray[float64]
                (
                    A_re,
                    A_im,
                ) = pyimpspec.analysis.kramers_kronig.matrix_inversion._generate_A_matrices(
                    w=self.w,
                    taus=self.taus,
                    add_capacitance=add_capacitance,
                    admittance=admittance,
                    abs_X_exp=abs(self.Z_exp ** (-1 if admittance else 1)),
                )

                circuit: Circuit
                circuit = pyimpspec.analysis.kramers_kronig.matrix_inversion._generate_circuit(
                    taus=self.taus,
                    add_capacitance=add_capacitance,
                    add_inductance=True,
                    admittance=admittance,
                )
                elements: List[Element] = circuit.get_elements(recursive=True)

                variables: NDArray[float64]
                variables = (
                    pyimpspec.analysis.kramers_kronig.matrix_inversion._imaginary_test(
                        A_im=A_im,
                        X_exp=self.Z_exp ** (-1 if admittance else 1),
                        f=self.f,
                        taus=self.taus,
                        add_capacitance=add_capacitance,
                        admittance=admittance,
                        weight=weight,
                        circuit=circuit,
                    )
                )

                self.assertIsInstance(variables, ndarray)
                self.assertEqual(variables.dtype, float64)
                self.assertEqual(len(variables.shape), 1)
                self.assertEqual(variables.shape[0], len(elements))

    def test_complex_test(self):
        admittance: bool
        for admittance in (False, True):
            add_capacitance: bool
            for add_capacitance in (False, True):
                A_re: NDArray[float64]
                A_im: NDArray[float64]
                (
                    A_re,
                    A_im,
                ) = pyimpspec.analysis.kramers_kronig.matrix_inversion._generate_A_matrices(
                    w=self.w,
                    taus=self.taus,
                    add_capacitance=add_capacitance,
                    admittance=admittance,
                    abs_X_exp=abs(self.Z_exp ** (-1 if admittance else 1)),
                )

                circuit: Circuit
                circuit = pyimpspec.analysis.kramers_kronig.matrix_inversion._generate_circuit(
                    taus=self.taus,
                    add_capacitance=add_capacitance,
                    add_inductance=True,
                    admittance=admittance,
                )
                elements: List[Element] = circuit.get_elements(recursive=True)

                variables: NDArray[float64]
                variables = (
                    pyimpspec.analysis.kramers_kronig.matrix_inversion._complex_test(
                        A_re=A_re,
                        A_im=A_im,
                        X_exp=self.Z_exp ** (-1 if admittance else 1),
                        taus=self.taus,
                        add_capacitance=add_capacitance,
                        admittance=admittance,
                        circuit=circuit,
                    )
                )

                self.assertIsInstance(variables, ndarray)
                self.assertEqual(variables.dtype, float64)
                self.assertEqual(len(variables.shape), 1)
                self.assertEqual(variables.shape[0], len(elements))

    def test_test_wrapper(self):
        test: str
        for test in ("complex", "real", "imaginary"):
            admittance: bool
            for admittance in (False, True):
                weight: NDArray[float64]
                weight = pyimpspec.analysis.kramers_kronig.utility._boukamp_weight(
                    Z=self.Z_exp,
                    admittance=admittance,
                )

                add_capacitance: bool
                for add_capacitance in (False, True):
                    num_RC: int
                    for num_RC in (2, 5, 10):
                        log_F_ext: float
                        for log_F_ext in (0.0, -0.5, 0.5):
                            n: int
                            circuit: Circuit
                            (
                                n,
                                circuit,
                            ) = pyimpspec.analysis.kramers_kronig.matrix_inversion._test_wrapper(
                                (
                                    test,
                                    self.f,
                                    self.Z_exp,
                                    weight,
                                    num_RC,
                                    add_capacitance,
                                    admittance,
                                    log_F_ext,
                                )
                            )
                            elements: List[Element] = circuit.get_elements(
                                recursive=True
                            )

                            self.assertIsInstance(n, int)
                            self.assertEqual(n, num_RC)
                            self.assertIsInstance(circuit, Circuit)
                            self.assertEqual(
                                len(elements),
                                num_RC + (3 if add_capacitance else 2),
                            )
                            self.assertEqual(
                                len([e for e in elements if isinstance(e, Resistor)]),
                                1,
                            )
                            self.assertEqual(
                                len(
                                    [
                                        e
                                        for e in elements
                                        if isinstance(
                                            e,
                                            (
                                                KramersKronigAdmittanceRC
                                                if admittance
                                                else KramersKronigRC
                                            ),
                                        )
                                    ]
                                ),
                                num_RC,
                            )
                            self.assertEqual(
                                len([e for e in elements if isinstance(e, Capacitor)]),
                                1 if add_capacitance else 0,
                            )
                            self.assertEqual(
                                len([e for e in elements if isinstance(e, Inductor)]),
                                1,
                            )


class KramersKronigCNLS(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.f: Frequencies = VALID_DATA.get_frequencies()
        cls.Z_exp: ComplexImpedances = VALID_DATA.get_impedances()

    def test_test_wrapper(self):
        method: str = "leastsq"
        max_nfev: int = 20

        admittance: bool
        for admittance in (False, True):
            weight: NDArray[float64]
            weight = pyimpspec.analysis.kramers_kronig.utility._boukamp_weight(
                Z=self.Z_exp,
                admittance=admittance,
            )

            add_capacitance: bool
            add_inductance: bool
            for add_capacitance, add_inductance in (
                (False, False),
                (True, False),
                (False, True),
                (True, True),
            ):
                num_RC: int
                for num_RC in (2, 5, 10):
                    log_F_ext: float
                    for log_F_ext in (0.0, -0.5, 0.5):
                        n: int
                        circuit: Circuit
                        (
                            n,
                            circuit,
                        ) = pyimpspec.analysis.kramers_kronig.cnls._test_wrapper(
                            (
                                self.f,
                                self.Z_exp,
                                weight,
                                num_RC,
                                add_capacitance,
                                add_inductance,
                                admittance,
                                log_F_ext,
                                method,
                                max_nfev,
                            )
                        )
                        elements: List[Element] = circuit.get_elements(recursive=True)

                        self.assertIsInstance(num_RC, int)
                        self.assertEqual(n, num_RC)
                        self.assertIsInstance(circuit, Circuit)
                        self.assertEqual(
                            len(elements),
                            num_RC
                            + (
                                3
                                if add_capacitance and add_inductance
                                else (2 if add_capacitance or add_inductance else 1)
                            ),
                        )
                        self.assertEqual(
                            len([e for e in elements if isinstance(e, Resistor)]), 1
                        )
                        self.assertEqual(
                            len(
                                [
                                    e
                                    for e in elements
                                    if isinstance(
                                        e,
                                        (
                                            KramersKronigAdmittanceRC
                                            if admittance
                                            else KramersKronigRC
                                        ),
                                    )
                                ]
                            ),
                            num_RC,
                        )
                        self.assertEqual(
                            len([e for e in elements if isinstance(e, Capacitor)]),
                            1 if add_capacitance else 0,
                        )
                        self.assertEqual(
                            len([e for e in elements if isinstance(e, Inductor)]),
                            1 if add_inductance else 0,
                        )


class KramersKronigEvaluateLogFExt(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.min_log_F_ext: float = -0.55
        cls.max_log_F_ext: float = 0.45

        cls.extensive_evaluations: List[
            Tuple[float, List[KramersKronigResult], float]
        ] = pyimpspec.analysis.kramers_kronig.evaluate_log_F_ext(
            VALID_DATA,
            test="complex",
            num_RCs=None,
            add_capacitance=True,
            add_inductance=True,
            admittance=False,
            min_log_F_ext=cls.min_log_F_ext,
            max_log_F_ext=cls.max_log_F_ext,
            log_F_ext=0.0,
            num_F_ext_evaluations=20,
            rapid_F_ext_evaluations=False,
            cnls_method="leastsq",
            max_nfev=0,
            timeout=60,
            num_procs=-1,
        )

        cls.rapid_evaluations: List[Tuple[float, List[KramersKronigResult], float]] = (
            pyimpspec.analysis.kramers_kronig.evaluate_log_F_ext(
                VALID_DATA,
                test="complex",
                num_RCs=None,
                add_capacitance=True,
                add_inductance=True,
                admittance=False,
                min_log_F_ext=cls.min_log_F_ext,
                max_log_F_ext=cls.max_log_F_ext,
                log_F_ext=0.0,
                num_F_ext_evaluations=20,
                rapid_F_ext_evaluations=True,
                cnls_method="leastsq",
                max_nfev=0,
                timeout=60,
                num_procs=-1,
            )
        )

    def test_return_values(self):
        for return_values in (
            self.extensive_evaluations,
            self.rapid_evaluations,
        ):
            self.assertIsInstance(return_values, list)
            self.assertTrue(all(map(lambda t: isinstance(t, tuple), return_values)))
            self.assertTrue(all(map(lambda t: len(t) == 3, return_values)))

            self.assertTrue(all(map(lambda t: isinstance(t[0], float), return_values)))
            self.assertTrue(
                all(
                    map(
                        lambda t: self.min_log_F_ext <= t[0] <= self.max_log_F_ext,
                        return_values,
                    )
                )
            )

            self.assertTrue(all(map(lambda t: isinstance(t[1], list), return_values)))
            self.assertTrue(
                all(
                    map(
                        lambda t: all(
                            map(
                                lambda result: isinstance(result, KramersKronigResult),
                                t[1],
                            )
                        ),
                        return_values,
                    )
                )
            )

            self.assertTrue(all(map(lambda t: isinstance(t[2], float), return_values)))
            self.assertEqual(min([t[2] for t in return_values]), return_values[0][2])

    def test_rapid_evaluations(self):
        baseline: Tuple[float, List[KramersKronigResult], float] = min(
            self.rapid_evaluations,
            key=lambda t: abs(t[0]),
        )
        optimized: Tuple[float, List[KramersKronigResult], float] = (
            self.rapid_evaluations[0]
        )

        self.assertTrue(
            all(
                map(
                    lambda t: (len(t[1]) < len(baseline[1]))
                    or t in (baseline, optimized),
                    self.rapid_evaluations,
                )
            )
        )
        self.assertTrue(
            all(
                map(
                    lambda t: (len(t[1]) < len(optimized[1]))
                    or t in (baseline, optimized),
                    self.rapid_evaluations,
                )
            )
        )

    def test_extensive_evaluations(self):
        baseline: Tuple[float, List[KramersKronigResult], float] = min(
            self.extensive_evaluations,
            key=lambda t: abs(t[0]),
        )

        self.assertTrue(
            all(
                map(
                    lambda t: (len(t[1]) == len(baseline[1])),
                    self.extensive_evaluations,
                )
            )
        )


class KramersKronigPerformExploratoryTests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results: Tuple[
            List[KramersKronigResult],
            Tuple[KramersKronigResult, Dict[int, float], int, int],
        ]
        cls.results = (
            pyimpspec.analysis.kramers_kronig.perform_exploratory_kramers_kronig_tests(
                VALID_DATA
            )
        )

    def test_admittance(self):
        results: Tuple[
            List[KramersKronigResult],
            Tuple[KramersKronigResult, Dict[int, float], int, int],
        ]
        results = pyimpspec.analysis.kramers_kronig.perform_exploratory_kramers_kronig_tests(
            VALID_DATA,
            admittance=False,
        )
        self.assertTrue(all(map(lambda t: t.admittance is False, results[0])))
        self.assertEqual(results[1][0].admittance, False)

        results = pyimpspec.analysis.kramers_kronig.perform_exploratory_kramers_kronig_tests(
            VALID_DATA,
            admittance=True,
        )
        self.assertTrue(all(map(lambda t: t.admittance is True, results[0])))
        self.assertEqual(results[1][0].admittance, True)

    def test_return_value(self):
        self.assertIsInstance(self.results, tuple)
        self.assertEqual(len(self.results), 2)
        self.assertIsInstance(self.results[0], list)
        self.assertGreater(len(self.results[0]), 0)
        self.assertTrue(
            all(
                map(
                    lambda t: isinstance(t, KramersKronigResult),
                    self.results[0],
                )
            )
        )
        self.assertIsInstance(self.results[1], tuple)
        self.assertEqual(len(self.results[1]), 4)
        self.assertIsInstance(self.results[1][0], KramersKronigResult)
        self.assertIsInstance(self.results[1][1], dict)
        self.assertTrue(
            all(
                map(
                    lambda k: isinstance(k, int),
                    self.results[1][1].keys(),
                )
            )
        )
        self.assertTrue(
            all(
                map(
                    lambda v: isinstance(v, float),
                    self.results[1][1].values(),
                )
            )
        )
        self.assertIsInstance(self.results[1][2], int)
        self.assertIsInstance(self.results[1][3], int)


class KramersKronigPerformTest(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result: KramersKronigResult = (
            pyimpspec.analysis.kramers_kronig.perform_kramers_kronig_test(
                VALID_DATA,
            )
        )

    def test_return_value(self):
        self.assertIsInstance(self.result, KramersKronigResult)
        self.assertEqual(self.result.test, "real")
        self.assertEqual(self.result.admittance, False)

    def test_test_type(self):
        num_RC: int = 17
        log_F_ext: float = 0.05

        test: str
        for test in (
            "complex",
            "real",
            "imaginary",
            "complex-inv",
            "real-inv",
            "imaginary-inv",
            "cnls",
        ):
            result: KramersKronigResult = (
                pyimpspec.analysis.kramers_kronig.perform_kramers_kronig_test(
                    VALID_DATA,
                    test=test,
                    admittance=False,
                    num_RC=num_RC,
                    log_F_ext=log_F_ext,
                    num_F_ext_evaluations=0,
                )
            )

            self.assertEqual(result.test, test)
            self.assertEqual(result.num_RC, num_RC)
            self.assertAlmostEqual(result.get_log_F_ext(), log_F_ext)
            self.assertEqual(result.admittance, False)

    def test_admittance_add_capacitance_inductance(self):
        num_RC: int = 17
        log_F_ext: float = 0.05

        admittance: bool
        for admittance in (False, True):
            add_capacitance: bool
            add_inductance: bool
            for add_capacitance, add_inductance in (
                (False, False),
                (True, False),
                (False, True),
                (True, True),
            ):
                result: KramersKronigResult = (
                    pyimpspec.analysis.kramers_kronig.perform_kramers_kronig_test(
                        VALID_DATA,
                        add_capacitance=add_capacitance,
                        add_inductance=add_inductance,
                        admittance=admittance,
                        num_RC=num_RC,
                        log_F_ext=log_F_ext,
                        num_F_ext_evaluations=0,
                    )
                )

                self.assertEqual(result.admittance, admittance)
                elements: List[Element] = result.circuit.get_elements()
                self.assertEqual(
                    len(
                        [
                            e
                            for e in elements
                            if isinstance(
                                e,
                                (
                                    KramersKronigAdmittanceRC
                                    if admittance
                                    else KramersKronigRC
                                ),
                            )
                        ]
                    ),
                    num_RC,
                )

                if admittance:
                    self.assertFalse(isnan(result.get_parallel_resistance()))
                    self.assertTrue(isnan(result.get_series_capacitance()))
                    self.assertTrue(isnan(result.get_series_inductance()))
                    self.assertNotEqual(
                        isnan(result.get_parallel_capacitance()),
                        add_capacitance,
                    )
                    self.assertNotEqual(
                        isnan(result.get_parallel_inductance()),
                        add_inductance,
                    )

                else:
                    self.assertFalse(isnan(result.get_series_resistance()))
                    self.assertTrue(isnan(result.get_parallel_capacitance()))
                    self.assertTrue(isnan(result.get_parallel_inductance()))
                    self.assertNotEqual(
                        isnan(result.get_series_capacitance()),
                        add_capacitance,
                    )
                    self.assertNotEqual(
                        isnan(result.get_series_inductance()),
                        add_inductance,
                    )

    def test_log_F_ext(self):
        min_log_F_ext: float = -0.25
        max_log_F_ext: float = 0.01
        num_F_ext_evaluations: int = 15

        result: KramersKronigResult = (
            pyimpspec.analysis.kramers_kronig.perform_kramers_kronig_test(
                VALID_DATA,
                test="complex",
                admittance=False,
                min_log_F_ext=min_log_F_ext,
                max_log_F_ext=max_log_F_ext,
                num_F_ext_evaluations=num_F_ext_evaluations,
            )
        )

        self.assertTrue(min_log_F_ext <= result.get_log_F_ext() <= max_log_F_ext)


class KramersKronigAlgorithms(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.min_log_F_ext: float = -0.25
        cls.max_log_F_ext: float = 0.35

        cls.Z_rapid_evaluations: List[
            Tuple[float, List[KramersKronigResult], float]
        ] = pyimpspec.analysis.kramers_kronig.evaluate_log_F_ext(
            VALID_DATA,
            test="complex",
            num_RCs=None,
            add_capacitance=True,
            add_inductance=True,
            admittance=False,
            min_log_F_ext=cls.min_log_F_ext,
            max_log_F_ext=cls.max_log_F_ext,
            log_F_ext=0.0,
            num_F_ext_evaluations=20,
            rapid_F_ext_evaluations=True,
            cnls_method="leastsq",
            max_nfev=0,
            timeout=60,
            num_procs=-1,
        )
        cls.Y_rapid_evaluations: List[
            Tuple[float, List[KramersKronigResult], float]
        ] = pyimpspec.analysis.kramers_kronig.evaluate_log_F_ext(
            VALID_DATA,
            test="complex",
            num_RCs=None,
            add_capacitance=True,
            add_inductance=True,
            admittance=True,
            min_log_F_ext=cls.min_log_F_ext,
            max_log_F_ext=cls.max_log_F_ext,
            log_F_ext=0.0,
            num_F_ext_evaluations=20,
            rapid_F_ext_evaluations=True,
            cnls_method="leastsq",
            max_nfev=0,
            timeout=60,
            num_procs=-1,
        )

    def test_suggest_num_RC_limits(self):
        tests: List[KramersKronigResult]
        for tests in (
            self.Z_rapid_evaluations[0][1],
            self.Y_rapid_evaluations[0][1],
        ):
            lower_limit: int
            upper_limit: int
            lower_limit, upper_limit = (
                pyimpspec.analysis.kramers_kronig.suggest_num_RC_limits(
                    tests,
                    lower_limit=0,
                    upper_limit=0,
                    limit_delta=0,
                )
            )
            self.assertIsInstance(lower_limit, int)
            self.assertIsInstance(upper_limit, int)
            self.assertGreater(
                lower_limit,
                min(tests, key=lambda t: t.num_RC).num_RC,
            )
            self.assertLess(
                upper_limit,
                max(tests, key=lambda t: t.num_RC).num_RC,
            )
            self.assertGreater(upper_limit, lower_limit)

            lower_limit, upper_limit = (
                pyimpspec.analysis.kramers_kronig.suggest_num_RC_limits(
                    tests,
                    lower_limit=4,
                    upper_limit=8,
                    limit_delta=5,
                )
            )
            self.assertEqual(lower_limit, 4)
            self.assertEqual(upper_limit, 8)

            lower_limit, upper_limit = (
                pyimpspec.analysis.kramers_kronig.suggest_num_RC_limits(
                    tests,
                    lower_limit=4,
                    upper_limit=15,
                    limit_delta=5,
                )
            )
            self.assertEqual(lower_limit, 4)
            self.assertEqual(upper_limit, 9)

    def test_suggest_num_RC(self):
        tests: List[KramersKronigResult]
        for tests in (
            self.Z_rapid_evaluations[0][1],
            self.Y_rapid_evaluations[0][1],
        ):
            i: int
            for i in range(0, 6 + 1):
                result: KramersKronigResult
                scores: Dict[int, float]
                lower_limit: int
                upper_limit: int
                result, scores, lower_limit, upper_limit = (
                    pyimpspec.analysis.kramers_kronig.suggest_num_RC(
                        tests,
                        methods=[] if i < 1 else [i],
                    )
                )

                self.assertIsInstance(result, KramersKronigResult)

                self.assertIsInstance(scores, dict)
                self.assertTrue(all(map(lambda k: isinstance(k, int), scores.keys())))
                self.assertTrue(
                    all(map(lambda v: isinstance(v, float), scores.values()))
                )
                self.assertTrue(all(map(lambda v: 0.0 <= v <= 1.0, scores.values())))

                self.assertIsInstance(lower_limit, int)
                self.assertIsInstance(upper_limit, int)
                self.assertGreater(upper_limit, lower_limit)
                self.assertLessEqual(result.num_RC, upper_limit)
                if i > 0:
                    self.assertGreaterEqual(result.num_RC, lower_limit)

            methods: List[int] = list(range(1, 6 + 1))

            with self.assertRaises(ValueError):
                result, scores, lower_limit, upper_limit = (
                    pyimpspec.analysis.kramers_kronig.suggest_num_RC(
                        tests,
                        methods=methods,
                    )
                )

            with self.assertRaises(ValueError):
                result, scores, lower_limit, upper_limit = (
                    pyimpspec.analysis.kramers_kronig.suggest_num_RC(
                        tests,
                        methods=methods,
                        use_mean=True,
                        use_sum=True,
                    )
                )

            for kwargs in (
                dict(use_mean=True),
                dict(use_sum=True),
                dict(use_ranking=True),
            ):
                result, scores, lower_limit, upper_limit = (
                    pyimpspec.analysis.kramers_kronig.suggest_num_RC(
                        tests,
                        methods=methods,
                        **kwargs,
                    )
                )

                self.assertIsInstance(result, KramersKronigResult)

                self.assertIsInstance(scores, dict)
                self.assertTrue(
                    all(
                        map(
                            lambda k: isinstance(k, int),
                            scores.keys(),
                        )
                    )
                )
                self.assertTrue(
                    all(
                        map(
                            lambda v: isinstance(v, float),
                            scores.values(),
                        )
                    )
                )
                self.assertTrue(
                    all(
                        map(
                            lambda v: 0.0 <= v <= max(methods),
                            scores.values(),
                        )
                    )
                )

                self.assertIsInstance(lower_limit, int)
                self.assertIsInstance(upper_limit, int)
                self.assertGreater(upper_limit, lower_limit)
                self.assertLessEqual(result.num_RC, upper_limit)
                self.assertGreaterEqual(result.num_RC, lower_limit)

    def test_suggest_representation(self):
        data: DataSet = pyimpspec.generate_mock_data(
            "CIRCUIT_3*",
            noise=5e-2,
            seed=42,
        )[0]
        suggestions: List[Tuple[KramersKronigResult, Dict[int, float], int, int]] = (
            list(
                map(
                    pyimpspec.analysis.kramers_kronig.suggest_num_RC,
                    (
                        pyimpspec.analysis.kramers_kronig.evaluate_log_F_ext(
                            data,
                            admittance=False,
                            log_F_ext=0.0,
                            num_F_ext_evaluations=0,
                        )[0][1],
                        pyimpspec.analysis.kramers_kronig.evaluate_log_F_ext(
                            data,
                            admittance=True,
                            log_F_ext=0.0,
                            num_F_ext_evaluations=0,
                        )[0][1],
                    ),
                )
            )
        )

        result: KramersKronigResult
        scores: Dict[int, float]
        lower_limit: int
        upper_limit: int
        result, scores, lower_limit, upper_limit = (
            pyimpspec.analysis.kramers_kronig.suggest_representation(suggestions)
        )

        self.assertIsInstance(result, KramersKronigResult)
        self.assertEqual(result.admittance, True)

        self.assertIsInstance(scores, dict)
        self.assertTrue(all(map(lambda k: isinstance(k, int), scores.keys())))
        self.assertTrue(all(map(lambda v: isinstance(v, float), scores.values())))
        self.assertTrue(all(map(lambda v: 0.0 <= v <= 1.0, scores.values())))

        self.assertIsInstance(lower_limit, int)
        self.assertIsInstance(upper_limit, int)
        self.assertLessEqual(result.num_RC, upper_limit)

    def test_method_1(self):
        tests: List[KramersKronigResult]
        for tests in (
            self.Z_rapid_evaluations[0][1],
            self.Y_rapid_evaluations[0][1],
        ):
            result: KramersKronigResult
            scores: Dict[int, float]
            lower_limit: int
            upper_limit: int

            # Original approach
            result, scores, lower_limit, upper_limit = (
                pyimpspec.analysis.kramers_kronig.suggest_num_RC(
                    tests,
                    methods=[1],
                    mu_criterion=0.75,
                    beta=0.0,
                    relative_scores=False,
                )
            )

            self.assertIsInstance(result, KramersKronigResult)

            self.assertTrue(all(map(lambda k: isinstance(k, int), scores.keys())))
            self.assertTrue(all(map(lambda v: isinstance(v, float), scores.values())))
            self.assertTrue(all(map(lambda v: 0.0 <= v <= 1.0, scores.values())))

            self.assertIsInstance(lower_limit, int)
            self.assertIsInstance(upper_limit, int)
            self.assertGreater(upper_limit, lower_limit)
            self.assertGreaterEqual(result.num_RC, lower_limit)
            self.assertLessEqual(result.num_RC, upper_limit)

            # Modified approach 1 (fit logistic function)
            result, scores, lower_limit, upper_limit = (
                pyimpspec.analysis.kramers_kronig.suggest_num_RC(
                    tests,
                    methods=[1],
                    mu_criterion=-1.0,
                    relative_scores=False,
                )
            )

            self.assertIsInstance(result, KramersKronigResult)

            self.assertTrue(all(map(lambda k: isinstance(k, int), scores.keys())))
            self.assertTrue(all(map(lambda v: isinstance(v, float), scores.values())))
            self.assertTrue(all(map(lambda v: 0.0 <= v <= 1.0, scores.values())))

            self.assertIsInstance(lower_limit, int)
            self.assertIsInstance(upper_limit, int)
            self.assertGreater(upper_limit, lower_limit)
            self.assertGreaterEqual(result.num_RC, lower_limit)
            self.assertLessEqual(result.num_RC, upper_limit)

            # Modified approach 2 (calculate score S)
            result, scores, lower_limit, upper_limit = (
                pyimpspec.analysis.kramers_kronig.suggest_num_RC(
                    tests,
                    methods=[1],
                    relative_scores=False,
                )
            )

            self.assertIsInstance(result, KramersKronigResult)

            self.assertTrue(all(map(lambda k: isinstance(k, int), scores.keys())))
            self.assertTrue(all(map(lambda v: isinstance(v, float), scores.values())))

            self.assertIsInstance(lower_limit, int)
            self.assertIsInstance(upper_limit, int)
            self.assertGreater(upper_limit, lower_limit)
            self.assertGreaterEqual(result.num_RC, lower_limit)
            self.assertLessEqual(result.num_RC, upper_limit)

    def test_method_2(self):
        tests: List[KramersKronigResult]
        for tests in (
            self.Z_rapid_evaluations[0][1],
            self.Y_rapid_evaluations[0][1],
        ):
            result: KramersKronigResult
            scores: Dict[int, float]
            lower_limit: int
            upper_limit: int

            result, scores, lower_limit, upper_limit = (
                pyimpspec.analysis.kramers_kronig.suggest_num_RC(
                    tests,
                    methods=[2],
                    relative_scores=False,
                )
            )

            self.assertIsInstance(result, KramersKronigResult)

            self.assertTrue(all(map(lambda k: isinstance(k, int), scores.keys())))
            self.assertTrue(all(map(lambda v: isinstance(v, float), scores.values())))

            self.assertIsInstance(lower_limit, int)
            self.assertIsInstance(upper_limit, int)
            self.assertGreater(upper_limit, lower_limit)
            self.assertGreaterEqual(result.num_RC, lower_limit)
            self.assertLessEqual(result.num_RC, upper_limit)

    def test_calculate_curvature(self):
        test: KramersKronigResult = self.Z_rapid_evaluations[0][1][-1]
        Z: ComplexImpedances = test.get_impedances()
        kappa: NDArray[float64] = calculate_curvatures(Z)

        i: int = 0
        for i in range(0, kappa.size):
            self.assertTrue(
                isclose(
                    kappa[i],
                    _fit_osculating_circle(Z[i], Z[i+1], Z[i+2]),
                )
            )

        self.assertGreater(i, 0)

    def test_method_3(self):
        tests: List[KramersKronigResult]
        for tests in (
            self.Z_rapid_evaluations[0][1],
            self.Y_rapid_evaluations[0][1],
        ):
            result: KramersKronigResult
            scores: Dict[int, float]
            lower_limit: int
            upper_limit: int

            # Original approach
            result, scores, lower_limit, upper_limit = (
                pyimpspec.analysis.kramers_kronig.suggest_num_RC(
                    tests,
                    methods=[3],
                    subdivision=0,
                    relative_scores=False,
                )
            )

            self.assertIsInstance(result, KramersKronigResult)

            self.assertTrue(all(map(lambda k: isinstance(k, int), scores.keys())))
            self.assertTrue(all(map(lambda v: isinstance(v, float), scores.values())))

            self.assertIsInstance(lower_limit, int)
            self.assertIsInstance(upper_limit, int)
            self.assertGreater(upper_limit, lower_limit)
            self.assertGreaterEqual(result.num_RC, lower_limit)
            self.assertLessEqual(result.num_RC, upper_limit)

            # Modified approach
            result, scores, lower_limit, upper_limit = (
                pyimpspec.analysis.kramers_kronig.suggest_num_RC(
                    tests,
                    methods=[3],
                    subdivision=4,
                    relative_scores=False,
                )
            )

            self.assertIsInstance(result, KramersKronigResult)

            self.assertTrue(all(map(lambda k: isinstance(k, int), scores.keys())))
            self.assertTrue(all(map(lambda v: isinstance(v, float), scores.values())))

            self.assertIsInstance(lower_limit, int)
            self.assertIsInstance(upper_limit, int)
            self.assertGreater(upper_limit, lower_limit)
            self.assertGreaterEqual(result.num_RC, lower_limit)
            self.assertLessEqual(result.num_RC, upper_limit)

    def test_method_4(self):
        tests: List[KramersKronigResult]
        for tests in (
            self.Z_rapid_evaluations[0][1],
            self.Y_rapid_evaluations[0][1],
        ):
            result: KramersKronigResult
            scores: Dict[int, float]
            lower_limit: int
            upper_limit: int

            # Original approach
            result, scores, lower_limit, upper_limit = (
                pyimpspec.analysis.kramers_kronig.suggest_num_RC(
                    tests,
                    methods=[4],
                    subdivision=0,
                    offset_factor=0.0,
                    relative_scores=False,
                )
            )

            self.assertIsInstance(result, KramersKronigResult)

            self.assertTrue(all(map(lambda k: isinstance(k, int), scores.keys())))
            self.assertTrue(all(map(lambda v: isinstance(v, float), scores.values())))

            self.assertIsInstance(lower_limit, int)
            self.assertIsInstance(upper_limit, int)
            self.assertGreater(upper_limit, lower_limit)
            self.assertGreaterEqual(result.num_RC, lower_limit)
            self.assertLessEqual(result.num_RC, upper_limit)

            # Modified approach
            result, scores, lower_limit, upper_limit = (
                pyimpspec.analysis.kramers_kronig.suggest_num_RC(
                    tests,
                    methods=[4],
                    subdivision=4,
                    offset_factor=1e-1,
                    relative_scores=False,
                )
            )

            self.assertIsInstance(result, KramersKronigResult)

            self.assertTrue(all(map(lambda k: isinstance(k, int), scores.keys())))
            self.assertTrue(all(map(lambda v: isinstance(v, float), scores.values())))

            self.assertIsInstance(lower_limit, int)
            self.assertIsInstance(upper_limit, int)
            self.assertGreater(upper_limit, lower_limit)
            self.assertGreaterEqual(result.num_RC, lower_limit)
            self.assertLessEqual(result.num_RC, upper_limit)

    def test_method_5(self):
        tests: List[KramersKronigResult]
        for tests in (
            self.Z_rapid_evaluations[0][1],
            self.Y_rapid_evaluations[0][1],
        ):
            result: KramersKronigResult
            scores: Dict[int, float]
            lower_limit: int
            upper_limit: int

            result, scores, lower_limit, upper_limit = (
                pyimpspec.analysis.kramers_kronig.suggest_num_RC(
                    tests,
                    methods=[5],
                    relative_scores=False,
                )
            )

            self.assertIsInstance(result, KramersKronigResult)

            self.assertTrue(all(map(lambda k: isinstance(k, int), scores.keys())))
            self.assertTrue(all(map(lambda v: isinstance(v, float), scores.values())))

            self.assertIsInstance(lower_limit, int)
            self.assertIsInstance(upper_limit, int)
            self.assertGreater(upper_limit, lower_limit)
            self.assertGreaterEqual(result.num_RC, lower_limit)
            self.assertLessEqual(result.num_RC, upper_limit)

    def test_method_6(self):
        tests: List[KramersKronigResult]
        for tests in (
            self.Z_rapid_evaluations[0][1],
            self.Y_rapid_evaluations[0][1],
        ):
            result: KramersKronigResult
            scores: Dict[int, float]
            lower_limit: int
            upper_limit: int

            result, scores, lower_limit, upper_limit = (
                pyimpspec.analysis.kramers_kronig.suggest_num_RC(
                    tests,
                    methods=[6],
                    relative_scores=False,
                )
            )

            self.assertIsInstance(result, KramersKronigResult)

            self.assertTrue(all(map(lambda k: isinstance(k, int), scores.keys())))
            self.assertTrue(all(map(lambda v: isinstance(v, float), scores.values())))

            self.assertIsInstance(lower_limit, int)
            self.assertIsInstance(upper_limit, int)
            self.assertGreater(upper_limit, lower_limit)
            self.assertGreaterEqual(result.num_RC, lower_limit)
            self.assertLessEqual(result.num_RC, upper_limit)


class KramersKronigTestResult(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results: List[KramersKronigResult] = []

        admittance: bool
        for admittance in (False, True):
            add_capacitance: bool
            add_inductance: bool
            for add_capacitance, add_inductance in (
                (False, False),
                (True, False),
                (False, True),
                (True, True),
            ):
                evaluations = pyimpspec.analysis.kramers_kronig.evaluate_log_F_ext(
                    VALID_DATA,
                    test="complex",
                    admittance=admittance,
                    add_capacitance=add_capacitance,
                    add_inductance=add_inductance,
                )
                cls.results.append(
                    pyimpspec.analysis.kramers_kronig.suggest_num_RC(evaluations[0][1])[
                        0
                    ]
                )

    def test_circuit_serialization(self):
        result: KramersKronigResult
        for result in self.results:
            cdc: str = result.circuit.serialize()
            add_capacitance: bool = not (
                isnan(result.get_series_capacitance())
                and isnan(result.get_parallel_capacitance())
            )
            add_inductance: bool = not (
                isnan(result.get_series_inductance())
                and isnan(result.get_parallel_inductance())
            )
            self.assertEqual(
                cdc.count("/inf/inf"),
                1 + result.num_RC * 2 + int(add_capacitance) + int(add_inductance),
            )

            circuit: Circuit = parse_cdc(cdc)
            self.assertEqual(cdc, circuit.serialize())

            element: Element
            for element in circuit.get_elements(recursive=True):
                key: str
                value: float
                for key, value in element.get_lower_limits().items():
                    self.assertTrue(isneginf(value))

                for key, value in element.get_upper_limits().items():
                    self.assertTrue(isposinf(value))

    def test_repr(self):
        result: KramersKronigResult
        for result in self.results:
            string: str = repr(result)

            self.assertTrue(string.startswith("KramersKronigResult"))
            self.assertTrue(("X=Z" in string) or ("X=Y" in string))
            self.assertTrue(f"log_F_ext={result.log_F_ext:.3f}" in string)
            self.assertTrue(f"num_RC={result.num_RC}" in string)

    def test_admittance(self):
        result: KramersKronigResult
        for result in self.results:
            self.assertIsInstance(result.admittance, bool)
            self.assertTrue(
                any(
                    (
                        isinstance(
                            element,
                            (
                                KramersKronigAdmittanceRC
                                if result.admittance
                                else KramersKronigRC
                            ),
                        )
                        for element in result.circuit.get_elements(recursive=True)
                    )
                ),
            )

    def test_get_label(self):
        result: KramersKronigResult
        for result in self.results:
            label: str = result.get_label()

            self.assertIsInstance(label, str)
            self.assertTrue(label.startswith("Y" if result.admittance else "Z"))

            elements: List[Element] = result.circuit.get_elements()
            add_capacitance: bool = any(
                (isinstance(element, Capacitor) for element in elements)
            )
            add_inductance: bool = any(
                (isinstance(element, Inductor) for element in elements)
            )

            if add_capacitance and add_inductance:
                self.assertTrue(", C+L" in label)
            elif add_capacitance:
                self.assertTrue(", C" in label)
            elif add_inductance:
                self.assertTrue(", L" in label)

            self.assertTrue(f" = {result.num_RC}" in label)

            log_F_ext: float = result.get_log_F_ext()
            if log_F_ext == 0.0:
                self.assertTrue("F_" not in label)
            else:
                self.assertTrue("F_" in label)

    def test_get_frequencies(self):
        result: KramersKronigResult
        for result in self.results:
            lengths: List[int] = []

            num_per_decade: Optional[int]
            for num_per_decade in (None, 0, 2, 100):
                frequencies: Frequencies
                if num_per_decade is None:
                    frequencies = result.get_frequencies()
                else:
                    frequencies = result.get_frequencies(num_per_decade=num_per_decade)

                self.assertIsInstance(frequencies, ndarray)
                self.assertEqual(frequencies.dtype, float64)
                self.assertEqual(len(frequencies.shape), 1)

                m: int = len(frequencies)
                self.assertGreater(m, 0)

                if num_per_decade == 0:
                    self.assertEqual(m, lengths[-1])
                elif num_per_decade == 2:
                    self.assertLess(m, lengths[-1])
                elif num_per_decade == 100:
                    self.assertGreater(m, lengths[0])
                    self.assertGreater(m, lengths[-1])
                elif num_per_decade is None:
                    pass
                else:
                    raise NotImplementedError()

                lengths.append(m)

    def test_get_impedances(self):
        result: KramersKronigResult
        for result in self.results:
            lengths: List[int] = []

            num_per_decade: Optional[int]
            for num_per_decade in (None, 0, 2, 100):
                impedances: ComplexImpedances
                if num_per_decade is None:
                    impedances = result.get_impedances()
                else:
                    impedances = result.get_impedances(num_per_decade=num_per_decade)

                self.assertIsInstance(impedances, ndarray)
                self.assertEqual(impedances.dtype, complex128)
                self.assertEqual(len(impedances.shape), 1)

                m: int = len(impedances)
                self.assertGreater(m, 0)

                if num_per_decade == 0:
                    self.assertEqual(m, lengths[-1])
                elif num_per_decade == 2:
                    self.assertLess(m, lengths[-1])
                elif num_per_decade == 100:
                    self.assertGreater(m, lengths[0])
                    self.assertGreater(m, lengths[-1])
                elif num_per_decade is None:
                    pass
                else:
                    raise NotImplementedError()

                lengths.append(m)

    def test_get_nyquist_data(self):
        result: KramersKronigResult
        for result in self.results:
            lengths: List[int] = []

            num_per_decade: Optional[int]
            for num_per_decade in (None, 0, 2, 100):
                nyquist_data: Tuple[Impedances, Impedances]
                if num_per_decade is None:
                    nyquist_data = result.get_nyquist_data()
                else:
                    nyquist_data = result.get_nyquist_data(
                        num_per_decade=num_per_decade
                    )

                self.assertIsInstance(nyquist_data, tuple)
                self.assertEqual(len(nyquist_data), 2)

                for values in nyquist_data:
                    self.assertIsInstance(values, ndarray)
                    self.assertEqual(values.dtype, float64)
                    self.assertEqual(len(values.shape), 1)

                m: int = len(nyquist_data[0])
                self.assertGreater(m, 0)
                self.assertTrue(all(map(lambda values: len(values) == m, nyquist_data)))

                if num_per_decade == 0:
                    self.assertEqual(m, lengths[-1])
                elif num_per_decade == 2:
                    self.assertLess(m, lengths[-1])
                elif num_per_decade == 100:
                    self.assertGreater(m, lengths[0])
                    self.assertGreater(m, lengths[-1])
                elif num_per_decade is None:
                    pass
                else:
                    raise NotImplementedError()

                lengths.append(m)

    def test_get_bode_data(self):
        result: KramersKronigResult
        for result in self.results:
            lengths: List[int] = []

            num_per_decade: Optional[int]
            for num_per_decade in (None, 0, 2, 100):
                bode_data: Tuple[Frequencies, Impedances, Phases]
                if num_per_decade is None:
                    bode_data = result.get_bode_data()
                else:
                    bode_data = result.get_bode_data(num_per_decade=num_per_decade)

                self.assertIsInstance(bode_data, tuple)
                self.assertEqual(len(bode_data), 3)

                for values in bode_data:
                    self.assertIsInstance(values, ndarray)
                    self.assertEqual(values.dtype, float64)
                    self.assertEqual(len(values.shape), 1)

                m: int = len(bode_data[0])
                self.assertGreater(m, 0)
                self.assertTrue(all(map(lambda values: len(values) == m, bode_data)))

                if num_per_decade == 0:
                    self.assertEqual(m, lengths[-1])
                elif num_per_decade == 2:
                    self.assertLess(m, lengths[-1])
                elif num_per_decade == 100:
                    self.assertGreater(m, lengths[0])
                    self.assertGreater(m, lengths[-1])
                elif num_per_decade is None:
                    pass
                else:
                    raise NotImplementedError()

                lengths.append(m)

    def test_get_residuals_data(self):
        result: KramersKronigResult
        for result in self.results:
            residuals_data: Tuple[Frequencies, Residuals, Residuals]
            residuals_data = result.get_residuals_data()

            self.assertIsInstance(residuals_data, tuple)
            self.assertEqual(len(residuals_data), 3)

            for values in residuals_data:
                self.assertIsInstance(values, ndarray)
                self.assertEqual(values.dtype, float64)
                self.assertEqual(len(values.shape), 1)

            m: int = len(residuals_data[0])
            self.assertGreater(m, 0)
            self.assertTrue(all(map(lambda values: len(values) == m, residuals_data)))

    def test_get_time_constants(self):
        result: KramersKronigResult
        for result in self.results:
            time_constants: TimeConstants = result.get_time_constants()

            self.assertIsInstance(time_constants, ndarray)
            self.assertEqual(time_constants.dtype, float64)
            self.assertEqual(len(time_constants.shape), 1)
            self.assertEqual(len(time_constants), result.num_RC)
            self.assertTrue(all(map(lambda tau: tau > 0, time_constants)))

    def test_get_log_F_ext(self):
        result: KramersKronigResult
        for result in self.results:
            log_F_ext: float = result.get_log_F_ext()

            self.assertIsInstance(log_F_ext, float)

    def test_perform_lilliefors_test(self):
        result: KramersKronigResult
        for result in self.results:
            p_values: Tuple[float, float] = result.perform_lilliefors_test()

            self.assertIsInstance(p_values, tuple)
            self.assertEqual(len(p_values), 2)

            p: float
            for p in p_values:
                self.assertIsInstance(p, float)
                self.assertTrue(0.0 <= p <= 1.0)

    def test_perform_shapiro_wilk_test(self):
        result: KramersKronigResult
        for result in self.results:
            p_values: Tuple[float, float] = result.perform_shapiro_wilk_test()

            self.assertIsInstance(p_values, tuple)
            self.assertEqual(len(p_values), 2)

            p: float
            for p in p_values:
                self.assertIsInstance(p, float)
                self.assertTrue(0.0 <= p <= 1.0)

    def test_perform_kolmogorov_smirnov_test(self):
        result: KramersKronigResult
        for result in self.results:
            p_values: Tuple[float, float] = result.perform_kolmogorov_smirnov_test()

            self.assertIsInstance(p_values, tuple)
            self.assertEqual(len(p_values), 2)

            p: float
            for p in p_values:
                self.assertIsInstance(p, float)
                self.assertTrue(0.0 <= p <= 1.0)

    def test_to_statistics_dataframe(self):
        result: KramersKronigResult
        for result in self.results:
            elements: List[Element] = result.circuit.get_elements()
            add_capacitance: bool = any(
                (isinstance(element, Capacitor) for element in elements)
            )
            add_inductance: bool = any(
                (isinstance(element, Inductor) for element in elements)
            )

            extended_statistics: int
            for extended_statistics in (0, 1, 2, 3):
                df: DataFrame = result.to_statistics_dataframe(
                    extended_statistics=extended_statistics
                )

                self.assertIsInstance(df, DataFrame)

                dictionary: Dict[str, dict] = df.to_dict()
                labels: Dict[int, str] = dictionary["Label"]
                values: Dict[int, float] = dictionary["Value"]

                self.assertTrue(
                    all(
                        map(
                            lambda key: isinstance(key, int),
                            labels.keys(),
                        )
                    )
                )
                self.assertTrue(
                    all(
                        map(
                            lambda value: isinstance(value, str),
                            labels.values(),
                        )
                    )
                )

                self.assertTrue(
                    all(
                        map(
                            lambda key: isinstance(key, int),
                            values.keys(),
                        )
                    )
                )
                self.assertTrue(
                    all(
                        map(
                            lambda value: isinstance(value, float),
                            values.values(),
                        )
                    )
                )

                self.assertEqual(len(labels), len(values))

                num_statistics: int = (
                    4 + (1 if add_capacitance else 0) + (1 if add_inductance else 0)
                )
                if extended_statistics > 0:
                    num_statistics += 10

                if extended_statistics > 1:
                    num_statistics += 4

                if extended_statistics > 2:
                    num_statistics += 3

                self.assertEqual(len(labels), num_statistics)

    def test_get_series_resistance(self):
        result: KramersKronigResult
        for result in self.results:
            value: float = result.get_series_resistance()

            self.assertTrue(_is_floating(value))
            self.assertEqual(isnan(value), result.admittance)

    def test_get_series_capacitance(self):
        result: KramersKronigResult
        for result in self.results:
            value: float = result.get_series_capacitance()

            self.assertTrue(_is_floating(value))

            elements: List[Element] = result.circuit.get_elements(recursive=True)
            add_capacitance: bool = any(
                (isinstance(element, Capacitor) for element in elements)
            )

            self.assertEqual(isnan(value), result.admittance or not add_capacitance)

    def test_get_series_inductance(self):
        result: KramersKronigResult
        for result in self.results:
            value: float = result.get_series_inductance()

            self.assertTrue(_is_floating(value))

            elements: List[Element] = result.circuit.get_elements(recursive=True)
            add_inductance: bool = any(
                (isinstance(element, Inductor) for element in elements)
            )

            self.assertEqual(isnan(value), result.admittance or not add_inductance)

    def test_get_parallel_resistance(self):
        result: KramersKronigResult
        for result in self.results:
            value: float = result.get_parallel_resistance()

            self.assertTrue(_is_floating(value))
            self.assertEqual(isnan(value), not result.admittance)

    def test_get_parallel_capacitance(self):
        result: KramersKronigResult
        for result in self.results:
            value: float = result.get_parallel_capacitance()

            self.assertTrue(_is_floating(value))

            elements: List[Element] = result.circuit.get_elements(recursive=True)
            add_capacitance: bool = any(
                (isinstance(element, Capacitor) for element in elements)
            )

            self.assertEqual(isnan(value), not result.admittance or not add_capacitance)

    def test_get_parallel_inductance(self):
        result: KramersKronigResult
        for result in self.results:
            value: float = result.get_parallel_inductance()

            self.assertTrue(_is_floating(value))

            elements: List[Element] = result.circuit.get_elements(recursive=True)
            add_inductance: bool = any(
                (isinstance(element, Inductor) for element in elements)
            )

            self.assertEqual(isnan(value), not result.admittance or not add_inductance)

    def test_get_estimated_percent_noise(self):
        result: KramersKronigResult
        for result in self.results:
            pct: float = result.get_estimated_percent_noise()

            self.assertIsInstance(pct, float)
            self.assertAlmostEqual(
                pct,
                pyimpspec.analysis.kramers_kronig.utility._estimate_pct_noise(
                    Z=result.get_impedances(),
                    pseudo_chisqr=result.pseudo_chisqr,
                ),
            )

    def test_matplotlib(self):
        result: KramersKronigResult = self.results[0]

        plotter: Callable
        for plotter in primitive_mpl_plotters:
            check_mpl_return_values(
                self,
                *plotter(data=result),
            )

        check_mpl_return_values(
            self,
            *mpl.plot_residuals(result),
        )

        check_mpl_return_values(
            self,
            *mpl.plot_residuals(
                result,
                colored_axes=True,
            ),
        )

        check_mpl_return_values(
            self,
            *mpl.plot_fit(
                result,
                data=VALID_DATA,
            ),
        )

        check_mpl_return_values(
            self,
            *mpl.plot_fit(
                result,
                data=VALID_DATA,
                colored_axes=True,
            ),
        )
