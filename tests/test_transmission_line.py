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
    Optional,
    Set,
    Tuple,
)
from unittest import TestCase
from numpy import (
    allclose,
    array,
    isinf,
    logspace,
    zeros,
)
from sympy import (
    Basic,
    Expr,
)
from pyimpspec.circuit.base import InfiniteImpedance
from pyimpspec import (
    Capacitor,
    Circuit,
    Element,
    Parallel,
    Resistor,
    Series,
    TransmissionLineModel,
    TransmissionLineModelBlockingCPE,
    TransmissionLineModelBlockingOpen,
    TransmissionLineModelBlockingShort,
    TransmissionLineModelNonblockingCPE,
    TransmissionLineModelNonblockingOpen,
    TransmissionLineModelNonblockingShort,
    parse_cdc,
)
from pyimpspec.exceptions import NotANumberImpedance
from pyimpspec.typing import (
    ComplexImpedance,
    ComplexImpedances,
    Frequencies,
)


class TestTransmissionLineModel(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.f: Frequencies = logspace(5, -5, num=11)
        cls.C: Capacitor = Capacitor()
        cls.R: Resistor = Resistor()
        cls.symbol: str = "Tlm"
        cls.default: TransmissionLineModel = cls.parse(cls.symbol)

    @classmethod
    def parse(cls, cdc: str, is_tlm: bool = True) -> Element:
        circuit: Circuit = parse_cdc(cdc)
        assert len(circuit.get_elements()) == 1, len(circuit.get_elements())
        tlm: Element = circuit.get_elements()[0]
        if is_tlm:
            assert isinstance(tlm, TransmissionLineModel), type(tlm)
        return tlm

    def extract_frequency(self, expr: Expr) -> Optional[Basic]:
        symbols: Set[Basic] = expr.free_symbols
        self.assertIsInstance(symbols, set)
        if len(symbols) == 0:
            return None
        self.assertEqual(len(symbols), 1)
        symbol: Basic = symbols.pop()
        self.assertEqual(str(symbol), "f")
        return symbol

    def compare_special_case(self, tlm: TransmissionLineModel, tlm_s: Element):
        self.assertTrue(
            allclose(
                tlm.get_impedances(self.f), tlm_s.get_impedances(self.f), rtol=1e-2
            ),
        )

    def compare_sympy(self, tlm: TransmissionLineModel) -> Tuple[Expr, Optional[Basic]]:
        expr: Expr = tlm.to_sympy(substitute=True)
        f: Optional[Basic] = self.extract_frequency(expr)
        Z: ComplexImpedances
        Z_expr: ComplexImpedances
        if f is not None:
            Z_expr = array(
                list(map(lambda _: complex(expr.subs(f, _)), self.f)),
                dtype=ComplexImpedance,
            )
            if isinf(Z_expr).all():
                with self.assertRaises(InfiniteImpedance):
                    tlm.get_impedances(self.f)
                self.assertTrue(all(map(lambda _: isinf(_), Z_expr)))
            else:
                Z = tlm.get_impedances(self.f)
                self.assertTrue(allclose(Z, Z_expr))
        else:
            Z_expr = complex(expr)
            Z_expr = array(list(map(lambda _: complex(expr), self.f)))
            if isinf(Z_expr).all():
                with self.assertRaises(InfiniteImpedance):
                    tlm.get_impedances(self.f)
                self.assertTrue(all(map(lambda _: isinf(_), Z_expr)))
            else:
                Z = tlm.get_impedances(self.f)
                self.assertTrue(allclose(Z, Z_expr))
        return (expr, f)

    def test_initializer(self):
        with self.assertRaises(TypeError):
            TransmissionLineModel(
                "X_1",
                Series([Resistor()]),
                "X_2",
                None,
            )
        TransmissionLineModel(
            X_1=Series([Resistor()]),
            X_2=None,
        )

    def test_simple_default_cdc(self):
        cdc: str = self.default.to_string()
        self.assertEqual(cdc, self.symbol)

    def test_extended_default_cdc(self):
        cdc: str = self.default.to_string(0)
        self.assertEqual(
            cdc,
            "Tlm{X_1=[R{R=1E+00/0E+00/inf}], X_2=short, Z_A=open, Z_B=open, Zeta=[Q{Y=5E-03/1E-24/1E+06,n=8E-01/0E+00/1E+00}], L=1E+00F/1E-24/inf}",
        )

    def test_get_set_label(self):
        self.assertEqual(self.default.get_label(), "")
        self.assertEqual(self.default.get_name(), self.symbol)
        self.default.set_label("test")
        self.assertEqual(self.default.get_label(), "test")
        self.assertEqual(self.default.get_name(), f"{self.symbol}_test")
        self.assertTrue(self.default.to_string(0).endswith(":test}"))
        self.default.set_label("")

    def test_shorted_X1_X2_Za_finite_rest_impedance(self):
        tlm = self.parse(
            self.symbol + "{X_1=short, X_2=short, Z_A=short, Z_B=C, Zeta=C}"
        )
        with self.assertRaises(NotImplementedError):
            tlm.get_impedances(self.f)
        with self.assertRaises(NotImplementedError):
            tlm.to_sympy(substitute=True)

    def test_shorted_X1_X2_Zb_finite_rest_impedance(self):
        tlm = self.parse(
            self.symbol + "{X_1=short, X_2=short, Z_A=C, Z_B=short, Zeta=C}"
        )
        with self.assertRaises(NotImplementedError):
            tlm.get_impedances(self.f)
        with self.assertRaises(NotImplementedError):
            tlm.to_sympy(substitute=True)

    def test_shorted_Z_A_Z_B_default_rest_impedance(self):
        tlm = self.parse(self.symbol + "{Z_A=short, Z_B=short}")
        with self.assertRaises(NotANumberImpedance):
            tlm.get_impedances(self.f)
        with self.assertRaises(ZeroDivisionError):
            tlm.to_sympy(substitute=True)

    def test_shorted_X1_X2_Zeta_finite_rest_impedance(self):
        tlm = self.parse(
            self.symbol + "{X_1=short, X_2=short, Z_A=C, Z_B=C, Zeta=short}"
        )
        with self.assertRaises(NotImplementedError):
            tlm.get_impedances(self.f)
        with self.assertRaises(NotImplementedError):
            tlm.to_sympy(substitute=True)

    def test_shorted_X1_X2_open_rest_impedance(self):
        tlm = self.parse(
            self.symbol + "{X_1=short, X_2=short, Z_A=open, Z_B=open, Zeta=open}"
        )
        with self.assertRaises(NotImplementedError):
            tlm.get_impedances(self.f)
        with self.assertRaises(NotImplementedError):
            tlm.to_sympy(substitute=True)

    def test_shorted_X1_X2_finite_Za_open_rest_impedance(self):
        tlm = self.parse(
            self.symbol + "{X_1=short, X_2=short, Z_A=C, Z_B=open, Zeta=open}"
        )
        with self.assertRaises(NotImplementedError):
            tlm.get_impedances(self.f)
        with self.assertRaises(NotImplementedError):
            tlm.to_sympy(substitute=True)

    def test_shorted_X1_X2_finite_Zb_open_rest_impedance(self):
        tlm = self.parse(
            self.symbol + "{X_1=short, X_2=short, Z_A=open, Z_B=C, Zeta=open}"
        )
        with self.assertRaises(NotImplementedError):
            tlm.get_impedances(self.f)
        with self.assertRaises(NotImplementedError):
            tlm.to_sympy(substitute=True)

    def test_shorted_X1_X2_finite_Zeta_open_rest_impedance(self):
        tlm = self.parse(
            self.symbol + "{X_1=short, X_2=short, Z_A=open, Z_B=open, Zeta=C}"
        )
        with self.assertRaises(NotImplementedError):
            tlm.get_impedances(self.f)
        with self.assertRaises(NotImplementedError):
            tlm.to_sympy(substitute=True)

    def test_shorted_X1_X2_finite_rest(self):
        tlm = self.parse(self.symbol + "{X_1=short, X_2=short, Z_A=C, Z_B=C, Zeta=C}")
        with self.assertRaises(NotImplementedError):
            tlm.get_impedances(self.f)
        with self.assertRaises(NotImplementedError):
            tlm.to_sympy(substitute=True)

    def test_open_Zeta_1(self):
        tlm = self.parse(self.symbol + "{X_1=R, X_2=R, Z_A=open, Z_B=C, Zeta=open}")
        with self.assertRaises(NotImplementedError):
            tlm.get_impedances(self.f)
        with self.assertRaises(NotImplementedError):
            tlm.to_sympy(substitute=True)

    def test_open_Zeta_2(self):
        tlm = self.parse(self.symbol + "{X_1=R, X_2=open, Z_A=C, Z_B=C, Zeta=open}")
        with self.assertRaises(NotImplementedError):
            tlm.get_impedances(self.f)
        with self.assertRaises(NotImplementedError):
            tlm.to_sympy(substitute=True)

    def test_open_Zeta_3(self):
        tlm = self.parse(self.symbol + "{X_1=R, X_2=R, Z_A=C, Z_B=open, Zeta=open}")
        with self.assertRaises(NotImplementedError):
            tlm.get_impedances(self.f)
        with self.assertRaises(NotImplementedError):
            tlm.to_sympy(substitute=True)

    def test_open_Zeta_4(self):
        tlm = self.parse(self.symbol + "{X_1=open, X_2=R, Z_A=C, Z_B=C, Zeta=open}")
        with self.assertRaises(NotImplementedError):
            tlm.get_impedances(self.f)
        with self.assertRaises(NotImplementedError):
            tlm.to_sympy(substitute=True)

    def test_open_Zeta_5(self):
        tlm = self.parse(self.symbol + "{X_1=R, X_2=R, Z_A=C, Z_B=C, Zeta=open}")
        with self.assertRaises(NotImplementedError):
            tlm.get_impedances(self.f)
        with self.assertRaises(NotImplementedError):
            tlm.to_sympy(substitute=True)

    def test_open_X1_X2(self):
        tlm = self.parse(self.symbol + "{X_1=open, X_2=open, Z_A=C, Z_B=C, Zeta=C}")
        with self.assertRaises(NotImplementedError):
            tlm.get_impedances(self.f)
        with self.assertRaises(NotImplementedError):
            tlm.to_sympy(substitute=True)

    def test_open_X1_finite_rest(self):
        tlm = self.parse(self.symbol + "{X_1=open, X_2=R, Z_A=C, Z_B=C, Zeta=C}")
        with self.assertRaises(NotImplementedError):
            tlm.get_impedances(self.f)
        with self.assertRaises(NotImplementedError):
            tlm.to_sympy(substitute=True)

    def test_open_X1_Zb_Zeta_finite_rest(self):
        tlm = self.parse(self.symbol + "{X_1=open, X_2=R, Z_A=C, Z_B=open, Zeta=open}")
        with self.assertRaises(NotImplementedError):
            tlm.get_impedances(self.f)
        with self.assertRaises(NotImplementedError):
            tlm.to_sympy(substitute=True)

    def test_open_X1_Za_finite_rest(self):
        tlm = self.parse(self.symbol + "{X_1=open, X_2=R, Z_A=open, Z_B=C, Zeta=C}")
        with self.assertRaises(NotImplementedError):
            tlm.get_impedances(self.f)
        with self.assertRaises(NotImplementedError):
            tlm.to_sympy(substitute=True)

    def test_open_X2_finite_rest(self):
        tlm = self.parse(self.symbol + "{X_1=R, X_2=open, Z_A=C, Z_B=C, Zeta=C}")
        with self.assertRaises(NotImplementedError):
            tlm.get_impedances(self.f)
        with self.assertRaises(NotImplementedError):
            tlm.to_sympy(substitute=True)

    def test_open_X2_Za_Zeta(self):
        tlm = self.parse(self.symbol + "{X_1=R, X_2=open, Z_A=open, Z_B=C, Zeta=open}")
        with self.assertRaises(NotImplementedError):
            tlm.get_impedances(self.f)
        with self.assertRaises(NotImplementedError):
            tlm.to_sympy(substitute=True)

    def test_open_X2_Zb_finite_rest(self):
        tlm = self.parse(self.symbol + "{X_1=R, X_2=open, Z_A=C, Z_B=open, Zeta=C}")
        with self.assertRaises(NotImplementedError):
            tlm.get_impedances(self.f)
        with self.assertRaises(NotImplementedError):
            tlm.to_sympy(substitute=True)

    def test_open_Za_Zb_1(self):
        tlm = self.parse(self.symbol + "{X_1=R, X_2=short, Z_A=open, Z_B=open, Zeta=C}")
        self.compare_sympy(tlm)

    def test_open_Za_Zb_2(self):
        tlm = self.parse(self.symbol + "{X_1=short, X_2=R, Z_A=open, Z_B=open, Zeta=C}")
        self.compare_sympy(tlm)

    def test_open_Za_Zb_3(self):
        tlm = self.parse(self.symbol + "{X_1=R, X_2=R, Z_A=open, Z_B=open, Zeta=C}")
        self.compare_sympy(tlm)

    def test_open_Za_finite_rest(self):
        tlm = self.parse(self.symbol + "{X_1=R, X_2=R, Z_A=open, Z_B=C, Zeta=C}")
        self.compare_sympy(tlm)

    def test_open_Zb_finite_rest(self):
        tlm = self.parse(self.symbol + "{X_1=R, X_2=R, Z_A=C, Z_B=open, Zeta=C}")
        self.compare_sympy(tlm)

    def test_all_finite(self):
        tlm = self.parse(self.symbol + "{X_1=R, X_2=R, Z_A=C, Z_B=C, Zeta=C}")
        self.compare_sympy(tlm)

    def test_all_shorted(self):
        tlm = self.parse(
            self.symbol + "{X_1=short, X_2=short, Z_A=short, Z_B=short, Zeta=short}"
        )
        with self.assertRaises(NotImplementedError):
            tlm.get_impedances(self.f)
        with self.assertRaises(NotImplementedError):
            tlm.to_sympy(substitute=True)

    def test_all_open_impedance(self):
        tlm = self.parse(
            self.symbol + "{X_1=open, X_2=open, Z_A=open, Z_B=open, Zeta=open}"
        )
        with self.assertRaises(NotImplementedError):
            self.assertTrue(tlm.get_impedances(self.f))

    def test_tlmbo(self):
        self.compare_special_case(
            self.parse(
                self.symbol
                + "{X_1=R{R=1.0}, X_2=short, Z_A=open, Z_B=open, Zeta=Q{Y=5e-3, n=0.8}}"
            ),
            TransmissionLineModelBlockingOpen(),
        )

    def test_tlmbq(self):
        self.compare_special_case(
            self.parse(
                self.symbol
                + "{X_1=R{R=1.0}, X_2=short, Z_A=open, Z_B=Q{Y=10e-3, n=0.7}, Zeta=Q{Y=5e-3, n=0.8}}"
            ),
            TransmissionLineModelBlockingCPE(),
        )

    def test_tlmbs(self):
        self.compare_special_case(
            self.parse(
                self.symbol
                + "{X_1=R{R=1.0}, X_2=short, Z_A=open, Z_B=short, Zeta=Q{Y=5e-3, n=0.8}}"
            ),
            TransmissionLineModelBlockingShort(),
        )

    def test_tlmno(self):
        self.compare_special_case(
            self.parse(
                self.symbol
                + "{X_1=R{R=1.0}, X_2=short, Z_A=open, Z_B=open, Zeta=(R{R=3.0}Q{Y=5e-3, n=0.8})}"
            ),
            TransmissionLineModelNonblockingOpen(),
        )

    def test_tlmnq(self):
        self.compare_special_case(
            self.parse(
                self.symbol
                + "{X_1=R{R=1.0}, X_2=short, Z_A=open, Z_B=(R{R=5.0}Q{Y=100e-3, n=0.7}), Zeta=(R{R=3.0}Q{Y=5e-3, n=0.8})}"
            ),
            TransmissionLineModelNonblockingCPE(),
        )

    def test_tlmns(self):
        self.compare_special_case(
            self.parse(
                self.symbol
                + "{X_1=R{R=1.0}, X_2=short, Z_A=open, Z_B=short, Zeta=(R{R=3.0}Q{Y=5e-3, n=0.8})}"
            ),
            TransmissionLineModelNonblockingShort(),
        )

    def test_shorted_X1_open_rest(self):
        tlm = self.parse(
            self.symbol + "{X_1=short, X_2=open, Z_A=open, Z_B=open, Zeta=open}"
        )
        with self.assertRaises(NotImplementedError):
            self.assertTrue(tlm.get_impedances(self.f))

    def test_shorted_X1_finite_rest(self):
        tlm = self.parse(self.symbol + "{X_1=short, X_2=R, Z_A=C, Z_B=C, Zeta=C}")
        self.compare_sympy(tlm)

    def test_shorted_X2_open_rest(self):
        tlm = self.parse(
            self.symbol + "{X_1=open, X_2=short, Z_A=open, Z_B=open, Zeta=open}"
        )
        with self.assertRaises(NotImplementedError):
            self.assertTrue(tlm.get_impedances(self.f))

    def test_shorted_X2_finite_rest(self):
        tlm = self.parse(self.symbol + "{X_1=R, X_2=short, Z_A=C, Z_B=C, Zeta=C}")
        self.compare_sympy(tlm)

    def test_get_subcircuit_descriptions(self):
        tlm: TransmissionLineModel = TransmissionLineModel()
        self.assertIsInstance(tlm.get_subcircuit_descriptions(), dict)
        description: str = (
            "The impedance of the liquid phase (i.e., ionic conductivity in the pore)"
        )
        self.assertEqual(tlm.get_subcircuit_description("X_1"), description)
        self.assertEqual(tlm.get_subcircuit_descriptions("X_1")["X_1"], description)
        self.assertEqual(tlm.get_subcircuit_descriptions(X_1=True)["X_1"], description)
        with self.assertRaises(KeyError):
            tlm.get_subcircuit_descriptions("Y")
        with self.assertRaises(KeyError):
            tlm.get_subcircuit_description("Y")
