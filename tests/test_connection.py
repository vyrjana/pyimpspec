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
    allclose,
    array,
    inf,
    zeros,
)
from schemdraw import Drawing
from sympy import Expr
from pyimpspec import (
    Connection,
    Parallel,
    Series,
    Element,
    Capacitor,
    Resistor,
    TransmissionLineModel,
)
from pyimpspec.circuit.base import InfiniteImpedance
from pyimpspec.typing import (
    ComplexImpedance,
    Frequency,
    Frequencies,
)
from pyimpspec.typing.helpers import List


class TestConnection(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.res1: Resistor = Resistor(R=250)
        cls.res2: Resistor = Resistor(R=400)
        cls.cap1: Capacitor = Capacitor()
        cls.series_nested_in_parallel: Parallel = Parallel(
            [cls.res1, Series([cls.res2, cls.cap1])]
        )
        cls.parallel_nested_in_series: Series = Series(
            [Parallel([cls.res1, cls.cap1]), cls.res2]
        )

    def test_impedance(self):
        f: Frequencies = array([1e-2, 1, 1e2], dtype=Frequency)
        self.assertTrue(
            allclose(
                self.series_nested_in_parallel.get_impedances(f),
                1
                / (
                    1 / self.res1.get_impedances(f)
                    + 1 / (self.res2.get_impedances(f) + self.cap1.get_impedances(f))
                ),
            )
        )
        self.assertTrue(
            allclose(
                self.parallel_nested_in_series.get_impedances(f),
                1 / (1 / self.res1.get_impedances(f) + 1 / self.cap1.get_impedances(f))
                + self.res2.get_impedances(f),
            )
        )
        self.assertFalse(
            allclose(
                self.series_nested_in_parallel.get_impedances(f),
                self.parallel_nested_in_series.get_impedances(f),
            )
        )

    def test_sympy(self):
        self.assertEqual(
            str(self.series_nested_in_parallel.to_sympy()),
            str(
                1
                / (
                    1 / self.res1.to_sympy(identifier=1)
                    + 1
                    / (
                        self.res2.to_sympy(identifier=2)
                        + self.cap1.to_sympy(identifier=1)
                    )
                )
            ),
        )
        self.assertEqual(
            str(self.parallel_nested_in_series.to_sympy()),
            str(
                1
                / (
                    1 / self.res1.to_sympy(identifier=1)
                    + 1 / self.cap1.to_sympy(identifier=1)
                )
                + self.res2.to_sympy(identifier=2)
            ),
        )
        self.assertNotEqual(
            str(self.series_nested_in_parallel.to_sympy()),
            str(self.parallel_nested_in_series.to_sympy()),
        )

    def test_sympy_impedance(self):
        fs: Frequencies = array([1e-2, 1, 1e2], dtype=Frequency)
        expr: Expr = self.series_nested_in_parallel.to_sympy(substitute=True)
        self.assertEqual(len(expr.free_symbols), 1)
        self.assertTrue(
            allclose(
                self.series_nested_in_parallel.get_impedances(fs),
                array(list(map(lambda _: complex(expr.subs("f", _)), fs))),
            )
        )

        expr = self.parallel_nested_in_series.to_sympy(substitute=True)
        self.assertEqual(len(expr.free_symbols), 1)
        self.assertTrue(
            allclose(
                self.parallel_nested_in_series.get_impedances(fs),
                array(list(map(lambda _: complex(expr.subs("f", _)), fs))),
            )
        )

    def test_get_connections(self):
        con: Connection = Series(
            [
                Resistor(),
                Parallel(
                    [
                        Capacitor(),
                        Series([Resistor(), TransmissionLineModel()]),
                    ]
                ),
            ]
        )
        connections: List[Connection] = con.get_connections()
        self.assertIsInstance(connections, list)
        self.assertEqual(len(connections), 2)
        self.assertTrue(
            all(map(lambda item: isinstance(item, Connection), connections))
        )

        connections = con.get_connections(recursive=True)
        self.assertIsInstance(connections, list)
        self.assertEqual(len(connections), 2)
        self.assertTrue(
            all(map(lambda item: isinstance(item, Connection), connections))
        )

        connections = con.get_connections(recursive=False)
        self.assertIsInstance(connections, list)
        self.assertEqual(len(connections), 1)
        self.assertTrue(
            all(map(lambda item: isinstance(item, Connection), connections))
        )

    def test_to_latex(self):
        latex: str = Series([Resistor(), Capacitor()]).to_latex()
        self.assertIsInstance(latex, str)
        self.assertEqual(latex, r"Z = R_{1} - \frac{j}{2 \pi C_{1} f}")

    def test_to_drawing(self):
        series: Series = Series([Resistor(), Capacitor()])
        drawing: Drawing = series.to_drawing()
        self.assertIsInstance(drawing, Drawing)

    def test_to_circuitikz(self):
        series: Series = Series([Resistor(), Capacitor()])
        circuitikz: str = series.to_circuitikz()
        self.assertIsInstance(circuitikz, str)


class TestSeries(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.f: Frequencies = array([1e-2, 1e0, 1e2], dtype=Frequency)
        cls.res: Resistor = Resistor()
        cls.cap: Capacitor = Capacitor()
        cls.tlm: TransmissionLineModel = TransmissionLineModel()
        cls.empty_series: Series = Series([])
        cls.single_element: Series = Series([cls.res])
        cls.multiple_elements: Series = Series([cls.res, cls.cap])
        cls.container: Series = Series([cls.res, cls.tlm])
        cls.shorted_series: Series = Series([cls.res, Resistor(R=0)])
        cls.open_series: Series = Series([cls.res, Resistor(R=inf)])

    def test_contains(self):
        self.assertTrue(self.res not in self.empty_series)
        self.assertTrue(self.res in self.single_element)
        self.assertTrue(self.cap not in self.single_element)
        self.assertTrue(self.res in self.multiple_elements)
        self.assertTrue(self.cap in self.multiple_elements)
        self.assertTrue(Resistor() not in self.empty_series)
        self.assertTrue(Resistor() not in self.single_element)
        self.assertTrue(Resistor() not in self.multiple_elements)
        self.assertTrue(not self.empty_series.contains(self.res))
        self.assertTrue(self.single_element.contains(self.res))
        self.assertTrue(not self.single_element.contains(self.cap))
        self.assertTrue(self.multiple_elements.contains(self.res))
        self.assertTrue(self.multiple_elements.contains(self.cap))
        self.assertTrue(not self.empty_series.contains(Resistor()))

    def test_len(self):
        self.assertEqual(len(self.empty_series), 0)
        self.assertEqual(len(self.single_element), 1)
        self.assertEqual(len(self.multiple_elements), 2)

    def test_repr(self):
        self.assertEqual(
            repr(self.multiple_elements),
            f"Series ({hex(id(self.multiple_elements))})",
        )

    def test_sympy(self):
        self.assertEqual(str(self.empty_series.to_sympy()), "0")
        self.assertEqual(
            str(self.single_element.to_sympy()),
            str(self.res.to_sympy(identifier=1)),
        )
        self.assertEqual(
            str(self.multiple_elements.to_sympy()),
            str(self.res.to_sympy(identifier=1) + self.cap.to_sympy(identifier=1)),
        )

    def test_empty_impedance(self):
        self.assertTrue(
            allclose(
                self.empty_series.get_impedances(self.f),
                zeros(self.f.shape, dtype=ComplexImpedance),
            )
        )

    def test_single_element_impedance(self):
        self.assertTrue(
            allclose(
                self.single_element.get_impedances(self.f),
                self.res.get_impedances(self.f),
            )
        )

    def test_multiple_element_impedance(self):
        self.assertTrue(
            allclose(
                self.multiple_elements.get_impedances(self.f),
                self.res.get_impedances(self.f) + self.cap.get_impedances(self.f),
            )
        )
        self.assertTrue(
            allclose(
                self.container.get_impedances(self.f),
                self.res.get_impedances(self.f) + self.tlm.get_impedances(self.f),
            )
        )

    def test_shorted_impedance(self):
        self.assertTrue(
            allclose(
                self.shorted_series.get_impedances(self.f),
                self.res.get_impedances(self.f),
            )
        )

    def test_open_impedance(self):
        with self.assertRaises(InfiniteImpedance):
            self.open_series.get_impedances(self.f)

    def test_get_elements_recursive(self):
        res: Resistor = Resistor()
        cap: Capacitor = Capacitor()
        tlm: TransmissionLineModel = TransmissionLineModel()
        ser: Series = Series([res, cap, tlm])
        elements: List[Element] = ser._get_elements_recursive()

        self.assertTrue(res in elements)
        self.assertTrue(cap in elements)
        self.assertTrue(tlm in elements)
        self.assertTrue(len(elements) > 3)


class TestParallel(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.f: Frequencies = array([1e-2, 1e0, 1e2], dtype=Frequency)
        cls.res: Resistor = Resistor()
        cls.cap: Capacitor = Capacitor()
        cls.tlm: TransmissionLineModel = TransmissionLineModel()
        cls.empty_parallel: Parallel = Parallel([])
        cls.single_element: Parallel = Parallel([cls.res])
        cls.multiple_elements: Parallel = Parallel([cls.res, cls.cap])
        cls.container: Parallel = Parallel([cls.res, cls.tlm])
        cls.shorted_parallel: Parallel = Parallel([cls.res, Resistor(R=0)])
        cls.partially_open_parallel: Parallel = Parallel([cls.res, Resistor(R=inf)])
        cls.open_parallel: Parallel = Parallel([Resistor(R=inf), Resistor(R=inf)])

    def test_contains(self):
        self.assertTrue(self.res not in self.empty_parallel)
        self.assertTrue(self.res in self.single_element)
        self.assertTrue(self.cap not in self.single_element)
        self.assertTrue(self.res in self.multiple_elements)
        self.assertTrue(self.cap in self.multiple_elements)
        self.assertTrue(Resistor() not in self.empty_parallel)
        self.assertTrue(Resistor() not in self.single_element)
        self.assertTrue(Resistor() not in self.multiple_elements)
        self.assertTrue(not self.empty_parallel.contains(self.res))
        self.assertTrue(self.single_element.contains(self.res))
        self.assertTrue(not self.single_element.contains(self.cap))
        self.assertTrue(self.multiple_elements.contains(self.res))
        self.assertTrue(self.multiple_elements.contains(self.cap))
        self.assertTrue(not self.empty_parallel.contains(Resistor()))

    def test_len(self):
        self.assertEqual(len(self.empty_parallel), 0)
        self.assertEqual(len(self.single_element), 1)
        self.assertEqual(len(self.multiple_elements), 2)

    def test_repr(self):
        self.assertEqual(
            repr(self.multiple_elements),
            f"Parallel ({hex(id(self.multiple_elements))})",
        )

    def test_sympy(self):
        self.assertEqual(str(self.empty_parallel.to_sympy()), "0")
        self.assertEqual(
            str(self.single_element.to_sympy()),
            str(self.res.to_sympy(identifier=1)),
        )
        self.assertEqual(
            str(self.multiple_elements.to_sympy()),
            str(
                1
                / (
                    1 / self.res.to_sympy(identifier=1)
                    + 1 / self.cap.to_sympy(identifier=1)
                )
            ),
        )

    def test_empty_impedance(self):
        self.assertTrue(
            allclose(
                self.empty_parallel.get_impedances(self.f),
                zeros(self.f.shape, dtype=ComplexImpedance),
            )
        )

    def test_single_element_impedance(self):
        self.assertTrue(
            allclose(
                self.single_element.get_impedances(self.f),
                self.res.get_impedances(self.f),
            )
        )

    def test_multiple_element_impedance(self):
        self.assertTrue(
            allclose(
                self.multiple_elements.get_impedances(self.f),
                1
                / (
                    1 / self.res.get_impedances(self.f)
                    + 1 / self.cap.get_impedances(self.f)
                ),
            )
        )
        self.assertTrue(
            allclose(
                self.container.get_impedances(self.f),
                1
                / (
                    1 / self.res.get_impedances(self.f)
                    + 1 / self.tlm.get_impedances(self.f)
                ),
            )
        )

    def test_shorted_impedance(self):
        self.assertTrue(
            allclose(
                self.shorted_parallel.get_impedances(self.f),
                zeros(self.f.shape, dtype=ComplexImpedance),
            )
        )

    def test_partially_open_impedance(self):
        self.assertTrue(
            allclose(
                self.partially_open_parallel.get_impedances(self.f),
                self.res.get_impedances(self.f),
            )
        )

    def test_fully_open_impedance(self):
        with self.assertRaises(InfiniteImpedance):
            self.open_parallel.get_impedances(self.f)

    def test_get_elements_recursive(self):
        res: Resistor = Resistor()
        cap: Capacitor = Capacitor()
        tlm: TransmissionLineModel = TransmissionLineModel()
        par: Parallel = Parallel([res, cap, tlm])
        elements: List[Element] = par._get_elements_recursive()

        self.assertTrue(res in elements)
        self.assertTrue(cap in elements)
        self.assertTrue(tlm in elements)
        self.assertTrue(len(elements) > 3)

    def test_to_latex(self):
        latex: str = Parallel([Resistor(), Capacitor()]).to_latex()
        self.assertIsInstance(latex, str)
        self.assertEqual(latex, r"Z = \frac{1}{2 j \pi C_{1} f + \frac{1}{R_{1}}}")

    def test_to_drawing(self):
        parallel: Parallel = Parallel([Resistor(), Capacitor()])
        drawing: Drawing = parallel.to_drawing()
        self.assertIsInstance(drawing, Drawing)

    def test_to_circuitikz(self):
        parallel: Parallel = Parallel([Resistor(), Capacitor()])
        circuitikz: str = parallel.to_circuitikz()
        self.assertIsInstance(circuitikz, str)
