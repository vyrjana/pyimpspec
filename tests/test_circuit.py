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

from contextlib import (
    redirect_stderr,
    redirect_stdout,
)
from io import StringIO
from re import finditer
from unittest import TestCase
from typing import (
    Callable,
    Dict,
    List,
    Match,
    Tuple,
    Type,
    Union,
)
from numpy import (
    allclose,
    array,
    logspace,
)
from pyimpspec.circuit.circuit import VERSION
from pyimpspec import (
    Capacitor,
    Circuit,
    CircuitBuilder,
    Connection,
    ConstantPhaseElement,
    DataSet,
    Element,
    Parallel,
    Resistor,
    Series,
    Warburg,
    get_elements,
    parse_cdc,
    simulate_spectrum,
)
from pyimpspec.exceptions import (
    ParsingError,
    ConnectionWithoutElements,
    DuplicateParameterDefinition,
    ExpectedNumericValue,
    ExpectedParameterIdentifier,
    InsufficientElementsInParallelConnection,
    InsufficientTokens,
    InvalidElementSymbol,
    InvalidNumericValue,
    InvalidParameterDefinition,
    InvalidParameterLowerLimit,
    InvalidParameterUpperLimit,
    TooManyParameterDefinitions,
    UnexpectedCharacter,
    UnexpectedIdentifier,
    UnexpectedToken,
)
from pyimpspec.circuit.parser import Parser
from pyimpspec.circuit.tokenizer import (
    Colon,
    Comma,
    Equals,
    Exclamation,
    FixedNumber,
    ForwardSlash,
    Identifier,
    LBracket,
    LCurly,
    LParen,
    Label,
    Number,
    Percent,
    RBracket,
    RCurly,
    RParen,
    Token,
    Tokenizer,
)
from schemdraw import Drawing
from sympy import Expr
from pyimpspec.typing import (
    ComplexImpedance,
    ComplexImpedances,
    Frequencies,
    Frequency,
)
from test_matplotlib import (
    check_mpl_return_values,
    mpl,
)


def redirect_output(func: Callable, stderr: bool = False) -> List[str]:
    buffer: StringIO = StringIO()
    if stderr:
        with redirect_stderr(buffer):
            func()
    else:
        with redirect_stdout(buffer):
            func()

    lines: List[str] = buffer.getvalue().split("\n")

    return list(map(str.strip, lines))


class TestCircuitBuilder(TestCase):
    def test_series(self):
        with CircuitBuilder() as builder:
            builder.add(Resistor())
            builder.add(Capacitor())

        self.assertEqual(builder.to_string(), "[RC]")

        with CircuitBuilder() as builder:
            builder += Resistor()
            builder += Capacitor()

        self.assertEqual(builder.to_string(), "[RC]")

        with self.assertRaises(ValueError):
            with CircuitBuilder() as builder:
                pass

    def test_parallel(self):
        with CircuitBuilder(parallel=True) as builder:
            builder.add(Resistor())
            builder.add(Capacitor())

        self.assertEqual(builder.to_string(), "[(RC)]")

        with CircuitBuilder(parallel=True) as builder:
            builder += Resistor()
            builder += Capacitor()

        self.assertEqual(builder.to_string(), "[(RC)]")

        with self.assertRaises(ValueError):
            with CircuitBuilder(parallel=True) as builder:
                builder.add(Resistor())

        with self.assertRaises(ValueError):
            with CircuitBuilder(parallel=True) as builder:
                builder += Resistor()

        with self.assertRaises(ValueError):
            with CircuitBuilder(parallel=True) as builder:
                pass

    def test_nested_connections(self):
        with CircuitBuilder() as builder:
            builder.add(Resistor())
            with builder.parallel() as parallel:
                parallel.add(Capacitor())
                parallel.add(Resistor())

        self.assertEqual(str(builder), "[R(CR)]")
        self.assertEqual(builder.to_string(), "[R(CR)]")

        with CircuitBuilder() as builder:
            builder += Resistor()
            with builder.parallel() as parallel:
                parallel += Capacitor()
                parallel += Resistor()

        self.assertEqual(builder.to_string(), "[R(CR)]")

        with CircuitBuilder() as builder:
            with builder.parallel() as parallel:
                parallel.add(Capacitor())
                parallel.add(Resistor())
            builder.add(Resistor())

        self.assertEqual(builder.to_string(), "[(CR)R]")

        with CircuitBuilder() as builder:
            with builder.parallel() as parallel:
                parallel += Capacitor()
                parallel += Resistor()
            builder += Resistor()

        self.assertEqual(builder.to_string(), "[(CR)R]")

        with CircuitBuilder() as builder:
            with builder.parallel() as parallel:
                with parallel.series() as series:
                    series.add(Resistor())
                    series.add(Capacitor())
                parallel.add(Resistor())

        self.assertEqual(builder.to_string(), "[([RC]R)]")

        with CircuitBuilder() as builder:
            with builder.parallel() as parallel:
                with parallel.series() as series:
                    series += Resistor()
                    series += Capacitor()
                parallel += Resistor()

        self.assertEqual(builder.to_string(), "[([RC]R)]")

    def test_parameters_and_labels(self):
        cdc: str = "[R{R=8.3E+01/2.0E+01/9.6E+01:test}C{C=4.0E-03F/1.0E-24/1.0E+03}]"
        with CircuitBuilder() as builder:
            R: Resistor = Resistor(R=83)
            R.set_lower_limits("R", 20)
            R.set_lower_limits(R=20)
            R.set_upper_limits("R", 96)
            R.set_upper_limits(R=96)
            R.set_label("test")
            builder.add(R)

            C: Capacitor = Capacitor(C=4e-3)
            C.set_fixed("C", True)
            C.set_fixed(C=True)
            builder.add(C)

        self.assertEqual(builder.to_string(1), cdc)

        with CircuitBuilder() as builder:
            builder += (
                Resistor(R=83)
                .set_lower_limits(R=20)
                .set_lower_limits("R", 20)
                .set_upper_limits(R=96)
                .set_upper_limits("R", 96)
                .set_label("test")
            )
            builder += Capacitor(C=4e-3).set_fixed(C=True).set_fixed("C", True)

        self.assertEqual(builder.to_string(1), cdc)


class TestTokenizer(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer: Tokenizer = Tokenizer()

    def check_token(self, Class: Type[Token], value: str):
        token: Token = self.tokenizer.process(value)[0]
        self.assertIsInstance(token, Class)
        self.assertEqual(token.value, value)

    def test_identifier(self):
        tokens: List[Token] = self.tokenizer.process("RCWoLTlmns")

        self.assertEqual(len(tokens), 5)
        self.assertTrue(all(map(lambda _: isinstance(_, Identifier), tokens)))

        self.assertEqual(tokens[0].value, "R")
        self.assertEqual(tokens[1].value, "C")
        self.assertEqual(tokens[2].value, "Wo")
        self.assertEqual(tokens[3].value, "L")
        self.assertEqual(tokens[4].value, "Tlmns")

    def test_label(self):
        tokens: List[Token] = self.tokenizer.process("R{:test}")
        self.assertEqual(len(tokens), 5)

        self.assertIsInstance(tokens[0], Identifier)
        self.assertEqual(tokens[0].value, "R")

        self.assertIsInstance(tokens[1], LCurly)
        self.assertIsInstance(tokens[2], Colon)

        self.assertIsInstance(tokens[3], Label)
        self.assertEqual(tokens[3].value, "test")

        self.assertIsInstance(tokens[4], RCurly)

    def test_numbers(self):
        tokens: List[Token] = self.tokenizer.process(
            "6,3.14159265,6.28e-3,62.8E+10,42f,36.5f,12F"
        )
        self.assertEqual(len(tokens), 13)

        self.assertIsInstance(tokens[0], Number)
        self.assertEqual(tokens[0].value, 6.0)

        self.assertIsInstance(tokens[1], Comma)

        self.assertIsInstance(tokens[2], Number)
        self.assertEqual(tokens[2].value, 3.14159265)

        self.assertIsInstance(tokens[3], Comma)

        self.assertIsInstance(tokens[4], Number)
        self.assertEqual(tokens[4].value, 0.00628)

        self.assertIsInstance(tokens[5], Comma)

        self.assertIsInstance(tokens[6], Number)
        self.assertEqual(tokens[6].value, 6.28e11)

        self.assertIsInstance(tokens[7], Comma)

        self.assertIsInstance(tokens[8], FixedNumber)
        self.assertEqual(tokens[8].value, 42.0)

        self.assertIsInstance(tokens[9], Comma)

        self.assertIsInstance(tokens[10], FixedNumber)
        self.assertEqual(tokens[10].value, 36.5)

        self.assertIsInstance(tokens[11], Comma)

        self.assertIsInstance(tokens[12], FixedNumber)
        self.assertEqual(tokens[12].value, 12.0)

    def test_brackets(self):
        self.check_token(LBracket, "[")
        self.check_token(RBracket, "]")

    def test_parentheses(self):
        self.check_token(LParen, "(")
        self.check_token(RParen, ")")

    def test_curly_braces(self):
        self.check_token(LCurly, "{")
        self.check_token(RCurly, "}")

    def test_equals(self):
        self.check_token(Equals, "=")

    def test_forward_slash(self):
        self.check_token(ForwardSlash, "/")

    def test_percent(self):
        self.check_token(Percent, "%")

    def test_comma(self):
        self.check_token(Comma, ",")

    def test_exclamation(self):
        self.check_token(Exclamation, "!")


class TestParser(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.valid_cdcs: List[str] = [
            "R",
            "RL",
            "(RL)",
            "([RL]C)",
            "(R[LC])",
            "(R[LC]W)",
            "(W[(RL)C])Q",
            "RLC",
            "RLCQ",
            "RLCQW",
            "(RLC)",
            "(RLCQ)",
            "(RLCQW)",
            "R(LCQW)",
            "RL(CQW)",
            "RLC(QW)",
            "(RLCQ)W",
            "(RLC)QW",
            "(RL)CQW",
            "R(LCQ)W",
            "R(LC)QW",
            "RL(CQ)W",
            "(R[WQ])",
            "(R[WQ]C)",
            "(R[W(LC)Q])",
            "([LC][RRQ])",
            "(R[WQ])([LC][RRQ])",
            "([RL][CW])",
            "R(RW)",
            "R(RW)C",
            "R(RWL)C",
            "R(RWL)(LQ)C",
            "R(RWL)C(LQ)",
            "R(LQ)C(RWL)",
            "R([RW]Q)C",
            "R(RW)(CQ)",
            "R([RW]Q[LRC])(CQ)",
            "R([RW][L(RQ)C]Q[LRC])(CQ)",
            "R([RW][L(WC)(RQ)C]Q[LRC])(CQ)",
            "(R[LCQW])",
            "(RL[CQW])",
            "(RLC[QW])",
            "(R[LCQ]W)",
            "(R[LC]QW)",
            "(RL[CQ]W)",
            "R(LC)(QW)",
            "(RL)C(QW)",
            "(RL)(CQ)W",
            "(RL)(CQW)",
            "(RLC)(QW)",
            "(R[LC])QW",
            "([RL]C)QW",
            "([RL]CQ)W",
            "([RL]CQW)",
            "([RLC]QW)",
            "([RLCQ]W)",
            "(R[(LC)QW])",
            "(R[L(CQ)W])",
            "(R[LC(QW)])",
            "(R[L(CQW)])",
            "(R[(LCQ)W])",
            "(R[(LC)Q]W)",
            "(R[L(CQ)]W)",
            "(RQ)RWL",
            "RWL(RQ)",
            "(R[QR])(LC)RW",
            "RW(LC)(RQ)",
            "RL(QW)L(RR)(RR)L(RR)C",
            "RL(QW)(L[(RR)(RR)L(RR)C])",
            "RL(QW)(L[(RR)(RR)L(RR)])",
            "Tlm{L=2.5, Z_A=open, Z_B=short}",
        ]
        cls.invalid_cdcs: List[Tuple[str, ParsingError]] = [
            (
                "R[]",
                ConnectionWithoutElements,
            ),
            (
                "R()",
                ConnectionWithoutElements,
            ),
            (
                "Q{n=0.5, n=0.9}",
                DuplicateParameterDefinition,
            ),
            (
                "[R(RL{L=",
                ExpectedNumericValue,
            ),
            (
                "R{R=,}",
                ExpectedNumericValue,
            ),
            (
                "R{=5}",
                ExpectedParameterIdentifier,
            ),
            (
                "R(R)",
                InsufficientElementsInParallelConnection,
            ),
            (
                "R{R",
                InsufficientTokens,
            ),
            (
                "bA",
                InvalidElementSymbol,
            ),
            (
                "Vtpas",
                InvalidElementSymbol,
            ),
            (
                "R{Pqt=2}",
                InvalidParameterDefinition,
            ),
            (
                "R{R=3/4}",
                InvalidParameterLowerLimit,
            ),
            (
                "R{R=3//2}",
                InvalidParameterUpperLimit,
            ),
            (
                "R{R=5, n=0.5}",
                TooManyParameterDefinitions,
            ),
            (
                "R{R}",
                UnexpectedToken,
            ),
            (
                "Tlm{L=inf}",
                InvalidParameterDefinition,
            ),
            (
                "Tlm{X_1=2.0}",
                InvalidParameterDefinition,
            ),
        ]

    def test_valid_cdcs(self):
        parser: Parser = Parser()
        freq: Frequencies = array([1e-5, 1, 1e5], dtype=Frequency)

        cdc: str
        for cdc in self.valid_cdcs:
            circuit: Circuit = parser.process(cdc)
            Z_regular: ComplexImpedances = circuit.get_impedances(freq)
            expr: Expr = circuit.to_sympy(substitute=True)
            Z_sympy: ComplexImpedances = array(
                list(map(lambda _: complex(expr.subs("f", _)), freq)),
                dtype=ComplexImpedance,
            )
            self.assertTrue(allclose(Z_regular.real, Z_sympy.real))
            self.assertTrue(allclose(Z_regular.imag, Z_sympy.imag))

        for cdc in self.valid_cdcs:
            parse_cdc(cdc)

        parse_cdc("")
        parse_cdc("[]")
        parse_cdc("R{R=50/50%/150%}")

    def test_invalid_cdcs(self):
        parser: Parser = Parser()
        cdc: str
        error: ParsingError
        for cdc, error in self.invalid_cdcs:
            with self.assertRaises(error, msg=cdc):
                parser.process(cdc)

        for cdc, error in self.invalid_cdcs:
            with self.assertRaises(error, msg=cdc):
                parse_cdc(cdc)

        with self.assertRaises(InsufficientTokens):
            parse_cdc("[")

        with self.assertRaises(UnexpectedCharacter):
            parse_cdc("R{:\\}")

        with self.assertRaises(UnexpectedToken):
            parse_cdc("R{:]")

        with self.assertRaises(UnexpectedToken):
            parse_cdc(":")

        with self.assertRaises(ExpectedNumericValue):
            parse_cdc("R{R=")

        with self.assertRaises(ExpectedParameterIdentifier):
            parse_cdc("R{")

        with self.assertRaises(InsufficientTokens):
            parse_cdc("(R")

    def test_nested_connections(self):
        CDCs: List[Tuple[str, str]] = [
            (
                "(R(LC))",
                "[(RLC)]",
            ),
            (
                "(R([LQ]C))",
                "[(R[LQ]C)]",
            ),
            (
                "[R[L][CQ]]",
                "[RLCQ]",
            ),
        ]
        parser: Parser = Parser()

        cdc_input: str
        cdc_output: str
        for cdc_input, cdc_output in CDCs:
            self.assertEqual(parser.process(cdc_input).to_string(), cdc_output)


class TestCircuits(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.elements: Dict[str, Element] = {k: v() for k, v in get_elements().items()}
        cls.element_circuits: Dict[str, Circuit] = {
            k: parse_cdc(k) for k in cls.elements
        }

    def test_parse_element_symbol(self):
        symbol: str
        circuit: Circuit
        for symbol, circuit in self.element_circuits.items():
            self.assertEqual(circuit.to_string(), f"[{symbol}]")

    def test_default_limits(self):
        symbol: str
        circuit: Circuit
        for symbol, circuit in self.element_circuits.items():
            element: Element = circuit.get_elements()[0]
            default_lower_limits: Dict[str, float] = element.get_default_lower_limits()
            default_upper_limits: Dict[str, float] = element.get_default_upper_limits()

            key: str
            value: float
            for key, value in element.get_lower_limits().items():
                self.assertTrue(allclose(value, default_lower_limits[key]))

            for key, value in element.get_upper_limits().items():
                self.assertTrue(allclose(value, default_upper_limits[key]))

    def test_impedance(self):
        symbol: str
        circuit: Circuit
        for symbol, circuit in self.element_circuits.items():
            element: Element = self.elements[symbol]
            self.assertTrue(
                allclose(
                    circuit.get_impedances(array([1.0])),
                    element.get_impedances(array([1.0])),
                )
            )

    def test_repr(self):
        circuit: Circuit
        for circuit in self.element_circuits.values():
            self.assertEqual(
                repr(circuit),
                f"Circuit ('{circuit.to_string()}', {hex(id(circuit))})",
            )

    def test_to_string(self):
        symbol: str
        circuit: Circuit
        for symbol, circuit in self.element_circuits.items():
            self.assertEqual(circuit.to_string(), f"[{symbol}]")

    def test_get_only_elements(self):
        symbol: str
        circuit: Circuit
        for symbol, circuit in self.element_circuits.items():
            element: Element = self.elements[symbol]
            elements: List[Element] = circuit.get_elements(recursive=True)
            self.assertEqual(len(elements), 1)
            element: Element = elements[0]
            self.assertIsInstance(elements[0], type(element))

    def test_define_parameter_value(self):
        value: float = 1.56e-3
        symbol: str
        element: Element
        for symbol, element in self.elements.items():
            key: str
            for key in element.get_default_values().keys():
                element = parse_cdc(f"{symbol}{{{key}={value}}}").get_elements(
                    recursive=True
                )[0]
                self.assertEqual(element.get_value(key), value)

    def test_define_label(self):
        symbol: str
        element: Element
        for symbol, element in self.elements.items():
            element = parse_cdc(f"{symbol}{{:test}}").get_elements(recursive=True)[0]
            self.assertEqual(element.get_label(), "test")
            self.assertEqual(element.get_name(), f"{symbol}_test")

    def test_element_names(self):
        circuit: Circuit = parse_cdc("R{:foo}(CR)")

        R_foo: Resistor
        C_1: Capacitor
        R_2: Resistor
        R_foo, C_1, R_2 = circuit.get_elements()
        self.assertEqual(R_foo.get_label(), "foo")
        self.assertEqual(C_1.get_label(), "")
        self.assertEqual(R_2.get_label(), "")

        self.assertEqual(circuit.get_element_name(R_foo), "R_foo")
        self.assertEqual(circuit.get_element_name(C_1), "C_1")
        self.assertEqual(circuit.get_element_name(R_2), "R_2")

    def test_implicit_series(self):
        circuit: Circuit = parse_cdc("R{R=250}R{R=500}")
        self.assertEqual(circuit.to_string(), "[RR]")
        self.assertTrue(
            allclose(
                circuit.get_impedances(array([1])),
                array([complex(250 + 500, 0)]),
            )
        )

    def test_explicit_series(self):
        circuit: Circuit = parse_cdc("[R{R=250}R{R=500}]")
        self.assertEqual(circuit.to_string(), "[RR]")
        self.assertTrue(
            allclose(
                circuit.get_impedances(array([1])),
                array([complex(250 + 500, 0)]),
            )
        )

    def test_parallel(self):
        circuit: Circuit = parse_cdc("(R{R=250}R{R=500})")
        self.assertEqual(circuit.to_string(), "[(RR)]")
        self.assertTrue(
            allclose(
                circuit.get_impedances(array([1])),
                array([complex(1 / (1 / 250 + 1 / 500), 0)]),
            )
        )

    def test_get_elements(self):
        circuit: Circuit = parse_cdc("(RCQ)")
        elements: List[Element] = circuit.get_elements(recursive=False)
        self.assertIsInstance(elements, list)
        self.assertEqual(len(elements), 0)

        elements = circuit.get_elements()
        self.assertIsInstance(elements, list)
        self.assertEqual(len(elements), 3)
        self.assertTrue(all(map(lambda item: isinstance(item, Element), elements)))
        self.assertIsInstance(elements[0], Resistor)
        self.assertIsInstance(elements[1], Capacitor)
        self.assertIsInstance(elements[2], ConstantPhaseElement)

        circuit = parse_cdc("R(RC)(RW)")
        elements = circuit.get_elements(recursive=False)
        self.assertIsInstance(elements, list)
        self.assertEqual(len(elements), 1)
        self.assertIsInstance(elements[0], Element)

        elements = circuit.get_elements()
        self.assertIsInstance(elements, list)
        self.assertEqual(len(elements), 5)
        self.assertTrue(all(map(lambda item: isinstance(item, Element), elements)))
        self.assertIsInstance(elements[0], Resistor)
        self.assertIsInstance(elements[1], Resistor)
        self.assertIsInstance(elements[2], Capacitor)
        self.assertIsInstance(elements[3], Resistor)
        self.assertIsInstance(elements[4], Warburg)

        self.assertEqual(circuit.get_elements(), circuit.get_elements(recursive=True))

    def test_get_connections(self):
        circuit: Circuit = parse_cdc("(RCQ)")
        connections: List[Connection] = circuit.get_connections(recursive=False)
        self.assertEqual(len(connections), 1)
        self.assertIsInstance(connections[0], Series)

        circuit = parse_cdc("R(RC)(RW)")
        connections = circuit.get_connections(recursive=False)
        self.assertEqual(len(connections), 1)
        self.assertIsInstance(connections[0], Series)

        connections = circuit.get_connections()
        self.assertEqual(len(connections), 3)
        self.assertTrue(
            all(map(lambda item: isinstance(item, Connection), connections))
        )
        self.assertIsInstance(connections[0], Series)
        self.assertIsInstance(connections[1], Parallel)
        self.assertIsInstance(connections[2], Parallel)

    def test_to_stack(self):
        circuit: Circuit = parse_cdc("(RCQ)")
        stack: List[Tuple[str, Union[Element, Connection]]] = circuit.to_stack()

        char: str
        elem: Union[Element, Connection]
        char, elem = stack.pop(0)
        self.assertEqual(char, "[")
        self.assertIsInstance(elem, Series)

        char, elem = stack.pop(0)
        self.assertEqual(char, "(")
        self.assertIsInstance(elem, Parallel)

        char, elem = stack.pop(0)
        self.assertEqual(char, "R")
        self.assertIsInstance(elem, Resistor)

        char, elem = stack.pop(0)
        self.assertEqual(char, "C")
        self.assertIsInstance(elem, Capacitor)

        char, elem = stack.pop(0)
        self.assertEqual(char, "Q")
        self.assertIsInstance(elem, ConstantPhaseElement)

        char, elem = stack.pop(0)
        self.assertEqual(char, ")")
        self.assertIsInstance(elem, Parallel)

        char, elem = stack.pop(0)
        self.assertEqual(char, "]")
        self.assertIsInstance(elem, Series)

    def test_to_latex(self):
        circuit: Circuit = parse_cdc("RC")
        self.assertEqual(
            circuit.to_latex(),
            r"Z = R_{0} - \frac{i}{2 \pi C_{1} f}",
        )

        circuit = parse_cdc("(RC)")
        self.assertEqual(
            circuit.to_latex(),
            r"Z = \frac{1}{2 i \pi C_{1} f + \frac{1}{R_{0}}}",
        )

    def test_to_drawing(self):
        circuit: Circuit = parse_cdc("RC")

        drawing: Drawing = circuit.to_drawing()
        self.assertIsInstance(drawing, Drawing)

        parse_cdc("(RC)L(QWLa)").to_drawing()
        parse_cdc("R([RW]C)").to_drawing()

    def test_to_circuitikz(self):
        self.assertIsInstance(parse_cdc("RC").to_circuitikz(), str)
        self.assertIsInstance(parse_cdc("(RC)").to_circuitikz(), str)
        self.assertIsInstance(parse_cdc("R([RW]C)").to_circuitikz(), str)
        self.assertEqual(
            type(parse_cdc("R([RW]C)").to_circuitikz(hide_labels=True)), str
        )

    def test_serialize_deserialize(self):
        cdc: str = "[R{R=2.5E+02/0.0E+00/inf}([R{R=5.0E+02/0.0E+00/inf}W{Y=1.0E-04/1.0E-24/inf,n=5.0E-01F/0.0E+00/1.0E+00}]C{C=1.0E-06/1.0E-24/1.0E+03})]"

        circuit: Circuit = parse_cdc(cdc)
        self.assertEqual(cdc, circuit.to_string(decimals=1))

        serialized_cdc: str = (
            f"!V={VERSION}!"
            + "[R{R=2.500000000000E+02/0.000000000000E+00/inf}([R{R=5.000000000000E+02/0.000000000000E+00/inf}W{Y=1.000000000000E-04/1.000000000000E-24/inf,n=5.000000000000E-01F/0.000000000000E+00/1.000000000000E+00}]C{C=1.000000000000E-06/1.000000000000E-24/1.000000000000E+03})]"
        )
        self.assertEqual(
            serialized_cdc,
            circuit.serialize(),
            msg=f"{serialized_cdc=}, {circuit.serialize()=}",
        )

        deserialized_circuit: Circuit = parse_cdc(serialized_cdc)
        self.assertEqual(cdc, deserialized_circuit.to_string(decimals=1))
        self.assertEqual(serialized_cdc, deserialized_circuit.serialize())

        with self.assertRaises(UnexpectedIdentifier):
            parse_cdc("!A=1!R")

        with self.assertRaises(InvalidNumericValue):
            parse_cdc("!V=0!R")

        with self.assertRaises(InvalidNumericValue):
            parse_cdc(f"!V={VERSION+1}!R")

        self.assertEqual(
            VERSION,
            1,  # Increment this value after incrementing VERSION
            msg=f"Implement CDC migrator and tests for V{VERSION-1}",
        )

    def test_simulate_spectrum(self):
        circuit: Circuit = parse_cdc("R([RW]C)")

        data: DataSet = simulate_spectrum(circuit, label="test")
        self.assertEqual(data.get_label(), "test")
        self.assertEqual(data.get_num_points(), 71)

        data = simulate_spectrum(circuit, frequencies=[1.0, 2.0])
        self.assertEqual(data.get_num_points(), 2)

    def test_matplotlib(self):
        circuit: Circuit = parse_cdc("R(RC)")
        f: Frequencies = logspace(4, 0, num=21)
        check_mpl_return_values(self, *mpl.plot_circuit(circuit, frequencies=f))
        check_mpl_return_values(
            self,
            *mpl.plot_circuit(
                circuit,
                frequencies=f,
                colored_axes=True,
            ),
        )


class TestDiagrams(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.circuit: Circuit = parse_cdc("R(RC)")
        elements = cls.circuit.get_elements()
        cls.custom_labels: Dict[Element, str] = {
            elements[0]: "Foo",
            elements[1]: "Bar",
            elements[2]: "Baz",
        }

    def schemdraw(self, **kwargs) -> str:
        svg: str = self.circuit.to_drawing(**kwargs).get_imagedata(fmt="svg").decode()
        self.assertIsInstance(svg, str)
        self.assertNotEqual(svg, "")

        return svg

    def circuitikz(self, **kwargs) -> str:
        tex: str = self.circuit.to_circuitikz(**kwargs)
        self.assertIsInstance(tex, str)
        self.assertNotEqual(tex, "")

        return tex

    def test_default(self):
        svg: str = self.schemdraw()
        self.assertRegex(svg, r"<!-- \$R_{\\rm 1}\$ -->")
        self.assertRegex(svg, r"<!-- \$R_{\\rm 2}\$ -->")
        self.assertRegex(svg, r"<!-- \$C_{\\rm 1}\$ -->")

        tex: str = self.circuitikz()
        self.assertRegex(tex, r"to\[R=\$R_{\\rm 1}\$\]")
        self.assertRegex(tex, r"to\[R=\$R_{\\rm 2}\$\]")
        self.assertRegex(tex, r"to\[capacitor=\$C_{\\rm 1}\$\]")

    def test_custom_labels(self):
        kwargs = {"custom_labels": self.custom_labels}

        svg: str = self.schemdraw(**kwargs)
        self.assertRegex(svg, r"<!-- Foo -->")
        self.assertRegex(svg, r"<!-- Bar -->")
        self.assertRegex(svg, r"<!-- Baz -->")

        tex: str = self.circuitikz(**kwargs)
        self.assertRegex(tex, r"to\[R=\$Foo\$\]")
        self.assertRegex(tex, r"to\[R=\$Bar\$\]")
        self.assertRegex(tex, r"to\[capacitor=\$Baz\$\]")

    def test_hide_labels(self):
        kwargs = {"hide_labels": True}

        svg: str = self.schemdraw(**kwargs)

        self.assertNotRegex(svg, r"<!-- \$R_{\\rm 1}\$ -->")
        self.assertNotRegex(svg, r"<!-- \$R_{\\rm 2}\$ -->")
        self.assertNotRegex(svg, r"<!-- \$C_{\\rm 1}\$ -->")

        tex: str = self.circuitikz(**kwargs)
        self.assertNotRegex(tex, r"to\[R=\$R_{\\rm 1}\$\]")
        self.assertNotRegex(tex, r"to\[R=\$R_{\\rm 2}\$\]")
        self.assertNotRegex(tex, r"to\[capacitor=\$C_{\\rm 1}\$\]")

    def test_left_terminal_label(self):
        kwargs = {"left_terminal_label": "Foo"}

        svg: str = self.schemdraw(**kwargs)
        self.assertRegex(svg, r"<!-- Foo -->")

        tex: str = self.circuitikz(**kwargs)
        self.assertRegex(tex, r"node\[above\]{Foo} to\[short, o-\]")

    def test_node_height(self):
        def parse_svg_coordinates(svg: str) -> List[Tuple[float, float]]:
            coordinates: List[Tuple[float, float]] = []
            match: Match
            for match in finditer(r"[LM] (?P<x>\d+.\d+) (?P<y>\d+.\d+)", svg):
                coordinates.append((float(match.group("x")), float(match.group("y"))))
            return coordinates

        kwargs = {"node_height": 1.28}

        default: List[Tuple[float, float]] = parse_svg_coordinates(self.schemdraw())
        altered: List[Tuple[float, float]] = parse_svg_coordinates(
            self.schemdraw(**kwargs)
        )
        self.assertEqual(len(default), len(altered))

        for old, new in zip(default, altered):
            self.assertAlmostEqual(old[0], new[0])
            self.assertGreaterEqual(old[1], new[1])

        tex: str = self.circuitikz(**kwargs)
        self.assertRegex(tex, r"\\draw \(3.0,-1.28\)")

    def test_node_width(self):
        kwargs = {"node_width": 2.56}

        tex: str = self.circuitikz(**kwargs)
        self.assertRegex(tex, r"\\draw \(2.56,-0.0\)")

    def test_right_terminal_label(self):
        kwargs = {"right_terminal_label": "Bar"}

        svg: str = self.schemdraw(**kwargs)
        self.assertRegex(svg, r"<!-- Bar -->")

        tex: str = self.circuitikz(**kwargs)
        self.assertRegex(tex, r"\) node\[above\]{Bar};")

    def test_running(self):
        kwargs = {"running": True}

        svg: str = self.schemdraw(**kwargs)
        self.assertRegex(svg, r"<!-- \$R_{\\rm 0}\$ -->")
        self.assertRegex(svg, r"<!-- \$R_{\\rm 1}\$ -->")
        self.assertRegex(svg, r"<!-- \$C_{\\rm 2}\$ -->")

        tex: str = self.circuitikz(**kwargs)
        self.assertRegex(tex, r"to\[R=\$R_{\\rm 0}\$\]")
        self.assertRegex(tex, r"to\[R=\$R_{\\rm 1}\$\]")
        self.assertRegex(tex, r"to\[capacitor=\$C_{\\rm 2}\$\]")
