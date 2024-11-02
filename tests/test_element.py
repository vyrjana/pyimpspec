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
from copy import (
    copy as copy_object,
    deepcopy as deepcopy_object,
)
from io import StringIO
from unittest import TestCase
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Type,
)
from numpy import (
    array,
    inf,
)
from pyimpspec import (
    Connection,
    Container,
    Element,
    ElementDefinition,
    ParameterDefinition,
    get_elements,
    register_element,
)
from pyimpspec.circuit.elements import (
    Capacitor,
    ConstantPhaseElement,
    DeLevieFiniteLength,
    Gerischer,
    GerischerAlternative,
    HavriliakNegami,
    HavriliakNegamiAlternative,
    Inductor,
    KramersKronigAdmittanceRC,
    KramersKronigRC,
    ModifiedInductor,
    Resistor,
    TransmissionLineModel,
    TransmissionLineModelBlockingCPE,
    TransmissionLineModelBlockingOpen,
    TransmissionLineModelBlockingShort,
    TransmissionLineModelNonblockingCPE,
    TransmissionLineModelNonblockingOpen,
    TransmissionLineModelNonblockingShort,
    Warburg,
    WarburgOpen,
    WarburgShort,
    ZARC,
)
from pyimpspec.circuit.registry import (
    _validate_impedances,
    remove_elements,
    reset as reset_registry,
    reset_default_parameter_values,
)
from pyimpspec.circuit.base import InfiniteLimit
from pyimpspec.typing import (
    Frequencies,
    Frequency,
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


class TestElement(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.res: Resistor = Resistor()
        cls.cpe: ConstantPhaseElement = ConstantPhaseElement()
        cls.elements: Dict[str, Type[Element]] = get_elements()
        cls.instances: Dict[str, Element] = {k: v() for k, v in cls.elements.items()}

    def validate_copy(self, original: Element, copy: Element):
        self.assertEqual(
            original.to_string(decimals=12),
            copy.to_string(decimals=12),
        )
        new_value: float = 1e0
        key: str
        value: float
        for key, value in original.get_values().items():
            self.assertEqual(copy.get_value(key), value)
            copy.set_values(key, new_value)
            self.assertEqual(original.get_value(key), value)
            self.assertEqual(copy.get_value(key), new_value)
        new_lower_limit: float = 1e-1
        for key, value in original.get_lower_limits().items():
            self.assertEqual(copy.get_lower_limit(key), value)
            copy.set_lower_limits(key, new_lower_limit)
            self.assertEqual(original.get_lower_limit(key), value)
            self.assertEqual(copy.get_lower_limit(key), new_lower_limit)
        new_upper_limit: float = 1e1
        for key, value in original.get_upper_limits().items():
            self.assertEqual(copy.get_upper_limit(key), value)
            copy.set_upper_limits(key, new_upper_limit)
            self.assertEqual(original.get_upper_limit(key), value)
            self.assertEqual(copy.get_upper_limit(key), new_upper_limit)
        flag: bool
        for key, flag in original.are_fixed().items():
            self.assertEqual(copy.is_fixed(key), flag)
            copy.set_fixed(key, not flag)
            self.assertEqual(original.is_fixed(key), flag)
            self.assertEqual(copy.is_fixed(key), not flag)
        self.assertNotEqual(
            original.to_string(decimals=12),
            copy.to_string(decimals=12),
        )

    def test_copy(self):
        original: ConstantPhaseElement = ConstantPhaseElement()
        self.validate_copy(original, copy_object(original))
        self.validate_copy(original, deepcopy_object(original))

    def test_sympy(self):
        Class: Type[Element]
        for Class in self.elements.values():
            _validate_impedances(Class)

    def test_latex(self):
        self.assertEqual(self.res.to_latex(), "Z = R")

    def test_get_set_limits(self):
        self.res.set_lower_limits(R=7)
        self.res.set_upper_limits(R=2.5e4)
        self.assertEqual(self.res.to_string(2), "R{R=1.00E+03/7.00E+00/2.50E+04}")
        self.assertEqual(self.res.get_lower_limit("R"), 7.0)
        self.assertEqual(self.res.get_lower_limits("R")["R"], 7.0)
        self.assertEqual(self.res.get_lower_limits(R=True)["R"], 7.0)
        self.assertEqual(self.res.get_upper_limit("R"), 2.5e4)
        self.assertEqual(self.res.get_upper_limits("R")["R"], 2.5e4)
        self.assertEqual(self.res.get_upper_limits(R=True)["R"], 2.5e4)

    def test_get_units(self):
        self.assertIsInstance(self.res.get_units(), dict)
        self.assertEqual(self.res.get_unit("R"), "ohm")
        self.assertEqual(self.res.get_units()["R"], "ohm")
        self.assertEqual(self.res.get_units("R")["R"], "ohm")
        self.assertEqual(self.res.get_units(R=True)["R"], "ohm")
        with self.assertRaises(KeyError):
            self.res.get_units("Y")
        with self.assertRaises(KeyError):
            self.res.get_unit("Y")

    def test_get_value_descriptions(self):
        self.assertIsInstance(self.res.get_value_descriptions(), dict)
        description: str = "Resistance"
        self.assertEqual(self.res.get_value_description("R"), description)
        self.assertEqual(self.res.get_value_descriptions()["R"], description)
        self.assertEqual(self.res.get_value_descriptions("R")["R"], description)
        self.assertEqual(self.res.get_value_descriptions(R=True)["R"], description)
        with self.assertRaises(KeyError):
            self.res.get_value_descriptions("Y")
        with self.assertRaises(KeyError):
            self.res.get_value_description("Y")

    def test_reset_parameters(self):
        self.cpe.reset_parameter("Y")
        self.cpe.reset_parameters()
        self.cpe.reset_parameters("n")
        self.cpe.reset_parameters(Y=True, n=False)

    def test_get_set_fixed(self):
        self.cpe.set_fixed(Y=False, n=True)
        fixed_by_default: Dict[str, bool] = self.cpe.are_fixed_by_default()
        self.assertIsInstance(fixed_by_default, dict)
        self.assertTrue(all(map(lambda _: type(_) is str, fixed_by_default.keys())))
        self.assertTrue(all(map(lambda _: type(_) is bool, fixed_by_default.values())))
        self.assertEqual(fixed_by_default["Y"], False)
        self.assertEqual(self.cpe.is_fixed_by_default("Y"), False)
        self.assertEqual(self.cpe.is_fixed_by_default("n"), False)
        self.assertEqual(self.cpe.is_fixed("Y"), False)
        self.assertEqual(self.cpe.are_fixed(Y=True)["Y"], False)
        self.assertEqual(self.cpe.are_fixed("Y")["Y"], False)
        self.assertEqual(self.cpe.is_fixed("n"), True)
        self.assertEqual(self.cpe.are_fixed("n")["n"], True)
        self.assertEqual(self.cpe.are_fixed(n=True)["n"], True)
        self.cpe.set_fixed("Y", True)
        self.assertEqual(self.cpe.is_fixed("Y"), True)
        self.assertEqual(self.cpe.are_fixed("Y")["Y"], True)
        self.assertEqual(self.cpe.are_fixed(Y=True)["Y"], True)
        self.cpe.reset_parameters(Y=True, n=False)
        self.assertEqual(self.cpe.is_fixed("Y"), False)
        self.assertEqual(self.cpe.are_fixed("Y")["Y"], False)
        self.assertEqual(self.cpe.are_fixed(Y=True)["Y"], False)
        self.assertEqual(self.cpe.is_fixed("n"), False)
        self.assertEqual(self.cpe.are_fixed("n")["n"], False)
        self.assertEqual(self.cpe.are_fixed(n=True)["n"], False)

    def test_get_set_value(self):
        self.cpe.set_values("n", 0.6, "Y", 5)
        self.assertEqual(self.cpe.get_value("Y"), 5.0)
        self.assertEqual(self.cpe.get_value("n"), 0.6)
        self.cpe.set_values(Y=2, n=1.4)
        self.assertEqual(self.cpe.get_values("Y")["Y"], 2.0)
        self.assertEqual(self.cpe.get_values("n")["n"], 1.4)

    def test_get_default_values(self):
        symbol: str
        Class: Type[Element]
        for symbol, Class in self.elements.items():
            defaults: Dict[str, float] = Class.get_default_values()
            self.assertIsInstance(defaults, dict, msg=symbol)
            self.assertTrue(len(defaults) > 0, msg=symbol)
            self.assertTrue(
                all(map(lambda _: type(_) is str, defaults.keys())),
                msg=symbol,
            )
            self.assertTrue(
                all(map(lambda _: type(_) is float, defaults.values())),
                msg=symbol,
            )
            element: Element = self.instances[symbol]
            key: str
            value: float
            for key, value in defaults.items():
                self.assertAlmostEqual(
                    element.get_default_value(key),
                    value,
                    msg=symbol,
                )

    def test_are_fixed_by_default(self):
        symbol: str
        Class: Type[Element]
        for symbol, Class in self.elements.items():
            default_fixed: Dict[str, bool] = Class.are_fixed_by_default()
            self.assertIsInstance(default_fixed, dict, msg=symbol)
            self.assertTrue(len(default_fixed) > 0, msg=symbol)
            self.assertTrue(
                all(map(lambda _: type(_) is str, default_fixed.keys())),
                msg=symbol,
            )
            self.assertTrue(
                all(
                    map(
                        lambda _: type(_) is bool,
                        default_fixed.values(),
                    )
                ),
                msg=symbol,
            )
            element: Element = self.instances[symbol]
            key: str
            value: bool
            for key, value in default_fixed.items():
                self.assertEqual(element.is_fixed_by_default(key), value, msg=symbol)
            for key, value in element.are_fixed_by_default(
                *default_fixed.keys()
            ).items():
                self.assertEqual(default_fixed[key], value)

    def test_repr(self):
        symbol: str
        element: Element
        for symbol, element in self.instances.items():
            self.assertEqual(
                repr(element), f"{element.get_name()} ({hex(id(element))})", msg=symbol
            )

    def test_get_extended_description(self):
        symbol: str
        Class: Type[Element]
        for symbol, Class in self.elements.items():
            self.assertNotEqual(Class.get_extended_description(), "", msg=symbol)

    def test_get_description(self):
        symbol: str
        Class: Type[Element]
        for symbol, Class in self.elements.items():
            self.assertNotEqual(Class.get_description(), "", msg=symbol)
            self.assertTrue(Class.get_description().startswith(symbol), msg=symbol)

    def test_get_default_lower_limits(self):
        symbol: str
        Class: Type[Element]
        for symbol, Class in self.elements.items():
            default_lower_limits: Dict[str, float] = Class.get_default_lower_limits()
            self.assertIsInstance(default_lower_limits, dict, msg=symbol)
            self.assertTrue(len(default_lower_limits) > 0, msg=symbol)
            self.assertTrue(
                all(map(lambda _: type(_) is str, default_lower_limits.keys())),
                msg=symbol,
            )
            self.assertTrue(
                all(
                    map(
                        lambda _: type(_) is float,
                        default_lower_limits.values(),
                    )
                ),
                msg=symbol,
            )
            key: str
            value: float
            for key, value in default_lower_limits.items():
                self.assertIsInstance(key, str, msg=symbol)
                self.assertIsInstance(value, float, msg=symbol)
                self.assertEqual(Class.get_default_lower_limit(key), value, msg=symbol)
            for key, value in Class.get_default_lower_limits(
                *default_lower_limits.keys()
            ).items():
                self.assertEqual(default_lower_limits[key], value)

    def test_get_default_upper_limits(self):
        symbol: str
        Class: Type[Element]
        for symbol, Class in self.elements.items():
            default_upper_limits: Dict[str, float] = Class.get_default_upper_limits()
            self.assertIsInstance(default_upper_limits, dict, msg=symbol)
            self.assertTrue(len(default_upper_limits) > 0, msg=symbol)
            self.assertTrue(
                all(map(lambda _: type(_) is str, default_upper_limits.keys())),
                msg=symbol,
            )
            self.assertTrue(
                all(
                    map(
                        lambda _: type(_) is float,
                        default_upper_limits.values(),
                    )
                ),
                msg=symbol,
            )
            key: str
            value: float
            for key, value in default_upper_limits.items():
                self.assertEqual(Class.get_default_upper_limit(key), value, msg=symbol)
            for key, value in Class.get_default_upper_limits(
                *default_upper_limits.keys()
            ).items():
                self.assertEqual(default_upper_limits[key], value)

    def test_get_set_label(self):
        i: int
        symbol: str
        element: Element
        for i, (symbol, element) in enumerate(self.instances.items()):
            self.assertEqual(
                element.get_label(),
                "",
            )
            self.assertEqual(
                element.get_name(),
                element.get_symbol(),
                msg=symbol,
            )
            with self.assertRaises(TypeError, msg=symbol):
                element.set_label(26)
            with self.assertRaises(ValueError, msg=symbol):
                element.set_label("26")
            element.set_label("test")
            self.assertEqual(
                element.get_label(),
                "test",
            )
            self.assertEqual(
                element.get_name(),
                f"{symbol}_test",
                msg=symbol,
            )

    def test_to_string(self):
        symbol: str
        element: Element
        for symbol, element in self.instances.items():
            # TODO: Implement more assertions
            self.assertTrue(element.to_string(1).endswith(":test}"), msg=symbol)

    def test_get_symbol(self):
        symbol: str
        Class: Type[Element]
        for symbol, Class in self.elements.items():
            self.assertEqual(symbol, Class.get_symbol(), msg=symbol)
            self.assertEqual(
                Class.get_symbol(),
                symbol,
                msg=symbol,
            )

    def test_get_set_values(self):
        symbol: str
        element: Element
        for symbol, element in self.instances.items():
            parameters: Dict[str, float] = element.get_values()
            self.assertIsInstance(parameters, dict, msg=symbol)
            self.assertTrue(len(parameters) > 0, msg=symbol)
            self.assertTrue(
                all(map(lambda _: type(_) is str, parameters.keys())), msg=symbol
            )
            self.assertTrue(
                all(map(lambda _: type(_) is float, parameters.values())), msg=symbol
            )
            key: str
            value: float
            for key, value in parameters.items():
                self.assertEqual(element.get_value(key), value)
                self.assertEqual(element.get_default_value(key), value)

    def test_limits(self):
        with self.assertRaises(InfiniteLimit):
            Capacitor().get_impedances(array([0.0], dtype=Frequency))
        with self.assertRaises(InfiniteLimit):
            Capacitor().get_impedances(array([5.0, 0.0, 2.5], dtype=Frequency))
        with self.assertRaises(InfiniteLimit):
            Inductor().get_impedances(array([inf], dtype=Frequency))
        with self.assertRaises(InfiniteLimit):
            Inductor().get_impedances(array([5.0, 2.5, inf, 1e-3], dtype=Frequency))

    def test_impedance(self):
        f: Frequencies = array([1e-3, 1, 1e3], dtype=Frequency)
        for Class in self.elements.values():
            element: Element = Class()
            element.get_impedances(f)
            # Try to calculate impedances at f == 0 or f == inf
            # SymPy may not be able to do it successfully in all cases
            try:
                element.get_impedances(array([0.0], dtype=Frequency))
            except InfiniteLimit:
                pass
            except NotImplementedError:
                pass
            try:
                element.get_impedances(array([inf], dtype=Frequency))
            except InfiniteLimit:
                pass
            except NotImplementedError:
                pass

    def test_get_elements(self):
        def check(
            symbols: List[str],
            elements: Dict[str, Type[Element]],
            symbol: str,
            Class: Type[Element],
        ):
            assert symbol in symbols, f"The following symbol is missing: '{symbol}'"
            symbols.remove(symbol)
            self.assertEqual(elements[symbol], Class)

        with self.assertRaises(TypeError):
            get_elements(default_only="test")

        elements: Dict[str, Type[Element]] = get_elements(private=True)
        symbols: List[str] = list(elements.keys())
        check(symbols, elements, "C", Capacitor)
        check(symbols, elements, "G", Gerischer)
        check(symbols, elements, "Ga", GerischerAlternative)
        check(symbols, elements, "H", HavriliakNegami)
        check(symbols, elements, "Ha", HavriliakNegamiAlternative)
        check(symbols, elements, "K", KramersKronigRC)
        check(symbols, elements, "Ky", KramersKronigAdmittanceRC)
        check(symbols, elements, "L", Inductor)
        check(symbols, elements, "La", ModifiedInductor)
        check(symbols, elements, "Ls", DeLevieFiniteLength)
        check(symbols, elements, "Q", ConstantPhaseElement)
        check(symbols, elements, "R", Resistor)
        check(symbols, elements, "Tlm", TransmissionLineModel)
        check(symbols, elements, "Tlmbo", TransmissionLineModelBlockingOpen)
        check(symbols, elements, "Tlmbq", TransmissionLineModelBlockingCPE)
        check(symbols, elements, "Tlmbs", TransmissionLineModelBlockingShort)
        check(symbols, elements, "Tlmno", TransmissionLineModelNonblockingOpen)
        check(symbols, elements, "Tlmnq", TransmissionLineModelNonblockingCPE)
        check(symbols, elements, "Tlmns", TransmissionLineModelNonblockingShort)
        check(symbols, elements, "W", Warburg)
        check(symbols, elements, "Wo", WarburgOpen)
        check(symbols, elements, "Ws", WarburgShort)
        check(symbols, elements, "Zarc", ZARC)
        self.assertEqual(len(symbols), 0, msg=symbols)


class UserDefinedIncomplete(Element):
    pass


class UserDefinedInvalid(Element):
    def _impedance(self, f: float) -> complex:
        return 1 + 1j


class UserDefined(Element):
    def _impedance(self, f: float, R: float, X: float) -> complex:
        return R + X * 1j * f


# TODO: Refactor
class TestRegistry(TestCase):
    def test_user_defined_element(self):
        with self.assertRaises(TypeError):
            register_element(UserDefined)

        kwargs: Dict[str, Any] = {
            "symbol": "U",
            "name": "User-defined element",
            "description": "A custom circuit element",
            "equation": "TODO",
            "parameters": [],
        }
        with self.assertRaises(TypeError):
            redirect_output(
                lambda: register_element(
                    ElementDefinition(
                        Class=UserDefinedIncomplete,
                        **kwargs,
                    ),
                ),
                stderr=True,
            )

        # The equation is only a placeholder (i.e., 'todo')
        lines: List[str] = redirect_output(
            lambda: register_element(
                ElementDefinition(
                    Class=UserDefinedInvalid,
                    **kwargs,
                ),
                validate_impedances=False,
            ),
            stderr=True,
        )
        self.assertTrue("The equation is missing!" in "\n".join(lines))

        # Equation cannot be turned into SymPy expression
        with self.assertRaises(TypeError):
            register_element(
                ElementDefinition(
                    Class=UserDefinedInvalid,
                    **kwargs,
                ),
            )

        # The equation is not correct since there are no parameters
        kwargs["equation"] = "R"
        with self.assertRaises(TypeError):
            register_element(
                ElementDefinition(
                    Class=UserDefinedInvalid,
                    **kwargs,
                ),
            )

        # Adding parameters but the output of the equation does not match
        # the output of the _impedance method
        kwargs["parameters"].append(
            ParameterDefinition(
                symbol="R",
                unit="ohm",
                description="The real part",
                value=5.0,
                lower_limit=0.0,
                upper_limit=10.0,
                fixed=False,
            )
        )

        with self.assertRaises(TypeError):
            register_element(
                ElementDefinition(
                    Class=UserDefinedInvalid,
                    **kwargs,
                ),
            )

        with self.assertRaises(TypeError):
            ParameterDefinition(
                symbol="R",
                unit="ohm",
                description="The real part",
                value="test",  # This should raise an error
                lower_limit=0.0,
                upper_limit=10.0,
                fixed=False,
            ),

        with self.assertRaises(ValueError):
            ParameterDefinition(
                symbol="R",
                unit="ohm",
                description="The real part",
                value=-1.0,  # value >= lower limit
                lower_limit=0.0,
                upper_limit=10.0,
                fixed=False,
            ),

        with self.assertRaises(ValueError):
            ParameterDefinition(
                symbol="R",
                unit="ohm",
                description="The real part",
                value=11.0,  # value <= upper limit
                lower_limit=0.0,
                upper_limit=10.0,
                fixed=False,
            ),

        # Duplicate symbols
        kwargs["parameters"].append(
            ParameterDefinition(
                symbol="R",
                unit="ohm",
                description="The real part",
                value=1.0,
                lower_limit=0.0,
                upper_limit=10.0,
                fixed=False,
            ),
        )
        with self.assertRaises(KeyError):
            register_element(
                ElementDefinition(
                    Class=UserDefinedInvalid,
                    **kwargs,
                )
            )
        kwargs["parameters"].pop()

        # Switching to the correct _impedance implementation
        kwargs["parameters"].append(
            ParameterDefinition(
                symbol="X",
                unit="ohm",
                description="The imaginary part",
                value=-5.0,
                lower_limit=-10.0,
                upper_limit=10.0,
                fixed=False,
            )
        )
        with self.assertRaises(ValueError):
            register_element(
                ElementDefinition(
                    Class=UserDefined,
                    **kwargs,
                )
            )

        # Fixing the equation but UserDefinedInvalid was successfully registered
        # when validate_impedances was set to False
        kwargs["equation"] = "R + X*I*f"
        with self.assertRaises(KeyError):
            register_element(
                ElementDefinition(
                    Class=UserDefined,
                    **kwargs,
                ),
            )

        # Removing the previous registration of UserDefinedInvalid
        with self.assertRaises(KeyError):
            register_element(
                ElementDefinition(
                    Class=UserDefined,
                    **kwargs,
                )
            )

        reset_registry()
        register_element(
            ElementDefinition(
                Class=UserDefined,
                **kwargs,
            )
        )

        self.assertLess(len(get_elements(default_only=True)), len(get_elements()))

        with self.assertRaises(ValueError):
            remove_elements(Resistor)

        with self.assertRaises(ValueError):
            remove_elements([Resistor, Capacitor])

        remove_elements(UserDefined)
        self.assertEqual(len(get_elements(default_only=True)), len(get_elements()))

    def test_reset_default_parameter_values(self):
        self.assertAlmostEqual(Resistor.get_default_value("R"), 1000.0)
        self.assertAlmostEqual(Capacitor.get_default_value("C"), 1.0e-6)

        Resistor.set_default_values(R=5.3)
        Capacitor.set_default_values(C=2.51e-3)
        self.assertAlmostEqual(Resistor.get_default_value("R"), 5.3)
        self.assertAlmostEqual(Capacitor.get_default_value("C"), 2.51e-3)

        reset_default_parameter_values(elements=[Resistor])
        self.assertAlmostEqual(Resistor.get_default_value("R"), 1000.0)
        self.assertAlmostEqual(Capacitor.get_default_value("C"), 2.51e-3)

        Resistor.set_default_values(R=42.0)
        reset_default_parameter_values()
        self.assertAlmostEqual(Resistor.get_default_value("R"), 1000.0)
        self.assertAlmostEqual(Capacitor.get_default_value("C"), 1.0e-6)


# TODO: Implement more tests
class TestContainer(TestCase):
    def validate_copy(self, original: Container, copy: Container):
        self.assertIsInstance(original, type(copy))
        self.assertEqual(
            original.to_string(decimals=12),
            copy.to_string(decimals=12),
        )
        for key in original.get_default_subcircuits().keys():
            original_con: Optional[Connection] = original.get_subcircuit(key)
            copy_con: Optional[Connection] = copy.get_subcircuit(key)
            self.assertIsInstance(original_con, type(copy_con))
            if original_con is None:
                continue
            self.assertNotEqual(id(original_con), id(copy_con))
            for original_elem, copy_elem in zip(
                original_con.get_elements(),
                copy_con.get_elements(),
            ):
                self.assertNotEqual(id(original_elem), id(copy_elem))
                self.assertEqual(
                    original_elem.to_string(decimals=12),
                    copy_elem.to_string(decimals=12),
                )
                for key, value in original_elem.get_values().items():
                    new_value: float = value + 1e2
                    copy_elem.set_values(key, new_value)
                    self.assertEqual(original_elem.get_value(key), value)
                    self.assertEqual(copy_elem.get_value(key), new_value)
        self.assertNotEqual(
            original.to_string(decimals=12),
            copy.to_string(decimals=12),
        )

    def test_copy(self):
        original: TransmissionLineModel = TransmissionLineModel()
        self.validate_copy(original, copy_object(original))
        self.validate_copy(original, deepcopy_object(original))
