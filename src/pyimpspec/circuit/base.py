# Copyright 2022 pyimpspec developers
# pyimpspec is licensed under the GPLv3 or later (https://www.gnu.org/licenses/gpl-3.0.html).
# The licenses of pyimpspec's dependencies and/or sources of portions of code are included in
# the LICENSES folder.

from collections import OrderedDict
from re import sub
from typing import Dict, List, Tuple, Union, Optional
from numpy import array, inf, ndarray
from sympy import Expr, latex, sympify


class Element:
    def __init__(self, keys: List[str]):
        self._label: str = ""
        self._fixed_parameters: Dict[str, bool] = {}
        self._lower_limits: Dict[str, float] = {}
        self._upper_limits: Dict[str, float] = {}
        # Make sure the keys exist in the relevant dictionaries
        key: str
        for key in keys:
            self._fixed_parameters[key] = False
            self._lower_limits[key] = -inf
            self._upper_limits[key] = inf
        self._identifier: int = -1

    def __repr__(self) -> str:
        return f"{self.get_label()} ({hex(id(self))})"

    @classmethod
    def get_extended_description(Class) -> str:
        assert (
            hasattr(Class, "__doc__")
            and type(Class.__doc__) is str
            and Class.__doc__.strip() != ""
        )
        return "\n".join(map(str.strip, Class.__doc__.split("\n"))).strip()

    @staticmethod
    def get_description() -> str:
        """
        Get a brief description of the element and its symbol.
        """
        raise Exception("Method has not been implemented!")

    @staticmethod
    def get_defaults() -> Dict[str, float]:
        """
        Get the default values for the element's parameters.

        Returns
        -------
        Dict[str, float]
        """
        raise Exception("Method has not been implemented!")

    @staticmethod
    def get_default_fixed() -> Dict[str, bool]:
        """
        Get whether or not the element's parameters are fixed by default.

        Returns
        -------
        Dict[str, bool]
        """
        raise Exception("Method has not been implemented!")

    @staticmethod
    def get_default_lower_limits() -> Dict[str, float]:
        """
        Get the default lower limits for the element's parameters.

        Returns
        -------
        Dict[str, float]
        """
        raise Exception("Method has not been implemented!")

    @staticmethod
    def get_default_upper_limits() -> Dict[str, float]:
        """
        Get the default upper limits for the element's parameters.

        Returns
        -------
        Dict[str, float]
        """
        raise Exception("Method has not been implemented!")

    def get_default_label(self) -> str:
        """
        Get the default label for this element.

        Returns
        -------
        str
        """
        if self._identifier >= 0:
            return f"{self.get_symbol()}_{self._identifier}"
        else:
            return self.get_symbol()

    def get_label(self) -> str:
        """
        Get the label assigned to a specific instance of the element.

        Returns
        -------
        str
        """
        if self._label == "":
            return self.get_default_label()
        return f"{self.get_symbol()}_{self._label}"

    def set_label(self, label: str):
        """
        Set the label assigned to a specific instance of the element.

        Parameters
        ----------
        label: str
            The new label.
        """
        assert type(label) is str, f"{label=}"
        self._label = label.strip()

    @staticmethod
    def get_symbol() -> str:
        """
        Get the symbol representing the element.

        Returns
        -------
        str
        """
        raise Exception("NOT YET IMPLEMENTED!")

    def to_string(self, decimals: int = -1) -> str:
        """
        Generates a string representation of the element.

        Parameters
        ----------
        decimals: int = -1
            The number of decimals used when formatting the current value and the limits for the
            element's parameters. -1 corresponds to no values being included in the output.

        Returns
        -------
        str
        """
        assert type(decimals) is int
        if decimals < 0:
            return self.get_symbol()
        parameters: List[str] = []
        for symbol, value in self.get_parameters().items():
            lower: float = self.get_lower_limit(symbol)
            upper: float = self.get_upper_limit(symbol)
            fixed: bool = self.is_fixed(symbol)
            string: str = f"{symbol}=" + (f"%.{decimals}E") % value
            if fixed:
                string += "F"
            elif lower != -inf and upper == inf:
                string += (f"/%.{decimals}E") % lower
            elif lower == -inf and upper != inf:
                string += (f"//%.{decimals}E") % upper
            elif lower != -inf and upper != inf:
                string += (f"/%.{decimals}E") % lower + (f"/%.{decimals}E") % upper
            parameters.append(string)
        cdc: str = self.get_symbol() + "{" + ",".join(parameters)
        if self._label != "":
            cdc += f":{self._label}"
        return cdc + "}"

    def _assign_identifier(self, current: int) -> int:
        """
        Get the internal identifier that is unique in the context of a circuit.
        Used internally when generating unique names for parameters when fitting a circuit to a
        data set.

        Parameters
        ----------
        current: int
            The most recently assigned identifier.
        """
        assert type(current) is int and current >= 0
        self._identifier = current
        return current + 1

    def get_identifier(self) -> int:
        """
        Get the internal identifier that is unique in the context of a circuit.
        Used internally when generating unique names for parameters when fitting a circuit to a
        data set.

        Returns
        -------
        int
        """
        return self._identifier

    def impedance(self, f: float) -> complex:
        """
        Calculates the complex impedance of the element at a specific frequency.

        Parameters
        ----------
        f: float
            Frequency in Hz

        Returns
        -------
        complex
        """
        assert f > 0 and f < inf
        return complex(-999, 999)

    def impedances(self, freq: Union[list, ndarray]) -> ndarray:
        """
        Calculates the complex impedance of the element at specific frequencies.

        Parameters
        ----------
        freq: Union[list, ndarray]
            Frequencies in Hz

        Returns
        -------
        numpy.ndarray
        """
        assert type(freq) is list or type(freq) is ndarray
        assert min(freq) > 0 and max(freq) < inf
        return array(list(map(self.impedance, freq)))

    def reset_parameters(self, keys: List[str]):
        """
        Resets the value, lower limit, upper limit, and fixed state of one or more parameters.

        Parameters
        ----------
        keys: List[str]
            Names of the parameters to reset.
        """
        assert type(keys) is list
        assert len(keys) > 0
        assert all(list(map(lambda _: type(_) is str, keys)))
        self.set_parameters({k: v for k, v in self.get_defaults().items() if k in keys})
        for k, v in self.get_default_lower_limits().items():
            if k not in keys:
                continue
            self.set_lower_limit(k, v)
        for k, v in self.get_default_upper_limits().items():
            if k not in keys:
                continue
            self.set_upper_limit(k, v)
        for k, v in self.get_default_fixed().items():
            if k not in keys:
                continue
            self.set_fixed(k, v)

    def get_parameters(self) -> OrderedDict[str, float]:
        """
        Get the current parameters of the element.

        Returns
        -------
        OrderedDict[str, float]
        """
        # This implementation is just for basic testing
        return OrderedDict(
            {
                "A": -999,
                "B": 999,
            }
        )

    def set_parameters(self, parameters: Dict[str, float]):
        """
        Set new values for the parameters of the element.

        Parameters
        ----------
        parameters: Dict[str, float]
        """
        raise Exception("Method has not been implemented!")

    def is_fixed(self, key: str) -> bool:
        """
        Check if an element parameter should have a fixed value when fitting a circuit to a data
        set.

        Parameters
        ----------
        key: str
            A key corresponding to an element parameter.

        Returns
        -------
        bool
            True if fixed and False if not fixed.
        """
        assert type(key) is str and key.strip() != ""
        return self._fixed_parameters[key]

    def set_fixed(self, key: str, value: bool):
        """
        Set whether or not an element parameter should have a fixed value when fitting a circuit
        to a data set.

        Parameters
        ----------
        key: str
            A key corresponding to an element parameter.
        value: bool
            True if the value should be fixed.
        """
        assert type(key) is str and key.strip() != ""
        assert key in self._fixed_parameters, f"{key=}"
        assert type(value) is bool, f"{value=}"
        self._fixed_parameters[key] = value

    def get_lower_limit(self, key: str) -> float:
        """
        Get the lower limit for the value of an element parameter when fitting a circuit to a data
        set.

        Parameters
        ----------
        key: str
            A key corresponding to an element parameter.

        Returns
        -------
        float
            The absence of a limit is represented by -numpy.inf.
        """
        assert type(key) is str and key.strip() != ""
        return self._lower_limits[key]

    def set_lower_limit(self, key: str, value: float):
        """
        Set the upper limit for the value of an element parameter when fitting a circuit to a data
        set.

        Parameters
        ----------
        key: str
            A key corresponding to an element parameter.
        value: float
            The new limit for the element parameter. The limit can be removed by setting the limit
            to be -numpy.inf.
        """
        assert type(key) is str and key.strip() != ""
        assert key in self._lower_limits, f"{key=}"
        self._lower_limits[key] = float(value)

    def get_upper_limit(self, key: str) -> float:
        """
        Get the upper limit for the value of an element parameter when fitting a circuit to a data
        set.

        Parameters
        ----------
        key: str
            A key corresponding to an element parameter.

        Returns
        -------
        float
            The absence of a limit is represented by numpy.inf.
        """
        assert type(key) is str and key.strip() != ""
        return self._upper_limits[key]

    def set_upper_limit(self, key: str, value: float):
        """
        Set the upper limit for the value of an element parameter when fitting a circuit to a data
        set.

        Parameters
        ----------
        key: str
            A key corresponding to an element parameter.
        value: float
            The new limit for the element parameter. The limit can be removed by setting the limit
            to be numpy.inf.
        """
        assert type(key) is str and key.strip() != ""
        assert key in self._upper_limits, f"{key=}"
        self._upper_limits[key] = float(value)

    def _str_expr(self, substitute: bool = False) -> str:
        raise Exception("Method has not been implemented!")

    def _subs_str_expr(
        self, string: str, parameters: OrderedDict[str, float], symbols_only: bool
    ) -> str:
        assert type(string) is str
        assert type(parameters) is OrderedDict
        assert type(symbols_only) is bool
        k: str
        v: float
        for k, v in parameters.items():
            repl: str = f"{v:.12E}"
            if symbols_only:
                if self._label != "":
                    repl = f"{k}_{self._label}"
                else:
                    assert (
                        self._identifier >= 0
                    ), "Assign an identifier, set a label, or create the element by parsing a circuit description code!"
                    repl = f"{k}_{self._identifier}"
            pattern: str = r"(?<![a-zA-Z])" + k + r"(?![a-zA-Z])"
            string = sub(pattern, repl, string)
        return string

    def to_sympy(self, substitute: bool = False) -> Expr:
        assert type(substitute) is bool
        return sympify(self._str_expr(substitute=substitute))

    def to_latex(self) -> str:
        return latex(self.to_sympy(substitute=False))


class Connection:
    def __init__(self, elements: List[Union[Element, "Connection"]]):
        assert type(elements) is list
        assert all(
            map(lambda _: isinstance(_, Connection) or isinstance(_, Element), elements)
        )
        self._elements: List[Union[Element, "Connection"]] = elements

    def __repr__(self) -> str:
        return f"{self.get_label()} ({hex(id(self))})"

    def __len__(self) -> int:
        return len(self._elements)

    def __contains__(self, element: Union[Element, "Connection"]) -> bool:
        for ec in self.get_elements(flattened=False):
            if ec == element:
                return True
        return False

    def contains(
        self, element: Union[Element, "Connection"], top_level: bool = False
    ) -> bool:
        if top_level:
            return element in self._elements
        return element in self

    def _assign_identifier(self, current: int) -> int:
        """
        Used internally when generating unique names for parameters when fitting a circuit to a
        data set.

        Parameters
        ----------
        current: int
            The most recently assigned identifier.

        Returns
        -------
        int
            The most recently assigned identifier.
        """
        element: Union[Element, "Connection"]
        for element in reversed(self._elements):
            current = element._assign_identifier(current)
        return current

    def get_label(self) -> str:
        raise Exception("Not yet implemented!")

    def to_string(self, decimals: int = -1) -> str:
        raise Exception("Not yet implemented!")

    def get_elements(
        self, flattened: bool = True
    ) -> List[Union[Element, "Connection"]]:
        """
        Get a list of elements and connections nested inside this connection.

        Returns
        -------
        List[Union[Element, Connection]]
        """
        if flattened:
            elements: List[Union[Element, "Connection"]] = []
            for element in self._elements:
                if isinstance(element, Connection):
                    elements.extend(reversed(element.get_elements(flattened=flattened)))
                else:
                    elements.append(element)
            return list(reversed(elements))
        return list(reversed(self._elements))

    def impedance(self, f: float) -> complex:
        """
        Calculates the complex impedance of the connection at a specific frequency.

        Parameters
        ----------
        f: float
            Frequency in Hz

        Returns
        -------
        complex
        """
        raise Exception("Method has not been implemented!")

    def impedances(self, freq: Union[list, ndarray]) -> ndarray:
        """
        Calculates the complex impedance of the element at specific frequencies.

        Parameters
        ----------
        freq: Union[list, ndarray]
            Frequencies in Hz

        Returns
        -------
        numpy.ndarray
        """
        assert type(freq) is list or type(freq) is ndarray
        assert min(freq) > 0 and max(freq) < inf
        return array(list(map(self.impedance, freq)))

    def get_parameters(self) -> Dict[int, OrderedDict[str, float]]:
        """
        Get the current element parameters of all elements nested inside this connection.

        Returns
        -------
        Dict[int, Dict[str, float]]
            The outer key is the unique identifier assigned to an element.
            The inner key is the symbol corresponding to an element parameter.
        """
        parameters: Dict[int, OrderedDict[str, float]] = {}
        element: Union[Element, "Connection"]
        for element in self._elements:
            if isinstance(element, Connection):
                parameters.update(element.get_parameters())
                continue
            parameters[element.get_identifier()] = element.get_parameters()
        return parameters

    def set_parameters(self, parameters: Dict[int, Dict[str, float]]):
        """
        Set new element parameters to some/all elements nested inside this connection.

        Parameters
        ----------
        parameters: Dict[int, Dict[str, float]
            The outer key is the unique identifier assigned to an element.
            The inner key is the symbol corresponding to an element parameter.
        """
        assert type(parameters) is dict
        element: Union[Element, "Connection"]
        for element in self._elements:
            if isinstance(element, Connection):
                element.set_parameters(parameters)
                continue
            ident: int = element.get_identifier()
            if ident not in parameters:
                continue
            element.set_parameters(parameters[ident])

    def get_element(self, ident: int) -> Optional[Element]:
        """
        Get a specific element based on its unique identifier.

        Parameters
        ----------
        ident: int
            The integer identifier that should be unique in the context of the circuit.

        Returns
        -------
        Optional[Element]
        """
        assert type(ident) is int
        element: Optional[Union[Element, "Connection"]]
        for element in self._elements:
            if isinstance(element, Connection):
                element = element.get_element(ident)  # type: ignore
                if element is not None:
                    return element  # type: ignore
                continue
            elif element is not None and element.get_identifier() == ident:
                return element  # type: ignore
        return None

    def to_stack(self, stack: List[Tuple[str, Union[Element, "Connection"]]]):
        raise Exception("Method has not been implemented!")

    def _str_expr(self, substitute: bool = False) -> str:
        raise Exception("Method has not been implemented!")

    def to_sympy(self, substitute: bool = False) -> Expr:
        assert type(substitute) is bool
        return sympify(self._str_expr(substitute=substitute))

    def to_latex(self) -> str:
        return latex(self.to_sympy(substitute=False))
