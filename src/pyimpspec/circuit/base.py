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

from abc import (
    ABC,
    abstractmethod,
)
from copy import deepcopy
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)
from numpy import (
    concatenate,
    delete,
    fromiter,
    indices as array_indices,
    isinf,
    isnan,
    isneginf,
    isposinf,
    unique,
    where,
    zeros,
)
from sympy import (
    Basic,
    Expr,
    latex,
    limit,
    sympify,
)
from pyimpspec.typing import (
    ComplexImpedance,
    ComplexImpedances,
    Frequency,
    Frequencies,
    Indices,
)
from pyimpspec.typing.helpers import (
    _cast_to_floating_array,
    _is_boolean,
    _is_floating_array,
    _is_integer,
)
from pyimpspec.exceptions import (
    InfiniteImpedance,
    InfiniteLimit,
    InvalidEquation,
    InvalidParameterKey,
    NotANumberImpedance,
)


def _calculate_limit(obj, f: Frequency) -> ComplexImpedance:
    expr: Expr = obj.to_sympy(substitute=True)
    symbols: Set[Basic] = expr.free_symbols

    Z: ComplexImpedance
    if len(symbols) == 0:
        Z = ComplexImpedance(expr)
    elif len(symbols) == 1:
        Z = ComplexImpedance(limit(expr, symbols.pop(), f))
    else:
        raise InvalidEquation("Invalid impedance expression! Too many free symbols")

    if isinf(Z):
        raise InfiniteLimit(
            f"The f -> {f} limit is not finite for the following expression when values are substituted: {str(obj.to_sympy())}"
        )
    elif isnan(Z):
        raise NotANumberImpedance()

    return Z.astype(ComplexImpedance)


def _calculate_impedances(
    obj: Union["Element", "Container", "Connection"],
    f: Frequencies,
) -> ComplexImpedances:
    if not _is_floating_array(f):
        f = _cast_to_floating_array(f)

    if min(f) < 0.0:
        raise ValueError("Negative frequencies are not supported")

    parameters: Dict[str, float]
    subcircuits: Dict[str, Optional[Connection]]

    func: Callable
    if isinstance(obj, Container):
        parameters = obj.get_values()
        subcircuits = obj.get_subcircuits()
        func = lambda _: obj._impedance(_, **parameters, **subcircuits)
    elif isinstance(obj, Element):
        parameters = obj.get_values()
        func = lambda _: obj._impedance(_, **parameters)
    elif isinstance(obj, Connection):
        parameters = {}
        func = lambda _: obj._impedance(_)
    else:
        raise NotImplementedError(f"Unsupported object: '{type(obj)}'")

    Z: ComplexImpedances = zeros(f.shape, dtype=ComplexImpedance)

    indices: Indices = array_indices(Z.shape)[0]
    limit_indices: Indices = unique(
        concatenate(
            (
                where(f == 0.0)[0],
                where(isinf(f))[0],
            ),
            axis=0,
        )
    )

    if limit_indices.size > 0:
        Z[limit_indices] = fromiter(
            (_calculate_limit(obj, _) for _ in f[limit_indices]),
            dtype=ComplexImpedance,
            count=limit_indices.size,
        )
        indices = delete(indices, limit_indices)

    if indices.size > 0:
        Z[indices] = func(f[indices])

    if isinf(Z).any():
        raise InfiniteImpedance("Encountered an infinite impedance")
    elif isnan(Z).any():
        raise NotANumberImpedance("Encountered an impedance that is not a number (NaN)")

    return Z.astype(ComplexImpedance)


class Element(ABC):
    """
    The base class for circuit elements.
    Classes implementing circuit elements should extend this base class and implement the ``_impedance`` method (note the underscore).
    The docstring for the class is generated when the class is registered via the :func:`~pyimpspec.register_element` function.
    """

    _symbol: str = ""
    _name: str = ""
    _description: str = ""
    _equation: str = ""
    _parameter_unit: Dict[str, str] = {}
    _parameter_description: Dict[str, str] = {}
    _parameter_default_value: Dict[str, float] = {}
    _parameter_default_lower_limit: Dict[str, float] = {}
    _parameter_default_upper_limit: Dict[str, float] = {}
    _parameter_default_fixed: Dict[str, bool] = {}
    _valid_kwargs_keys: Set[str] = set()

    def __init__(self, **kwargs):
        if kwargs and not set(kwargs.keys()).issubset(self._valid_kwargs_keys):
            raise InvalidParameterKey(
                "Invalid parameter (or subcircuit) key detected! The valid keys are: "
                + ", ".join(self._valid_kwargs_keys)
            )

        self._label: str = ""
        self._parameter_value: Dict[str, float] = {}

        key: str
        value: float
        for key, value in self._parameter_default_value.items():
            if key not in kwargs:
                self._parameter_value[key] = value
            else:
                self._parameter_value[key] = float(kwargs[key])

        self._parameter_lower_limit: Dict[
            str, float
        ] = self._parameter_default_lower_limit.copy()
        self._parameter_upper_limit: Dict[
            str, float
        ] = self._parameter_default_upper_limit.copy()
        self._parameter_fixed: Dict[str, bool] = self._parameter_default_fixed.copy()

    def __copy__(self) -> "Element":
        return (
            type(self)()
            .set_lower_limits(**self.get_lower_limits())
            .set_upper_limits(**self.get_upper_limits())
            .set_values(**self.get_values())
            .set_fixed(**self.are_fixed())
            .set_label(self._label)
        )

    def __deepcopy__(self, memo: dict) -> "Element":
        ident: int = id(self)
        copy: Optional["Element"] = memo.get(ident)

        if copy is None:
            copy = self.__copy__()
            memo[ident] = copy

        return copy

    def __repr__(self) -> str:
        return f"{self.get_name()} ({hex(id(self))})"

    def __str__(self) -> str:
        return self.get_description()

    @classmethod
    def get_extended_description(cls) -> str:
        """
        Get an extended description of this element.

        Returns
        -------
        str
        """
        if not isinstance(cls.__doc__, str):
            raise TypeError(
                f"Expected the element's description to be a string instead of {cls.__doc__=}"
            )

        return cls.__doc__

    @classmethod
    def get_description(cls) -> str:
        """
        Get a brief description of this element.

        Returns
        -------
        str
        """
        return f"{cls._symbol}: {cls._name}"  # type: ignore

    @classmethod
    def get_default_values(cls, *args, **kwargs) -> Dict[str, float]:
        """
        Get the default values for this element's parameters as a dictionary.

        Returns
        -------
        Dict[str, float]
        """
        if not (args or kwargs):
            return cls._parameter_default_value.copy()

        results: Dict[str, float] = {}

        key: Any
        for key in set(list(args) + list(kwargs.keys())):
            results[key] = cls._parameter_default_value[key]

        return results

    @classmethod
    def get_default_value(cls, key: str) -> float:
        """
        Get the default value for a specific parameter.

        Parameters
        ----------
        key: str
            A key corresponding to a parameter.

        Returns
        -------
        float
        """
        return cls.get_default_values(key)[key]

    @classmethod
    def set_default_values(cls, *args, **kwargs):
        """
        Set the default values for this element's parameters.

        Parameters
        ----------
        *args
            Pairs of string keys and numeric values corresponding to parameters (e.g., `set_values("R", 1.0, "Y", 1e-9)`).

        **kwargs
            String keys and numeric values corresponding to parameters (e.g., `set_values(R=1.0, Y=1e-9)`).
        """
        pairs: dict = kwargs.copy()

        key: Any
        value: Any
        if args:
            if len(args) % 2 != 0:
                raise ValueError(f"Expected pairs of arguments instead of {args=}")

            args_list: List[Any] = list(args)
            while args_list:
                key = args_list.pop(0)
                value = args_list.pop(0)
                if key in pairs:
                    raise KeyError(
                        f"The key-value pair {key=} was already defined as a keyword argument"
                    )
                else:
                    pairs[key] = value

        for key, value in pairs.items():
            if key not in cls._parameter_default_value:
                raise KeyError(
                    f"Expected a key that exists in {cls._parameter_default_value.keys()} instead of {key=}"
                )

            cls._parameter_default_value[key] = float(value)

    @classmethod
    def get_symbol(cls) -> str:
        """
        Get the symbol for this element.
        The symbol is used to represent this type of element in a circuit description code.

        Returns
        -------
        str
        """
        return cls._symbol

    def _sympy(
        self,
        substitute: bool,
        identifier: int,
        values: Dict[str, float],
    ) -> Expr:
        return sympify(self._equation)

    def to_sympy(self, substitute: bool = False, identifier: int = -1) -> Expr:
        """
        Get the |Expr| object for the impedance of this element.

        Parameters
        ----------
        substitute: bool, optional
            Substitute the numeric values for the variables.

        identifier: int, optional
            The identifier of the element.

        Returns
        -------
        |Expr|
        """
        if not _is_boolean(substitute):
            raise TypeError(f"Expected a boolean instead of {substitute=}")

        if not _is_integer(identifier):
            raise TypeError(f"Expected an integer instead of {identifier=}")

        substitutions: Dict[str, Union[str, float]] = {}
        values: Dict[str, float] = self.get_values()

        key: str
        value: float
        for key, value in values.items():
            repl: Union[str, float]

            if not substitute:
                if self._label != "":
                    repl = f"{key}_{self._label}"
                elif identifier >= 0:
                    repl = f"{key}_{identifier}"
                else:
                    repl = f"{key}"

            elif isposinf(value):
                repl = "oo"

            elif isneginf(value):
                repl = "-oo"

            else:
                repl = value

            substitutions[key] = repl

        return self._sympy(
            substitute=substitute,
            identifier=identifier,
            values=values,
        ).subs(substitutions)

    def to_latex(self) -> str:
        """
        Get the equation for the impedance of this element as a LaTeX-compatible string.

        Returns
        -------
        str
        """
        return f"Z = {latex(self.to_sympy(substitute=False), imaginary_unit='j')}"

    def get_label(self) -> str:
        """
        Get the label assigned to a specific instance of the element.

        Returns
        -------
        str
        """
        return self._label

    def set_label(self, label: str) -> "Element":
        """
        Set the label assigned to a specific instance of the element.

        Parameters
        ----------
        label: str
            The new label.

        Returns
        -------
        Element
        """
        if not isinstance(label, str):
            raise TypeError(f"Expected a string instead of {label=}")

        label = label.strip()

        if label != "":
            if not all(map(str.isascii, label)):
                raise ValueError(
                    f"Expected the label to only contain ASCII characters instead of {label=}"
                )

            if all(map(str.isdigit, label)):
                raise ValueError(
                    f"Expected the label to not only contain digits instead of {label=}"
                )

        self._label = label

        return self

    def get_name(self) -> str:
        """
        Get the display name for this element, which consists of the element's symbol and its optional label.

        Returns
        -------
        str
        """
        if self._label == "":
            return self.get_symbol()

        return f"{self.get_symbol()}_{self._label}"

    def serialize(self, decimals: int = 12) -> str:
        return self.to_string(decimals=decimals)

    def to_string(self, decimals: int = -1) -> str:
        """
        Generates a string representation of the element.

        Parameters
        ----------
        decimals: int, optional
            The number of decimals used when formatting the current value and the limits for the element's parameters.
            -1 corresponds to no values being included in the output.

        Returns
        -------
        str
        """
        if not _is_integer(decimals):
            raise TypeError(f"Expected an integer instead of {decimals=}")

        if decimals < 0:
            return self.get_symbol()

        lower_limits: Dict[str, float] = self.get_lower_limits()
        upper_limits: Dict[str, float] = self.get_upper_limits()
        fixed_values: Dict[str, bool] = self.are_fixed()
        parameters: List[str] = []

        symbol: str
        value: float
        for symbol, value in self.get_values().items():
            lower: float = lower_limits[symbol]
            upper: float = upper_limits[symbol]
            fixed: bool = fixed_values[symbol]
            string: str = f"{symbol}=" + (f"%.{decimals}E") % value

            if fixed:
                string += "F"

            if isinf(lower):
                string += "/inf"
            else:
                string += (f"/%.{decimals}E") % lower

            if isinf(upper):
                string += "/inf"
            else:
                string += (f"/%.{decimals}E") % upper

            parameters.append(string)

        cdc: str = self.get_symbol() + "{" + ",".join(parameters)
        if self._label != "":
            cdc += f":{self._label}"

        return cdc + "}"

    def reset_parameters(self, *args, **kwargs):
        """
        Resets the value, lower limit, upper limit, and fixed state of one or more parameters.

        Parameters
        ----------
        *args
            One or more string keys corresponding to parameters.

        **kwargs
            String keys corresponding to parameters.
            The values can be anything.
        """
        self.set_values(**self.get_default_values(*args, **kwargs))
        self.set_lower_limits(**self.get_default_lower_limits(*args, **kwargs))
        self.set_upper_limits(**self.get_default_upper_limits(*args, **kwargs))
        self.set_fixed(**self.are_fixed_by_default(*args, **kwargs))

    def reset_parameter(self, key: str):
        """
        Resets the value, lower limit, upper limit, and fixed state of one parameter.

        Parameters
        ----------
        key: str
            A string key corresponding to a parameter.
        """
        self.set_values(key, self.get_default_value(key))
        self.set_lower_limits(key, self.get_default_lower_limit(key))
        self.set_upper_limits(key, self.get_default_upper_limit(key))
        self.set_fixed(key, self.is_fixed_by_default(key))

    def are_fixed(self, *args, **kwargs) -> Dict[str, bool]:
        """
        Get a dictionary that maps parameter keys to whether or not those parameters currently have fixed values.

        Parameters
        ----------
        *args
            String keys corresponding to parameters.

        **kwargs
            String keys corresponding to parameters.
            The values can be anything.

        Returns
        -------
        Dict[str, bool]
        """
        if not (args or kwargs):
            return self._parameter_fixed.copy()

        results: Dict[str, bool] = {}

        key: Any
        for key in set(list(args) + list(kwargs.keys())):
            results[key] = self._parameter_fixed[key]

        return results

    @classmethod
    def are_fixed_by_default(cls, *args, **kwargs) -> Dict[str, bool]:
        """
        Get a dictionary that maps parameter keys to whether or not those parameters have fixed values by default.

        Parameters
        ----------
        *args
            String keys corresponding to parameters.

        **kwargs
            String keys corresponding to parameters.
            The values can be anything.

        Returns
        -------
        Dict[str, bool]
        """
        if not (args or kwargs):
            return cls._parameter_default_fixed.copy()

        results: Dict[str, bool] = {}

        key: Any
        for key in set(list(args) + list(kwargs.keys())):
            results[key] = cls._parameter_default_fixed[key]

        return results

    def is_fixed(self, key: str) -> bool:
        """
        Get whether or not a specific parameter currently has a fixed value.

        Parameters
        ----------
        key: str
            A key corresponding to a parameter.

        Returns
        -------
        bool
        """
        return self.are_fixed(key)[key]

    @classmethod
    def is_fixed_by_default(cls, key: str) -> bool:
        """
        Get whether or not a specific parameter has a fixed value by default.

        Parameters
        ----------
        key: str
            A key corresponding to a parameter.

        Returns
        -------
        bool
        """
        return cls.are_fixed_by_default(key)[key]

    def set_fixed(self, *args, **kwargs) -> "Element":
        """
        Set parameters to have fixed values.

        Parameters
        ----------
        *args
            Pairs of string keys and boolean values corresponding to parameters (e.g., `set_fixed("R", True, "Y", False)`).

        **kwargs
            String keys and boolean values corresponding to parameters (e.g., `set_fixed(R=True, Y=False)`).

        Returns
        -------
        Element
        """
        pairs: dict = kwargs.copy()

        key: Any
        value: Any
        if args:
            if len(args) % 2 != 0:
                raise ValueError(f"Expected pairs of arguments instead of {args=}")

            args_list: List[Any] = list(args)
            while args_list:
                key = args_list.pop(0)
                value = args_list.pop(0)
                if key in pairs:
                    raise KeyError(
                        f"The key-value pair {key=} was already defined as a keyword argument"
                    )
                else:
                    pairs[key] = value

        for key, value in pairs.items():
            if key not in self._parameter_fixed:
                raise KeyError(
                    f"Expected a key that exists in {self._parameter_fixed} instead of {key=}"
                )

            if not _is_boolean(value):
                raise TypeError(f"Expected a boolean instead of {value=}")

            self._parameter_fixed[key] = value

        return self

    def get_lower_limits(self, *args, **kwargs) -> Dict[str, float]:
        """
        Get a dictionary that maps parameter keys to their current lower limits.

        Parameters
        ----------
        *args
            String keys corresponding to parameters.

        **kwargs
            String keys corresponding to parameters.
            The values can be anything.

        Returns
        -------
        Dict[str, float]
        """
        if not (args or kwargs):
            return self._parameter_lower_limit.copy()

        results: Dict[str, float] = {}

        key: Any
        for key in set(list(args) + list(kwargs.keys())):
            results[key] = self._parameter_lower_limit[key]

        return results

    def get_lower_limit(self, key: str) -> float:
        """
        Get the current lower limit for a specific parameter.

        Parameters
        ----------
        key: str
            A key corresponding to a parameter.

        Returns
        -------
        float
        """
        return self.get_lower_limits(key)[key]

    @classmethod
    def get_default_lower_limits(cls, *args, **kwargs) -> Dict[str, float]:
        """
        Get a dictionary that maps parameter keys to their default lower limits.

        Parameters
        ----------
        *args
            String keys corresponding to parameters.

        **kwargs
            String keys corresponding to parameters.
            The values can be anything.

        Returns
        -------
        Dict[str, float]
        """
        if not (args or kwargs):
            return cls._parameter_default_lower_limit.copy()

        results: Dict[str, float] = {}

        key: Any
        for key in set(list(args) + list(kwargs.keys())):
            results[key] = cls._parameter_default_lower_limit[key]

        return results

    @classmethod
    def get_default_lower_limit(cls, key: str) -> float:
        """
        Get the default lower limit for a specific parameter.

        Parameters
        ----------
        key: str
            A key corresponding to a parameter.

        Returns
        -------
        float
        """
        return cls.get_default_lower_limits(key)[key]

    def set_lower_limits(self, *args, **kwargs) -> "Element":
        """
        Set lower limits for parameters.
        Lower limits are used during circuit fitting.

        Parameters
        ----------
        *args
            Pairs of string keys and numeric values corresponding to parameters (e.g., `set_lower_limits("R", 1.0, "Y", 1e-9)`).

        **kwargs
            String keys and numeric values corresponding to parameters (e.g., `set_lower_limits(R=1.0, Y=1e-9)`)..

        Returns
        -------
        Element
        """
        pairs: dict = kwargs.copy()

        key: Any
        value: Any
        if args:
            if len(args) % 2 != 0:
                raise ValueError(f"Expected pairs of arguments instead of {args=}")

            args_list: List[Any] = list(args)
            while args_list:
                key = args_list.pop(0)
                value = args_list.pop(0)
                if key in pairs:
                    raise KeyError(
                        f"The key-value pair {key=} was already defined as a keyword argument"
                    )
                else:
                    pairs[key] = value

        for key, value in pairs.items():
            if key not in self._parameter_lower_limit:
                raise KeyError(
                    f"Expected a key that exists in {self._parameter_lower_limit} instead of {key=}"
                )

            value = float(value)
            if value >= self._parameter_upper_limit[key]:
                raise ValueError(
                    f"Expected the new value of {key=} ({value}) to be less than the current upper limit of {self._parameter_upper_limit[key]}"
                )

            if self._parameter_value[key] < value:
                self._parameter_value[key] = value

            self._parameter_lower_limit[key] = value

        return self

    def get_upper_limits(self, *args, **kwargs) -> Dict[str, float]:
        """
        Get a dictionary that maps parameter keys to their current upper limits.

        Parameters
        ----------
        *args
            String keys corresponding to parameters.

        **kwargs
            String keys corresponding to parameters.
            The values can be anything.

        Returns
        -------
        Dict[str, float]
        """
        if not (args or kwargs):
            return self._parameter_upper_limit.copy()

        results: Dict[str, float] = {}

        key: Any
        for key in set(list(args) + list(kwargs.keys())):
            results[key] = self._parameter_upper_limit[key]

        return results

    def get_upper_limit(self, key: str) -> float:
        """
        Get the current upper limit for a specific parameter.

        Parameters
        ----------
        key: str
            A key corresponding to a parameter.

        Returns
        -------
        float
        """
        return self.get_upper_limits(key)[key]

    @classmethod
    def get_default_upper_limits(cls, *args, **kwargs) -> Dict[str, float]:
        """
        Get a dictionary that maps parameter keys to their default upper limits.

        Parameters
        ----------
        *args
            String keys corresponding to parameters.

        **kwargs
            String keys corresponding to parameters.
            The values can be anything.

        Returns
        -------
        Dict[str, float]
        """
        if not (args or kwargs):
            return cls._parameter_default_upper_limit.copy()

        results: Dict[str, float] = {}

        key: Any
        for key in set(list(args) + list(kwargs.keys())):
            results[key] = cls._parameter_default_upper_limit[key]

        return results

    @classmethod
    def get_default_upper_limit(cls, key: str) -> float:
        """
        Get the default upper limit for a specific parameter.

        Parameters
        ----------
        key: str
            A key corresponding to a parameter.

        Returns
        -------
        float
        """
        return cls.get_default_upper_limits(key)[key]

    def set_upper_limits(self, *args, **kwargs) -> "Element":
        """
        Set upper limits for parameters.
        Upper limits are used during circuit fitting.

        Parameters
        ----------
        *args
            Pairs of string keys and numeric values corresponding to parameters (e.g., `set_upper_limits("R", 1.0, "Y", 1e-9)`).

        **kwargs
            String keys and numeric values corresponding to parameters (e.g., `set_upper_limits(R=1.0, Y=1e-9)`).

        Returns
        -------
        Element
        """
        pairs: dict = kwargs.copy()

        key: Any
        value: Any
        if args:
            if len(args) % 2 != 0:
                raise ValueError(f"Expected pairs of arguments instead of {args=}")

            args_list: List[Any] = list(args)
            while args_list:
                key = args_list.pop(0)
                value = args_list.pop(0)
                if key in pairs:
                    raise KeyError(
                        f"The key-value pair {key=} was already defined as a keyword argument"
                    )
                else:
                    pairs[key] = value

        for key, value in pairs.items():
            if key not in self._parameter_upper_limit:
                raise KeyError(
                    f"Expected a key that exists in {self._parameter_upper_limit.keys()} instead of {key=}"
                )

            value = float(value)
            if value <= self._parameter_lower_limit[key]:
                raise ValueError(
                    f"Expected the new value of {key=} ({value}) to be greater than the current lower limit of {self._parameter_lower_limit[key]}"
                )

            if self._parameter_value[key] > value:
                self._parameter_value[key] = value

            self._parameter_upper_limit[key] = value

        return self

    def get_values(self, *args, **kwargs) -> Dict[str, float]:
        """
        Get a dictionary that maps parameter keys to their current values.

        Parameters
        ----------
        *args
            String keys corresponding to parameters.

        **kwargs
            String keys corresponding to parameters.
            The values can be anything.

        Returns
        -------
        Dict[str, float]
        """
        if not (args or kwargs):
            return self._parameter_value.copy()

        results: Dict[str, float] = {}

        key: Any
        for key in set(list(args) + list(kwargs.keys())):
            results[key] = self._parameter_value[key]

        return results

    def get_value(self, key: str) -> float:
        """
        Get the current value for a specific parameter.

        Parameters
        ----------
        key: str
            A key corresponding to a parameter.

        Returns
        -------
        float
        """
        return self.get_values(key)[key]

    def set_values(self, *args, **kwargs) -> "Element":
        """
        Set values for parameters.

        Parameters
        ----------
        *args
            Pairs of string keys and numeric values corresponding to parameters (e.g., `set_values("R", 1.0, "Y", 1e-9)`).

        **kwargs
            String keys and numeric values corresponding to parameters (e.g., `set_values(R=1.0, Y=1e-9)`).

        Returns
        -------
        Element
        """
        pairs: dict = kwargs.copy()

        key: Any
        value: Any
        if args:
            if len(args) % 2 != 0:
                raise ValueError(f"Expected pairs of arguments instead of {args=}")

            args_list: List[Any] = list(args)
            while args_list:
                key = args_list.pop(0)
                value = args_list.pop(0)
                if key in pairs:
                    raise KeyError(
                        f"The key-value pair {key=} was already defined as a keyword argument"
                    )
                else:
                    pairs[key] = value

        for key, value in pairs.items():
            if key not in self._parameter_value:
                raise KeyError(
                    f"Expected a key that exists in {self._parameter_value.keys()} instead of {key=}"
                )

            self._parameter_value[key] = float(value)

        return self

    @classmethod
    def get_units(cls, *args, **kwargs) -> Dict[str, str]:
        """
        Get a dictionary that maps parameter keys to their corresponding units.

        Parameters
        ----------
        *args
            String keys corresponding to parameters.

        **kwargs
            String keys corresponding to parameters.
            The values can be anything.

        Returns
        -------
        Dict[str, str]
        """
        if not (args or kwargs):
            return cls._parameter_unit.copy()

        results: Dict[str, str] = {}

        key: Any
        for key in set(list(args) + list(kwargs.keys())):
            results[key] = cls._parameter_unit[key]

        return results

    @classmethod
    def get_unit(cls, key: str) -> str:
        """
        Get the unit for a specific parameter.

        Parameters
        ----------
        key: str
            String key for a parameter.

        Returns
        -------
        str
        """
        return cls.get_units(key)[key]

    @classmethod
    def get_value_descriptions(cls, *args, **kwargs) -> Dict[str, str]:
        """
        Get a dictionary that maps parameter keys to their corresponding descriptions.

        Parameters
        ----------
        *args
            String keys corresponding to parameters.

        **kwargs
            String keys corresponding to parameters.
            The values can be anything.

        Returns
        -------
        Dict[str, str]
        """
        if not (args or kwargs):
            return cls._parameter_description.copy()

        results: Dict[str, str] = {}

        key: Any
        for key in set(list(args) + list(kwargs.keys())):
            results[key] = cls._parameter_description[key]

        return results

    @classmethod
    def get_value_description(cls, key: str) -> str:
        """
        Get the description for a specific parameter.

        Parameters
        ----------
        key: str
            String key for a parameter.

        Returns
        -------
        str
        """
        return cls.get_value_descriptions(key)[key]

    @abstractmethod
    def _impedance(self, f: Frequencies, **kwargs) -> ComplexImpedances:
        """
        The method that performs the actual computation of the element's impedance at a given excitation frequency.
        This is the method that should be overridden by classes that implement circuit elements.

        Parameters
        ----------
        f: |Frequencies|
            The excitation frequencies in hertz.

        **kwargs
            The element parameters.

        Returns
        -------
        |ComplexImpedances|
        """
        pass

    def get_impedances(self, frequencies: Frequencies) -> ComplexImpedances:
        """
        Calculate the impedance of this element at multiple frequencies.

        Parameters
        ----------
        frequencies: |Frequencies|
            The excitation frequencies in hertz.

        Returns
        -------
        |ComplexImpedances|
        """
        return _calculate_impedances(self, frequencies)


class Connection(ABC):
    def __init__(self, elements: List[Union[Element, "Connection"]]):
        if not isinstance(elements, list) and all(
            map(lambda _: isinstance(_, Connection) or isinstance(_, Element), elements)
        ):
            raise TypeError(
                f"Expected a list of Element and/or Connection instances instead of {elements=}"
            )

        self._elements: List[Union[Element, "Connection"]] = elements

    def __copy__(self) -> "Connection":
        return type(self)([_.__copy__() for _ in self._elements])

    def __deepcopy__(self, memo: dict) -> "Connection":
        ident: int = id(self)
        copy: Optional["Connection"] = memo.get(ident)

        if copy is None:
            copy = type(self)([_.__deepcopy__(memo) for _ in self._elements])
            memo[ident] = copy

        return copy

    def __iter__(self) -> List[Union[Element, "Connection"]]:
        return iter(self._elements)

    def __repr__(self) -> str:
        return f"TODO ({hex(id(self))})"

    def __len__(self) -> int:
        return len(self._elements)

    def __contains__(self, item: Union[Element, "Connection"]) -> bool:
        element_or_connection: Union[Element, "Connection"]
        for element_or_connection in self._elements:
            if element_or_connection is item:
                return True
            elif isinstance(element_or_connection, Connection):
                if item in element_or_connection:
                    return True
            elif isinstance(element_or_connection, Container):
                if item in element_or_connection:
                    return True

        return False

    def _get_all_items_recursive(self) -> List[Union[Element, "Connection"]]:
        items: List[Union[Element, "Connection"]] = []

        element_or_connection: Union[Element, "Connection"]
        for element_or_connection in self._elements:
            if isinstance(element_or_connection, Connection):
                items.extend(element_or_connection._get_all_items_recursive())
            else:
                items.append(element_or_connection)

        return items

    def _get_elements_recursive(self) -> List[Element]:
        connection_type: Type["Connection"] = type(self).__bases__[0]
        queue: List[Union[Element, "Connection"]] = self._get_all_items_recursive()
        elements: List[Element] = []

        while queue:
            element: Union[Element, "Connection"] = queue.pop(0)

            if isinstance(element, connection_type):
                queue.extend(element._get_elements_recursive())
                continue

            if not isinstance(element, Element):
                raise TypeError(f"Expected an Element instead of {element=}")

            if element not in elements:
                elements.append(element)

            if isinstance(element, Container):
                queue.extend(
                    filter(
                        lambda connection: connection is not None,
                        element.get_subcircuits().values(),
                    )
                )

        if len(elements) != len(set(elements)):
            raise ValueError("Detected duplicate elements")

        return elements

    def generate_element_identifiers(self, running: bool) -> Dict[Element, int]:
        """
        Generate a mapping of elements to their corresponding integer identifiers.

        Parameters
        ----------
        running: bool
            If true, then the identifiers are simply a running count from 0 to N. Primarily intended for use within pyimpspec.
            If false, then the identifiers represent what number instance of a particular element type an element is (e.g., the second resistor of three resistors would have 2 as its identifier). Primarily intended for use in anything that most users would see (e.g., circuit diagrams and parameter tables).

        Returns
        -------
        Dict[Element, int]
        """
        if running:
            return {
                element: i for i, element in enumerate(self._get_elements_recursive())
            }

        elements: List[Element] = self._get_elements_recursive()
        identifiers: Dict[Element, int] = {}

        element: Element
        counts: Dict[str, int] = {element.get_symbol(): 0 for element in elements}

        for element in elements:
            symbol: str = element.get_symbol()
            i: int = counts[symbol] + 1
            counts[symbol] = i
            identifiers[element] = i

        return identifiers

    def contains(
        self,
        element_or_connection: Union[Element, "Connection"],
        top_level: bool = False,
    ) -> bool:
        """
        Check if this connection contains a specific Element or Connection instance.

        Parameters
        ----------
        element_or_connection: Union[Element, Connection]
            The Element or Connection instance to check for.

        top_level: bool, optional
            Whether to only check in the current Connection instance instead of also checking in any nested Connection instances.

        Returns
        -------
        bool
        """
        if top_level:
            return any((item is element_or_connection for item in self._elements))

        return element_or_connection in self

    def append(self, element_or_connection: Union[Element, "Connection"]):
        """
        Append an element/connection to this connection.

        Parameters
        ----------
        element_or_connection: Union[Element, Connection]
        """
        self._elements.append(element_or_connection)

    def extend(self, elements_or_connections: List[Union[Element, "Connection"]]):
        """
        Extend this connection with a list of elements and/or connections.

        Parameters
        ----------
        elements_or_connections: List[Union[Element, Connection]]
        """
        self._elements.extend(elements_or_connections)

    def insert(self, i: int, element_or_connection: Union[Element, "Connection"]):
        """
        Insert an element/connection into position i of this connection.

        Parameters
        ----------
        i: int

        element_or_connection: Union[Element, Connection]
        """
        self._elements.insert(i, element_or_connection)

    def remove(self, element_or_connection: Union[Element, "Connection"]):
        """
        Remove a specific element/connection from this connection.

        Parameters
        ----------
        element_or_connection: Union[Element, Connection]
        """
        self._elements.remove(element_or_connection)

    def pop(self, i: int) -> Union[Element, "Connection"]:
        """
        Pop the element/connection at position i in this connection.

        Parameters
        ----------
        i: int

        Returns
        -------
        Union[Element, Connection]
        """
        return self._elements.pop(i)

    def clear(self):
        """
        Remove all elements and/or connections from this connection.
        """
        self._elements.clear()

    def index(
        self,
        element_or_connection: Union[Element, "Connection"],
        start: int = 0,
        end: int = -1,
    ) -> int:
        """
        Get the index of an element/connection in this connection.

        Parameters
        ----------
        element_or_connection: Union[Element, Connection]

        start: int, optional

        end: int, optional

        Returns
        -------
        int
        """
        if end < 0:
            end = len(self._elements)

        return self._elements.index(element_or_connection, start, end)

    def count(self) -> int:
        """
        Get the number of elements and/or connections in this connection.

        Returns
        -------
        int
        """
        return len(self._elements)

    def serialize(self, decimals: int = 12) -> str:
        return self.to_string(decimals=decimals)

    @abstractmethod
    def to_string(self, decimals: int = -1) -> str:
        """
        Generate the circuit description code for this connection.

        Parameters
        ----------
        decimals: int, optional
            The number of decimals used in numeric values.

        Returns
        -------
        str
        """
        pass

    def get_connections(self, recursive: bool = True) -> List["Connection"]:
        """
        Get the connections in this connection.

        Parameters
        ----------
        recursive: bool, optional
            If True and this Connection contains other Connection instances, then all nested Connect instances are returned.
            If False, then only the Connection instances within the top level of this Connection are returned.

        Returns
        -------
        List[Connection]
        """
        if recursive:
            connections: List["Connection"] = []
            for item in self._elements:
                if isinstance(item, Connection):
                    connections.append(item)
                    connections.extend(item.get_connections(recursive=recursive))

            return connections

        return [item for item in self._elements if isinstance(item, Connection)]

    def get_elements(self, recursive: bool = True) -> List[Element]:
        """
        Get the elements in this connection.

        Parameters
        ----------
        recursive: bool, optional
            If True and this Connection contains other Connection instances, then all nested elements are returned.
            If False, then only the Element instances within the top level of this Connection are returned.

        Returns
        -------
        List[Element]
        """
        if recursive:
            elements: List[Element] = []
            for item in self._elements:
                if isinstance(item, Connection):
                    elements.extend(item.get_elements(recursive=recursive))
                else:
                    elements.append(item)

            return elements

        return [item for item in self._elements if isinstance(item, Element)]

    @abstractmethod
    def _impedance(self, f: Frequencies) -> ComplexImpedances:
        """
        The method that performs the actual computation of the connection's impedance at a given excitation frequency.
        This is the method that should be overridden by classes that implement circuit connections.

        Parameters
        ----------
        f: |Frequencies|
            The excitation frequency in hertz.

        Returns
        -------
        |ComplexImpedances|
        """
        pass

    def get_impedances(self, frequencies: Frequencies) -> ComplexImpedances:
        """
        Calculate the impedance of this connection at multiple frequencies.

        Parameters
        ----------
        frequencies: |Frequencies|
            The excitation frequencies in hertz.

        Returns
        -------
        |ComplexImpedances|
        """
        return _calculate_impedances(self, frequencies)

    @abstractmethod
    def to_stack(self, stack: List[Tuple[str, Union[Element, "Connection"]]]):
        pass

    @abstractmethod
    def to_sympy(
        self,
        substitute: bool = False,
        identifiers: Optional[Dict[Element, int]] = None,
    ) -> Expr:
        """
        Get the |Expr| object for the impedance of this connection.

        Parameters
        ----------
        substitute: bool, optional
            Substitute the numeric values for the variables.

        identifiers: Optional[Dict[Element, int]], optional
            A mapping of elements to their identifiers.

        Returns
        -------
        |Expr|
        """
        pass

    def to_latex(self) -> str:
        """
        Get the equation for the impedance of this connection as a LaTeX-compatible string.

        Returns
        -------
        str
        """
        return f"Z = {latex(self.to_sympy(substitute=False), imaginary_unit='j')}"

    def to_drawing(self) -> "Drawing":  # noqa: F821
        # Dynamically set to pyimpspec.circuit.diagrams.to_drawing
        raise NotImplementedError()

    def to_circuitikz(self) -> str:
        # Dynamically set to pyimpspec.circuit.diagrams.to_circuitikz
        raise NotImplementedError()

    def get_element_name(
        self,
        element: Element,
        identifiers: Optional[Dict[Element, int]] = None,
    ) -> str:
        """
        Get the name of the element with consideration for any overriding label assigned to the element or the type-specific count in the context of this connection.

        Parameters
        ----------
        element: Element
            The element whose name should be returned.

        identifiers: Optional[Dict[Element, int]], optional
            The identifiers to use when determining the name of the provided element.

        Returns
        -------
        str
        """
        if element not in self:
            raise ValueError(f"This connection does not contain {element=}")

        name: str = element.get_name()
        symbol: str = element.get_symbol()

        if name != symbol:
            return name

        if identifiers is None:
            identifiers = self.generate_element_identifiers(running=False)

        if element not in identifiers:
            raise ValueError(f"{element=} does not exist in {identifiers=}")

        return f"{symbol}_{identifiers[element]}"


class Container(Element):
    """
    A subclass of :class:`~pyimpspec.circuit.base.Element` that adds support for subcircuits.
    """

    _subcircuit_unit: Dict[str, str] = {}
    _subcircuit_description: Dict[str, str] = {}
    _subcircuit_default_value: Dict[str, Optional[Connection]] = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._subcircuit_value: Dict[str, Optional[Connection]] = {}

        key: str
        value: Optional[Connection]
        for key, value in self._subcircuit_default_value.items():
            if key not in kwargs:
                self._subcircuit_value[key] = (
                    deepcopy(value) if value is not None else value
                )
            else:
                value = kwargs[key]
                if isinstance(value, Connection) or value is None:
                    self._subcircuit_value[key] = value
                else:
                    raise TypeError(
                        f"Expected the {key=} subcircuit to be a Connection or None instead of {value=}"
                    )

    def __contains__(self, item: Union[Element, Connection]) -> bool:
        con: Optional[Connection]
        for con in self._subcircuit_value.values():
            if con is None:
                continue
            if item in con:
                return True

        return False

    def __copy__(self) -> "Container":
        return (
            type(self)(
                **self.get_values(),
                **{
                    k: (v.__copy__() if v is not None else v)
                    for k, v in self.get_subcircuits().items()
                },
            )
            .set_lower_limits(**self.get_lower_limits())
            .set_upper_limits(**self.get_upper_limits())
            .set_fixed(**self.are_fixed())
            .set_label(self._label)
        )

    def __deepcopy__(self, memo: dict) -> "Container":
        ident: int = id(self)
        copy: Optional["Container"] = memo.get(ident)

        if copy is None:
            copy = (
                type(self)(
                    **self.get_values(),
                    **{
                        k: (v.__deepcopy__(memo) if v is not None else None)
                        for k, v in self.get_subcircuits().items()
                    },
                )
                .set_lower_limits(**self.get_lower_limits())
                .set_upper_limits(**self.get_upper_limits())
                .set_fixed(**self.are_fixed())
                .set_label(self._label)
            )
            memo[ident] = copy

        if copy is None:
            raise ValueError("Expected {copy=} to not be None")

        return copy

    def _sympy(
        self,
        substitute: bool,
        identifiers: Dict[Element, int],
        values: Dict[str, float],
        subcircuits: Dict[str, Optional[Connection]],
    ) -> Expr:
        return sympify(self._equation)

    def to_sympy(
        self,
        substitute: bool = False,
        identifiers: Optional[Dict[Element, int]] = None,
    ) -> Expr:
        """
        Get the SymPy expression for the impedance of this element.

        Parameters
        ----------
        substitute: bool, optional
            Substitute the numeric values for the variables.

        identifiers: Optional[Dict[Element, int]], optional
            A mapping of elements to their identifiers.

        Returns
        -------
        `sympy.Expr`_
        """
        if not _is_boolean(substitute):
            raise TypeError(f"Expected a boolean instead of {substitute=}")

        if identifiers is None:
            identifiers = self.generate_element_identifiers(running=False)

        if not isinstance(identifiers, dict):
            raise TypeError(
                f"Expected identifiers to be a dictionary instead of {identifiers=}"
            )

        identifier: int = identifiers[self]
        substitutions: Dict[str, Union[str, float, Expr]] = {}
        values: Dict[str, float] = self.get_values()
        subcircuits: Dict[str, Optional[Connection]] = self.get_subcircuits()

        key: str
        value: float
        for key, value in values.items():
            repl: Union[str, float, Expr]
            if not substitute:
                if self._label != "":
                    repl = f"{key}_{self._label}"
                elif identifier >= 0:
                    repl = f"{key}_{identifier}"
                else:
                    repl = f"{key}"
            elif isposinf(value):
                repl = "oo"
            elif isneginf(value):
                repl = "-oo"
            else:
                repl = value

            substitutions[key] = repl

        con: Optional[Connection]
        for key, con in subcircuits.items():
            if con is None:
                repl = "oo"
            else:
                repl = con.to_sympy(substitute=substitute, identifiers=identifiers)

            substitutions[key] = repl

        return self._sympy(
            substitute=substitute,
            identifiers=identifiers,
            values=values,
            subcircuits=subcircuits,
        ).subs(substitutions)

    def generate_element_identifiers(self, running: bool) -> Dict[Element, int]:
        """
        Generate a mapping of elements to their corresponding integer identifiers.

        Parameters
        ----------
        running: bool
            If true, then the identifiers are simply a running count from 0 to N. Primarily intended for use within pyimpspec.
            If false, then the identifiers represent what number instance of a particular element type an element is (e.g., the second resistor of three resistors would have 2 as its identifier). Primarily intended for use in anything that most users would see (e.g., circuit diagrams and parameter tables).

        Returns
        -------
        Dict[Element, int]
        """
        identifiers: Dict[Element, int] = {}
        subcircuits: List[Connection] = []
        counts: Dict[str, int] = {}

        def process_element(element: Element):
            if running:
                identifiers[element] = len(identifiers)
            else:
                symbol: str = element.get_symbol()
                if symbol not in counts:
                    counts[symbol] = 0
                i: int = counts[symbol] + 1
                counts[symbol] = i
                identifiers[element] = i

            if isinstance(element, Container):
                subcircuits.extend(
                    filter(
                        lambda connection: connection is not None,
                        element.get_subcircuits().values(),
                    )
                )

        process_element(self)
        counts[self.get_symbol()] = 0
        identifiers[self] = -1

        connection: Optional[Connection]
        for connection in self.get_subcircuits().values():
            if connection is None:
                continue
            [process_element(_) for _ in connection.get_elements(recursive=True)]

        return identifiers

    def to_string(self, decimals: int = -1) -> str:
        cdc: str = super().to_string(decimals=decimals)

        if decimals < 0 or not cdc.endswith("}"):
            return cdc

        index: int = cdc.find("{") + 1
        if index < 1:
            raise ValueError(f"Expected the CDC to begin with '{{' instead of {cdc=}")

        ending: str
        cdc, ending = cdc[:index], cdc[index:]

        key: str
        for key in sorted(self._subcircuit_value.keys()):
            con: Optional[Connection] = self._subcircuit_value[key]
            if con is None:
                cdc += f"{key}=open, "
            elif len(con.get_elements()) == 0:
                cdc += f"{key}=short, "
            else:
                cdc += f"{key}={con.to_string(decimals=decimals)}, "

        if ending[0] == ":" or ending[0] == "}":
            cdc = cdc[:-2]

        return cdc + ending

    @classmethod
    def get_units(cls, *args, **kwargs) -> Dict[str, str]:
        """
        Get a dictionary that maps parameter and/or subcircuit keys to their corresponding units.

        Parameters
        ----------
        *args
            String keys corresponding to parameters and/or subcircuits.

        **kwargs
            String keys corresponding to parameters and/or subcircuits.
            The values can be anything.

        Returns
        -------
        Dict[str, str]
        """
        results: Dict[str, str] = {}

        if not (args or kwargs):
            results.update(cls._parameter_unit)
            results.update(cls._subcircuit_unit)

        else:
            key: Any
            for key in set(list(args) + list(kwargs.keys())):
                if key in cls._parameter_unit:
                    results[key] = cls._parameter_unit[key]
                elif key in cls._subcircuit_unit:
                    results[key] = cls._subcircuit_unit[key]
                else:
                    raise KeyError(
                        f"Expected a key that exists in either {cls._parameter_unit.keys()} or {cls._subcircuit_unit} instead of {key=}"
                    )

        return results

    @classmethod
    def get_unit(cls, key: str) -> str:
        """
        Get the unit for a specific parameter or subcircuit.

        Parameters
        ----------
        key: str
            String key for a parameter or subcircuit.

        Returns
        -------
        str
        """
        return cls.get_units(key)[key]

    @classmethod
    def get_subcircuit_descriptions(cls, *args, **kwargs) -> Dict[str, str]:
        """
        Get a dictionary that maps subcircuit keys to their corresponding descriptions.

        Parameters
        ----------
        *args
            String keys corresponding to subcircuits.

        **kwargs
            String keys corresponding to subcircuits.
            The values can be anything.

        Returns
        -------
        Dict[str, str]
        """
        if not (args or kwargs):
            return cls._subcircuit_description.copy()

        results: Dict[str, str] = {}

        key: Any
        for key in set(list(args) + list(kwargs.keys())):
            results[key] = cls._subcircuit_description[key]

        return results

    @classmethod
    def get_subcircuit_description(cls, key: str) -> str:
        """
        Get the description for a specific subcircuit.

        Parameters
        ----------
        key: str
            String key for a subcircuit.

        Returns
        -------
        str
        """
        return cls.get_subcircuit_descriptions(key)[key]

    @classmethod
    def get_default_subcircuits(
        cls, *args, **kwargs
    ) -> Dict[str, Optional[Connection]]:
        """
        Get the default values for this element's parameters as a dictionary.

        Returns
        -------
        Dict[str, Optional[Connection]]
        """
        if not (args or kwargs):
            return cls._subcircuit_default_value.copy()

        results: Dict[str, Optional[Connection]] = {}

        key: Any
        for key in set(list(args) + list(kwargs.keys())):
            results[key] = cls._subcircuit_default_value[key]

        return results

    @classmethod
    def get_default_subcircuit(cls, key: str) -> Optional[Connection]:
        """
        Get the default value for a specific subcircuit.

        Parameters
        ----------
        key: str
            A key corresponding to a subcircuit.

        Returns
        -------
        float
        """
        return cls.get_default_subcircuits(key)[key]

    def get_subcircuits(self, *args, **kwargs) -> Dict[str, Optional[Connection]]:
        """
        Get a dictionary that maps subcircuit keys to their current values.

        Parameters
        ----------
        *args
            String keys corresponding to subcircuits.

        **kwargs
            String keys corresponding to subcircuits.
            The values can be anything.

        Returns
        -------
        Dict[str, Optional[Connection]]
        """
        if not (args or kwargs):
            return self._subcircuit_value.copy()

        results: Dict[str, Optional[Connection]] = {}

        key: Any
        for key in set(list(args) + list(kwargs.keys())):
            results[key] = self._subcircuit_value[key]

        return results

    def get_subcircuit(self, key: str) -> Optional[Connection]:
        """
        Get the current subcircuit matching the given key.

        Parameters
        ----------
        key: str
            A key corresponding to a subcircuit.

        Returns
        -------
        Optional[Connection]
        """
        return self.get_subcircuits(key)[key]

    def set_subcircuits(self, *args, **kwargs) -> "Element":
        """
        Set values for subcircuits.

        Parameters
        ----------
        *args
            Pairs of string keys and numeric values corresponding to subcircuits(e.g., `set_subcircuits("X", None, "Y", Series([Resistor()]), "Z", Series([]))`).

        **kwargs
            String keys and numeric values corresponding to parameters (e.g., `set_values(R=None, Y=Series([Resistor()]), Z=Series([]))`).

        Returns
        -------
        Element
        """
        pairs: dict = kwargs.copy()

        key: Any
        value: Any
        if args:
            if len(args) % 2 != 0:
                raise ValueError(f"Expected pairs of arguments instead of {args=}")

            args_list: List[Any] = list(args)
            while args_list:
                key = args_list.pop(0)
                value = args_list.pop(0)
                if key in pairs:
                    raise KeyError(
                        f"The key-value pair {key=} was already defined as a keyword argument"
                    )
                else:
                    pairs[key] = value

        for key, value in pairs.items():
            if key not in self._subcircuit_value:
                raise KeyError(
                    f"Expected a key that exists in {self._subcircuit_value.keys()} instead of {key=}"
                )

            if isinstance(value, Connection) or value is None:
                self._subcircuit_value[key] = value
            else:
                raise TypeError(
                    f"Expected the {key=} subcircuit to be a Connection or None instead of {value=}"
                )

        return self
