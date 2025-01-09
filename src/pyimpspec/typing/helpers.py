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

from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    IO,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)
from numpy import (
    array,
    bool_,
    complex128,
    complexfloating,
    float64,
    floating,
    int64,
    integer,
    issubdtype,
    ndarray,
)
from numpy.typing import NDArray


def _is_numpy_array(obj: Any) -> bool:
    return isinstance(obj, ndarray)


def _is_boolean(obj: Any) -> bool:
    return isinstance(obj, bool) or isinstance(obj, bool_)


def _is_integer(obj: Any) -> bool:
    return isinstance(obj, int) or issubdtype(type(obj), integer)


def _is_integer_array(obj: Any) -> bool:
    if not _is_numpy_array(obj):
        return False
    else:
        return issubdtype(obj.dtype, integer)


def _is_integer_list(obj: Any) -> bool:
    if not isinstance(obj, list):
        return False
    else:
        return all(map(_is_integer, obj))


def _cast_to_integer_array(obj: Any) -> NDArray[int64]:
    if _is_numpy_array(obj):
        return obj.astype(int64)
    elif isinstance(obj, list):
        return array(obj, dtype=int64)
    else:
        return array([obj], dtype=int64)


def _is_floating(obj: Any) -> bool:
    return isinstance(obj, float) or issubdtype(type(obj), floating)


def _is_floating_array(obj: Any) -> bool:
    if not _is_numpy_array(obj):
        return False
    else:
        return issubdtype(obj.dtype, floating)


def _is_floating_list(obj: Any) -> bool:
    if not isinstance(obj, list):
        return False
    else:
        return all(map(_is_floating, obj))


def _cast_to_floating_array(obj: Any) -> NDArray[float64]:
    if _is_numpy_array(obj):
        return obj.astype(float64)
    elif isinstance(obj, list):
        return array(obj, dtype=float64)
    else:
        return array([obj], dtype=float64)


def _is_complex(obj: Any) -> bool:
    return isinstance(obj, complex) or issubdtype(type(obj), complexfloating)


def _is_complex_array(obj: Any) -> bool:
    if not _is_numpy_array(obj):
        return False
    else:
        return issubdtype(obj.dtype, complexfloating)


def _is_complex_list(obj: Any) -> bool:
    if not isinstance(obj, list):
        return False
    else:
        return all(map(_is_complex, obj))


def _cast_to_complex_array(obj: Any) -> NDArray[complex128]:
    if _is_numpy_array(obj):
        return obj.astype(complex128)
    elif isinstance(obj, list):
        return array(obj, dtype=complex128)
    else:
        return array([obj], dtype=complex128)
