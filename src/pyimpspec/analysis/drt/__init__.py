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

from typing import List
from pyimpspec.data import DataSet
from pyimpspec.exceptions import DRTError
from .result import (
    DRTPeaks,
    DRTPeak,
    DRTResult,
)
from .tr_nnls import (
    TRNNLSResult,
    calculate_drt_tr_nnls,
)
from .tr_rbf import (
    TRRBFResult,
    calculate_drt_tr_rbf,
)
from .bht import (
    BHTResult,
    calculate_drt_bht,
)
from .mrq_fit import (
    MRQFitResult,
    calculate_drt_mrq_fit,
)
from .lm import (
    LMResult,
    calculate_drt_lm,
)


_METHODS: List[str] = [
    "bht",
    "lm",
    "mrq-fit",
    "tr-nnls",
    "tr-rbf",
]


def calculate_drt(
    data: DataSet,
    method: str = "tr-nnls",
    **kwargs,
) -> DRTResult:
    """
    Calculates the distribution of relaxation times (DRT) for a given data set using one of the supported methods (see the 'method' parameter below for more details).

    Parameters
    ----------
    data: DataSet
        The data set to use in the calculations.

    method: str, optional
        Valid values include:

        - "bht"
        - "lm"
        - "mrq-fit"
        - "tr-nnls"
        - "tr-rbf"

    **kwargs
        Additional keyword arguments are passed to the chosen method's function.
        See the documentation for those functions for more information about their parameters.

    Returns
    -------
    DRTResult
    """
    if not isinstance(method, str):
        raise TypeError(f"Expected a string instead of {method=}")
    elif method not in _METHODS:
        raise ValueError(
            f"Unsupported method: '{method}'! Valid value include: '"
            + "', '".join(_METHODS)
            + "'."
        )

    if (
        method == "tr-nnls"
        and kwargs.get("mode") is not None
        and kwargs["mode"] == "complex"
    ):
        kwargs["mode"] = "real"

    return {
        "bht": calculate_drt_bht,
        "lm": calculate_drt_lm,
        "mrq-fit": calculate_drt_mrq_fit,
        "tr-nnls": calculate_drt_tr_nnls,
        "tr-rbf": calculate_drt_tr_rbf,
    }[method](data=data, **kwargs)
