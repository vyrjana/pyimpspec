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

from dataclasses import dataclass
from typing import (
    Dict,
    Optional,
    Union,
)
from numpy import (
    array,
    full,
    inf,
    isinf,
    pi,
    where,
)
from sympy import (
    Expr,
    oo,
    sympify,
    cosh as sympy_cosh,
    coth as sympy_coth,
    sinh as sympy_sinh,
    sqrt as sympy_sqrt,
)
from .functions import (
    cosh,
    coth,
    sinh,
    sqrt,
    tanh,
)
from .base import (
    Connection,
    Container,
    Element,
)
from .series import Series
from .resistor import Resistor
from .constant_phase_element import ConstantPhaseElement
from .registry import (
    ContainerDefinition,
    ElementDefinition,
    ParameterDefinition,
    SubcircuitDefinition,
    register_element,
)
from pyimpspec.typing import (
    ComplexImpedance,
    ComplexImpedances,
    Frequencies,
)


# References
# - 1. Bisquert (2000), DOI: 10.1039/B001708F
# - 2. Bisquert et al. (2000), DOI: 10.1021/jp993148h
# - 3. Bisquert et al. (2000), DOI: 10.1016/S1388-2481(00)00089-8


class TransmissionLineModelBlockingOpen(Element):
    # Model (b) from [1]
    def _impedance(
        self,
        f: Frequencies,
        R_i: float,
        Y: float,
        n: float,
        L: float,
    ) -> ComplexImpedances:
        w: Frequencies = 2 * pi * f
        w_L: float = 1 / ((R_i * Y * L**2) ** (1 / n))  # Eq. 30 [1]
        alpha: ComplexImpedances = (1j * w / w_L) ** (n / 2)

        # Eq. 31 [1]
        return R_i / alpha * coth(alpha)


register_element(
    ElementDefinition(
        Class=TransmissionLineModelBlockingOpen,
        symbol="Tlmbo",
        name="Blocking porous electrode, open",
        description="Transmission line, blocking porous electrode with perfectly reflective inner boundary. Model (b) in DOI:10.1039/B001708F.",
        equation="R_i*(I*2*pi*f/(((R_i*Y*L^2)^(1/n))^-1))^(-n/2) * coth((I*2*pi*f/(((R_i*Y*L^2)^(1/n))^-1))^(n/2))",
        parameters=[
            ParameterDefinition(
                symbol="R_i",
                unit="ohm/m",
                description="Ionic resistance",
                value=1.0,
                lower_limit=0.0,
                upper_limit=inf,
                fixed=False,
            ),
            ParameterDefinition(
                symbol="Y",
                unit="S*s^n/m",
                description="Interfacial 'admittance'",
                value=5e-3,
                lower_limit=1e-24,
                upper_limit=inf,
                fixed=False,
            ),
            ParameterDefinition(
                symbol="n",
                unit="",
                description="Phase angle in -pi/2 radians (interface)",
                value=0.8,
                lower_limit=0.0,
                upper_limit=1.0,
                fixed=False,
            ),
            ParameterDefinition(
                symbol="L",
                unit="m",
                description="Pore length",
                value=1.0,
                lower_limit=1e-24,
                upper_limit=inf,
                fixed=True,
            ),
        ],
    ),
)


class TransmissionLineModelBlockingCPE(Element):
    # Model (d) [1] with modifications
    def _impedance(
        self,
        f: Frequencies,
        R_i: float,
        Y: float,
        n: float,
        Y_B: float,
        n_B: float,
        L: float,
    ) -> ComplexImpedances:
        w: Frequencies = 2 * pi * f
        w_L: float = 1 / ((R_i * Y * L**2) ** (1 / n))  # Eq. 30 [1]
        Z_B: ComplexImpedances = 1 / (Y_B * (1j * w) ** n_B)  # Eq. 35 [1]
        alpha: ComplexImpedances = 1j * w / w_L
        beta: ComplexImpedances = alpha ** (n / 2)
        ct_beta: ComplexImpedances = coth(beta)

        # Eq. 34 [1] with modification to account for interfacial CPE
        # alpha**n instead of just alpha
        return (
            R_i
            * (1 + Z_B / R_i * ct_beta * beta)
            / (alpha**n * Z_B / R_i + ct_beta * beta)
        )


register_element(
    ElementDefinition(
        Class=TransmissionLineModelBlockingCPE,
        symbol="Tlmbq",
        name="Blocking porous electrode, CPE",
        description="Transmission line, blocking porous electrode with reflective inner boundary. Model (d) in DOI:10.1039/B001708F.",
        equation="R_i*(1 + ((Y_B*(I*2*pi*f)^(n_B))^-1)/R_i * coth((I*2*pi*f/(((R_i*Y*L^2)^(1/n))^-1))^(n/2)) * (I*2*pi*f/(((R_i*Y*L^2)^(1/n))^-1))^(n/2)) / ((I*2*pi*f/(((R_i*Y*L^2)^(1/n))^-1))^n * ((Y_B*(I*2*pi*f)^(n_B))^-1)/R_i + coth((I*2*pi*f/(((R_i*Y*L^2)^(1/n))^-1))^(n/2)) * (I*2*pi*f/(((R_i*Y*L^2)^(1/n))^-1))^(n/2))",
        parameters=[
            ParameterDefinition(
                symbol="R_i",
                unit="ohm/m",
                description="Ionic resistance",
                value=1.0,
                lower_limit=0.0,
                upper_limit=inf,
                fixed=False,
            ),
            ParameterDefinition(
                symbol="Y",
                unit="S*s^n/m",
                description="Interfacial 'admittance'",
                value=5e-3,
                lower_limit=1e-24,
                upper_limit=inf,
                fixed=False,
            ),
            ParameterDefinition(
                symbol="n",
                unit="",
                description="Phase angle in -pi/2 radians (interface)",
                value=0.8,
                lower_limit=0.0,
                upper_limit=1.0,
                fixed=False,
            ),
            ParameterDefinition(
                symbol="Y_B",
                unit="S*s^n/m",
                description="Inner boundary 'admittance'",
                value=10e-3,
                lower_limit=1e-24,
                upper_limit=inf,
                fixed=False,
            ),
            ParameterDefinition(
                symbol="n_B",
                unit="",
                description="Phase angle in -pi/2 radians (inner boundary)",
                value=0.7,
                lower_limit=0.0,
                upper_limit=1.0,
                fixed=False,
            ),
            ParameterDefinition(
                symbol="L",
                unit="m",
                description="Pore length",
                value=1.0,
                lower_limit=1e-24,
                upper_limit=inf,
                fixed=True,
            ),
        ],
    ),
)


class TransmissionLineModelBlockingShort(Element):
    # Model (d) [1] with modifications
    def _impedance(
        self,
        f: Frequencies,
        R_i: float,
        Y: float,
        n: float,
        L: float,
    ) -> ComplexImpedances:
        w: Frequencies = 2 * pi * f
        w_L: float = 1 / ((R_i * Y * L**2) ** (1 / n))  # Eq. 30 [1]
        alpha: ComplexImpedances = (1j * w / w_L) ** (n / 2)

        # Eq. 34 [1] with Z_B == 0
        return R_i / (coth(alpha) * alpha)


register_element(
    ElementDefinition(
        Class=TransmissionLineModelBlockingShort,
        symbol="Tlmbs",
        name="Blocking porous electrode, short",
        description="Transmission line, blocking porous electrode with absorbing inner boundary. Model (d) in DOI:10.1039/B001708F with a shorted Z_B (i.e., Z_B == 0).",
        equation="R_i / (coth((I*2*pi*f/(((R_i*Y*L^2)^(1/n))^-1))^(n/2)) * (I*2*pi*f/(((R_i*Y*L^2)^(1/n))^-1))^(n/2))",
        parameters=[
            ParameterDefinition(
                symbol="R_i",
                unit="ohm/m",
                description="Ionic resistance",
                value=1.0,
                lower_limit=0.0,
                upper_limit=inf,
                fixed=False,
            ),
            ParameterDefinition(
                symbol="Y",
                unit="S*s^n/m",
                description="Interfacial 'admittance'",
                value=5e-3,
                lower_limit=1e-24,
                upper_limit=inf,
                fixed=False,
            ),
            ParameterDefinition(
                symbol="n",
                unit="",
                description="Phase angle in -pi/2 radians (interface)",
                value=0.8,
                lower_limit=0.0,
                upper_limit=1.0,
                fixed=False,
            ),
            ParameterDefinition(
                symbol="L",
                unit="m",
                description="Pore length",
                value=1.0,
                lower_limit=1e-24,
                upper_limit=inf,
                fixed=True,
            ),
        ],
    ),
)


class TransmissionLineModelNonblockingOpen(Element):
    # Model (f) [1] with corrected form from [2]
    def _impedance(
        self,
        f: Frequencies,
        R_i: float,
        R_ct: float,
        Y: float,
        n: float,
        L: float,
    ) -> ComplexImpedances:
        w: Frequencies = 2 * pi * f
        w_L: float = 1 / ((R_i * Y * L**2) ** (1 / n))  # Eq. 30 [1]
        w_ct: float = 1 / ((R_ct * Y) ** (1 / n))  # Eq. 42 [1]
        alpha: ComplexImpedances = 1 + (1j * w / w_ct) ** n

        # Eq. 42 [2] (corrected form of eq. 40 [1] that has a typo)
        return sqrt((R_i * R_ct) / alpha) * coth((w_ct / w_L) ** (n / 2) * sqrt(alpha))


register_element(
    ElementDefinition(
        Class=TransmissionLineModelNonblockingOpen,
        symbol="Tlmno",
        name="Non-blocking porous electrode, open",
        description="Transmission line, non-blocking porous electrode with perfectly reflective inner boundary. Also known as 'Bisquert, open'. Model (f) in DOI:10.1039/B001708F with corrected form from DOI:10.1021/jp993148h.",
        equation="(R_i*R_ct / (1+(I*2*pi*f/(((R_ct*Y)^(1/n))^-1))^n))^(1/2) * coth((((R_ct/L*Y)^(1/n))^-1/((R_i*Y*L^2)^(1/n))^-1)^(n/2) * (1+(I*2*pi*f/(((R_ct*Y)^(1/n))^-1))^n)^(1/2))",
        parameters=[
            ParameterDefinition(
                symbol="R_i",
                unit="ohm/m",
                description="Ionic resistance",
                value=1.0,
                lower_limit=0.0,
                upper_limit=inf,
                fixed=False,
            ),
            ParameterDefinition(
                symbol="R_ct",
                unit="ohm*m",
                description="Interfacial charge transfer resistance",
                value=3.0,
                lower_limit=0.0,
                upper_limit=inf,
                fixed=False,
            ),
            ParameterDefinition(
                symbol="Y",
                unit="S*s^n/m",
                description="Interfacial 'admittance'",
                value=5e-3,
                lower_limit=1e-24,
                upper_limit=inf,
                fixed=False,
            ),
            ParameterDefinition(
                symbol="n",
                unit="",
                description="Phase angle in -pi/2 radians (interface)",
                value=0.8,
                lower_limit=0.0,
                upper_limit=1.0,
                fixed=False,
            ),
            ParameterDefinition(
                symbol="L",
                unit="m",
                description="Pore length",
                value=1.0,
                lower_limit=1e-24,
                upper_limit=inf,
                fixed=True,
            ),
        ],
    ),
)


class TransmissionLineModelNonblockingCPE(Element):
    # Model (g) [1]
    def _impedance(
        self,
        f: Frequencies,
        R_i: float,
        R_ct: float,
        Y: float,
        n: float,
        R_B: float,
        Y_B: float,
        n_B: float,
        L: float,
    ) -> ComplexImpedances:
        w: Frequencies = 2 * pi * f
        alpha: ComplexImpedances = Y * R_ct * (1j * w) ** n
        beta: ComplexImpedances = Y_B * R_B * (1j * w) ** n_B
        # Eq. 43 [1] and eq. 39 [1] inserted into eq. 18 [1]
        Zint: ComplexImpedances = sqrt(R_ct / (R_i * (alpha + 1)))
        ct_LZint: ComplexImpedances = coth(L / Zint)

        return (
            R_ct
            * (R_B * ct_LZint + R_i * Zint * (beta + 1))
            / (R_B * Zint * (alpha + 1) + R_ct * (beta + 1) * ct_LZint)
        )


register_element(
    ElementDefinition(
        Class=TransmissionLineModelNonblockingCPE,
        symbol="Tlmnq",
        name="Non-blocking porous electrode, CPE",
        description="Transmission line, non-blocking porous electrode with reflective inner boundary. Model (g) in DOI:10.1039/B001708F.",
        equation="R_ct * (R_B * coth(L/sqrt(R_ct/(R_i*(Y*R_ct*(I*2*pi*f)^n+1)))) + R_i*sqrt(R_ct/(R_i*(Y*R_ct*(I*2*pi*f)^n+1))) * (Y_B*R_B*(I*2*pi*f)^n_B+1)) / (R_B*sqrt(R_ct/(R_i*(Y*R_ct*(I*2*pi*f)^n+1))) * (Y*R_ct*(I*2*pi*f)^n+1) + R_ct*(Y_B*R_B*(I*2*pi*f)^n_B+1) * coth(L/sqrt(R_ct/(R_i*(Y*R_ct*(I*2*pi*f)^n+1)))))",
        parameters=[
            ParameterDefinition(
                symbol="R_i",
                unit="ohm/m",
                description="Ionic resistance",
                value=1.0,
                lower_limit=0.0,
                upper_limit=inf,
                fixed=False,
            ),
            ParameterDefinition(
                symbol="R_ct",
                unit="ohm*m",
                description="Interfacial charge transfer resistance",
                value=3.0,
                lower_limit=0.0,
                upper_limit=inf,
                fixed=False,
            ),
            ParameterDefinition(
                symbol="Y",
                unit="S*s^n/m",
                description="Interfacial 'admittance'",
                value=5e-3,
                lower_limit=1e-24,
                upper_limit=inf,
                fixed=False,
            ),
            ParameterDefinition(
                symbol="n",
                unit="",
                description="Phase angle in -pi/2 radians (interface)",
                value=0.8,
                lower_limit=0.0,
                upper_limit=1.0,
                fixed=False,
            ),
            ParameterDefinition(
                symbol="R_B",
                unit="ohm*m",
                description="Inner boundary charge transfer resistance",
                value=5.0,
                lower_limit=0.0,
                upper_limit=inf,
                fixed=False,
            ),
            ParameterDefinition(
                symbol="Y_B",
                unit="S*s^n/m",
                description="Inner boundary 'admittance'",
                value=100e-3,
                lower_limit=1e-24,
                upper_limit=inf,
                fixed=False,
            ),
            ParameterDefinition(
                symbol="n_B",
                unit="",
                description="Phase angle in -pi/2 radians (inner boundary)",
                value=0.7,
                lower_limit=0.0,
                upper_limit=1.0,
                fixed=False,
            ),
            ParameterDefinition(
                symbol="L",
                unit="m",
                description="Pore length",
                value=1.0,
                lower_limit=1e-24,
                upper_limit=inf,
                fixed=True,
            ),
        ],
    ),
)


class TransmissionLineModelNonblockingShort(Element):
    # Model (g) [1] with modifications
    def _impedance(
        self,
        f: Frequencies,
        R_i: float,
        R_ct: float,
        Y: float,
        n: float,
        L: float,
    ) -> ComplexImpedances:
        w: Frequencies = 2 * pi * f
        # Eq. 43 [1] and eq. 39 [1] inserted into eq. 18 [1] and Z_B == 0
        Zint: ComplexImpedances = sqrt(R_ct / (R_i * (Y * R_ct * (1j * w) ** n + 1)))

        return R_i * Zint * tanh(L / Zint)


register_element(
    ElementDefinition(
        Class=TransmissionLineModelNonblockingShort,
        symbol="Tlmns",
        name="Non-blocking porous electrode, short",
        description="Transmission line, non-blocking porous electrode with absorbing inner boundary. Also known as 'Bisquert, short'. Model (g) in DOI:10.1039/B001708F with a shorted Z_B (i.e., Z_B == 0).",
        equation="R_i*sqrt(R_ct/(R_i*(Y*R_ct*(I*2*pi*f)^n+1))) * tanh(L/sqrt(R_ct/(R_i*(Y*R_ct*(I*2*pi*f)^n+1))))",
        parameters=[
            ParameterDefinition(
                symbol="R_i",
                unit="ohm/m",
                description="Ionic resistance",
                value=1.0,
                lower_limit=0.0,
                upper_limit=inf,
                fixed=False,
            ),
            ParameterDefinition(
                symbol="R_ct",
                unit="ohm*m",
                description="Interfacial charge transfer resistance",
                value=3.0,
                lower_limit=0.0,
                upper_limit=inf,
                fixed=False,
            ),
            ParameterDefinition(
                symbol="Y",
                unit="S*s^n/m",
                description="Interfacial 'admittance'",
                value=5e-3,
                lower_limit=1e-24,
                upper_limit=inf,
                fixed=False,
            ),
            ParameterDefinition(
                symbol="n",
                unit="",
                description="Phase angle in -pi/2 radians (interface)",
                value=0.8,
                lower_limit=0.0,
                upper_limit=1.0,
                fixed=False,
            ),
            ParameterDefinition(
                symbol="L",
                unit="m",
                description="Pore length",
                value=1.0,
                lower_limit=1e-24,
                upper_limit=inf,
                fixed=True,
            ),
        ],
    ),
)


@dataclass
class Subcircuit:
    impedances: ComplexImpedances
    is_open: bool
    is_short: bool
    expr: Union[Expr, int] = None

    def update_expr(
        self,
        connection: Optional[Connection],
        substitute: bool,
        identifiers: Dict[Element, int],
    ):
        if self.is_open:
            if connection is None:
                self.expr = oo
            else:
                raise TypeError(
                    f"Expected connection to be None instead of {connection=}"
                )

        elif self.is_short:
            if connection is not None:
                self.expr = 0
            else:
                raise TypeError(
                    f"Expected connection to not be None instead of {connection=}"
                )

        else:
            if connection is not None:
                self.expr = connection.to_sympy(
                    substitute=substitute,
                    identifiers=identifiers,
                )
            else:
                raise TypeError(
                    f"Expected connection to not be None instead of {connection=}"
                )


def _evaluate_subcircuit(
    con: Optional[Connection],
    f: Frequencies,
    open_connection: Optional[ComplexImpedances],
) -> Subcircuit:
    if con is None:
        if open_connection is not None:
            return Subcircuit(
                impedances=open_connection,
                is_open=True,
                is_short=False,
            )
        else:
            raise TypeError(
                f"Expected open_connection to not be None instead of {open_connection=}"
            )

    Z: ComplexImpedances = con._impedance(f)

    Z_is_inf = isinf(Z)
    if Z_is_inf.any():
        if Z_is_inf.all():
            return Subcircuit(
                impedances=Z,
                is_open=True,
                is_short=False,
            )
        else:
            raise ValueError(f"Expected all impedances to be infinite instead of {Z=}")

    return Subcircuit(
        impedances=Z,
        is_open=False,
        is_short=where(Z == 0.0)[0].size == f.size,
    )


class TransmissionLineModel(Container):
    def _impedance(
        self,
        f: Frequencies,
        X_1: Optional[Connection],
        X_2: Optional[Connection],
        Z_A: Optional[Connection],
        Z_B: Optional[Connection],
        Zeta: Optional[Connection],
        L: float,
    ) -> ComplexImpedances:
        open_connection: Optional[ComplexImpedances] = None
        if any(map(lambda _: _ is None, (X_1, X_2, Z_A, Z_B, Zeta))):
            open_connection = full(
                f.shape,
                inf,
                dtype=ComplexImpedance,
            )

        x1: Subcircuit = _evaluate_subcircuit(X_1, f, open_connection)
        x2: Subcircuit = _evaluate_subcircuit(X_2, f, open_connection)
        za: Subcircuit = _evaluate_subcircuit(Z_A, f, open_connection)
        zb: Subcircuit = _evaluate_subcircuit(Z_B, f, open_connection)
        ze: Subcircuit = _evaluate_subcircuit(Zeta, f, open_connection)

        if x1.is_open:
            raise NotImplementedError("X_1 cannot be open!")
        elif x2.is_open:
            raise NotImplementedError("X_2 cannot be open!")
        elif x1.is_short and x2.is_short:
            raise NotImplementedError(
                "Both X_1 and X_2 cannot be short at the same time!"
            )
        elif ze.is_open:
            raise NotImplementedError("Zeta cannot be open!")
        elif ze.is_short:
            raise NotImplementedError("Zeta cannot be short!")

        lm: ComplexImpedances = sqrt(ze.impedances / (x1.impedances + x2.impedances))
        cs: ComplexImpedances = cosh(L / lm)
        ct: ComplexImpedances = coth(L / lm)
        s: ComplexImpedances = sinh(L / lm)

        x: Subcircuit
        if za.is_open and zb.is_open:
            # Both boundaries are perfectly reflecting (i.e., open)
            if x1.is_short or x2.is_short:
                return self._eq20((x2 if x1.is_short else x1).impedances, lm, ct)

            return self._eq8(x1.impedances, x2.impedances, L, lm, s, ct)

        elif za.is_open or zb.is_open:
            # One of the boundaries is perfectly reflecting (i.e., open)
            z: Subcircuit = zb if za.is_open else za
            if x1.is_short or x2.is_short:
                # One of the phases is a short (or has a much smaller impedance
                # than the other)
                x = x2 if x1.is_short else x1
                if zb.is_short if za.is_open else za.is_short:
                    # The other boundary is a short
                    return self._eq18_variant(x.impedances, lm, ct)

                return self._eq18(x.impedances, z.impedances, lm, ct)

            return self._eq17(x1.impedances, x2.impedances, z.impedances, L, lm, cs, s)

        elif x1.is_short or x2.is_short:
            # One of the phases is a short (or has a much smaller impedance than
            # the other)
            # Eq. 19 [1]
            return self._eq19(
                (x2 if x1.is_short else x1).impedances,
                za.impedances,
                zb.impedances,
                lm,
                ct,
            )

        return self._eq16(
            x1.impedances,
            x2.impedances,
            za.impedances,
            zb.impedances,
            L,
            lm,
            cs,
            s,
        )

    def _eq8(
        self,
        x1: ComplexImpedances,
        x2: ComplexImpedances,
        L: float,
        lm: ComplexImpedances,
        s: ComplexImpedances,
        ct: ComplexImpedances,
    ) -> ComplexImpedances:
        # Eq. 8 [1]
        return (
            x1 * x2 / (x1 + x2) * (L + 2 * lm / s)
            + lm * (x1**2 + x2**2) / (x1 + x2) * ct
        )

    def _eq16(
        self,
        x1: ComplexImpedances,
        x2: ComplexImpedances,
        za: ComplexImpedances,
        zb: ComplexImpedances,
        L: float,
        lm: ComplexImpedances,
        cs: ComplexImpedances,
        s: ComplexImpedances,
    ) -> ComplexImpedances:
        # Eq. 16 [1]
        return (
            (x1 + x2) ** -1
            * (
                lm * (x1 + x2) * s
                + (za + zb) * cs
                + za * zb * s * (lm * (x1 + x2)) ** -1
            )
            ** -1
            * (
                L * lm * x1 * x2 * (x1 + x2) * s
                + x1 * (lm * x1 * s + L * x2 * cs) * za
                + x2 * (lm * x2 * s + L * x1 * cs) * zb
                + (x1 + x2) ** -1
                * (2 * x1 * x2 + (x1**2 + x2**2) * cs + L / lm * x1 * x2 * s)
                * za
                * zb
            )
        )

    def _eq17(
        self,
        x1: ComplexImpedances,
        x2: ComplexImpedances,
        Z: ComplexImpedances,
        L: float,
        lm: ComplexImpedances,
        cs: ComplexImpedances,
        s: ComplexImpedances,
    ) -> ComplexImpedances:
        # Eq. 17 [1]
        return (
            (x1 + x2) ** -1
            * (cs + Z * s / (lm * (x1 + x2))) ** -1
            * (
                x1 * (lm * x1 * s + L * x2 * cs)
                + (x1 + x2) ** -1
                * (2 * x1 * x2 + (x1**2 + x2**2) * cs + L / lm * x1 * x2 * s)
                * Z
            )
        )

    def _eq18(
        self,
        X: ComplexImpedances,
        Z: ComplexImpedances,
        lm: ComplexImpedances,
        ct: ComplexImpedances,
    ) -> ComplexImpedances:
        # Eq. 18 [1]
        return X * (1 + Z / (lm * X) * ct) / (Z / (lm**2 * X) + ct / lm)

    def _eq18_variant(
        self,
        X: ComplexImpedances,
        lm: ComplexImpedances,
        ct: ComplexImpedances,
    ) -> ComplexImpedances:
        # Based on eq. 18 [1]
        return X * lm / ct

    def _eq19(
        self,
        X: ComplexImpedances,
        za: ComplexImpedances,
        zb: ComplexImpedances,
        lm: ComplexImpedances,
        ct: ComplexImpedances,
    ) -> ComplexImpedances:
        return (
            1 / za
            + (X * (1 + zb / (lm * X) * ct) / (zb / (lm**2 * X) + ct / lm)) ** -1
        ) ** -1

    def _eq20(
        self,
        X: ComplexImpedances,
        lm: ComplexImpedances,
        ct: ComplexImpedances,
    ) -> ComplexImpedances:
        # Eq. 20 [1]
        return X * lm * ct

    def _sympy(
        self,
        substitute: bool,
        identifiers: Dict[Element, int],
        values: Dict[str, float],
        subcircuits: Dict[str, Optional[Connection]],
    ) -> Expr:
        L = sympify("L")
        if substitute:
            L.subs("L", values["L"])

        X_1: Optional[Connection] = subcircuits["X_1"]
        X_2: Optional[Connection] = subcircuits["X_2"]
        Z_A: Optional[Connection] = subcircuits["Z_A"]
        Z_B: Optional[Connection] = subcircuits["Z_B"]
        Zeta: Optional[Connection] = subcircuits["Zeta"]
        f: Frequencies = array([1.0])
        open_connection: ComplexImpedances = full(
            f.shape,
            inf,
            dtype=ComplexImpedance,
        )

        x1: Subcircuit = _evaluate_subcircuit(X_1, f, open_connection)
        x2: Subcircuit = _evaluate_subcircuit(X_2, f, open_connection)
        za: Subcircuit = _evaluate_subcircuit(Z_A, f, open_connection)
        zb: Subcircuit = _evaluate_subcircuit(Z_B, f, open_connection)
        ze: Subcircuit = _evaluate_subcircuit(Zeta, f, open_connection)

        if x1.is_open:
            raise NotImplementedError("X_1 cannot be open!")
        elif x2.is_open:
            raise NotImplementedError("X_2 cannot be open!")
        elif x1.is_short and x2.is_short:
            raise NotImplementedError(
                "Both X_1 and X_2 cannot be short at the same time!"
            )
        elif ze.is_open:
            raise NotImplementedError("Zeta cannot be open!")
        elif ze.is_short:
            raise NotImplementedError("Zeta cannot be short!")

        x1.update_expr(
            connection=X_1,
            substitute=substitute,
            identifiers=identifiers,
        )
        x2.update_expr(
            connection=X_2,
            substitute=substitute,
            identifiers=identifiers,
        )
        za.update_expr(
            connection=Z_A,
            substitute=substitute,
            identifiers=identifiers,
        )
        zb.update_expr(
            connection=Z_B,
            substitute=substitute,
            identifiers=identifiers,
        )
        ze.update_expr(
            connection=Zeta,
            substitute=substitute,
            identifiers=identifiers,
        )

        lm = sympy_sqrt(ze.expr / (x1.expr + x2.expr))
        Cs = sympy_cosh(L / lm)
        Ct = sympy_coth(L / lm)
        S = sympy_sinh(L / lm)

        x: Subcircuit
        z: Subcircuit
        if za.is_open and zb.is_open:
            # Both boundaries are perfectly reflecting (i.e., open)
            if x1.is_short or x2.is_short:
                # Eq. 20 [1]
                x = x2 if x1.is_short else x1
                return lm * x.expr * Ct
            else:
                # Eq. 8 [1]
                return (x1.expr * x2.expr) / (x1.expr + x2.expr) * (
                    L + (2 * lm) / S
                ) + lm * (x1.expr**2 + x2.expr**2) / (x1.expr + x2.expr) * Ct

        elif za.is_open or zb.is_open:
            # One of the boundaries is perfectly reflecting (i.e., open)
            z = zb if za.is_open else za
            if x1.is_short or x2.is_short:
                # One of the phases is a short (or has a much smaller impedance
                # than the other)
                x = x2 if x1.is_short else x1
                if z.is_short:
                    # The other boundary is a short
                    # Based on eq. 18 [1]
                    return x.expr * lm / Ct

                else:
                    # Eq. 18 [1]
                    return (
                        x.expr
                        * (1 + z.expr / (lm * x.expr) * Ct)
                        / (z.expr / (lm**2 * x.expr) + Ct / lm)
                    )

            else:
                # Eq. 17 [1]
                return (
                    (x1.expr + x2.expr) ** -1
                    * (Cs + z.expr * S / (lm * (x1.expr + x2.expr))) ** -1
                    * (
                        x1.expr * (lm * x1.expr * S + L * x2.expr * Cs)
                        + (x1.expr + x2.expr) ** -1
                        * (
                            2 * x1.expr * x2.expr
                            + (x1.expr**2 + x2.expr**2) * Cs
                            + L / lm * x1.expr * x2.expr * S
                        )
                        * z.expr
                    )
                )

        elif x1.is_short or x2.is_short:
            # Eq. 19 [1]
            x = x2 if x1.is_short else x1
            return (
                za.expr**-1
                + (
                    x.expr
                    * (1 + zb.expr / (lm * x.expr) * Ct)
                    / (zb.expr / (lm**2 * x.expr) + Ct / lm)
                )
                ** -1
            ) ** -1

        # Eq. 16 [1]
        return (
            (x1.expr + x2.expr) ** -1
            * (
                lm * (x1.expr + x2.expr) * S
                + (za.expr + zb.expr) * Cs
                + za.expr * zb.expr * S / (lm * (x1.expr + x2.expr))
            )
            ** -1
            * (
                L * lm * x1.expr * x2.expr * (x1.expr + x2.expr) * S
                + x1.expr * (lm * x1.expr * S + L * x2.expr * Cs) * za.expr
                + x2.expr * (lm * x2.expr * S + L * x1.expr * Cs) * zb.expr
                + (x1.expr + x2.expr) ** -1
                * (
                    2 * x1.expr * x2.expr
                    + (x1.expr**2 + x2.expr**2) * Cs
                    + L / lm * x1.expr * x2.expr * S
                )
                * za.expr
                * zb.expr
            )
        )


register_element(
    ContainerDefinition(
        Class=TransmissionLineModel,
        symbol="Tlm",
        name="Transmission line model, general",
        description="""A general model for a transmission line. The units of the subcircuits' parameters vary depending on the connections. Note the pore length parameter, L, for this element and how it may affect the aforementioned units. The equation that is used depends on the configuration of subcircuits that have been chosen. See DOI:10.1039/B001708F for more information about this element.

|no-equation|""",
        equation=(
            "(X_1+X_2)^-1 * (lm*(X_1+X_2)*S + (Z_A+Z_B)*C+1/(lm*(X_1+X_2))*Z_A*Z_B*S)^-1 * (L*lm*X_1*X_2*(X_1+X_2)*S + X_1*(lm*X_1*S+L*X_2*C)*Z_A + X_2*(lm*X_2*S+L*X_1*C)*Z_B + 1/(X_1+X_2)*(2*X_1*X_2+(X_1^2+X_2^2)*C + L/lm*X_1*X_2*S)*Z_A*Z_B)".replace(
                "S", "sinh(L/lm)"
            )
            .replace("C", "cosh(L/lm)")
            .replace("lm", "sqrt(Zeta/(X_1+X_2))")
        ),
        parameters=[
            ParameterDefinition(
                symbol="L",
                unit="m",
                description="Pore length",
                value=1.0,
                lower_limit=1e-24,
                upper_limit=inf,
                fixed=True,
            ),
        ],
        subcircuits=[
            SubcircuitDefinition(
                symbol="X_1",
                unit="",
                description="The impedance of the liquid phase (i.e., ionic conductivity in the pore)",
                value=Series([Resistor(R=1.0)]),
            ),
            SubcircuitDefinition(
                symbol="X_2",
                unit="",
                description="The impedance of the solid phase (i.e., electronic conductivity of the film)",
                value=Series([]),
            ),
            SubcircuitDefinition(
                symbol="Z_A",
                unit="",
                description="The impedance at the outer boundary of the porous electrode (i.e., at the electrolyte-film interface outside of the pore)",
                value=None,
            ),
            SubcircuitDefinition(
                symbol="Z_B",
                unit="",
                description="The impedance at the inner boundary of the porous electrode (i.e., at the electrolyte-substrate interface in the pore)",
                value=None,
            ),
            SubcircuitDefinition(
                symbol="Zeta",
                unit="",
                description="The interfacial impedance (i.e., at the electrolyte-film interface of the pore wall)",
                value=Series([ConstantPhaseElement(Y=5e-3, n=0.8)]),
            ),
        ],
    ),
)
