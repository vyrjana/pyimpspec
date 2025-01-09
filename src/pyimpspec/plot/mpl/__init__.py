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

from .bht_scores import plot_bht_scores
from .bode import plot_bode
from .circuit import plot_circuit
from .real_imaginary import plot_real_imaginary
from .data import plot_data
from .drt import plot_drt
from .fit import plot_fit
from .gamma import plot_gamma
from .imaginary import plot_imaginary
from .kramers_kronig import (
    plot_num_RC_suggestion,
    plot_num_RC_suggestion_method,
    plot_kramers_kronig_tests,
    plot_log_F_ext,
)
from .magnitude import plot_magnitude
from .pseudo_chisqr import plot_pseudo_chisqr
from .nyquist import plot_nyquist
from .phase import plot_phase
from .real import plot_real
from .residuals import plot_residuals


def show(*args, **kwargs):
    """
    Wrapper for calling ``matplotlib.pyplot.show()``.
    """
    import matplotlib.pyplot as plt

    plt.show(*args, **kwargs)
