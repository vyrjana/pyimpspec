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
    Callable,
    List,
)
from pyimpspec.plot import mpl
from pyimpspec.plot.mpl.helpers import (
    _UNFILLED_MARKERS as UNFILLED_MARKERS,
    _get_marker_color_args,
)
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib

matplotlib.use("Agg")

if len(UNFILLED_MARKERS) == 0:
    _get_marker_color_args("o", "blue")
    assert len(UNFILLED_MARKERS) > 0


def check_mpl_return_values(self, figure, axes):
    plt.close()
    self.assertTrue(type(figure) is Figure)
    self.assertIsInstance(axes, list)
    self.assertTrue(len(axes) > 0)
    self.assertTrue(all(map(lambda _: isinstance(_, Axes), axes)))


primitive_mpl_plotters: List[Callable] = [
    mpl.plot_real,
    mpl.plot_imaginary,
    mpl.plot_real_imaginary,
    mpl.plot_magnitude,
    mpl.plot_phase,
    mpl.plot_bode,
    mpl.plot_nyquist,
]
