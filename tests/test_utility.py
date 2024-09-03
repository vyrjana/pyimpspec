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
from numpy import inf
from pyimpspec import (
    get_default_num_procs,
    set_default_num_procs,
)
from pyimpspec.analysis.utility import _interpolate
from pyimpspec.typing import Frequencies


class Utility(TestCase):
    def test_default_num_procs(self):
        set_default_num_procs(num_procs=1)
        self.assertEqual(get_default_num_procs(), 1)

        set_default_num_procs(num_procs=99)
        self.assertEqual(get_default_num_procs(), 99)

        set_default_num_procs(num_procs=-1)
        self.assertNotEqual(get_default_num_procs(), 99)
        self.assertNotEqual(get_default_num_procs(), -1)

    def test_interpolate(self):
        # Too few frequencies
        with self.assertRaises(ValueError):
            _interpolate([5.2], 2)

        # Invalid num_per_decade value
        with self.assertRaises(ValueError):
            _interpolate([5.2, 2.1], 0)

        # Invalid num_per_decade type
        with self.assertRaises(TypeError):
            _interpolate([1, 1000], 10.0)

        # The lowest frequency is zero
        with self.assertRaises(ValueError):
            _interpolate([0.0, 5.2, 2.1], 10)

        # Negative frequency
        with self.assertRaises(ValueError):
            _interpolate([-1.0, 5.2, 2.1], 10)

        # The highest frequency is infinity
        with self.assertRaises(ValueError):
            _interpolate([326.1, inf], 10)

        # Valid cases
        for args in (
            (1e5, 1e-2, 10, 71),
            (1e3, 1e-1, 9, 37),
            (1.24346e4, 8.23563e-2, 7, 36),
        ):
            max_f: float
            min_f: float
            num_per_decade: int
            num_points: int
            max_f, min_f, num_per_decade, num_points = args
            f: Frequencies = _interpolate([min_f, max_f], num_per_decade)

            self.assertAlmostEqual(max(f), max_f)
            self.assertAlmostEqual(min(f), min_f)
            self.assertEqual(len(f), num_points)
