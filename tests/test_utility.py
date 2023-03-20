# pyimpspec is licensed under the GPLv3 or later (https://www.gnu.org/licenses/gpl-3.0.html).
# Copyright 2023 pyimpspec developers
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
from pyimpspec import (
    get_default_num_procs,
    set_default_num_procs,
)


class Utility(TestCase):
    def test_default_num_procs(self):
        set_default_num_procs(num_procs=1)
        self.assertEqual(get_default_num_procs(), 1)
        set_default_num_procs(num_procs=99)
        self.assertEqual(get_default_num_procs(), 99)
        set_default_num_procs(num_procs=-1)
        self.assertNotEqual(get_default_num_procs(), 99)
        self.assertNotEqual(get_default_num_procs(), -1)
