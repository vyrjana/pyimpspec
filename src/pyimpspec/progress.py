# pyimpspec is licensed under the GPLv3 or later (https://www.gnu.org/licenses/gpl-3.0.html).
# Copyright 2022 pyimpspec developers
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
    Optional,
)


_CALLBACK: Optional[Callable] = None


def register(callback: Callable):
    assert callable(callback)
    global _CALLBACK
    _CALLBACK = callback


def update(*args, **kwargs):
    if not callable(_CALLBACK):
        return
    _CALLBACK(*args, **kwargs)


_RECENT_PROGRESS: float = -1.0


def update_every_N_percent(i: int, total: int = 1, N: float = 10.0, *args, **kwargs):
    global _RECENT_PROGRESS
    if i == 0:
        _RECENT_PROGRESS = -1.0
    step: float = N / 100.0
    progress: float = i / total
    if _RECENT_PROGRESS < 0.0 or progress >= _RECENT_PROGRESS + step:
        if _RECENT_PROGRESS < 0.0:
            _RECENT_PROGRESS = 0.0
        else:
            _RECENT_PROGRESS += step
        update(progress=_RECENT_PROGRESS, *args, **kwargs)
    if i >= total:
        _RECENT_PROGRESS = -1.0
