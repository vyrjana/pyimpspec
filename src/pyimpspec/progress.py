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

from typing import (
    Callable,
    Dict,
)


_CALLBACKS: Dict[int, Callable] = {}
_COUNTER: int = 0


def register(callback: Callable) -> int:
    """
    Register a callback function that should be invoked when information about progress is emitted.
    The callback function should have `*args` and `**kwargs` as its arguments.
    An integer identifier is returned and this value can be used to unregister the callback function.

    Parameters
    ----------
    callback: Callable

    Returns
    -------
    int
    """
    assert callable(callback)
    global _COUNTER
    global _CALLBACKS
    _COUNTER += 1
    _CALLBACKS[_COUNTER] = callback
    return _COUNTER


def unregister(identifier: int) -> bool:
    """
    Unregister a callback function based on the identifier that was returned when the callback function was registered.
    Returns True if the callback function was successfully unregistered.

    Parameters
    ----------
    identifier: int

    Returns
    -------
    bool
    """
    global _CALLBACKS
    assert isinstance(identifier, int) and identifier > 0
    if identifier in _CALLBACKS:
        del _CALLBACKS[identifier]
        return True
    return False


def _update(*args, **kwargs):
    callback: Callable
    for callback in _CALLBACKS.values():
        callback(*args, **kwargs)


_RECENT_PROGRESS: float = -1.0


def _update_every_N_percent(i: int, total: int = 1, N: float = 1.0, *args, **kwargs):
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
        _update(progress=_RECENT_PROGRESS, *args, **kwargs)
    if i >= total:
        _RECENT_PROGRESS = -1.0


class Progress:
    """
    Context manager class that can be used to emit information related to progress status.
    The *args and **kwargs parameters are passed on to the callback functions.

    Parameters
    ----------
    message: str
        The status message to emit.

    total: int, optional
        The total number of steps, which is used to emit a percentage.

    *args

    **kwargs
    """

    def __init__(self, message: str, total: int = 1, *args, **kwargs):
        self._i: int = 0
        self._message: str = message
        self._total: int = total
        self._args: tuple = args
        self._kwargs: dict = kwargs

    def __enter__(self) -> "Progress":
        self._update()
        return self

    def __exit__(self, *args, **kwargs):
        self.increment()

    def _update(self):
        _update_every_N_percent(
            i=self._i,
            total=self._total,
            message=self._message,
            *self._args,
            **self._kwargs
        )

    def set(self, i: int):
        """
        Set the progress in terms of steps (i / total).

        Parameters
        ----------
        i: int
            The step index.
        """
        assert i <= self._total, (i, self._total)
        self._i = i
        self._update()

    def increment(self, step: int = 1):
        """
        Increment the progress by a step size ((i + step) / total).

        Parameters
        ----------
        step: int, optional
            The size of the step to take when called.
        """
        self._i += step
        assert self._i <= self._total, (self._i, self._total)
        self._update()

    def set_message(self, message: str, i: int = -1, total: int = -1):
        """
        Set the status message (and progress and total).

        Parameters
        ----------
        message: str
            The new message.

        i: int, optional
            The new step index.

        total: int, optional
            The new total number of steps
        """
        self._message = message
        if i >= 0:
            self._i = 0
        if total >= 0:
            assert total > 0, total
            self._total = total
        self._update()
