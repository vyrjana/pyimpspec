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

from pyimpspec.typing.helpers import (
    Callable,
    Dict,
    _is_integer,
)


_CALLBACKS: Dict[int, Callable] = {}
_COUNTER: int = 0


def register(callback: Callable) -> int:
    """
    Register a callback function that should be invoked when a progress update is emitted.

    Parameters
    ----------
    callback: Callable
        The callback function should have ``*args`` and ``**kwargs`` as its arguments.

    Returns
    -------
    int
        An identifier/handle that can be used to unregister the callback function.
    """
    global _COUNTER
    global _CALLBACKS

    if not callable(callback):
        raise TypeError(f"Expected a callable instead of {callback=}")

    _COUNTER += 1
    _CALLBACKS[_COUNTER] = callback

    return _COUNTER


def unregister(identifier: int) -> bool:
    """
    Unregister a callback function based on the identifier/handle that was returned when the callback function was registered.

    Parameters
    ----------
    identifier: int
        The identifier/handle for the callback function that was registered at some point.

    Returns
    -------
    bool
        True if the callback function was successfully unregistered.
    """
    global _CALLBACKS

    if not _is_integer(identifier):
        raise TypeError(f"Expected an integer instead of {identifier=}")
    elif identifier <= 0:
        raise ValueError(f"Expected an integer greater than zero instead of {identifier=}")

    if identifier in _CALLBACKS:
        del _CALLBACKS[identifier]
        return True

    return False


def _update(*args, **kwargs):
    callback: Callable
    for callback in _CALLBACKS.values():
        callback(*args, **kwargs)


_RECENT_PROGRESS: float = -1.0


def _update_every_N_percent(
    i: int,
    total: int = 1,
    N: float = 1.0,
    force: bool = False,
    *args,
    **kwargs,
):
    global _RECENT_PROGRESS
    if i == 0:
        _RECENT_PROGRESS = -1.0

    step: float = N / 100.0
    progress: float = i / total
    if (_RECENT_PROGRESS < 0.0) or (progress >= _RECENT_PROGRESS + step):
        if _RECENT_PROGRESS < 0.0:
            _RECENT_PROGRESS = 0.0
        else:
            _RECENT_PROGRESS += step

        _update(progress=_RECENT_PROGRESS, *args, **kwargs)
    elif force:
        _update(progress=progress, *args, **kwargs)

    if i >= total:
        _RECENT_PROGRESS = -1.0


class Progress:
    """
    Context manager class that can be used to emit information related to progress status.
    This is used in various parts of the pyimpspec source code to handle the state related to progress and for emitting progress updates as necessary.
    The ``*args`` and ``**kwargs`` parameters are passed on to the callback functions whenever a progress update is emitted.

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

    def _update(self, force: bool = False):
        _update_every_N_percent(
            i=self._i,
            total=self._total,
            message=self._message,
            force=force,
            *self._args,
            **self._kwargs,
        )

    def get_total(self) -> int:
        """
        Get the current total number of steps.

        Returns
        -------
        int
        """
        return self._total

    def get(self) -> int:
        """
        Get the current step.

        Returns
        -------
        int
        """
        return self._i

    def set(self, i: int):
        """
        Set the progress in terms of steps (i / total).

        Parameters
        ----------
        i: int
            The step index.
        """
        if not (i <= self._total):
            raise ValueError(f"Expected {i=} <= {self._total=}")

        self._i = i
        self._update()

    def increment(self, step: int = 1, force: bool = False):
        """
        Increment the progress by a step size ((i + step) / total).

        Parameters
        ----------
        step: int, optional
            The size of the step to take when called.

        force: bool, optional
            Force a progress update to be emitted.
        """
        self._i += step
        if not (self._i <= self._total):
            raise ValueError(f"Expected {self._i=} <= {self._total=}")

        self._update(force=force)

    def set_message(
        self,
        message: str,
        i: int = -1,
        total: int = -1,
        force: bool = True,
    ):
        """
        Set the status message and optionally also the progress and total number of steps.

        Parameters
        ----------
        message: str
            The new message.

        i: int, optional
            The new step index.

        total: int, optional
            The new total number of steps

        force: bool, optional
            Force a progress update to be emitted.
        """
        self._message = message
        if i >= 0:
            self._i = 0

        if total >= 0:
            self._total = total

        self._update(force=force)


_PROGRESS_MESSAGE: str = ""


def _default_handler(*args, **kwargs):
    global _PROGRESS_MESSAGE

    pct: str = f"{kwargs['progress'] * 100.0:.0f}%".rjust(4)
    message: str = kwargs["message"]

    _PROGRESS_MESSAGE = f"{pct}: {message}".ljust(len(_PROGRESS_MESSAGE.rstrip()))

    print(_PROGRESS_MESSAGE, end="\r")


def register_default_handler():
    """
    Register the default handler for progress updates.
    Formats the incoming information and prints it to stdout.
    The output ends with a carriage return (``\\r``) instead of a newline (``\\n``).
    """
    register(_default_handler)


def clear_default_handler_output():
    """
    Print a blank line with the same length as the previously printed formatted output.
    The output ends with a carriage return (``\\r``) instead of a newline (``\\n``).
    """
    global _PROGRESS_MESSAGE

    _PROGRESS_MESSAGE = "".ljust(len(_PROGRESS_MESSAGE.rstrip()))

    print(_PROGRESS_MESSAGE, end="\r")
