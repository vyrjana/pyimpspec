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

from argparse import (
    ArgumentParser,
    Namespace,
)
import sys
from typing import (
    Callable,
    Dict,
)
from .config import (
    command as config_command,
    get_argument_parser,
    parse_cli_args,
)


def main():
    parser: ArgumentParser = get_argument_parser()
    args: Namespace = parse_cli_args(parser, sys.argv[1:])
    if args.version:
        from pyimpspec.version import PACKAGE_VERSION

        print(PACKAGE_VERSION)
        return
    from .circuit import command as circuit_command
    from .drt import command as drt_command
    from .fit import command as fit_command
    from .license import command as license_command
    from .parse import command as parse_command
    from .plot import command as plot_command
    from .test import command as test_command
    from .zhit import command as zhit_command

    commands: Dict[str, Callable] = {
        "parse": parse_command,
        "plot": plot_command,
        "test": test_command,
        "zhit": zhit_command,
        "drt": drt_command,
        "fit": fit_command,
        "circuit": circuit_command,
        "config": config_command,
        "show": license_command,
    }
    if args.command in commands:
        if hasattr(args, "suppress_progress") and not args.suppress_progress:
            from pyimpspec.progress import register_default_handler

            register_default_handler()
        try:
            commands[args.command](parser, args)
        except KeyboardInterrupt:
            return
    else:
        parser.print_help()
