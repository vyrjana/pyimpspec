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
    _HelpAction,
)
from importlib.machinery import SourceFileLoader
from json import (
    dump as dump_json,
    load as load_json,
)
from os import makedirs
from os.path import exists
from types import ModuleType
from typing import (
    Any,
    Callable,
    Dict,
    IO,
    List,
    Optional,
)
from .args import (
    circuit_args,
    drt_args,
    fit_args,
    license_args,
    parse_args,
    plot_args,
    test_args,
    zhit_args,
)
from .utility import (
    get_config_dir,
    get_config_path,
    print_command_help,
)

# from .circuit import command as circuit_command
# from .drt import command as drt_command
# from .fit import command as fit_command
# from .license import command as license_command
# from .parse import command as parse_command
# from .plot import command as plot_command
# from .test import command as test_command
# from .zhit import command as zhit_command


_IGNORE_USER_CONFIG: bool = False  # This is used to make testing easier
VERSION: int = 1


def args(parser: ArgumentParser):
    parser.add_argument(
        "--save",
        action="store_true",
        dest="save_file",
        help=f"Save the default configuration to '{get_config_path()}'.",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        dest="update_file",
        help=f"Update the configuration file stored in '{get_config_path()}'.",
    )
    parser.add_argument(
        "--num-procs",
        action="store_true",
        dest="num_procs",
        help="Print the default number of processes that pyimpspec would use on this system based on the available libraries and environment variables.",
    )


config_args: Callable = args


def get_defaults() -> dict:
    defaults: dict = {}
    parser: ArgumentParser = get_argument_parser()

    cmd: str
    subparser: ArgumentParser
    for cmd, subparser in parser._actions[1]._name_parser_map.items():
        if cmd == "config":
            continue
        elif cmd == "show":
            continue

        defaults[cmd] = {}

        for action in subparser._actions:
            if type(action) is _HelpAction or action.default is None:
                continue

            defaults[cmd][action.dest] = action.default

    return defaults


def merge_configs(src: dict, dst: dict):
    k: str
    v: Any
    for k, v in src.items():
        if k not in dst:
            continue

        if type(v) is dict:
            merge_configs(v, dst[k])
        else:
            dst[k] = src[k]


def command(parser: ArgumentParser, args: Namespace, print_func: Callable = print):
    if args.save_file or args.update_file:
        folder: str = get_config_dir()
        if not exists(folder):
            makedirs(folder)

        output: dict = get_defaults()
        output["user_defined_elements"] = ""
        output["num_procs"] = 0

        if args.update_file:
            merge_configs(get_config(), output)

        output["version"] = VERSION

        fp: IO
        with open(get_config_path(), "w") as fp:
            dump_json(output, fp, indent=4, sort_keys=True)

    elif args.num_procs:
        from pyimpspec.analysis.utility import get_default_num_procs

        print_func(get_default_num_procs())

    else:
        print_command_help(parser, args.command)


def get_argument_parser() -> ArgumentParser:
    parser: ArgumentParser = ArgumentParser(
        prog="pyimpspec",
        allow_abbrev=False,
        description="""
pyimpspec Copyright (C) 2024 pyimpspec developers
This program comes with ABSOLUTELY NO WARRANTY; for details type `show w'.
This is free software, and you are welcome to redistribute it
under certain conditions; type `show c' for details.

There are some special values that can be used as input paths to load example data:
"<EXAMPLE>", "<RANDLES>", "<DRIFTING>".
        """.strip(),
    )
    subparsers = parser.add_subparsers(dest="command")
    parse_args(
        subparsers.add_parser(
            "parse",
            description="""
Parse one or more data files.
""".strip(),
        )
    )
    plot_args(
        subparsers.add_parser(
            "plot",
            description="""
Plot one or more impedance spectra individually or overlaid.
""".strip(),
        )
    )
    test_args(
        subparsers.add_parser(
            "test",
            description="""
Validate one or more impedance spectra using Kramers-Kronig transforms.
""".strip(),
        )
    )
    zhit_args(
        subparsers.add_parser(
            "zhit",
            description="""
Reconstruct the modulus data of impedance spectra from the phase data using the Z-HIT algorithm.
""".strip(),
        )
    )
    drt_args(
        subparsers.add_parser(
            "drt",
            description="""
Calculate the distribution of relaxation times for one or more impedance spectra.
""".strip(),
        )
    )
    fit_args(
        subparsers.add_parser(
            "fit",
            description="""
Fit an equivalent circuit to one or more impedance spectra.
""".strip(),
        )
    )
    circuit_args(
        subparsers.add_parser(
            "circuit",
            description="""
Generate circuit diagrams for one or more circuits. Alternatively, simulate the impedance spectra of one or more circuits.
""".strip(),
        )
    )
    config_args(
        subparsers.add_parser(
            "config",
            description=f"""
The configuration file can be used to override the default behaviors of the pyimpspec CLI.
Configuration file path: {get_config_path()}
""".strip(),
        )
    )
    license_args(
        subparsers.add_parser(
            "show",
            description="""
Command-line interface for the pyimpspec Python package, which can be used to validate, analyze, and visualize impedance spectra.
""".strip(),
        )
    )
    parser.add_argument(
        "--version",
        action="store_true",
        dest="version",
        help="Print the current version.",
    )

    return parser


def load_user_defined_elements(path: Optional[str]):
    if path is None or path == "":
        return

    if not exists(path):
        raise FileNotFoundError(
            f"Failed to load user-defined elements from '{path}' because the file does not exist!"
        )

    loader = SourceFileLoader("user_defined_elements", path)
    mod = ModuleType(loader.name)
    loader.exec_module(mod)


def set_num_procs_override(num_procs: int):
    from pyimpspec.analysis.utility import set_default_num_procs

    set_default_num_procs(num_procs)


def parse_cli_args(parser: ArgumentParser, argv: List[str]) -> Namespace:
    args: Namespace = parser.parse_args(argv)
    if _IGNORE_USER_CONFIG:
        return args

    config: dict = get_config()
    load_user_defined_elements(config.get("user_defined_elements"))

    num_procs: int = config.get("num_procs", -1)
    if num_procs > 0:
        set_num_procs_override(num_procs)

    config = config.get(args.command, {})
    if not config:
        return args

    subparser: ArgumentParser = parser._actions[1]._name_parser_map[args.command]

    for act in subparser._actions:
        if isinstance(act, _HelpAction) or act.default is None:
            continue
        elif act.dest not in config:
            continue
        act.default = config[act.dest]

    return parser.parse_args(argv)


def _v1_migrator(config: dict) -> dict:
    if config["version"] > 1:
        raise NotImplementedError(f"Implement migrator to CLI config version {config['version']}")

    return config


def migrate_config(config: dict) -> dict:
    version: int = config.get("version", -1)
    if version < 1:
        return {}

    if not (0 < version <= VERSION):
        raise NotImplementedError(f"Unsupported config version {version=}")

    migrators: Dict[int, Callable] = {
        1: _v1_migrator,
    }

    v: int
    migrator: Callable
    for v, migrator in migrators.items():
        if v < version:
            continue
        config = migrator(config)

    del config["version"]

    return config


def get_config() -> dict:
    config_path: str = get_config_path()
    if not exists(config_path):
        return {}

    fp: IO
    with open(config_path, "r") as fp:
        config: dict = load_json(fp)

    return migrate_config(config)
