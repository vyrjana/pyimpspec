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
from os import makedirs
from os.path import (
    abspath,
    basename,
    exists,
    isdir,
    join,
    splitext,
)
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)
from pyimpspec.typing.helpers import _is_integer


def _cm_to_in(string: str) -> float:
    return float(string[:-2]) / 2.54


def _px_to_in(string: str, dpi: int) -> float:
    return float(string[:-2]) / dpi


def set_figure_size(width: str, height: str, dpi: int):
    from matplotlib import rcParams

    width_inches: float
    if width.endswith("in"):
        width_inches = float(width[:-2])
    elif width.endswith("cm"):
        width_inches = _cm_to_in(width)
    elif width.endswith("px"):
        width_inches = _px_to_in(width, dpi)
    else:
        width_inches = float(width)

    height_inches: float
    if height.endswith("in"):
        height_inches = float(height[:-2])
    elif height.endswith("cm"):
        height_inches = _cm_to_in(height)
    elif height.endswith("px"):
        height_inches = _px_to_in(height, dpi)
    else:
        height_inches = float(height)

    rcParams["figure.figsize"] = (width_inches, height_inches)


def apply_filters(
    data: "DataSet",  # noqa: F821
    args: Namespace,
):
    if args.low_pass_cutoff > 0.0:
        data.low_pass(args.low_pass_cutoff)

    if args.high_pass_cutoff > 0.0:
        data.high_pass(args.high_pass_cutoff)

    if data.get_num_points() < 1:
        raise ValueError(
            f"The applied low- and/or high-pass filters have masked all data points in the '{data.get_label()}' data set parsed from '{data.get_path()}'!"
        )

    if len(args.exclude_indices) > 0:
        data.set_mask({i: True for i in args.exclude_indices})

    if data.get_num_points() < 1:
        raise ValueError(
            f"All data points have been masked after excluding data points based on indices in the '{data.get_label()}' data set parsed from '{data.get_path()}'!"
        )


COLORS: List[str] = []


def get_color(cli_colors: List[str]) -> str:
    global COLORS

    if len(COLORS) == 0:
        COLORS = (
            [
                "#000000",
                "#0077BB",
                "#CC3311",
                "#EE3377",
                "#EE7733",
                "#009988",
                "#33BBEE",
            ]
            if not cli_colors
            else cli_colors
        )

    COLORS.append(COLORS.pop(0))

    return COLORS[-1]


MARKERS: List[str] = []


def get_marker(cli_markers: List[str]) -> str:
    global MARKERS

    if len(MARKERS) == 0:
        MARKERS = (
            [
                "o",
                "s",
                "v",
                "^",
                ">",
                "<",
            ]
            if not cli_markers
            else cli_markers
        )

    MARKERS.append(MARKERS.pop(0))

    return MARKERS[-1]


def validate_text_format(fmt: str) -> str:
    formats: List[str] = [
        "csv",
        "md",
        "tex",
        "json",
    ]

    if fmt not in formats:
        raise ValueError(
            f"Unsupported output format: '{fmt}'! Valid formats: {', '.join(formats)}"
        )

    return fmt


def get_text_extension(fmt: str) -> str:
    return validate_text_format(
        {
            "latex": "tex",
            "markdown": "md",
        }.get(fmt, fmt)
    )


def get_output_path(
    data: "DataSet",  # noqa: F821
    result: Optional[
        Union["KramersKronigResult", "ZHITResult", "DRTResult", "FitResult"]  # noqa: F821
    ],  # noqa: F821
    is_plot: bool,
    args: Namespace,
    i: int = -1,
) -> str:
    from pyimpspec import (
        DRTResult,
        FitResult,
        KramersKronigResult,
        ZHITResult,
    )

    output_dir: str = validate_output_dir(args.output_dir)

    name: str = get_output_name(data.get_path(), args.output_name, i)
    if isinstance(result, KramersKronigResult):
        name += "-test"

    elif isinstance(result, ZHITResult):
        name += "-zhit"

    elif isinstance(result, DRTResult):
        name += "-drt"

    elif isinstance(result, FitResult):
        name += "-fit"

    elif result is None:
        name += "-data"

    extension: str = (
        args.plot_format if is_plot else get_text_extension(args.output_format)
    )

    if extension.startswith("."):
        extension = extension[1:]

    return join(output_dir, f"{name}.{extension}")


def format_text(
    df: "DataFrame",  # noqa: F821
    args: Namespace,
) -> str:
    output_extension: str = get_text_extension(args.output_format)

    output: str
    if output_extension == "csv":
        output = df.to_csv(index=args.output_indices)

    elif output_extension == "tex":
        if args.output_indices:
            output = df.style.to_latex()
        else:
            output = df.style.hide(axis="index").to_latex()

    elif output_extension == "md":
        output = df.to_markdown(
            index=args.output_indices,
            floatfmt=f".{args.output_significant_digits}g",
        )

    elif output_extension == "json":
        output = df.to_json()

    else:
        raise NotImplementedError(f"Unsupported format: {output_extension}")

    return output


def _parse_identity(identity: str) -> Tuple[str, dict]:
    kwarg_types = {
        "noise": float,
        "num_per_decade": int,
        "log_max_f": float,
        "log_min_f": float,
        "seed": int,
        "drift": float,
    }
    assertion_message: str = "'\n- '".join(kwarg_types.keys())
    kwargs: dict = {}

    i: int = identity.rfind(":")
    if ":" in identity and i > max(map(lambda s: identity.rfind(s), ("}", "]", ")"))):
        for arg in map(str.strip, identity[i + 1:].split(",")):
            key, value = arg.split("=")
            if key in kwarg_types:
                kwargs[key] = kwarg_types[key](value)
            else:
                raise KeyError(
                    f"Invalid keyword argument '{key}'! Valid keyword arguments are:\n- '{assertion_message}'"
                )

        identity = identity[:i]

    return (identity, kwargs)


def get_mock_circuits(identity: str) -> List["Circuit"]:  # noqa: F821
    from pyimpspec.mock_data import generate_mock_circuits

    kwargs: dict
    identity, kwargs = _parse_identity(identity)

    return generate_mock_circuits(identity, **kwargs)


def get_mock_data(identity: str) -> List["DataSet"]:  # noqa: F821
    from pyimpspec.mock_data import generate_mock_data

    kwargs: dict
    identity, kwargs = _parse_identity(identity)

    return generate_mock_data(identity, **kwargs)


def validate_input_paths(paths: List[str]) -> List[str]:
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate input paths detected!")

    path: str
    for path in paths:
        is_mock_data: bool = path.startswith("<") and path.endswith(">")

        if is_mock_data:
            get_mock_data(path[1:-1])
            continue

        elif not exists(path):
            raise FileNotFoundError(path)

    return paths[:]


def validate_output_dir(path: str) -> str:
    path = abspath(path)

    if not exists(path):
        makedirs(path)

    elif not isdir(path):
        raise NotADirectoryError(path)

    return path


def parse_circuits(args: Namespace) -> List["Circuit"]:  # noqa: F821
    from pyimpspec import (
        Circuit,
        parse_cdc,
    )

    circuits: List[Circuit] = []

    i: int
    cdc: str
    for i, cdc in enumerate(args.input):
        if cdc.startswith("<") and cdc.endswith(">"):
            circuit: Circuit
            for circuit in get_mock_circuits(cdc[1:-1]):
                circuits.append(circuit)
        else:
            circuits.append(parse_cdc(cdc))

    return circuits


def parse_inputs(args: Namespace) -> Dict[str, List["DataSet"]]:  # noqa: F821
    from pyimpspec import (
        DataSet,
        parse_data,
    )

    validate_input_paths(args.input)

    all_data_sets: Dict[str, List[DataSet]] = {}

    i: int
    path: str
    for i, path in enumerate(args.input):
        if path.startswith("<") and path.endswith(">"):
            data: DataSet
            for data in get_mock_data(path[1:-1]):
                label: str = data.get_label()
                if label in all_data_sets:
                    all_data_sets[label].append(data)
                else:
                    all_data_sets[label] = [data]
        elif len(args.nth_data_set) > 0:
            all_data_sets[path] = [
                data for i, data in enumerate(parse_data(path))
                if i in args.nth_data_set
            ]
        else:
            all_data_sets[path] = parse_data(path)

    return all_data_sets


def get_output_name(path: str, defaults: List[str] = [], i: int = -1) -> str:
    if not isinstance(path, str):
        raise TypeError(f"Expected a string instead of {path=}")

    if not isinstance(defaults, list):
        raise TypeError(f"Expected a list instead of {defaults=}")

    if not _is_integer(i):
        raise TypeError(f"Expected an integer instead of {i=}")

    generated: str = splitext(basename(path))[0]
    generated = f"{generated}-{i+1}" if i >= 0 else generated

    if i >= len(defaults):
        return generated

    elif defaults:
        return defaults[i if i >= 0 else 0] or generated

    return generated


def print_command_help(parser: ArgumentParser, command: str):
    subparser: ArgumentParser = parser._actions[1]._name_parser_map[command]
    subparser.print_help()


def get_config_dir() -> str:
    from xdg import xdg_config_home

    return join(xdg_config_home(), "pyimpspec")


def get_config_path() -> str:
    return join(get_config_dir(), "config.json")
