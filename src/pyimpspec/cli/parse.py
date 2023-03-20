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

from argparse import (
    ArgumentParser,
    Namespace,
)
from typing import (
    Callable,
    Dict,
    IO,
    List,
)
from .utility import (
    apply_filters,
    format_text,
    get_mock_data,
    get_output_path,
    validate_input_paths,
)


def command(parser: ArgumentParser, args: Namespace, print_func: Callable = print):
    validate_input_paths(args.input)

    from pyimpspec import (
        DataSet,
        parse_data,
    )

    num_paths: int = len(args.input)
    num_paths_remaining: int = num_paths
    all_data_sets: Dict[str, List[DataSet]] = {}
    i: int
    path: str
    for i, path in enumerate(args.input):
        if path.startswith("<") and path.endswith(">"):
            all_data_sets[path] = [get_mock_data(path[1:-1])]
        else:
            all_data_sets[path] = parse_data(path)
    data_sets: List[DataSet]
    for path, data_sets in all_data_sets.items():
        num_data: int = len(data_sets)
        list(map(lambda _: apply_filters(_, args), data_sets))
        i: int
        data: DataSet
        for i, data in enumerate(data_sets):
            if num_paths > 1 or num_data > 1:
                print_func(f"{path}: {data.get_label() or i}")
            output: str = format_text(data.to_dataframe(), args).rstrip()
            if args.output:
                output_path: str = get_output_path(
                    data=data,
                    result=None,
                    is_plot=False,
                    args=args,
                    i=i,
                )
                fp: IO
                with open(output_path, "w") as fp:
                    fp.write(output)
            else:
                print_func(output)
            if i < num_data - 1 and not args.output:
                print_func("")
        if num_paths_remaining > 1 or 1 < num_data - 1:
            print_func("")
        num_paths_remaining -= 1
