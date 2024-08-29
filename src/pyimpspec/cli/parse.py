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
from typing import (
    Callable,
    Dict,
    IO,
    List,
)
from .utility import (
    apply_filters,
    format_text,
    get_output_path,
    parse_inputs,
)


def command(parser: ArgumentParser, args: Namespace, print_func: Callable = print):
    from pyimpspec import DataSet

    all_data_sets: Dict[str, List[DataSet]] = parse_inputs(args)
    num_paths: int = len(all_data_sets)
    num_paths_remaining: int = num_paths

    data_sets: List[DataSet]
    if args.average_data_sets:
        to_average: List[DataSet] = []
        for data_sets in all_data_sets.values():
            to_average.extend(data_sets)
        all_data_sets.clear()
        all_data_sets["Average"] = [DataSet.average(to_average)]

    for path, data_sets in all_data_sets.items():
        num_data: int = len(data_sets)
        list(map(lambda _: apply_filters(_, args), data_sets))

        for i, data in enumerate(data_sets):
            if num_paths > 1 or num_data > 1:
                if len(data_sets) > 1:
                    print_func(f"{path}: {data.get_label() or i}")
                else:
                    print_func(f"{path}")

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
