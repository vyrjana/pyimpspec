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
    Tuple,
)
from matplotlib import get_backend
import matplotlib.pyplot as plt
from .utility import (
    apply_filters,
    clear_progress,
    format_text,
    get_mock_data,
    get_output_path,
    set_figure_size,
    validate_input_paths,
)


def single_test(
    data: "DataSet",  # noqa: F821
    args: Namespace,
) -> Tuple["TestResult", Callable]:  # noqa: F821
    from pyimpspec import (
        TestResult,
        mpl,
        perform_test,
    )

    test: TestResult = perform_test(
        data,
        test=args.test,
        num_RC=args.num_RC,
        mu_criterion=args.mu_criterion,
        add_capacitance=args.add_capacitance,
        add_inductance=args.add_inductance,
        method=args.method,
        max_nfev=args.max_nfev,
        num_procs=args.num_procs,
    )
    return (
        test,
        lambda t=test, d=data: mpl.plot_fit(
            test,
            data=d,
            title=f"{data.get_label()}\n{test.get_label()}" if args.plot_title else "",
            legend=not args.plot_no_legend,
            colored_axes=args.plot_colored_axes,
        ),
    )


def exploratory_tests(
    data: "DataSet",  # noqa: F821
    args: Namespace,
) -> Tuple["TestResult", Callable]:  # noqa: F821
    from pyimpspec import (
        TestResult,
        mpl,
        perform_exploratory_tests,
    )

    num_RCs: List[int] = []
    if args.max_num_RC > 0:
        num_RCs = list(range(2, args.max_num_RC))
    tests: List[TestResult] = perform_exploratory_tests(
        data,
        test=args.test,
        num_RCs=num_RCs,
        mu_criterion=args.mu_criterion,
        add_capacitance=args.add_capacitance,
        add_inductance=args.add_inductance,
        method=args.method,
        max_nfev=args.max_nfev,
        num_procs=args.num_procs,
    )
    test: TestResult = tests[0]
    return (
        test,
        lambda t=tests, d=data: mpl.plot_tests(
            t,
            args.mu_criterion,
            data=d,
            title=f"{data.get_label()}\n{test.get_label()}" if args.plot_title else "",
            legend=not args.plot_no_legend,
            colored_axes=args.plot_colored_axes,
        ),
    )


def command(parser: ArgumentParser, args: Namespace, print_func: Callable = print):
    validate_input_paths(args.input)

    from pyimpspec import (
        DataSet,
        TestResult,
        parse_data,
    )

    all_data_sets: Dict[str, List[DataSet]] = {}
    num_paths: int = len(args.input)
    num_paths_remaining: int = num_paths
    i: int
    path: str
    for i, path in enumerate(args.input):
        if path.startswith("<") and path.endswith(">"):
            all_data_sets[path] = [get_mock_data(path[1:-1])]
        else:
            all_data_sets[path] = parse_data(path)
    agg_backend: bool = get_backend().lower() == "agg"
    set_figure_size(args.plot_width, args.plot_height, args.plot_dpi)
    data_sets: List[DataSet]
    for path, data_sets in all_data_sets.items():
        list(map(lambda _: apply_filters(_, args), data_sets))
        num_data: int = len(data_sets)
        i: int
        data: DataSet
        for i, data in enumerate(data_sets):
            if num_paths > 1 or num_data > 1:
                print_func(f"{path}: {data.get_label() or i}")
            test: TestResult
            plot: Callable
            if args.automatic or args.num_RC > 0:
                test, plot = single_test(data, args)
            else:
                test, plot = exploratory_tests(data, args)
            clear_progress()
            report: str = format_text(test.to_statistics_dataframe(), args)
            fig, ax = plot()
            fig.tight_layout()
            if args.output:
                output_path: str = get_output_path(
                    data=data,
                    result=test,
                    is_plot=True,
                    args=args,
                    i=i,
                )
                fig.savefig(output_path, dpi=args.plot_dpi)
                output_path = get_output_path(
                    data=data,
                    result=test,
                    is_plot=False,
                    args=args,
                    i=i,
                )
                fp: IO
                with open(output_path, "w") as fp:
                    fp.write(report)
            elif not agg_backend:
                print_func(report)
                plt.show()
            else:
                print_func(report)
            plt.close()
            if (num_paths_remaining > 1 or i < num_data - 1) and not args.output:
                print_func("")
            num_paths_remaining -= 1
