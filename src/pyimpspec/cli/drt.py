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
from numpy import log10 as log
from .utility import (
    apply_filters,
    clear_progress,
    format_text,
    get_color,
    get_mock_data,
    get_output_path,
    set_figure_size,
    validate_input_paths,
)


def overlay_plot(
    all_data_sets: Dict[str, List["DataSet"]],  # noqa: F821
    args: Namespace,
    print_func: Callable,
):
    from pyimpspec import (
        DRTResult,
        DataSet,
        calculate_drt,
        mpl,
        parse_cdc,
    )

    assert args.output_name[0] != "" or not args.output, "Expected an output name!"
    drts: List[Tuple[DataSet, DRTResult]] = []
    fragments: List[str] = []
    num_paths: int = len(all_data_sets)
    num_paths_remaining: int = num_paths
    path: str
    data_sets: List[DataSet]
    for path, data_sets in all_data_sets.items():
        list(map(lambda _: apply_filters(_, args), data_sets))
        num_data: int = len(data_sets)
        i: int
        data: DataSet
        for i, data in enumerate(data_sets):
            label: str = f"{path}: {data.get_label() or i}"
            drt: DRTResult = calculate_drt(
                data,
                method=args.method,
                mode=args.mode,
                lambda_value=args.lambda_value,
                rbf_type=args.rbf_type,
                derivative_order=args.derivative_order,
                rbf_shape=args.rbf_shape,
                shape_coeff=args.shape_coeff,
                inductance=args.inductance,
                credible_intervals=args.credible_intervals,
                timeout=args.timeout,
                num_samples=args.num_samples,
                num_attempts=args.num_attempts,
                maximum_symmetry=args.maximum_symmetry,
                circuit=parse_cdc(args.circuit),
                gaussian_width=args.gaussian_width,
                num_per_decade=args.num_per_decade,
                max_nfev=args.max_nfev,
                num_procs=args.num_procs,
            )
            drts.append(
                (
                    data,
                    drt,
                )
            )
            clear_progress()
            fragments.append(label)
            fragments.append(format_text(drt.to_statistics_dataframe(), args))
            if args.peak_threshold >= 0.0:
                fragments.append(
                    format_text(
                        drts[-1][1].to_peaks_dataframe(threshold=args.peak_threshold),
                        args,
                    )
                )
            if args.method == "bht":
                fragments.append(format_text(drt.to_scores_dataframe(), args))
            if num_paths_remaining > 0 or i < num_data - 1:
                fragments.append("\n")
            num_paths_remaining -= 1
    report: str = "\n".join(fragments)
    data, drt = drts.pop(0)
    color: str = get_color(args.plot_color)
    figure, axes = mpl.plot_gamma(
        drt,
        label=data.get_label(),
        legend=False,
        peak_threshold=args.peak_threshold,
        colors={"gamma": color},
    )
    axis = axes[0]
    for (data, drt) in drts:
        color = get_color(args.plot_color)
        mpl.plot_gamma(
            drt,
            label=data.get_label(),
            legend=False,
            peak_threshold=args.peak_threshold,
            colors={"gamma": color},
            figure=figure,
            axes=axes,
        )
    if not args.plot_no_legend:
        axis.legend()
    figure.tight_layout()
    if args.output:
        output_path: str = get_output_path(
            data=data,
            result=drt,
            is_plot=True,
            args=args,
        )
        figure.savefig(output_path, dpi=args.plot_dpi)
        output_path = get_output_path(
            data=data,
            result=drt,
            is_plot=False,
            args=args,
        )
        fp: IO
        with open(output_path, "w") as fp:
            fp.write(report)
    else:
        print_func(report)
        plt.show()
    plt.close()


def individual_plots(
    all_data_sets: Dict[str, List["DataSet"]],  # noqa: F821
    args: Namespace,
    print_func: Callable,
):
    from pyimpspec import (
        DRTResult,
        DataSet,
        calculate_drt,
        mpl,
        parse_cdc,
    )

    agg_backend: bool = get_backend().lower() == "agg"
    num_paths: int = len(all_data_sets)
    num_paths_remaining: int = num_paths
    path: str
    data_sets: List[DataSet]
    for path, data_sets in all_data_sets.items():
        list(map(lambda _: apply_filters(_, args), data_sets))
        num_data = len(data_sets)
        i: int
        data: DataSet
        for i, data in enumerate(data_sets):
            if num_paths > 1 or num_data > 1:
                print_func(f"{path}: {data.get_label() or i}")
            drt: DRTResult = calculate_drt(
                data,
                method=args.method,
                mode=args.mode,
                lambda_value=args.lambda_value,
                rbf_type=args.rbf_type,
                derivative_order=args.derivative_order,
                rbf_shape=args.rbf_shape,
                shape_coeff=args.shape_coeff,
                inductance=args.inductance,
                credible_intervals=args.credible_intervals,
                timeout=args.timeout,
                num_samples=args.num_samples,
                num_attempts=args.num_attempts,
                maximum_symmetry=args.maximum_symmetry,
                circuit=parse_cdc(args.circuit),
                gaussian_width=args.gaussian_width,
                num_per_decade=args.num_per_decade,
                max_nfev=args.max_nfev,
                num_procs=args.num_procs,
            )
            clear_progress()
            label: str = f"{data.get_label()}\n{drt.get_label()}"
            figure, axes = mpl.plot_drt(
                drt,
                data=data,
                title=label if args.plot_title else "",
                legend=not args.plot_no_legend,
                colored_axes=args.plot_colored_axes,
                peak_threshold=args.peak_threshold,
            )
            figure.tight_layout()
            fragments: List[str] = []
            fragments.append(format_text(drt.to_statistics_dataframe(), args))
            if args.peak_threshold >= 0.0:
                fragments.append(
                    format_text(
                        drt.to_peaks_dataframe(threshold=args.peak_threshold), args
                    )
                )
            if args.method == "bht":
                fragments.append(format_text(drt.to_scores_dataframe(), args))
            report: str = "\n\n".join(fragments)
            if args.output:
                output_path: str = get_output_path(
                    data=data,
                    result=drt,
                    is_plot=True,
                    args=args,
                    i=i,
                )
                figure.savefig(output_path, dpi=args.plot_dpi)
                output_path = get_output_path(
                    data=data,
                    result=drt,
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


def command(parser: ArgumentParser, args: Namespace, print_func: Callable = print):
    validate_input_paths(args.input)

    from pyimpspec import (
        DataSet,
        parse_data,
    )

    set_figure_size(args.plot_width, args.plot_height, args.plot_dpi)
    if args.method == "mrq-fit":
        assert args.circuit.strip() != "", "Expected a circuit description code!"
    all_data_sets: Dict[str, List[DataSet]] = {}
    i: int
    path: str
    for i, path in enumerate(args.input):
        if path.startswith("<") and path.endswith(">"):
            all_data_sets[path] = [get_mock_data(path[1:-1])]
        else:
            all_data_sets[path] = parse_data(path)
    if args.plot_overlay:
        overlay_plot(all_data_sets, args, print_func)
    else:
        individual_plots(all_data_sets, args, print_func)
