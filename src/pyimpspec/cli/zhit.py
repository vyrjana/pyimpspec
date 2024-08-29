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
from matplotlib import get_backend
import matplotlib.pyplot as plt
from pyimpspec.progress import clear_default_handler_output
from pyimpspec.typing.helpers import (
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
    set_figure_size,
)


def command(parser: ArgumentParser, args: Namespace, print_func: Callable = print):
    from pyimpspec import (
        DataSet,
        ZHITResult,
        mpl,
        perform_zhit,
    )

    all_data_sets: Dict[str, List[DataSet]] = parse_inputs(args)
    num_paths: int = len(all_data_sets)
    num_paths_remaining: int = num_paths

    agg_backend: bool = get_backend().lower() == "agg"
    set_figure_size(args.plot_width, args.plot_height, args.plot_dpi)

    plot_types: Dict[str, Callable] = {
        "bode": mpl.plot_bode,
        "fit": mpl.plot_fit,
        "imaginary": mpl.plot_imaginary,
        "magnitude": mpl.plot_magnitude,
        "nyquist": mpl.plot_nyquist,
        "phase": mpl.plot_phase,
        "real": mpl.plot_real,
        "real-imaginary": mpl.plot_real_imaginary,
    }
    plot: Callable = plot_types[args.plot_type]

    colors: Dict[str, str] = {
        "impedance": "black",
        "real": "black",
        "imaginary": "black",
        "magnitude": "black",
        "phase": "black",
    }

    data_sets: List[DataSet]
    for path, data_sets in all_data_sets.items():
        list(map(lambda _: apply_filters(_, args), data_sets))
        num_data: int = len(data_sets)

        for i, data in enumerate(data_sets):
            if num_paths > 1 or num_data > 1:
                if len(data_sets) > 1:
                    print_func(f"{path}: {data.get_label() or i}")
                else:
                    print_func(f"{path}")

            zhit: ZHITResult = perform_zhit(
                data,
                smoothing=args.smoothing,
                interpolation=args.interpolation,
                window=args.weights_window,
                center=args.weights_center,
                width=args.weights_width,
                num_points=args.num_points,
                polynomial_order=args.polynomial_order,
                num_iterations=args.num_iterations,
                admittance=args.admittance,
                num_procs=args.num_procs,
            )

            clear_default_handler_output()
            report: str = format_text(zhit.to_statistics_dataframe(), args)

            if plot == mpl.plot_fit:
                figure, axes = plot(
                    zhit,
                    data=data,
                    line=True,
                    legend=not args.plot_no_legend,
                    title=f"{data.get_label()}\n{zhit.get_label()}",
                    colored_axes=args.plot_colored_axes,
                    admittance=args.plot_admittance,
                )

            else:
                figure, axes = plot(
                    zhit,
                    label="",
                    line=False,
                    markers={
                        "magnitude": ".",
                        "phase": ".",
                        "real": ".",
                        "imaginary": ".",
                        "impedance": ".",
                    },
                    admittance=args.plot_admittance,
                    legend=False,
                )
                figure, axes = plot(
                    data,
                    colors=colors,
                    legend=False,
                    admittance=args.plot_admittance,
                    figure=figure,
                    axes=axes,
                )
                figure, axes = plot(
                    zhit,
                    line=True,
                    legend=not args.plot_no_legend,
                    title=f"{data.get_label()}\n{zhit.get_label()}",
                    colored_axes=args.plot_colored_axes,
                    admittance=args.plot_admittance,
                    figure=figure,
                    axes=axes,
                )

            figure.tight_layout()

            if args.output:
                output_path: str = get_output_path(
                    data=data,
                    result=zhit,
                    is_plot=True,
                    args=args,
                    i=i,
                )

                figure.savefig(output_path, dpi=args.plot_dpi)

                output_path = get_output_path(
                    data=data,
                    result=zhit,
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
