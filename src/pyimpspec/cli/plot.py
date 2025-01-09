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
    List,
)
from matplotlib import get_backend
import matplotlib.pyplot as plt
from .utility import (
    apply_filters,
    get_color,
    get_marker,
    get_output_path,
    parse_inputs,
    set_figure_size,
)


def overlay_plot(
    all_data_sets: Dict[str, List["DataSet"]],  # noqa: F821
    plot: Callable,
    args: Namespace,
):
    from pyimpspec import DataSet

    if not (args.output_name[0] != "" or not args.output):
        raise ValueError("Expected an output name")

    data_sets: List[DataSet] = []

    path: str
    for path in all_data_sets.keys():
        data_sets.extend(all_data_sets[path])

    list(map(lambda _: apply_filters(_, args), data_sets))
    data: DataSet = data_sets.pop(0)
    output_path: str = get_output_path(data=data, result=None, is_plot=True, args=args)
    color: str = get_color(args.plot_color)
    marker_0: str = get_marker(args.plot_marker)
    marker_1: str = get_marker(args.plot_marker)
    marker_2: str = get_marker(args.plot_marker)

    kwargs = {
        "label": data.get_label(),
        "legend": not args.plot_no_legend,
        "colored_axes": args.plot_colored_axes,
        "colors": {
            "magnitude": color,
            "phase": color,
            "real": color,
            "imaginary": color,
            "impedance": color,
        },
        "markers": {
            "impedance": marker_0,
            "magnitude": marker_1,
            "phase": marker_2,
            "real": marker_1,
            "imaginary": marker_2,
        },
        "adjust_axes": False,
        "admittance": args.plot_admittance,
    }
    figure, axes = plot(
        data,
        **kwargs,
    )
    kwargs.update(
        {
            "figure": figure,
            "axes": axes,
        }
    )

    while data_sets:
        data = data_sets.pop(0)
        color = get_color(args.plot_color)
        marker_0 = get_marker(args.plot_marker)
        marker_1 = get_marker(args.plot_marker)
        marker_2 = get_marker(args.plot_marker)
        kwargs.update(
            {
                "label": data.get_label(),
                "colors": {
                    "magnitude": color,
                    "phase": color,
                    "real": color,
                    "imaginary": color,
                    "impedance": color,
                },
                "markers": {
                    "impedance": marker_0,
                    "magnitude": marker_1,
                    "phase": marker_2,
                    "real": marker_1,
                    "imaginary": marker_2,
                },
                "adjust_axes": len(data_sets) == 0,
            }
        )
        plot(
            data,
            **kwargs,
        )

    figure.tight_layout()

    if args.output:
        figure.savefig(output_path, dpi=args.plot_dpi)
    elif not get_backend().lower() == "agg":
        plt.show()

    plt.close()


def individual_plots(
    all_data_sets: Dict[str, List["DataSet"]],  # noqa: F821
    plot: Callable,
    args: Namespace,
    print_func: Callable,
):
    from pyimpspec import (
        DataSet,
        mpl,
    )

    agg_backend: bool = get_backend().lower() == "agg"
    kwargs = {
        "legend": not args.plot_no_legend,
        "colored_axes": args.plot_colored_axes,
        "admittance": args.plot_admittance,
    }
    if not (plot == mpl.plot_bode or plot == mpl.plot_real_imaginary or plot == mpl.plot_data):
        kwargs["legend"] = not args.plot_no_legend if not args.plot_title else False

    num_paths: int = len(all_data_sets)
    path: str
    data_sets: List[DataSet]
    for path, data_sets in all_data_sets.items():
        list(map(lambda _: apply_filters(_, args), data_sets))
        num_data: int = len(data_sets)

        i: int
        data: DataSet
        for i, data in enumerate(data_sets):
            if num_paths > 1 or num_data > 1:
                if len(data_sets) > 1:
                    print_func(f"{path}: {data.get_label() or i}")
                else:
                    print_func(f"{path}")

            kwargs["label"] = "" if args.plot_title else data.get_label()
            figure, axes = plot(
                data,
                **kwargs,
            )
            if args.plot_title:
                figure.suptitle(data.get_label())
            figure.tight_layout()

            if args.output:
                output_path: str = get_output_path(
                    data=data,
                    result=None,
                    is_plot=True,
                    args=args,
                    i=i,
                )
                figure.savefig(output_path, dpi=args.plot_dpi)

            elif not agg_backend:
                plt.show()

            plt.close()


def command(parser: ArgumentParser, args: Namespace, print_func: Callable = print):
    from pyimpspec import (
        DataSet,
        mpl,
    )

    all_data_sets: Dict[str, List[DataSet]] = parse_inputs(args)

    set_figure_size(args.plot_width, args.plot_height, args.plot_dpi)

    plot_types: Dict[str, Callable] = {
        "bode": mpl.plot_bode,
        "data": mpl.plot_data,
        "imaginary": mpl.plot_imaginary,
        "magnitude": mpl.plot_magnitude,
        "nyquist": mpl.plot_nyquist,
        "phase": mpl.plot_phase,
        "real": mpl.plot_real,
        "real-imaginary": mpl.plot_real_imaginary,
    }

    if args.plot_overlay:
        del plot_types["data"]
        if args.plot_type == "data":
            args.plot_type = "nyquist"

    if args.plot_type not in plot_types:
        raise ValueError(
            f"Unsupported plot type: '{args.plot_type}'! Valid plot types: {', '.join(plot_types.keys())}"
        )

    plot: Callable = plot_types[args.plot_type]
    if args.plot_overlay:
        overlay_plot(all_data_sets, plot, args)
    else:
        individual_plots(all_data_sets, plot, args, print_func)
