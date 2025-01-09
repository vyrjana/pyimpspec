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
    Optional,
    Tuple,
)
from .utility import (
    apply_filters,
    format_text,
    get_color,
    get_output_path,
    parse_inputs,
    set_figure_size,
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
    from pyimpspec.analysis.drt import LMResult

    if not (args.output_name[0] != "" or not args.output):
        raise ValueError("Expected an output name")

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
                cross_validation=args.cross_validation,
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
                max_iter=args.max_iter,
                model_order=args.model_order,
                model_order_method=args.model_order_method,
                num_procs=args.num_procs,
            )
            drts.append(
                (
                    data,
                    drt,
                )
            )
            clear_default_handler_output()

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
    for data, drt in drts:
        color = get_color(args.plot_color)
        mpl.plot_gamma(
            drt,
            label=data.get_label(),
            legend=False,
            peak_threshold=args.peak_threshold,
            colors={"gamma": color},
            figure=figure,
            axes=axes,
            frequency=args.plot_frequency,
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
    from pandas import DataFrame
    from pyimpspec import (
        DRTPeaks,
        DRTResult,
        DataSet,
        calculate_drt,
        mpl,
        parse_cdc,
    )
    from pyimpspec.analysis.drt import LMResult
    from pyimpspec.plot.colors import COLOR_BLACK
    from pyimpspec.plot.mpl.helpers import _color_axis

    agg_backend: bool = get_backend().lower() == "agg"
    num_paths: int = len(all_data_sets)
    num_paths_remaining: int = num_paths

    plot_types: Dict[str, Callable] = {
        "bode": mpl.plot_bode,
        "drt": mpl.plot_drt,
        "gamma": mpl.plot_gamma,
        "imaginary": mpl.plot_imaginary,
        "magnitude": mpl.plot_magnitude,
        "nyquist": mpl.plot_nyquist,
        "phase": mpl.plot_phase,
        "real": mpl.plot_real,
        "real-imaginary": mpl.plot_real_imaginary,
    }
    plot: Callable = plot_types[args.plot_type]
    kwargs = {
        "legend": not args.plot_no_legend,
        "colored_axes": args.plot_colored_axes,
        "admittance": args.plot_admittance,
    }

    path: str
    data_sets: List[DataSet]
    for path, data_sets in all_data_sets.items():
        list(map(lambda _: apply_filters(_, args), data_sets))
        num_data = len(data_sets)

        i: int
        data: DataSet
        for i, data in enumerate(data_sets):
            if num_paths > 1 or num_data > 1:
                if len(data_sets) > 1:
                    print_func(f"{path}: {data.get_label() or i}")
                else:
                    print_func(f"{path}")

            drt: DRTResult = calculate_drt(
                data,
                method=args.method,
                mode=args.mode,
                lambda_value=args.lambda_value,
                cross_validation=args.cross_validation,
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
                max_iter=args.max_iter,
                model_order=args.model_order,
                model_order_method=args.model_order_method,
                num_procs=args.num_procs,
            )

            peaks: Optional[DRTPeaks] = None
            if (
                args.analyze_peaks
                and hasattr(drt, "analyze_peaks")
                and not isinstance(drt, LMResult)
            ):
                 peaks = drt.analyze_peaks(
                    num_peaks=args.num_peaks,
                    peak_positions=args.peak_positions or None,
                    disallow_skew=args.disallow_skew,
                )

            clear_default_handler_output()

            label: str = f"{data.get_label()}\n{drt.get_label()}"
            if plot is mpl.plot_gamma:
                figure, axes = mpl.plot_gamma(
                    drt,
                    title=label if args.plot_title else "",
                    peak_threshold=args.peak_threshold,
                    frequency=args.plot_frequency,
                    **kwargs,
                )
            else:
                figure, axes = mpl.plot_drt(
                    drt,
                    data=data,
                    title=label if args.plot_title else "",
                    peak_threshold=args.peak_threshold,
                    frequency=args.plot_frequency,
                    **kwargs,
                )

            if plot not in (mpl.plot_drt, mpl.plot_gamma):
                subset_axes = [axes[0], axes[1]]
                for ax in subset_axes:
                    ax.set_xlim(None, None)
                    ax.set_xscale("linear")
                    ax.set_ylim(None, None)
                    ax.set_yscale("linear")

                subset_axes[0].clear()
                subset_axes[1].clear()
                if plot in (
                    mpl.plot_nyquist,
                    mpl.plot_magnitude,
                    mpl.plot_phase,
                    mpl.plot_real,
                    mpl.plot_imaginary,
                ):
                    figure.delaxes(subset_axes.pop(1))

                if len(subset_axes) == 2:
                    subset_axes[1].yaxis.set_label_position("right")

                if plot is mpl.plot_nyquist:
                    _color_axis(subset_axes[0], COLOR_BLACK, left=True, right=False)

                plot(
                    drt,
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
                    figure=figure,
                    axes=subset_axes,
                )
                plot(
                    data,
                    label="Data" if args.plot_title else None,
                    colors={
                        "magnitude": COLOR_BLACK,
                        "phase": COLOR_BLACK,
                        "real": COLOR_BLACK,
                        "imaginary": COLOR_BLACK,
                        "impedance": COLOR_BLACK,
                    },
                    figure=figure,
                    axes=subset_axes,
                    **kwargs,
                )
                plot(
                    drt,
                    label="Fit" if args.plot_title else None,
                    line=True,
                    figure=figure,
                    axes=subset_axes,
                    **kwargs,
                )

            figure.tight_layout()

            fragments: List[str] = []
            fragments.append(format_text(drt.to_statistics_dataframe(), args))
            if args.peak_threshold >= 0.0:
                fragments.append(
                    format_text(
                        drt.to_peaks_dataframe(threshold=args.peak_threshold),
                        args,
                    )
                )

            if args.method == "bht":
                fragments.append(format_text(drt.to_scores_dataframe(), args))

            if (
                args.analyze_peaks
                and peaks is not None
                and not isinstance(drt, LMResult)
            ):
                if isinstance(peaks, tuple):
                    for df in (p.to_peaks_dataframe() for p in peaks):
                        fragments.append(format_text(df, args))
                else:
                    fragments.append(format_text(peaks.to_peaks_dataframe(), args))

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
    from pyimpspec import DataSet

    set_figure_size(args.plot_width, args.plot_height, args.plot_dpi)

    if args.method == "mrq-fit":
        if args.circuit.strip() == "":
            raise ValueError(
                f"Expected a non-empty string for the circuit description code instead of {args.circuit=}"
            )

    all_data_sets: Dict[str, List[DataSet]] = parse_inputs(args)

    if args.plot_overlay:
        overlay_plot(all_data_sets, args, print_func)
    else:
        individual_plots(all_data_sets, args, print_func)
