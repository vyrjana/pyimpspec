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
from pyimpspec.plot.colors import COLOR_BLACK
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
    get_output_path,
    parse_inputs,
    set_figure_size,
)


def plot_num_RC_suggestion_method(
    method: int,
    tests: List["KramersKronigResult"],  # noqa: F821
    suggestion: Tuple["KramersKronigResult", Dict[int, float], int, int],  # noqa: F821
    data: "DataSet",  # noqa: F821
    **kwargs,
) -> Tuple["Figure", List["Axes"]]:  # noqa: F821
    from pyimpspec import mpl
    from matplotlib.ticker import AutoLocator

    figure, axes = mpl.plot_kramers_kronig_tests(
        tests=tests,
        suggestion=suggestion,
        data=data,
        **kwargs,
    )

    axes[0].clear()
    axes[1].remove()
    axes.pop(1)
    axes.insert(1, axes[0].twinx())
    axes[1].yaxis.set_major_locator(AutoLocator())

    color_num_RC = kwargs.get("colors", {}).get("num_RC", COLOR_BLACK)
    test: "KramersKronigResult"  # noqa: F821
    test = suggestion[0]
    axes[1].axvline(
        test.num_RC,
        color=color_num_RC,
        linestyle="--",
        label=r"$N_{\tau\rm, opt} = " + str(test.num_RC) + "$",
    )

    _ = mpl.plot_num_RC_suggestion_method(
        tests=tests,
        method=method,
        lower_limit=suggestion[2],
        upper_limit=suggestion[3],
        figure=figure,
        axes=axes[0:2],
        **kwargs,
    )

    return (
        figure,
        axes,
    )


def exploratory_tests(
    data: "DataSet",  # noqa: F821
    agg_backend: bool,
    args: Namespace,
    print_func: Callable = print,
) -> Tuple["KramersKronigResult", Callable]:  # noqa: F821
    from pyimpspec.analysis.kramers_kronig import (
        KramersKronigResult,
        evaluate_log_F_ext,
        suggest_num_RC,
        suggest_representation,
    )
    from pyimpspec import mpl

    num_RCs: Optional[List[int]] = None
    if args.max_num_RC > 0:
        num_RCs = list(range(2, args.max_num_RC + 1))
        args.num_F_ext_evaluations = 0

    if all(
        (
            args.plot_log_F_ext_3d,
            args.plot_log_F_ext_2d,
        )
    ):
        raise ValueError(
            f"Expected either {args.plot_log_F_ext_3d=} or {args.plot_log_F_ext_2d=}, but not both at the same time"
        )

    if all((args.impedance, args.admittance)):
        raise ValueError(
            f"Either {args.impedance=} or {args.admittance=} can be True, but not both at the same time!"
        )

    representations: List[
        Tuple[
            Tuple[float, List[KramersKronigResult], float],
            Tuple[KramersKronigResult, Dict[int, float], int, int],
        ]
    ] = []

    options: List[bool] = (
        [True] if args.admittance else ([False] if args.impedance else [False, True])
    )

    err: Optional[Exception] = None
    admittance: bool
    for admittance in options:
        evaluations: List[Tuple[float, List[KramersKronigResult], float]]
        try:
            evaluations = evaluate_log_F_ext(
                data,
                test=args.test,
                num_RCs=num_RCs,
                add_capacitance=not args.no_capacitance,
                add_inductance=not args.no_inductance,
                admittance=admittance,
                min_log_F_ext=args.min_log_F_ext,
                max_log_F_ext=args.max_log_F_ext,
                log_F_ext=args.log_F_ext,
                num_F_ext_evaluations=args.num_F_ext_evaluations,
                rapid_F_ext_evaluations=not args.no_rapid_F_ext_evaluations,
                cnls_method=args.cnls_method,
                max_nfev=args.max_nfev,
                timeout=args.timeout,
                num_procs=args.num_procs,
            )
        except ValueError as e:
            err = e
            continue

        log_F_ext: float
        tests: List[KramersKronigResult]
        statistic: float
        log_F_ext, tests, statistic = evaluations[0]

        suggestion: Tuple[KramersKronigResult, Dict[int, float], int, int]
        suggestion = suggest_num_RC(
            tests,
            lower_limit=args.lower_limit if args.lower_limit > 0 else 0,
            upper_limit=args.upper_limit,
            limit_delta=args.limit_delta,
            mu_criterion=args.mu_criterion,
            beta=args.beta,
            methods=(
                args.suggestion_methods if len(args.suggestion_methods) > 0 else None
            ),
            use_sum=args.use_sum,
            use_mean=args.use_mean,
            use_ranking=args.use_ranking,
        )

        representations.append((evaluations, suggestion))

    if len(representations) == 0:
        if err is not None:
            raise err
        else:
            raise ValueError(
                "Expected to successfully test at least one representation"
            )

    test: KramersKronigResult
    if len(representations) > 1:
        test = suggest_representation([t[1] for t in representations])[0]
        for evaluations, suggestion in representations:
            if test in suggestion:
                break

    log_F_ext, tests, statistic = evaluations[0]
    test = suggestion[0]

    if len(evaluations) > 1 and (args.plot_log_F_ext_3d or args.plot_log_F_ext_2d):
        if agg_backend:
            print_func(
                "Cannot plot the time constant extensions when matplotlib is set to use the Agg backend!"
            )
        else:
            fig, ax = mpl.plot_log_F_ext(
                evaluations,
                projection="2d" if args.plot_log_F_ext_2d else "3d",
                legend=not args.plot_no_legend,
            )
            fig.tight_layout()
            plt.show()
            plt.close()

    if args.num_RC > 0:
        if not (tests[0].num_RC <= args.num_RC <= tests[-1].num_RC):
            raise ValueError(
                f"Expected the specified number of RC elements to be {tests[0].num_RC} <= {args.num_RC=} <= {tests[-1].num_RC}"
            )

        suggestion = (
            [t for t in tests if t.num_RC == args.num_RC][0],
            {args.num_RC: 0.0},
            (
                max(args.lower_limit, min(tests, key=lambda t: t.num_RC).num_RC)
                if args.lower_limit != 0
                else suggestion[2]
            ),
            (
                min(args.upper_limit, max(tests, key=lambda t: t.num_RC).num_RC)
                if args.upper_limit != 0
                else suggestion[3]
            ),
        )

    test: KramersKronigResult = suggestion[0]
    title: str = f"{data.get_label()}\n{test.get_label()}"
    kwargs = {
        "admittance": test.admittance if args.plot_immittance else args.plot_admittance,
        "title": title,
        "legend": not args.plot_no_legend,
        "colored_axes": args.plot_colored_axes,
        "moving_average_width": args.moving_average_width,
        "estimate_noise": args.plot_estimated_noise,
    }

    if args.plot_auto_limited_residuals:
        kwargs["limit"] = -1.0

    if len(args.suggestion_methods) == 1 and not args.plot_pseudo_chi_squared:
        return (
            test,
            lambda t=tests, s=suggestion, d=data: plot_num_RC_suggestion_method(
                tests=t,
                suggestion=s,
                data=d,
                method=args.suggestion_methods[0],
                mu_criterion=args.mu_criterion,
                beta=args.beta,
                **kwargs,
            ),
        )

    return (
        test,
        lambda t=tests, s=suggestion, d=data: mpl.plot_kramers_kronig_tests(
            tests=t,
            suggestion=s,
            data=d,
            **kwargs,
        ),
    )


def command(parser: ArgumentParser, args: Namespace, print_func: Callable = print):
    from pyimpspec import (
        DataSet,
        KramersKronigResult,
    )

    all_data_sets: Dict[str, List[DataSet]] = parse_inputs(args)
    num_paths: int = len(all_data_sets)
    num_paths_remaining: int = num_paths

    agg_backend: bool = get_backend().lower() == "agg"
    set_figure_size(args.plot_width, args.plot_height, args.plot_dpi)

    for path, data_sets in all_data_sets.items():
        list(map(lambda _: apply_filters(_, args), data_sets))
        num_data: int = len(data_sets)

        for i, data in enumerate(data_sets):
            if num_paths > 1 or num_data > 1:
                if len(data_sets) > 1:
                    print_func(f"{path}: {data.get_label() or i}")
                else:
                    print_func(f"{path}")

            test: KramersKronigResult
            plot: Callable
            test, plot = exploratory_tests(data, agg_backend, args, print_func)
            clear_default_handler_output()

            report: str = format_text(
                test.to_statistics_dataframe(
                    extended_statistics=args.extended_statistics,
                ),
                args,
            )

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
