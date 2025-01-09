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
from os.path import (
    abspath,
    join,
)
from typing import (
    Callable,
    Dict,
    IO,
    List,
    Type,
    Union,
)
import matplotlib.pyplot as plt
from matplotlib import get_backend
from numpy import (
    all as array_all,
    array,
    inf,
    isinf,
)
from .utility import (
    format_text,
    get_color,
    get_marker,
    get_output_name,
    get_text_extension,
    parse_circuits,
    set_figure_size,
    validate_output_dir,
)


def print_circuit_limits(cdc: str, args: Namespace, print_func: Callable):
    from pyimpspec import (
        Circuit,
        ComplexImpedance,
        Element,
        parse_cdc,
    )
    from pyimpspec.exceptions import InfiniteLimit

    print_func(f"CDC: {cdc}")
    circuit: Union[Circuit, Element] = parse_cdc(cdc)
    if len(circuit.get_elements()) == 1:
        circuit = circuit.get_elements()[0]

    if args.min_frequency == 0.0:
        try:
            dc_limit: ComplexImpedance = circuit.get_impedances(array([0.0]))[0]
            print_func(f"- Z(f -> 0) = {dc_limit:.3e}")
        except InfiniteLimit as e:
            print_func("- Z(f -> 0) = inf")
            print_func(f"  {e}")

    if isinf(args.max_frequency):
        try:
            inf_limit: ComplexImpedance = circuit.get_impedances(array([inf]))[0]
            print_func(f"- Z(f -> inf) = {inf_limit:.3e}")
        except InfiniteLimit as e:
            print_func("- Z(f -> inf) = inf")
            print_func(f"  {e}")


def print_elements(args: Namespace, print_func: Callable):
    from pyimpspec import (
        Circuit,
        DataSet,
        Element,
        get_elements,
        mpl,
        parse_cdc,
        simulate_spectrum,
    )
    from pyimpspec.analysis.utility import _interpolate

    elements: Dict[str, Type[Element]] = get_elements()

    symbol: str
    Class: Type[Element]
    if not args.element:
        print_func("Supported circuit elements:")
        for symbol, Class in get_elements().items():
            print_func(f"- {Class.get_description()}")
        return
    num_elements: int = len(args.element)

    i: int
    for i, symbol in enumerate(args.element):
        if symbol not in elements:
            raise ValueError(
                f"Expected an element symbol that exists in {elements} instead of {symbol=}"
            )

        if args.simulate:
            print_limits: bool = args.min_frequency == 0.0 or isinf(args.max_frequency)
            if print_limits:
                print_circuit_limits(symbol, args, print_func)

            else:
                circuit: Circuit = parse_cdc(symbol)
                data_sets: List[DataSet] = [
                    simulate_spectrum(
                        circuit,
                        _interpolate(
                            [args.max_frequency, args.min_frequency],
                            args.num_per_decade,
                        ),
                        label=circuit.to_string(),
                    )
                ]

                marked_data_sets: List[DataSet] = []
                if args.mark_frequency:
                    marked_data_sets.append(
                        simulate_spectrum(
                            circuit,
                            args.mark_frequency,
                            label="",
                        )
                    )

                plot: Callable = {
                    "data": mpl.plot_data,
                    "nyquist": mpl.plot_nyquist,
                    "bode": mpl.plot_bode,
                    "magnitude": mpl.plot_magnitude,
                    "phase": mpl.plot_phase,
                    "real-imaginary": mpl.plot_real_imaginary,
                    "real": mpl.plot_real,
                    "imaginary": mpl.plot_imaginary,
                }[args.plot_type]
                individual_plots(
                    data_sets,
                    marked_data_sets,
                    plot,
                    args,
                    print_func=lambda *a, **k: (a, k),
                )

        else:
            Class = elements[symbol]
            element: Element = Class()

            if args.sympy or args.latex:
                from sympy import (
                    Expr,
                    latex,
                    pprint,
                    sympify,
                )

                expr: Expr = sympify(Class._equation, evaluate=False)
                if args.sympy:
                    pprint(expr)
                elif args.latex:
                    print_func(f"Z = {latex(expr, imaginary_unit='j')}")

            else:
                print_func(element.get_extended_description())

        if num_elements > 1 and i < num_elements - 1:
            print_func("\n")


def overlay_plot(
    data_sets: List["DataSet"],  # noqa: F821
    marked_data_sets: List["DataSet"],  # noqa: F821
    plot: Callable,
    args: Namespace,
):
    from pyimpspec import (
        ComplexImpedances,
        DataSet,
        mpl,
    )

    if not (args.output_name[0] != "" or not args.output):
        raise ValueError("Expected an output name")

    data: DataSet = data_sets.pop(0)
    Z: ComplexImpedances = data.get_impedances()
    color: str = get_color(args.plot_color)
    marker: str = get_marker(args.plot_marker)
    kwargs: dict = {
        "line": not array_all(Z == Z[0]),
        "legend": not args.plot_no_legend,
        "colors": {
            "impedance": color,
            "real": color,
            "imaginary": color,
            "magnitude": color,
            "phase": color,
        },
        "markers": {
            "impedance": marker,
            "real": marker,
            "imaginary": marker,
            "magnitude": marker,
            "phase": marker,
        },
        "colored_axes": args.plot_colored_axes,
        "admittance": args.plot_admittance,
    }

    figure, axes = plot(
        data,
        **kwargs,
    )
    kwargs["colored_axes"] = False
    kwargs.update(
        {
            "figure": figure,
            "axes": axes,
        }
    )

    marked_kwargs: dict
    if args.mark_frequency:
        marked_kwargs = kwargs.copy()
        marked_kwargs.update(
            {
                "line": False,
                "legend": False,
            }
        )
        data = marked_data_sets.pop(0)
        plot(
            data,
            **marked_kwargs,
        )

        if plot == mpl.plot_nyquist and args.annotate_frequency:
            for f, Z in zip(data.get_frequencies(), data.get_impedances()):
                axes[0].annotate(
                    f"{f:.1g} Hz",
                    (
                        Z.real,
                        -Z.imag,
                    ),
                    xytext=(
                        10,
                        -10,
                    ),
                    textcoords="offset points",
                    color=color,
                )

    i: int
    for i, data in enumerate(data_sets):
        Z = data.get_impedances()
        color: str = get_color(args.plot_color)
        marker: str = get_marker(args.plot_marker)
        kwargs.update(
            {
                "line": not array_all(Z == Z[0]),
                "color": color,
                "colors": {
                    "impedance": color,
                    "real": color,
                    "imaginary": color,
                    "magnitude": color,
                    "phase": color,
                },
                "markers": {
                    "impedance": marker,
                    "real": marker,
                    "imaginary": marker,
                    "magnitude": marker,
                    "phase": marker,
                },
            }
        )
        plot(
            data,
            **kwargs,
        )

        if args.mark_frequency:
            marked_kwargs = kwargs.copy()
            marked_kwargs.update(
                {
                    "line": False,
                    "legend": False,
                }
            )
            plot(
                marked_data_sets[i],
                **marked_kwargs,
            )

            if plot == mpl.plot_nyquist and args.annotate_frequency:
                for f, Z in zip(
                    marked_data_sets[i].get_frequencies(),
                    marked_data_sets[i].get_impedances(),
                ):
                    axes[0].annotate(
                        f"{f:.1g} Hz",
                        (
                            Z.real,
                            -Z.imag,
                        ),
                        xytext=(
                            10,
                            -10,
                        ),
                        textcoords="offset points",
                        color=color,
                    )

    figure.tight_layout()

    if args.output:
        extension: str = args.plot_format
        if extension.startswith("."):
            extension = extension[1:]

        output_path: str = abspath(
            join(
                args.output_dir,
                f"{args.output_name[0]}.{extension}",
            )
        )
        figure.savefig(output_path, dpi=args.plot_dpi)

    elif not get_backend().lower() == "agg":
        plt.show()

    plt.close()


def individual_plots(
    data_sets: List["DataSet"],  # noqa: F821
    marked_data_sets: List["DataSet"],  # noqa: F821
    plot: Callable,
    args: Namespace,
    print_func: Callable,
):
    from pyimpspec import (
        ComplexImpedances,
        DataSet,
        mpl,
    )

    if not (len(args.output_name) == len(args.input) or not args.output):
        raise ValueError(f"Expected {len(args.input)} output names")

    agg_backend: bool = get_backend().lower() == "agg"
    kwargs = {
        "legend": not args.plot_no_legend,
        "colored_axes": args.plot_colored_axes,
        "admittance": args.plot_admittance,
    }
    num_data: int = len(data_sets)

    i: int
    data: DataSet
    for i, data in enumerate(data_sets):
        print_func(data.get_label())
        kwargs["title"] = data.get_label() if args.plot_title else None
        Z: ComplexImpedances = data.get_impedances()
        kwargs["line"] = not array_all(Z == Z[0])

        if "figure" in kwargs:
            del kwargs["figure"]
        if "axes" in kwargs:
            del kwargs["axes"]
        figure, axes = plot(data, **kwargs)

        if args.mark_frequency:
            marked_kwargs = kwargs.copy()
            marked_kwargs.update(
                {
                    "line": False,
                    "legend": False,
                    "figure": figure,
                    "axes": axes,
                }
            )
            plot(
                marked_data_sets[i],
                **marked_kwargs,
            )

            if plot == mpl.plot_nyquist and args.annotate_frequency:
                for f, Z in zip(
                    marked_data_sets[i].get_frequencies(),
                    marked_data_sets[i].get_impedances(),
                ):
                    axes[0].annotate(
                        f"{f:.1g} Hz",
                        xy=(
                            Z.real,
                            -Z.imag,
                        ),
                        xytext=(
                            10,
                            -10,
                        ),
                        textcoords="offset points",
                    )

        result: str = format_text(data.to_dataframe(), args)
        figure.tight_layout()

        if args.output:
            extension: str = args.plot_format
            if extension.startswith("."):
                extension = extension[1:]

            output_path: str = abspath(
                join(
                    args.output_dir,
                    f"{args.output_name[i]}.{extension}",
                )
            )
            figure.savefig(output_path, dpi=args.plot_dpi)

            extension = get_text_extension(args.output_format)
            if extension.startswith("."):
                extension = extension[1:]

            output_path = abspath(
                join(
                    args.output_dir,
                    f"{args.output_name[i]}.{extension}",
                )
            )
            fp: IO
            with open(output_path, "w") as fp:
                fp.write(result)

        elif not agg_backend:
            print_func(result)
            plt.show()

        plt.close()

        if i < num_data - 1:
            print_func("")


def simulate_spectra(args: Namespace, print_func: Callable):
    from pyimpspec import (
        Circuit,
        DataSet,
        mpl,
        parse_cdc,
        simulate_spectrum,
    )
    from pyimpspec.analysis.utility import _interpolate

    data_sets: List[DataSet] = []
    marked_data_sets: List[DataSet] = []

    print_limits: bool = args.min_frequency == 0.0 or isinf(args.max_frequency)
    if print_limits:
        num_circuits: int = len(args.input)

        i: int
        cdc: str
        for i, cdc in enumerate(args.input):
            print_circuit_limits(cdc, args, print_func)
            if num_circuits > 1 and i < num_circuits - 1:
                print_func("\n")

        return

    circuit: Circuit
    for circuit in parse_circuits(args):
        data_sets.append(
            simulate_spectrum(
                circuit,
                _interpolate(
                    [args.max_frequency, args.min_frequency], args.num_per_decade
                ),
                label=circuit.to_string(),
            )
        )

        if args.mark_frequency:
            marked_data_sets.append(
                simulate_spectrum(
                    circuit,
                    args.mark_frequency,
                    label="",
                )
            )

    plot_types: Dict[str, Callable] = {
        "imaginary": mpl.plot_imaginary,
        "magnitude": mpl.plot_magnitude,
        "nyquist": mpl.plot_nyquist,
        "phase": mpl.plot_phase,
        "real": mpl.plot_real,
    }

    if not args.plot_overlay:
        plot_types.update(
            {
                "bode": mpl.plot_bode,
                "real-imaginary": mpl.plot_real_imaginary,
            }
        )

    plot: Callable = plot_types[args.plot_type]
    if args.plot_overlay:
        overlay_plot(data_sets, marked_data_sets, plot, args)
    else:
        individual_plots(data_sets, marked_data_sets, plot, args, print_func)


def circuit_diagrams(args: Namespace):
    from schemdraw import Drawing
    from pyimpspec import Circuit

    if not (
        not args.output
        or (
            len(args.output_name) == len(args.input)
            and not any(map(lambda _: _.strip() == "", args.output_name))
        )
    ):
        raise ValueError(f"Expected {len(args.input)} output name(s)!")

    agg_backend: bool = get_backend().lower() == "agg"

    i: int
    circuit: Circuit
    for i, circuit in enumerate(parse_circuits(args)):
        drawing: Drawing = circuit.to_drawing(
            node_height=args.node_height,
            left_terminal_label=args.left_terminal_label,
            right_terminal_label=args.right_terminal_label,
            hide_labels=args.hide_labels,
            running=args.running_count,
        )

        if args.output:
            output_name: str = get_output_name(circuit.to_string(), args.output_name, i)
            extension: str = args.plot_format
            if extension.startswith("."):
                extension = extension[1:]

            output_path: str = abspath(
                join(
                    args.output_dir,
                    f"{output_name or circuit.to_string()}.{extension}",
                )
            )
            drawing.save(output_path, dpi=args.plot_dpi)

        elif not agg_backend:
            drawing.draw()


def command(parser: ArgumentParser, args: Namespace, print_func: Callable = print):
    if args.output:
        validate_output_dir(args.output_dir)

    if not args.input:
        print_elements(args, print_func)
        return

    set_figure_size(args.plot_width, args.plot_height, args.plot_dpi)

    if args.output:
        if args.plot_overlay or len(args.input) == 1:
            if len(args.output_name) != 1:
                raise ValueError("Expected an output name")

        elif len(args.input) > 1:
            if len(args.output_name) != len(args.input):
                raise ValueError(f"Expected {len(args.input)} output names")

    if args.simulate:
        simulate_spectra(args, print_func)

    elif args.sympy or args.latex:
        from sympy import pprint
        from pyimpspec import Circuit

        circuit: Circuit
        for circuit in parse_circuits(args):
            if args.sympy:
                pprint(circuit.to_sympy())
            elif args.latex:
                print_func(circuit.to_latex())

    else:
        circuit_diagrams(args)
