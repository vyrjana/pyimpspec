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

from argparse import ArgumentParser, Namespace
from contextlib import redirect_stdout
from io import StringIO
from os.path import join
from tempfile import gettempdir
from typing import (
    Any,
    Callable,
    Dict,
    List,
)
from unittest import TestCase
import pyimpspec.cli.config as config_module
from pyimpspec.cli.circuit import command as circuit_command
from pyimpspec.cli.drt import command as drt_command
from pyimpspec.cli.fit import command as fit_command
from pyimpspec.cli.parse import command as parse_command
from pyimpspec.cli.plot import command as plot_command
from pyimpspec.cli.license import command as show_command
from pyimpspec.cli.test import command as test_command
from pyimpspec.cli.zhit import command as zhit_command
from pyimpspec.cli.config import (
    command as config_command,
    get_argument_parser,
)
from pyimpspec.cli.args import (
    add_output_args,
    add_plot_args,
)
from pyimpspec.cli.utility import (
    get_output_name,
    set_figure_size,
    validate_input_paths,
    validate_output_dir,
)
import matplotlib
from matplotlib import rcParams

config_module._IGNORE_USER_CONFIG = True
matplotlib.use("Agg")  # Suppress plot windows


def redirect_output(func: Callable) -> List[str]:
    buffer: StringIO = StringIO()
    with redirect_stdout(buffer):
        func()
    lines: List[str] = buffer.getvalue().split("\n")
    return list(map(str.strip, lines))


COMMAND_LINE_ARGUMENTS: Dict[str, List[Namespace]] = {}


class TestArguments(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output_dir: str = join(gettempdir(), "pyimpspec-test-cli-output")
        cls.commands: List[str] = [
            "parse",
            "plot",
            "test",
            "zhit",
            "drt",
            "fit",
            "circuit",
            "config",
            "show",
        ]

        COMMAND_LINE_ARGUMENTS.update({_: [] for _ in cls.commands})

        cls.parser: ArgumentParser = get_argument_parser()
        cls.parsers: Dict[str, Callable] = {
            "parse": lambda *a: cls.parser.parse_args(["parse", *a]),
            "plot": lambda *a: cls.parser.parse_args(["plot", *a]),
            "test": lambda *a: cls.parser.parse_args(["test", *a]),
            "zhit": lambda *a: cls.parser.parse_args(["zhit", *a]),
            "drt": lambda *a: cls.parser.parse_args(["drt", *a]),
            "fit": lambda *a: cls.parser.parse_args(["fit", "R(RC)(RW)", *a]),
            "circuit": lambda *a: cls.parser.parse_args(["circuit", *a]),
            "config": lambda *a: cls.parser.parse_args(["config", *a]),
            "show": lambda *a: cls.parser.parse_args(["show", *a]),
        }
        cls.data_paths: List[str] = [
            "./data-comma.csv",
            "./data.dta",
        ]

    def test_input(self):
        skip: List[str] = [
            "circuit",
            "config",
            "show",
        ]
        self.assertEqual(
            set(self.commands) - set(self.parsers.keys()) - set(skip),
            set(),
            msg=set(self.parsers.keys()) - set(skip),
        )

        command: str
        parser: Callable
        for command, parser in self.parsers.items():
            if command in skip:
                continue

            args: Namespace
            # Defaults
            args = parser(*self.data_paths)

            self.assertTrue(args.input == self.data_paths)
            self.assertEqual(args.low_pass_cutoff, -1.0)
            self.assertEqual(args.high_pass_cutoff, -1.0)

            COMMAND_LINE_ARGUMENTS[command].append(args)

            for args in [
                # Long names
                parser(
                    *self.data_paths,
                    "--low-pass-filter",
                    "7.1e3",
                    "--high-pass-filter",
                    "5.21",
                ),
                # Short names
                parser(
                    *self.data_paths,
                    "-lpf",
                    "7.1e3",
                    "-hpf",
                    "5.21",
                ),
            ]:
                self.assertTrue(args.input == self.data_paths)
                self.assertEqual(args.low_pass_cutoff, 7.1e3)
                self.assertEqual(args.high_pass_cutoff, 5.21)

            COMMAND_LINE_ARGUMENTS[command].append(args)

    def test_output(self):
        skip: List[str] = [
            "config",
            "show",
        ]
        self.assertEqual(
            set(self.commands) - set(self.parsers.keys()) - set(skip),
            set(),
            msg=set(self.parsers.keys()) - set(skip),
        )

        command: str
        parser: Callable
        for command, parser in self.parsers.items():
            if command in skip:
                continue

            paths: List[str] = self.data_paths if command != "circuit" else []
            args: Namespace
            # Defaults
            args = parser(*paths)

            if command != "circuit":
                self.assertTrue(args.input == self.data_paths)
            self.assertEqual(args.output, False)
            self.assertEqual(args.output_format, "markdown")
            self.assertEqual(args.output_name, [""])
            self.assertEqual(args.output_significant_digits, 6)
            self.assertEqual(args.output_dir, ".")

            COMMAND_LINE_ARGUMENTS[command].append(args)

            for args in [
                # Long names
                parser(
                    *paths,
                    "--output-to",
                    "--output-format",
                    "csv",
                    "--output-name",
                    "foo",
                    "--output-significant-digits",
                    "3",
                    "--output-dir",
                    self.output_dir,
                ),
                # Short names
                parser(
                    *paths,
                    "-ot",
                    "-of",
                    "csv",
                    "-on",
                    "foo",
                    "-osd",
                    "3",
                    "-od",
                    self.output_dir,
                ),
            ]:
                if command != "circuit":
                    self.assertTrue(args.input == self.data_paths)
                self.assertEqual(args.output, True)
                self.assertEqual(args.output_format, "csv")
                self.assertEqual(args.output_name, ["foo"])
                self.assertEqual(args.output_significant_digits, 3)
                self.assertEqual(args.output_dir, self.output_dir)

            COMMAND_LINE_ARGUMENTS[command].append(args)

    def test_plot(self):
        skip: List[str] = [
            "parse",
            "config",
            "show",
        ]
        self.assertEqual(
            set(self.commands) - set(self.parsers.keys()) - set(skip),
            set(),
            msg=set(self.parsers.keys()) - set(skip),
        )

        command: str
        parser: Callable
        for command, parser in self.parsers.items():
            if command in skip:
                continue

            paths: List[str] = self.data_paths if command != "circuit" else []
            args: Namespace
            # Defaults
            args = parser(*paths)

            if command != "circuit":
                self.assertTrue(args.input == self.data_paths)
            self.assertEqual(args.plot_width, "12.8in")
            self.assertEqual(args.plot_height, "7.2in")
            self.assertEqual(args.plot_dpi, 100)
            self.assertEqual(args.plot_format, "png")
            self.assertEqual(args.plot_overlay, False)
            self.assertEqual(args.plot_no_legend, False)
            self.assertEqual(args.plot_colored_axes, False)
            self.assertEqual(args.plot_title, False)
            self.assertEqual(args.plot_color, [])
            self.assertEqual(args.plot_marker, [])
            COMMAND_LINE_ARGUMENTS[command].append(args)

            for args in [
                # Long names
                parser(
                    *paths,
                    "--plot-width",
                    "1280px",
                    "--plot-height",
                    "720px",
                    "--plot-dpi",
                    "300",
                    "--plot-format",
                    "svg",
                    "--plot-overlay",
                    "--plot-no-legend",
                    "--plot-colored-axes",
                    "--plot-title",
                    "--plot-colors",
                    "black",
                    "blue",
                    "red",
                    "--plot-markers",
                    "o",
                    "s",
                    "+",
                ),
                # Short names
                parser(
                    *paths,
                    "-pw",
                    "1280px",
                    "-ph",
                    "720px",
                    "-dpi",
                    "300",
                    "-pf",
                    "svg",
                    "-po",
                    "-pnl",
                    "-pca",
                    "-pT",
                    "-pc",
                    "black",
                    "blue",
                    "red",
                    "-pm",
                    "o",
                    "s",
                    "+",
                ),
            ]:
                if command != "circuit":
                    self.assertTrue(args.input == self.data_paths)
                self.assertEqual(args.plot_width, "1280px")
                self.assertEqual(args.plot_height, "720px")
                self.assertEqual(args.plot_dpi, 300)
                self.assertEqual(args.plot_format, "svg")
                self.assertEqual(args.plot_overlay, True)
                self.assertEqual(args.plot_no_legend, True)
                self.assertEqual(args.plot_colored_axes, True)
                self.assertEqual(args.plot_title, True)
                self.assertEqual(args.plot_color, ["black", "blue", "red"])
                self.assertEqual(args.plot_marker, ["o", "s", "+"])

            COMMAND_LINE_ARGUMENTS[command].append(args)


class TestParse(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parser: ArgumentParser = get_argument_parser()

    def test_args(self):
        for args in COMMAND_LINE_ARGUMENTS["parse"]:
            lines: List[str] = []
            parse_command(
                self.parser,
                args,
                print_func=lambda *a, **k: lines.extend(a),
            )


class TestPlot(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parser: ArgumentParser = get_argument_parser()
        cls.data_paths: List[str] = [
            "./data-comma.csv",
            "./data.dta",
        ]

    def test_args(self):
        parser: Callable = lambda *a: self.parser.parse_args(
            ["plot", *self.data_paths, *a]
        )
        args: Namespace = parser("--plot-type", "bode")
        self.assertEqual(args.plot_type, "bode")
        COMMAND_LINE_ARGUMENTS["plot"].extend(
            [
                args,
                parser("-pt", "nyquist"),
                parser("-pt", "data"),
                parser("-pt", "magnitude"),
                parser("-pt", "phase"),
                parser("-pt", "real-imaginary"),
                parser("-pt", "real"),
                parser("-pt", "imaginary"),
            ]
        )

        for args in COMMAND_LINE_ARGUMENTS["plot"]:
            lines: List[str] = []
            plot_command(
                self.parser,
                args,
                print_func=lambda *a, **k: lines.extend(a),
            )


class TestTest(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parser: ArgumentParser = get_argument_parser()
        cls.data_paths: List[str] = [
            "./data-comma.csv",
            "./data.dta",
        ]

    def test_args(self):
        parser: Callable = lambda *a: self.parser.parse_args(
            ["test", *self.data_paths, *a]
        )

        long: str
        short: str
        dest: str
        value: Any
        for long, short, dest, value in [
            ("--num-RC", "-n", "num_RC", 6),
            ("--max-num-RC", "-N", "max_num_RC", 8),
            ("--mu-criterion", "--mu-criterion", "mu_criterion", 0.3),
            ("--beta", "--beta", "beta", 0.91),
            ("--suggestion-methods", "-sm", "suggestion_methods", [3]),
            ("--mean", "--mean", "use_mean", True),
            ("--sum", "--sum", "use_sum", True),
            ("--ranking", "--ranking", "use_ranking", True),
            ("--no-capacitance", "-C", "no_capacitance", True),
            ("--no-inductance", "-L", "no_inductance", True),
            ("--admittance", "-Y", "admittance", True),
            ("--impedance", "-Z", "impedance", True),
            ("--min-log-F-ext", "-mlFe", "min_log_F_ext", -0.4),
            ("--max-log-F-ext", "-MlFe", "max_log_F_ext", 0.6),
            ("--log-F-ext", "-lFe", "log_F_ext", 0.1),
            ("--num-F-ext-evaluations", "-nFee", "num_F_ext_evaluations", 15),
            (
                "--no-rapid-F-ext-evaluations",
                "-nrFe",
                "no_rapid_F_ext_evaluations",
                True,
            ),
            ("--lower-limit", "-ll", "lower_limit", 2),
            ("--upper-limit", "-ul", "upper_limit", 12),
            ("--test", "-t", "test", "real"),
            ("--cnls-method", "--cnls-method", "cnls_method", "least_squares"),
            ("--max-nfev", "--max-nfev", "max_nfev", 2),
            ("--timeout", "-T", "timeout", 32),
            ("--num-procs", "--num-procs", "num_procs", 3),
            ("--plot-pseudo-chi-squared", "-ppcs", "plot_pseudo_chi_squared", True),
            ("--plot-moving-average-width", "-pmaw", "moving_average_width", 7),
            ("--plot-immittance", "-pX", "plot_immittance", True),
            ("--plot-estimated-noise", "-pen", "plot_estimated_noise", True),
            ("--plot-log-F-ext-3d", "-plFe3d", "plot_log_F_ext_3d", True),
            ("--plot-log-F-ext-2d", "-plFe2d", "plot_log_F_ext_2d", True),
        ]:
            args_long: Namespace
            if isinstance(value, bool):
                args_long = parser(long)
            elif isinstance(value, list):
                args_long = parser(long, " ".join((str(v) for v in value)))
            else:
                args_long = parser(long, str(value))

            self.assertEqual(getattr(args_long, dest), value)

            if short != "":
                args_short: Namespace
                if isinstance(value, bool):
                    args_short = parser(short)
                elif isinstance(value, list):
                    args_short = parser(short, " ".join((str(v) for v in value)))
                else:
                    args_short = parser(short, str(value))

                self.assertEqual(getattr(args_long, dest), getattr(args_short, dest))

            COMMAND_LINE_ARGUMENTS["test"].append(args_long)

        args: Namespace
        for args in COMMAND_LINE_ARGUMENTS["test"]:
            lines: List[str] = []
            test_command(
                self.parser,
                args,
                print_func=lambda *a, **k: lines.extend(a),
            )


class TestZHIT(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parser: ArgumentParser = get_argument_parser()
        cls.data_paths: List[str] = [
            "./data-comma.csv",
            "./data.dta",
        ]

    def test_args(self):
        parser: Callable = lambda *a: self.parser.parse_args(
            ["zhit", *self.data_paths, *a]
        )

        long: str
        short: str
        dest: str
        value: Any
        for long, short, dest, value in [
            ("--plot-type", "-pt", "plot_type", "nyquist"),
            ("--smoothing", "-s", "smoothing", "savgol"),
            ("--interpolation", "-i", "interpolation", "cubic"),
            ("--window", "-w", "weights_window", "hann"),
            ("--window-center", "-wc", "weights_center", 2.0),
            ("--window-width", "-ww", "weights_width", 1.0),
            ("--num-points", "-np", "num_points", 3),
            ("--polynomial-order", "-p", "polynomial_order", 2),
            ("--num-iterations", "-ni", "num_iterations", 5),
            ("--num-procs", "", "num_procs", 2),
        ]:
            args_long: Namespace
            if isinstance(value, bool):
                args_long = parser(long)
            else:
                args_long = parser(long, str(value))
            self.assertEqual(getattr(args_long, dest), value)

            if short != "":
                args_short: Namespace
                if isinstance(value, bool):
                    args_short = parser(short)
                else:
                    args_short = parser(short, str(value))
                self.assertEqual(getattr(args_long, dest), getattr(args_short, dest))

            COMMAND_LINE_ARGUMENTS["zhit"].append(args_long)

        args: Namespace
        for args in COMMAND_LINE_ARGUMENTS["zhit"]:
            lines: List[str] = []
            zhit_command(
                self.parser,
                args,
                print_func=lambda *a, **k: lines.extend(a),
            )


# TODO: Command-specific arguments
class TestDRT(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parser: ArgumentParser = get_argument_parser()
        cls.data_paths: List[str] = [
            "./data-comma.csv",
            "./data.dta",
        ]

    def test_args(self):
        parser: Callable = lambda *a: self.parser.parse_args(
            ["drt", *self.data_paths, *a]
        )

        long: str
        short: str
        dest: str
        value: Any
        for long, short, dest, value in [
            ("--method", "-me", "method", "tr-rbf"),
            ("--mode", "-mo", "mode", "real"),
            ("--lambda-value", "-lv", "lambda_value", 1e-5),
            ("--rbf-type", "-rt", "rbf_type", "cauchy"),
            ("--derivative-order", "-do", "derivative_order", 2),
            ("--rbf-shape", "-rs", "rbf_shape", "factor"),
            ("--shape-coeff", "-sc", "shape_coeff", 0.75),
            ("--inductance", "-L", "inductance", True),
            ("--credible-intervals", "-ci", "credible_intervals", True),
            ("--timeout", "-T", "timeout", 5),
            ("--num-samples", "-ns", "num_samples", 2500),
            ("--num-attempts", "-na", "num_attempts", 2),
            ("--max-symmetry", "-ms", "maximum_symmetry", 0.6),
            ("--circuit", "-c", "circuit", "R(RC)(RQ)"),
            ("--gaussian-width", "-gw", "gaussian_width", 0.1),
            ("--num-per-decade", "-npd", "num_per_decade", 5),
            ("--threshold", "-t", "peak_threshold", 0.2),
            ("--max-nfev", "", "max_nfev", 4),
            ("--max-iter", "--max-iter", "max_iter", 100000),
            ("--num-procs", "", "num_procs", 2),
            ("--plot-frequency", "-pF", "plot_frequency", True),
            ("--analyze-peaks", "-ap", "analyze_peaks", True),
            ("--num-peaks", "-np", "num_peaks", 2),
            ("--peak-positions", "-pp", "peak_positions", [1e-1, 0.2]),
            ("--disallow-skew", "-ds", "disallow_skew", True),
        ]:
            args_long: Namespace
            if isinstance(value, bool):
                args_long = parser(long)
            elif isinstance(value, list):
                args_long = parser(long, *map(str, value))
            else:
                args_long = parser(long, str(value))
            self.assertEqual(getattr(args_long, dest), value)

            if short != "":
                args_short: Namespace
                if isinstance(value, bool):
                    args_short = parser(short)
                elif isinstance(value, list):
                    args_short = parser(short, *map(str, value))
                else:
                    args_short = parser(short, str(value))
                self.assertEqual(getattr(args_long, dest), getattr(args_short, dest))

            COMMAND_LINE_ARGUMENTS["drt"].append(args_long)

        args: Namespace
        for args in COMMAND_LINE_ARGUMENTS["drt"]:
            lines: List[str] = []
            drt_command(
                self.parser,
                args,
                print_func=lambda *a, **k: lines.extend(a),
            )


# TODO: Command-specific arguments
class TestFit(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parser: ArgumentParser = get_argument_parser()
        cls.data_paths: List[str] = [
            "./data-comma.csv",
            "./data.dta",
        ]

    def test_args(self):
        parser: Callable = lambda *a: self.parser.parse_args(
            ["fit", "R(RC)(RW)", *self.data_paths, *a]
        )

        long: str
        short: str
        dest: str
        value: Any
        for long, short, dest, value in [
            ("--method", "-me", "method", "least_squares"),
            ("--weight", "-we", "weight", "proportional"),
            ("--max-nfev", "", "max_nfev", 5),
            ("--num-procs", "", "num_procs", 2),
        ]:
            args_long: Namespace
            if isinstance(value, bool):
                args_long = parser(long)
            else:
                args_long = parser(long, str(value))
            self.assertEqual(getattr(args_long, dest), value)

            if short != "":
                args_short: Namespace
                if isinstance(value, bool):
                    args_short = parser(short)
                else:
                    args_short = parser(short, str(value))
                self.assertEqual(getattr(args_long, dest), getattr(args_short, dest))

            COMMAND_LINE_ARGUMENTS["fit"].append(args_long)

        args: Namespace
        for args in COMMAND_LINE_ARGUMENTS["fit"]:
            lines: List[str] = []
            fit_command(
                self.parser,
                args,
                print_func=lambda *a, **k: lines.extend(a),
            )


# TODO: Command-specific arguments
class TestCircuit(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parser: ArgumentParser = get_argument_parser()

    def test_args(self):
        parser: Callable = lambda *a: self.parser.parse_args(["circuit", *a])

        long: str
        short: str
        dest: str
        value: Any
        for long, short, dest, value in [
            # ("circuit", "", "circuit", []),
            # ("--element", "-e", "element", []),
            ("--simulate", "-s", "simulate", True),
            ("--min-frequency", "-f", "min_frequency", 1.3e0),
            ("--max-frequency", "-F", "max_frequency", 5e3),
            ("--num-per-decade", "-npd", "num_per_decade", 25),
            # ("--mark-frequency", "-mf", "mark_frequency", []),
            ("--annotate-frequency", "-af", "annotate_frequency", True),
            ("--node-height", "-nh", "node_height", 1.2),
            ("--left-terminal-label", "-ltl", "left_terminal_label", "LEFT"),
            ("--right-terminal-label", "-rtl", "right_terminal_label", "RIGHT"),
            ("--hide-labels", "-H", "hide_labels", True),
            ("--plot-type", "-pt", "plot_type", "phase"),
            ("--sympy", "-S", "sympy", True),
            ("--latex", "-L", "latex", True),
            ("--running-count", "-rc", "running_count", True),
        ]:
            args_long: Namespace
            if isinstance(value, bool):
                args_long = parser(long)
            else:
                args_long = parser(long, str(value))
            self.assertEqual(getattr(args_long, dest), value)

            if short != "":
                args_short: Namespace
                if isinstance(value, bool):
                    args_short = parser(short)
                else:
                    args_short = parser(short, str(value))
                self.assertEqual(getattr(args_long, dest), getattr(args_short, dest))

            COMMAND_LINE_ARGUMENTS["circuit"].append(args_long)

        args: Namespace
        for args in COMMAND_LINE_ARGUMENTS["circuit"]:
            lines: List[str] = []
            circuit_command(
                self.parser,
                args,
                print_func=lambda *a, **k: lines.extend(a),
            )


# TODO: Command-specific arguments
class TestConfig(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parser: ArgumentParser = get_argument_parser()

    def test_args(self):
        parser: Callable = lambda *a: self.parser.parse_args(["config", *a])

        long: str
        short: str
        dest: str
        value: Any
        for long, short, dest, value in [
            ("--save", "", "save_file", True),
            ("--update", "", "update_file", True),
        ]:
            args_long: Namespace
            if isinstance(value, bool):
                args_long = parser(long)
            else:
                args_long = parser(long, str(value))
            self.assertEqual(getattr(args_long, dest), value)

            if short != "":
                args_short: Namespace
                if isinstance(value, bool):
                    args_short = parser(short)
                else:
                    args_short = parser(short, str(value))
                self.assertEqual(getattr(args_long, dest), getattr(args_short, dest))

            COMMAND_LINE_ARGUMENTS["config"].append(args_long)

        args: Namespace
        for args in COMMAND_LINE_ARGUMENTS["config"]:
            lines: List[str] = []
            config_command(
                self.parser,
                args,
                print_func=lambda *a, **k: lines.extend(a),
            )


# TODO: Command-specific arguments
# item
class TestShow(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parser: ArgumentParser = get_argument_parser()

    def test_args(self):
        args: Namespace
        for args in COMMAND_LINE_ARGUMENTS["show"]:
            lines: List[str] = []
            show_command(
                self.parser,
                args,
                print_func=lambda *a, **k: lines.extend(a),
            )


class TestUtility(TestCase):
    def test_validate_input_paths(self):
        paths: List[str] = ["./data-comma.csv", "./case-fragment.csv"]
        validate_input_paths(paths)
        with self.assertRaises(FileNotFoundError):
            validate_input_paths(paths + ["./invalid-path"])

    def test_validate_output_paths(self):
        with self.assertRaises(NotADirectoryError):
            validate_output_dir("./data-comma.csv")

    def test_get_output_name(self):
        self.assertEqual(get_output_name("./data-comma.csv"), "data-comma")
        self.assertEqual(get_output_name("./data-comma.csv", i=0), "data-comma-1")
        self.assertEqual(get_output_name("./data-comma.csv", i=1), "data-comma-2")
        self.assertEqual(get_output_name("./data-comma.csv", [""]), "data-comma")
        self.assertEqual(get_output_name("./data-comma.csv", [""], i=0), "data-comma-1")
        self.assertEqual(get_output_name("./data-comma.csv", [""], i=1), "data-comma-2")
        self.assertEqual(get_output_name("./data-comma.csv", ["test"]), "test")
        self.assertEqual(get_output_name("./data-comma.csv", ["test"], i=0), "test")
        self.assertEqual(
            get_output_name("./data-comma.csv", ["test"], i=1),
            "data-comma-2",
        )
        self.assertEqual(
            get_output_name("./data-comma.csv", ["foo", "bar"], i=1),
            "bar",
        )
