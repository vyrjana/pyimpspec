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
    SUPPRESS,
)
from pyimpspec.analysis.drt import _METHODS as DRT_METHODS
from pyimpspec.analysis.drt.tr_rbf import (
    _RBF_TYPES,
    _RBF_SHAPES,
    _CROSS_VALIDATION_METHODS,
)
from pyimpspec.analysis.drt.lm import _MODEL_ORDER_METHODS
from pyimpspec.analysis.fitting import (
    _METHODS as FIT_METHODS,
    _WEIGHT_FUNCTIONS as FIT_WEIGHTS,
)


def add_input_args(parser: ArgumentParser):
    parser.add_argument(
        "input",
        metavar="PATH",
        nargs="+",
        type=str,
        help="The path(s) to data file(s) that should be parsed.",
    )
    parser.add_argument(
        "--low-pass-filter",
        "-lpf",
        metavar="FLOAT",
        type=float,
        dest="low_pass_cutoff",
        default=-1.0,
        help="The cutoff frequency for the low-pass filter.",
    )
    parser.add_argument(
        "--high-pass-filter",
        "-hpf",
        metavar="FLOAT",
        type=float,
        dest="high_pass_cutoff",
        default=-1.0,
        help="The cutoff frequency for the high-pass filter.",
    )
    parser.add_argument(
        "--exclude-indices",
        "-ei",
        metavar="INTEGER",
        nargs="+",
        type=int,
        dest="exclude_indices",
        default=[],
        help="The zero-based indices of data points to exclude. These indices correspond to the indices listed prior to any masking (i.e., when low- or high-pass filters have not yet been applied).",
    )
    parser.add_argument(
        "--nth-data-set",
        "-nds",
        nargs="+",
        type=int,
        dest="nth_data_set",
        default=[],
        help="The zero-based indices of the data sets in a file to include while ignoring any others.",
    )


def add_output_args(
    parser: ArgumentParser,
    multiple_names: bool = False,
):
    parser.add_argument(
        "--output-to",
        "-ot",
        dest="output",
        action="store_true",
        help="Output to file(s) in the output directory in the output format. Otherwise the output is simply printed.",
    )
    parser.add_argument(
        "--output-format",
        "-of",
        metavar="STRING",
        dest="output_format",
        type=str,
        default="markdown",
        help="The output format to use: 'markdown'/'md', 'csv', 'json', or 'latex'/'tex'. Defaults to 'markdown'.",
    )
    parser.add_argument(
        "--output-name",
        "-on",
        nargs="+" if multiple_names else 1,
        metavar="STRING",
        dest="output_name",
        type=str,
        default=[""],
        help="The name of the output file. Defaults to the name of the input file.",
    )
    parser.add_argument(
        "--output-significant-digits",
        "-osd",
        metavar="INTEGER",
        dest="output_significant_digits",
        type=int,
        default=6,
        help="The number of significant digits in the output",
    )
    parser.add_argument(
        "--output-dir",
        "-od",
        metavar="STRING",
        dest="output_dir",
        type=str,
        default=".",
        help="The path to the output directory. Defaults to the current working directory.",
    )
    parser.add_argument(
        "--output-indices",
        "-oi",
        action="store_true",
        dest="output_indices",
        help="Include zero-based indices in the text output.",
    )
    parser.add_argument(
        "--suppress-progress",
        dest="suppress_progress",
        action="store_true",
        help="Suppress printing progress messages.",
    )


def add_plot_args(
    parser: ArgumentParser,
    overlay: bool = True,
    title: bool = False,
    colors: bool = True,
    markers: bool = True,
):
    parser.add_argument(
        "--plot-width",
        "-pw",
        metavar="STRING",
        dest="plot_width",
        type=str,
        default="12.8in",
        help="The width of the figures. The following units can be specified: 'in' for inches, 'cm' for centimeters, and 'px' for pixels. Defaults to '12.8in'.",
    )
    parser.add_argument(
        "--plot-height",
        "-ph",
        metavar="STRING",
        dest="plot_height",
        type=str,
        default="7.2in",
        help="The height of the figures. The following units can be specified ('in' for inches, 'cm' for centimeters, and 'px' for pixels. Defaults to '7.2in'.",
    )
    parser.add_argument(
        "--plot-dpi",
        "-dpi",
        metavar="INTEGER",
        dest="plot_dpi",
        type=int,
        default=100,
        help="The number of dots (or pixels) per inch. Defaults to 100.",
    )
    parser.add_argument(
        "--plot-format",
        "-pf",
        metavar="STRING",
        dest="plot_format",
        type=str,
        default="png",
        help="The output format to use for plots. Defaults to PNG.",
    )
    parser.add_argument(
        "--plot-overlay",
        "-po",
        dest="plot_overlay",
        action="store_true",
        help=(
            "Overlay plots instead of generating separate plots."
            if overlay
            else SUPPRESS
        ),
    )
    parser.add_argument(
        "--plot-no-legend",
        "-pnl",
        dest="plot_no_legend",
        action="store_true",
        help="Omit the legend from the plot.",
    )
    parser.add_argument(
        "--plot-colored-axes",
        "-pca",
        dest="plot_colored_axes",
        action="store_true",
        help="Color the y-axes (if applicable to the plot type).",
    )
    parser.add_argument(
        "--plot-title",
        "-pT",
        dest="plot_title",
        action="store_true",
        help=("Add title to plot." if title else SUPPRESS),
    )
    parser.add_argument(
        "--plot-colors",
        "-pc",
        dest="plot_color",
        metavar="STRING",
        type=str,
        nargs="+",
        default=[],
        help=(
            "Specify colors to use instead of the default color scheme."
            if colors
            else SUPPRESS
        ),
    )
    parser.add_argument(
        "--plot-markers",
        "-pm",
        dest="plot_marker",
        metavar="STRING",
        type=str,
        nargs="+",
        default=[],
        help=(
            "Specify markers to use instead of the default marker scheme."
            if markers
            else SUPPRESS
        ),
    )
    parser.add_argument(
        "--plot-admittance",
        "-pY",
        dest="plot_admittance",
        action="store_true",
        help="Plot the admittance representation of the immittance spectrum.",
    )


def circuit_args(parser: ArgumentParser):
    parser.add_argument(
        "input",
        metavar="CDC",
        nargs="*",
        type=str,
        help="One or more circuit description codes.",
    )
    parser.add_argument(
        "--element",
        "-e",
        dest="element",
        metavar="SYMBOL",
        nargs="+",
        type=str,
        help="The symbols of one or more elements to get more information about.",
    )
    parser.add_argument(
        "--simulate",
        "-s",
        action="store_true",
        help="Simulate the impedance response(s) of the circuit(s).",
    )
    parser.add_argument(
        "--min-frequency",
        "-f",
        dest="min_frequency",
        type=float,
        default=1e-2,
        help="The minimum frequency to use when simulating impedance responses. Defaults to 1e-2 (i.e., 10 mHz).",
    )
    parser.add_argument(
        "--max-frequency",
        "-F",
        dest="max_frequency",
        type=float,
        default=1e5,
        help="The maximum frequency to use when simulating impedance responses. Defaults to 1e5 (i.e., 100 kHz).",
    )
    parser.add_argument(
        "--num-per-decade",
        "-npd",
        dest="num_per_decade",
        type=int,
        default=100,
        help="The number of points per decade to use when simulating impedance responses. Defaults to 100 points per decade.",
    )
    parser.add_argument(
        "--mark-frequency",
        "-mf",
        metavar="mark_frequency",
        nargs="*",
        type=float,
        default=[],
        help="Add markers at one or more frequencies to the simulated spectra.",
    )
    parser.add_argument(
        "--annotate-frequency",
        "-af",
        dest="annotate_frequency",
        action="store_true",
        help="If there are marked frequencies, then also annotate them with the corresponding frequencies.",
    )
    parser.add_argument(
        "--node-height",
        "-nh",
        metavar="node_height",
        type=float,
        default=1.5,
        help="The height of a node (i.e., one vertical step) in a circuit diagram.",
    )
    parser.add_argument(
        "--left-terminal-label",
        "-ltl",
        dest="left_terminal_label",
        type=str,
        default="",
        help="The label of the left-hand side terminal in circuit diagrams. Defaults to ''.",
    )
    parser.add_argument(
        "--right-terminal-label",
        "-rtl",
        dest="right_terminal_label",
        type=str,
        default="",
        help="The label of the right-hand side terminal in circuit diagrams. Defaults to ''.",
    )
    parser.add_argument(
        "--hide-labels",
        "-H",
        dest="hide_labels",
        action="store_true",
        help="Hide all terminal and element labels.",
    )
    parser.add_argument(
        "--plot-type",
        "-pt",
        metavar="STRING",
        dest="plot_type",
        type=str,
        default="nyquist",
        help="The type of plot to generate: 'bode', 'imaginary', 'magnitude', 'nyquist', 'phase', 'real', or 'real-imaginary'. Defaults to 'nyquist'.",
    )
    parser.add_argument(
        "--sympy",
        "-S",
        dest="sympy",
        action="store_true",
        help="Print the SymPy expression for the impedance of an element or a circuit.",
    )
    parser.add_argument(
        "--latex",
        "-L",
        dest="latex",
        action="store_true",
        help="Print the LaTeX math equation for the impedance of an element or a circuit.",
    )
    parser.add_argument(
        "--running-count",
        "-rc",
        dest="running_count",
        action="store_true",
        help="Use a running count (0..N) for circuit elements.",
    )
    add_output_args(parser, multiple_names=True)
    add_plot_args(parser, overlay=False, title=False)


def drt_args(parser: ArgumentParser):
    add_input_args(parser)
    parser.add_argument(
        "--method",
        "-me",
        dest="method",
        metavar="STRING",
        type=str,
        default="tr-nnls",
        help="The DRT method to use for the calculations. Valid values: "
        + ", ".join(sorted(map(lambda _: f"'{_}'", DRT_METHODS)))
        + ". Defaults to 'tr-nnls'.",
    )
    parser.add_argument(
        "--mode",
        "-mo",
        dest="mode",
        metavar="STRING",
        type=str,
        default="complex",
        help="Which parts of the data to use: 'complex', 'real', 'imaginary'. Defaults to 'complex'.",
    )
    parser.add_argument(
        "--lambda-value",
        "-lv",
        dest="lambda_value",
        metavar="FLOAT",
        type=float,
        default=-1.0,
        help="The regularization parameter, lambda, used by the methods that employ Tikhonov regularization. If the value is 0.0 or less, then an attempt will be made to automatically find a suitable value. Defaults to -1.0.",
    )
    parser.add_argument(
        "--cross-validation",
        "-cv",
        dest="cross_validation",
        metavar="STRING",
        type=str,
        default="mgcv",
        help="The cross-validation method to use when picking a suitable lambda value (TR-RBF method only). Valid values include: "
        + ", ".join(sorted(map(lambda _: f"'{_}'", _CROSS_VALIDATION_METHODS.keys())))
        + ". An empty string (i.e., '') causes --lambda-value to be used directly. Defaults to 'mgcv'.",
    )
    parser.add_argument(
        "--rbf-type",
        "-rt",
        dest="rbf_type",
        metavar="STRING",
        type=str,
        default="gaussian",
        help="The radial basis function type to use (TR-RBF and BHT methods only). Valid values: "
        + ", ".join(sorted(map(lambda _: f"'{_}'", _RBF_TYPES)))
        + ". Defaults to 'gaussian'.",
    )
    parser.add_argument(
        "--derivative-order",
        "-do",
        dest="derivative_order",
        metavar="INTEGER",
        type=int,
        default=1,
        help="The order of the derivative used during discretization (TR-RBF and BHT methods only). Defaults to 1.",
    )
    parser.add_argument(
        "--rbf-shape",
        "-rs",
        dest="rbf_shape",
        metavar="STRING",
        type=str,
        default="fwhm",
        help="The shape control of the radial basis functions. Valid values: "
        + ", ".join(sorted(map(lambda _: f"'{_}'", _RBF_SHAPES)))
        + ". Defaults to 'fwhm'.",
    )
    parser.add_argument(
        "--shape-coeff",
        "-sc",
        dest="shape_coeff",
        metavar="FLOAT",
        type=float,
        default=0.5,
        help="The full width at half maximum (FWHM) coefficient affecting the chosen shape type (TR-RBF and BHT methods only). Defaults to 0.5.",
    )
    parser.add_argument(
        "--inductance",
        "-L",
        dest="inductance",
        action="store_true",
        help="If true, then an inductive element is included in the calculations (TR-RBF method only).",
    )
    parser.add_argument(
        "--credible-intervals",
        "-ci",
        dest="credible_intervals",
        action="store_true",
        help="If true, then the credible intervals are also calculated for the DRT results according to Bayesian statistics (TR-RBF method only).",
    )
    parser.add_argument(
        "--timeout",
        "-T",
        dest="timeout",
        metavar="INTEGER",
        type=int,
        default=60,
        help="The number of seconds to wait for the calculation of credible intervals to complete. Defaults to 60.",
    )
    parser.add_argument(
        "--num-samples",
        "-ns",
        dest="num_samples",
        metavar="INTEGER",
        type=int,
        default=2000,
        help="The number of samples drawn when calculating the Bayesian credible intervals (TR-RBF method) or the Jensen-Shannon distance (BHT method). A greater number provides better accuracy but requires more time. Defaults to 2000.",
    )
    parser.add_argument(
        "--num-attempts",
        "-na",
        dest="num_attempts",
        metavar="INTEGER",
        type=int,
        default=10,
        help="The minimum number of attempts to make when trying to find suitable random initial values (BHT method only). A greater number should provide better results at the expense of time. Defaults to 10.",
    )
    parser.add_argument(
        "--max-symmetry",
        "-ms",
        dest="maximum_symmetry",
        metavar="FLOAT",
        type=float,
        default=0.5,
        help="A maximum limit (between 0.0 and 1.0) for the relative vertical symmetry of the DRT. A high degree of symmetry is common for results where the gamma value oscillates wildly (e.g., due to a small regularization parameter). A low value for the limit should improve the results but may cause the BHT method to take longer to finish.",
    )
    parser.add_argument(
        "--circuit",
        "-c",
        dest="circuit",
        metavar="STRING",
        default="",
        type=str,
        help="A circuit that contains one or more '(RQ)' or '(RC)' elements connected in series (m(RQ)fit method only). An optional series resistance may also be included. For example, a circuit with a CDC representation of 'R(RQ)(RQ)(RC)' would be a valid circuit. It is highly recommended that the provided circuit has already been fitted. However, if all of the various parameters of the provided circuit are at their default values, then an attempt will be made to fit the circuit to the data.",
    )
    parser.add_argument(
        "--gaussian-width",
        "-gw",
        dest="gaussian_width",
        metavar="FLOAT",
        type=float,
        default=0.15,
        help="The width of the Gaussian curve that is used to approximate the DRT of an '(RC)' element (m(RQ)fit method only).",
    )
    parser.add_argument(
        "--num-per-decade",
        "-npd",
        dest="num_per_decade",
        metavar="INTEGER",
        type=int,
        default=100,
        help="The number of points per decade to use when calculating a DRT (m(RQ)fit method only).",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        dest="peak_threshold",
        metavar="FLOAT",
        type=float,
        default=-1.0,
        help="The threshold to use for detecting peaks (0.0 to 1.0 relative to the highest peak's magnitude). Defaults to -1.0.",
    )
    parser.add_argument(
        "--max-nfev",
        dest="max_nfev",
        default=-1,
        type=int,
        help="The maximum number of function evaluations to use when fitting a circuit (m(RQ)fit method only). A value below 1 means no limit. Defaults to -1.",
    )
    parser.add_argument(
        "--max-iter",
        dest="max_iter",
        default=-1,
        type=int,
        help="The maximum number of iterations (TR-NNLS method only). A value below 1 means no limit. Defaults to -1.",
    )
    parser.add_argument(
        "--model-order",
        "-k",
        dest="model_order",
        metavar="INTEGER",
        type=int,
        default=0,
        help="The model order (k) to use (Loewner method only). Defaults to 0.",
    )
    parser.add_argument(
        "--model-order-method",
        "-km",
        dest="model_order_method",
        metavar="STRING",
        type=str,
        default="matrix_rank",
        help="The approach to use to automatically pick the model order (k) if a model order is not specified (Loewner method only). " 
        + ", ".join(sorted(map(lambda _: f"'{_}'", _MODEL_ORDER_METHODS))) 
        + ". Defaults to 'matrix_rank'.",
    )
    parser.add_argument(
        "--num-procs",
        dest="num_procs",
        metavar="INTEGER",
        default=-1,
        type=int,
        help="""
The maximum number of parallel processes to use.
A value less than 1 results in an attempt to figure out a suitable value based on, e.g., the number of cores detected.
Negative values can be used to select, e.g., one less than the maximum.
Defaults to -1.
""".strip(),
    )
    parser.add_argument(
        "--analyze-peaks",
        "-ap",
        dest="analyze_peaks",
        action="store_true",
        help="Perform peak analyses by fitting skew normal distributions.",
    )
    parser.add_argument(
        "--num-peaks",
        "-np",
        dest="num_peaks",
        type=int,
        default=0,
        help="The number of peaks to include in the peak analysis. The tallest peaks are prioritized. Only applicable when peak positions are not provided manually.",
    )
    parser.add_argument(
        "--peak-positions",
        "-pp",
        dest="peak_positions",
        metavar="FLOAT",
        type=float,
        nargs="*",
        default=[],
        help="The positions of the peaks to analyze. If not provided, then peaks and their positions are detected automatically.",
    )
    parser.add_argument(
        "--disallow-skew",
        "-ds",
        dest="disallow_skew",
        action="store_true",
        help="If true, then normal distributions are used instead of skew normal distributions when analyzing peaks.",
    )
    add_output_args(parser)
    add_plot_args(parser)
    parser.add_argument(
        "--plot-type",
        "-pt",
        metavar="STRING",
        dest="plot_type",
        type=str,
        default="drt",
        help="The type of plot to generate: 'bode', 'drt', 'imaginary', 'magnitude', 'nyquist', 'phase', 'real', or 'real-imaginary'. Defaults to 'drt'.",
    )
    parser.add_argument(
        "--plot-frequency",
        "-pF",
        dest="plot_frequency",
        action="store_true",
        help="Plot gamma vs frequency instead of time constant.",
    )


def fit_args(parser: ArgumentParser):
    parser.add_argument(
        "circuit",
        type=str,
        metavar="CDC",
        help="A circuit description code (CDC).",
    )
    add_input_args(parser)
    parser.add_argument(
        "--method",
        "-me",
        metavar="STRING",
        dest="method",
        type=str,
        default="auto",
        help="The iterative method to use. If set to 'auto', then all supported methods are attempted. Valid values: 'auto', "
        + ", ".join(sorted(map(lambda _: f"'{_}'", FIT_METHODS)))
        + ". Defaults to 'auto'.",
    )
    parser.add_argument(
        "--weight",
        "-we",
        metavar="STRING",
        dest="weight",
        type=str,
        default="auto",
        help="The weight to use. If set to 'auto', then all supported weights are attempted. Valid values: 'auto', "
        + ", ".join(sorted(map(lambda _: f"'{_}'", FIT_WEIGHTS)))
        + ". Defaults to 'auto'.",
    )
    parser.add_argument(
        "--max-nfev",
        metavar="INTEGER",
        dest="max_nfev",
        default=-1,
        type=int,
        help="The maximum number of function evaluations to use. A value below 1 imposes no limit. Defaults to -1.",
    )
    parser.add_argument(
        "--num-procs",
        metavar="INTEGER",
        dest="num_procs",
        default=0,
        type=int,
        help="""
The maximum number of parallel processes to use when method and/or weight are set to "auto".
A value less than 1 results in an attempt to figure out a suitable value based on, e.g., the number of cores detected.
""".strip(),
    )
    parser.add_argument(
        "--timeout",
        "-T",
        metavar="INTEGER",
        dest="timeout",
        default=0,
        type=int,
        help="""
The amount of time in seconds that a single fit is allowed to take before being timed out.
If this values is less than one, then no time limit is imposed.
Defaults to 0.
""".strip(),
    )
    parser.add_argument(
        "--num-refinements",
        "-nr",
        metavar="INTEGER",
        dest="num_refinements",
        default=0,
        type=int,
        help="""
The number of times to re-use the fitted values as the initial values in another attempt to fit the circuit.
It might not be possible to estimate the uncertainties of the fitted parameters if the fit is refined.
Defaults to 0.
""".strip(),
    )
    parser.add_argument(
        "--running-count",
        "-rc",
        dest="running_count",
        action="store_true",
        help="Use a running count (0..N) for circuit elements.",
    )
    parser.add_argument(
        "--plot-type",
        "-pt",
        metavar="STRING",
        dest="plot_type",
        type=str,
        default="fit",
        help="The type of plot to generate: 'bode', 'fit', 'imaginary', 'magnitude', 'nyquist', 'phase', 'real', or 'real-imaginary'. Defaults to 'fit'.",
    )
    add_output_args(parser)
    add_plot_args(parser, overlay=False)


def license_args(parser: ArgumentParser):
    parser.add_argument(
        "item",
        type=str,
        help="The item to show: c (terms and conditions), w (warranty disclaimer).",
    )


def parse_args(parser: ArgumentParser):
    add_input_args(parser)
    parser.add_argument(
        "--average",
        "-a",
        action="store_true",
        dest="average_data_sets",
        help="Parse the input files and output the average impedance spectrum.",
    )
    add_output_args(parser)


def plot_args(parser: ArgumentParser):
    add_input_args(parser)
    parser.add_argument(
        "--plot-type",
        "-pt",
        metavar="STRING",
        dest="plot_type",
        type=str,
        default="nyquist",
        help="The type of plot to generate: 'bode', 'data', 'imaginary', 'magnitude', 'nyquist', 'phase', 'real', or 'real-imaginary'. Defaults to 'nyquist'.",
    )
    add_output_args(parser)
    add_plot_args(parser)


def test_args(parser: ArgumentParser):
    add_input_args(parser)
    parser.add_argument(
        "--num-RC",
        "-n",
        dest="num_RC",
        metavar="INTEGER",
        default=0,
        type=int,
        help="The number of RC elements use. A value greater than or equal to 1 uses the specific number. Otherwise, the number of RC elements to use is determined automatically. Defaults to 0.",
    )
    parser.add_argument(
        "--max-num-RC",
        "-N",
        dest="max_num_RC",
        metavar="INTEGER",
        default=0,
        type=int,
        help="The maximum number of RC elements to test when --num-RC is less than 1. A value less than 1 imposes no limit. Defaults to 0.",
    )
    parser.add_argument(
        "--no-capacitance",
        "-C",
        dest="no_capacitance",
        action="store_true",
        help="Do not add a capacitance to the circuit that is fitted as part of the Kramers-Kronig test.",
    )
    parser.add_argument(
        "--no-inductance",
        "-L",
        dest="no_inductance",
        action="store_true",
        help="Do not add an inductance to the circuit that is fitted as part of the Kramers-Kronig test.",
    )
    parser.add_argument(
        "--admittance",
        "-Y",
        dest="admittance",
        action="store_true",
        help="Perfom Kramers-Kronig tests on the admittance representation of the immittance spectrum.",
    )
    parser.add_argument(
        "--impedance",
        "-Z",
        dest="impedance",
        action="store_true",
        help="Perfom Kramers-Kronig tests on the impedance representation of the immittance spectrum.",
    )
    parser.add_argument(
        "--min-log-F-ext",
        "-mlFe",
        dest="min_log_F_ext",
        type=float,
        default=-1.0,
        metavar="FLOAT",
        help="The lower limit of log Fext, which affects the lower and upper limits of the range of time constants. If the the number of extension evaluations is zero, then this argument is used directly as log Fext. Defaults to -1.0.",
    )
    parser.add_argument(
        "--max-log-F-ext",
        "-MlFe",
        dest="max_log_F_ext",
        type=float,
        default=1.0,
        metavar="FLOAT",
        help="The upper limit of log Fext, which affects the lower and upper limits of the range of time constants. Defaults to 1.0.",
    )
    parser.add_argument(
        "--log-F-ext",
        "-lFe",
        dest="log_F_ext",
        type=float,
        default=0.0,
        metavar="FLOAT",
        help="The log Fext value to use if the number of extension evaluations is set to zero. Defaults to 0.0.",
    )
    parser.add_argument(
        "--num-F-ext-evaluations",
        "-nFee",
        dest="num_F_ext_evaluations",
        type=int,
        default=20,
        metavar="INTEGER",
        help="The number of evaluations to perform when automatically estimating the optimal log Fext. Values greater than zero mean that an approach based on least squares fitting is used and values less than zero mean that an approach based on splines is used. Defaults to 20.",
    )
    parser.add_argument(
        "--no-rapid-F-ext-evaluations",
        "-nrFe",
        dest="no_rapid_F_ext_evaluations",
        action="store_true",
        help="Evaluate the full range of num_RC values for each log Fext instead of just the bare minimum.",
    )
    parser.add_argument(
        "--extended-statistics",
        "-es",
        dest="extended_statistics",
        type=int,
        default=3,
        help="Report the standard deviations of residuals, p-values for tests of the null hypothesis that the real or the imaginary parts of the residuals are normally distributed and, where applicable, have a mean of zero and a standard deviation equal to the estimated standard deviation. Higher values include more statistics, but they may take increasingly more time to calculate. Defaults to 3.",
    )
    parser.add_argument(
        "--test",
        "-t",
        dest="test",
        metavar="STRING",
        type=str,
        default="real",
        help="The type of test to perform. Three linear Kramers-Kronig tests are supported: 'complex', 'imaginary', 'real'. Alternative implementations of the tests using matrix inversion are also available via, e.g., 'complex-inv'. An implementation of the complex test using complex non-linear least squares fitting is available via 'cnls'. Defaults to 'real'.",
    )
    parser.add_argument(
        "--mu-criterion",
        dest="mu_criterion",
        metavar="FLOAT",
        default=0.85,
        type=float,
        help="The mu-criterion that is used when determining which number of RC elements to choose when applying the method (method 1) described by Schönleber et al. (DOI: 10.1016/j.electacta.2014.01.034). The value can range from 0.0 (underfitting) to 1.0 (overfitting). Defaults to 0.85.",
    )
    parser.add_argument(
        "--beta",
        dest="beta",
        metavar="FLOAT",
        default=0.75,
        type=float,
        help="The exponent beta that is used to adjust the penalty due to the difference between the mu-criterion and mu. Only relevant when using method 1, which is implemented in a modified form in pyimpspec. Defaults to 0.75.",
    )
    parser.add_argument(
        "--lower-limit",
        "-ll",
        dest="lower_limit",
        metavar="INTEGER",
        type=int,
        default=0,
        help="The lower limit for the number of RC elements to use when suggesting the optimal number of RC elements. A value less than 1 results in an attempt to determine the lower limit automatically. Defaults to 0.",
    )
    parser.add_argument(
        "--upper-limit",
        "-ul",
        dest="upper_limit",
        metavar="INTEGER",
        type=int,
        default=0,
        help="The upper limit for the number of RC elements to use when suggesting the optimal number of RC elements. A value less than 1 results in an attempt to determine the upper limit automatically. Defaults to 0.",
    )
    parser.add_argument(
        "--limit-delta",
        "-ld",
        dest="limit_delta",
        metavar="INTEGER",
        type=int,
        default=0,
        help="An alternative way of defining the upper limit as lower limit + delta. Only used if the value is greater than zero. Defaults to 0.",
    )
    parser.add_argument(
        "--suggestion-methods",
        "-sm",
        dest="suggestion_methods",
        metavar="INTEGER",
        type=int,
        nargs="*",
        default=[],
        help="Choose to specific methods to use when suggesting the number of RC circuits to use. See the API documentation for 'suggest_num_RC' for details.",
    )
    parser.add_argument(
        "--mean",
        dest="use_mean",
        action="store_true",
        help="Choose the optimal number of RC elements based on the mean of the top suggestion provided by each method.",
    )
    parser.add_argument(
        "--ranking",
        dest="use_ranking",
        action="store_true",
        help="Choose the optimal number of RC elements based on assigning scores after each method has ranked the different options.",
    )
    parser.add_argument(
        "--sum",
        dest="use_sum",
        action="store_true",
        help="Choose the optimal number of RC elements based on the sum of the scores provided by each suggestion method.",
    )
    parser.add_argument(
        "--cnls-method",
        metavar="STRING",
        dest="cnls_method",
        type=str,
        default="leastsq",
        help="The iterative method to use when performing CNLS tests. Valid values: "
        + ", ".join(sorted(map(lambda _: f"'{_}'", FIT_METHODS)))
        + ". Defaults to 'leastsq'.",
    )
    parser.add_argument(
        "--max-nfev",
        metavar="INTEGER",
        dest="max_nfev",
        default=0,
        type=int,
        help="The maximum number of function evaluations to use. A value below 1 imposes no limit. Defaults to 0.",
    )
    parser.add_argument(
        "--timeout",
        "-T",
        dest="timeout",
        metavar="INTEGER",
        type=int,
        default=60,
        help="The number of seconds to wait for a CNLS test to finish. Defaults to 60.",
    )
    parser.add_argument(
        "--num-procs",
        dest="num_procs",
        metavar="INTEGER",
        default=-1,
        type=int,
        help="""
The maximum number of parallel processes to use.
A value less than 1 results in an attempt to figure out a suitable value based on, e.g., the number of cores detected.
Negative values can be used to select, e.g., one less than the maximum.
Defaults to -1.
""".strip(),
    )
    add_output_args(parser)
    add_plot_args(parser, overlay=False, colors=False, markers=False)
    parser.add_argument(
        "--plot-immittance",
        "-pX",
        dest="plot_immittance",
        action="store_true",
        help="Plot the same data representation that was used to validate the immittance spectrum.",
    )
    parser.add_argument(
        "--plot-pseudo-chi-squared",
        "-ppcs",
        dest="plot_pseudo_chi_squared",
        action="store_true",
        help="Plot the pseudo chi-squared plot regardless of which suggestion method has been chosen.",
    )
    parser.add_argument(
        "--plot-estimated-noise",
        "-pen",
        dest="plot_estimated_noise",
        action="store_true",
        help="Include the estimated standard deviation for the noise in the plots. The noise is estimated based on the pseudo chi-squared value and the number of excitation frequencies. The noise is assumed to be normally distributed and affect the real and imaginary part equally.",
    )
    parser.add_argument(
        "--plot-moving-average-width",
        "-pmaw",
        dest="moving_average_width",
        metavar="INTEGER",
        default=0,
        type=int,
        help="The width of the moving average. The value must be an odd integer number greater than or equal to 3. Otherwise, the moving averages are not plotted. Defaults to 0.",
    )
    parser.add_argument(
        "--plot-log-F-ext-3d",
        "-plFe3d",
        dest="plot_log_F_ext_3d",
        action="store_true",
        help="Plot pseudo chi-squared versus the number of RC elements and the number of decades for the evaluated time constant extensions. Requires that multiple extensions were evaluated.",
    )
    parser.add_argument(
        "--plot-log-F-ext-2d",
        "-plFe2d",
        dest="plot_log_F_ext_2d",
        action="store_true",
        help="Similar to '--plot-log-F-ext-3d' but as a 2D plot of pseudo chi-squared versus the number of RC elements.",
    )
    parser.add_argument(
        "--plot-auto-limited-residuals",
        "-palr",
        dest="plot_auto_limited_residuals",
        action="store_true",
        help="Automatically adjust the limits of the plot of the relative residuals.",
    )


def zhit_args(parser: ArgumentParser):
    add_input_args(parser)
    parser.add_argument(
        "--smoothing",
        "-s",
        metavar="STRING",
        type=str,
        default="modsinc",
        dest="smoothing",
        help="The type of smoothing to apply: 'none', 'lowess' (locally weighted scatterplot smoothing), 'modsinc' (modified sinc kernel), 'savgol' (Savitzky-Golay), 'whithend' (Whittaker-Henderson) or 'auto'. Defaults to 'modsinc'.",
    )
    parser.add_argument(
        "--interpolation",
        "-i",
        metavar="STRING",
        type=str,
        default="makima",
        dest="interpolation",
        help="The type of interpolation to apply: 'akima' (Akima spline), 'makima' (modified Akima spline), 'cubic' (cubic spline), 'pchip' (Piecewise Cubic Hermite Interpolating Polynomial), or 'auto'. Defaults to 'makima'.",
    )
    parser.add_argument(
        "--window",
        "-w",
        metavar="STRING",
        type=str,
        default="auto",
        dest="weights_window",
        help="The window function to use when determining the weights for the Mod(Z) offset adjustment. See scipy.signal.windows for valid values (any window functions with only 'M' and 'sym' as parameters). Defaults to 'auto', which tries all window functions and selects one based on the best fit.",
    )
    parser.add_argument(
        "--window-center",
        "-wc",
        metavar="FLOAT",
        type=float,
        default="1.5",
        dest="weights_center",
        help="The center of the window function on the logarithmic frequency scale. Defaults to 1.5.",
    )
    parser.add_argument(
        "--window-width",
        "-ww",
        metavar="FLOAT",
        type=float,
        default="3.0",
        dest="weights_width",
        help="The width of the window function on the logarithmic frequency scale. Defaults to 3.0.",
    )
    parser.add_argument(
        "--num-points",
        "-np",
        metavar="INTEGER",
        dest="num_points",
        default=3,
        type=int,
        help="The number of points to take into account while smoothing any given point.",
    )
    parser.add_argument(
        "--polynomial-order",
        "-p",
        metavar="INTEGER",
        dest="polynomial_order",
        default=2,
        type=int,
        help="The order of the polynomial used when smoothing (modified sinc kernel, Savitzky-Golay, and Whittaker-Henderson only). Must be an even number when using the modified sinc kernel.",
    )
    parser.add_argument(
        "--num-iterations",
        "-ni",
        metavar="INTEGER",
        dest="num_iterations",
        default=3,
        type=int,
        help="The number of iterations to perform while smoothing (LOWESS only)",
    )
    parser.add_argument(
        "--admittance",
        "-Y",
        dest="admittance",
        action="store_true",
        help="Perfom ZHIT analyses on the admittance representation of the immittance spectrum.",
    )
    parser.add_argument(
        "--plot-type",
        "-pt",
        metavar="STRING",
        dest="plot_type",
        type=str,
        default="fit",
        help="The type of plot to generate: 'bode', 'fit', 'imaginary', 'magnitude', 'nyquist', 'phase', 'real', or 'real-imaginary'. Defaults to 'fit'.",
    )
    parser.add_argument(
        "--num-procs",
        metavar="INTEGER",
        dest="num_procs",
        default=0,
        type=int,
        help="""
The maximum number of parallel processes to use when performing a test.
A value less than 1 results in an attempt to figure out a suitable value based on, e.g., the number of cores detected.
""".strip(),
    )
    add_output_args(parser)
    add_plot_args(parser, overlay=False, colors=False, markers=False)
