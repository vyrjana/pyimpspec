#!/usr/bin/env python3
from os import getcwd
from os.path import join
from typing import IO
from api_documenter import process, process_classes, process_functions  # github.com/vyrjana/python-api-documenter
import pyimpspec
import pyimpspec.plot.mpl


def write_file(path: str, content: str):
    fp: IO
    with open(path, "w") as fp:
        fp.write(content)


if __name__ == "__main__":
    # PDF
    write_file(
        join(getcwd(), "API.md"),
        r"""---
header-includes:
    \usepackage{geometry}
    \geometry{a4paper, margin=2.5cm}
---
""" + process(
            title="pyimpspec - API reference",
            modules_to_document=[
                pyimpspec,
                pyimpspec.plot.mpl,
            ],
            minimal_classes=[
                pyimpspec.FittingError,
                pyimpspec.ParsingError,
                pyimpspec.UnexpectedCharacter,
                pyimpspec.Capacitor,
                pyimpspec.ConstantPhaseElement,
                pyimpspec.Gerischer,
                pyimpspec.HavriliakNegami,
                pyimpspec.Inductor,
                pyimpspec.Parallel,
                pyimpspec.Resistor,
                pyimpspec.Series,
                pyimpspec.Warburg,
                pyimpspec.WarburgOpen,
                pyimpspec.WarburgShort,
                pyimpspec.DeLevieFiniteLength,
            ],
            latex_pagebreak=True,
        )
    )
    # Main module
    write_file(
        join(getcwd(), "API-main.md"),
        process(
            title="pyimpspec",
            modules_to_document=[
                pyimpspec,
            ],
            objects_to_ignore=[
                pyimpspec.DataSet,
                pyimpspec.Circuit,
                pyimpspec.Element,
                pyimpspec.Connection,
                pyimpspec.FittedParameter,
                pyimpspec.FittingError,
                pyimpspec.FittingResult,
                pyimpspec.KramersKronigResult,
                pyimpspec.ParsingError,
                pyimpspec.UnexpectedCharacter,
                pyimpspec.Capacitor,
                pyimpspec.ConstantPhaseElement,
                pyimpspec.Gerischer,
                pyimpspec.HavriliakNegami,
                pyimpspec.Inductor,
                pyimpspec.Parallel,
                pyimpspec.Resistor,
                pyimpspec.Series,
                pyimpspec.Warburg,
                pyimpspec.WarburgOpen,
                pyimpspec.WarburgShort,
                pyimpspec.DeLevieFiniteLength,
            ],
        ),
    )
    # Data sets
    write_file(
        join(getcwd(), "API-data-sets.md"),
        process_classes(
            classes_to_document=[
                pyimpspec.DataSet,
            ],
            module_name="pyimpspec",
        ),
    )
    # Kramers-Kronig results
    write_file(
        join(getcwd(), "API-kk-results.md"),
        process_classes(
            classes_to_document=[
                pyimpspec.KramersKronigResult,
            ],
            module_name="pyimpspec",
        ),
    )
    # Fitting results
    write_file(
        join(getcwd(), "API-fitting-results.md"),
        process_classes(
            classes_to_document=[
                pyimpspec.FittingResult,
            ],
            module_name="pyimpspec",
        ),
    )
    # Circuits
    write_file(
        join(getcwd(), "API-circuits.md"),
        process_classes(
            classes_to_document=[
                pyimpspec.Circuit,
                pyimpspec.Connection,
                pyimpspec.Parallel,
                pyimpspec.Series,
            ],
            module_name="pyimpspec",
            minimal_classes=[
                pyimpspec.Parallel,
                pyimpspec.Series,
            ],
        ),
    )
    # Elements
    write_file(
        join(getcwd(), "API-elements.md"),
        process_classes(
            classes_to_document=[
                pyimpspec.Element,
                pyimpspec.Capacitor,
                pyimpspec.ConstantPhaseElement,
                pyimpspec.Gerischer,
                pyimpspec.HavriliakNegami,
                pyimpspec.Inductor,
                pyimpspec.Parallel,
                pyimpspec.Resistor,
                pyimpspec.Series,
                pyimpspec.Warburg,
                pyimpspec.WarburgOpen,
                pyimpspec.WarburgShort,
                pyimpspec.DeLevieFiniteLength,
            ],
            module_name="pyimpspec",
            minimal_classes=[
                pyimpspec.Capacitor,
                pyimpspec.ConstantPhaseElement,
                pyimpspec.Gerischer,
                pyimpspec.HavriliakNegami,
                pyimpspec.Inductor,
                pyimpspec.Parallel,
                pyimpspec.Resistor,
                pyimpspec.Series,
                pyimpspec.Warburg,
                pyimpspec.WarburgOpen,
                pyimpspec.WarburgShort,
                pyimpspec.DeLevieFiniteLength,
            ],
        ),
    )
    # Plotting - matplotlib
    write_file(
        join(getcwd(), "API-plot.mpl.md"),
        process(
            title="pyimpspec.plot.mpl",
            modules_to_document=[
                pyimpspec.plot.mpl,
            ],
        ),
    )
