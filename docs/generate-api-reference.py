#!/usr/bin/env python3
from os import getcwd, makedirs
from os.path import exists, join
from typing import IO
from api_documenter import (
    process,
    process_classes,
    process_functions,
)  # github.com/vyrjana/python-api-documenter
import pyimpspec
import pyimpspec.plot.mpl


def write_file(path: str, content: str):
    fp: IO
    with open(path, "w") as fp:
        fp.write(content)


def jekyll_header(title: str, link: str) -> str:
    return f"""---
layout: documentation
title: API - {title}
permalink: /api/{link}/
---
"""


if __name__ == "__main__":
    # PDF
    write_file(
        join(getcwd(), "API.md"),
        r"""---
header-includes:
    \usepackage{geometry}
    \geometry{a4paper, margin=2.5cm}
---
"""
        + process(
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
        ),
    )
    # Jekyll
    root_folder: str = join(getcwd(), "documentation")
    if not exists(root_folder):
        makedirs(root_folder)
    root_url: str = "https://vyrjana.github.io/pyimpspec/api"
    # - index
    write_file(
        join(root_folder, "index.md"),
        f"""---
layout: documentation
title: API documentation
permalink: /api/
---

## API documentation

Check out [this Jupyter notebook](https://github.com/vyrjana/pyimpspec/blob/main/examples/examples.ipynb) for examples of how to use the various functions and classes.

- [High-level functions]({root_url}/high-level-functions)
- [Data set]({root_url}/data-set)
- [Kramers-Kronig testing]({root_url}/kramers-kronig)
- [Circuit]({root_url}/circuit)
    - [Elements]({root_url}/elements)
    - [Base element]({root_url}/base-element)
    - [Connections]({root_url}/connections)
- [Fitting]({root_url}/fitting)
- Plotting
  - [matplotlib]({root_url}/plot-mpl)
""",
    )
    # - main module
    write_file(
        join(root_folder, "high-level-functions.md"),
        jekyll_header("high-level functions", "high-level-functions")
        + process(
            title="",
            modules_to_document=[
                pyimpspec,
            ],
            description="""
The functions presented here are the recommended way of reading data, creating circuits, performing tests, etc.
Check the other pages for information about the objects returned by the functions presented here.
            """,
            objects_to_ignore=[
                pyimpspec.Capacitor,
                pyimpspec.Circuit,
                pyimpspec.Connection,
                pyimpspec.ConstantPhaseElement,
                pyimpspec.DataSet,
                pyimpspec.DeLevieFiniteLength,
                pyimpspec.Element,
                pyimpspec.FittedParameter,
                pyimpspec.FittingError,
                pyimpspec.FittingResult,
                pyimpspec.Gerischer,
                pyimpspec.HavriliakNegami,
                pyimpspec.Inductor,
                pyimpspec.KramersKronigResult,
                pyimpspec.Parallel,
                pyimpspec.ParsingError,
                pyimpspec.Resistor,
                pyimpspec.Series,
                pyimpspec.UnexpectedCharacter,
                pyimpspec.UnsupportedFileFormat,
                pyimpspec.Warburg,
                pyimpspec.WarburgOpen,
                pyimpspec.WarburgShort,
            ],
        ),
    )
    # Data sets
    write_file(
        join(root_folder, "data-set.md"),
        jekyll_header("data set", "data-set")
        + f"""
Check the page for [high-level functions]({root_url}/high-level-functions) for the recommended way of reading data to generate a `DataSet` object.

"""
        + process_classes(
            classes_to_document=[
                pyimpspec.DataSet,
                pyimpspec.UnsupportedFileFormat,
            ],
            module_name="pyimpspec",
        ),
    )
    # Kramers-Kronig results
    write_file(
        join(root_folder, "kramers-kronig.md"),
        jekyll_header("Kramers-Kronig testing", "kramers-kronig")
        + f"""
Check the page for [high-level functions]({root_url}/high-level-functions) for the recommended ways to perform a Kramers-Kronig test to generate a `KramersKronigResult` object.

"""
        + process_classes(
            classes_to_document=[
                pyimpspec.KramersKronigResult,
            ],
            module_name="pyimpspec",
        ),
    )
    # Fitting results
    write_file(
        join(root_folder, "fitting.md"),
        jekyll_header("fitting", "fitting")
        + f"""
Check the page for [high-level functions]({root_url}/high-level-functions) for the recommended way to perform an equivalent circuit fit to generate a `FittingResult` object.

"""
        + process_classes(
            classes_to_document=[
                pyimpspec.FittingResult,
                pyimpspec.FittedParameter,
                pyimpspec.FittingError,
            ],
            module_name="pyimpspec",
        ),
    )
    # Circuits
    write_file(
        join(root_folder, "circuit.md"),
        jekyll_header("circuit", "circuit")
        + f"""
Check the page for [high-level functions]({root_url}/high-level-functions) for the recommended way of parsing a circuit description code (CDC) to generate a `Circuit` object.

"""
        + process_classes(
            classes_to_document=[
                pyimpspec.Circuit,
                pyimpspec.ParsingError,
                pyimpspec.UnexpectedCharacter,
            ],
            module_name="pyimpspec",
        ),
    )
    # Connections
    write_file(
        join(root_folder, "connections.md"),
        jekyll_header("connections", "connections")
        + f"""
These are used inside of [`Circuit`]({root_url}/circuit) objects.

"""
        + process_classes(
            classes_to_document=[
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
        join(root_folder, "base-element.md"),
        jekyll_header("base element", "base-element")
        + f"""
This is the page for the base class for all [elements]({root_url}/elements).

"""
        + process_classes(
            classes_to_document=[
                pyimpspec.Element,
            ],
            module_name="pyimpspec",
        ),
    )
    write_file(
        join(root_folder, "elements.md"),
        jekyll_header("elements", "elements")
        + f"""
These are used inside of [`Circuit`]({root_url}/circuit) and [`Connection`]({root_url}/connections) objects.
Check the page for the [base element class]({root_url}/base-element) for information about the methods that are available for various elements.

"""
        + process_classes(
            classes_to_document=[
                pyimpspec.Capacitor,
                pyimpspec.ConstantPhaseElement,
                pyimpspec.Gerischer,
                pyimpspec.HavriliakNegami,
                pyimpspec.Inductor,
                pyimpspec.Resistor,
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
                pyimpspec.Resistor,
                pyimpspec.Warburg,
                pyimpspec.WarburgOpen,
                pyimpspec.WarburgShort,
                pyimpspec.DeLevieFiniteLength,
            ],
        ),
    )
    # Plotting - matplotlib
    write_file(
        join(root_folder, "plot-mpl.md"),
        jekyll_header("plotting - matplotlib", "plot-mpl")
        + process(
            title="",
            modules_to_document=[
                pyimpspec.plot.mpl,
            ],
            description="""
These functions are for basic visualization of various objects (e.g., `DataSet`, `KramersKronigResult`, and `FittingResult`) using the [matplotlib](https://matplotlib.org/) package.
            """,
        ),
    )
