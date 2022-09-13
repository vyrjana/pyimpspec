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
- [Distribution of relaxation times]({root_url}/drt)
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
                # Data sets and results
                pyimpspec.DataSet,
                pyimpspec.TestResult,
                pyimpspec.FitResult,
                pyimpspec.FittedParameter,
                pyimpspec.FittingError,
                pyimpspec.DRTResult,
                pyimpspec.DRTError,
                # Circuits and connections
                pyimpspec.Circuit,
                pyimpspec.CircuitBuilder,
                pyimpspec.Connection,
                pyimpspec.Parallel,
                pyimpspec.Series,
                # Circuit elements
                pyimpspec.Element,
                pyimpspec.Capacitor,
                pyimpspec.ConstantPhaseElement,
                pyimpspec.DeLevieFiniteLength,
                pyimpspec.Gerischer,
                pyimpspec.HavriliakNegami,
                pyimpspec.Inductor,
                pyimpspec.Resistor,
                pyimpspec.Warburg,
                pyimpspec.WarburgOpen,
                pyimpspec.WarburgShort,
                # Exceptions
                pyimpspec.ParsingError,
                pyimpspec.UnexpectedCharacter,
                pyimpspec.UnsupportedFileFormat,
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
Check the page for [high-level functions]({root_url}/high-level-functions) for the recommended ways to perform a Kramers-Kronig test to generate a `TestResult` object.

"""
        + process_classes(
            classes_to_document=[
                pyimpspec.TestResult,
            ],
            module_name="pyimpspec",
        ),
    )
    # Fitting results
    write_file(
        join(root_folder, "fitting.md"),
        jekyll_header("fitting", "fitting")
        + f"""
Check the page for [high-level functions]({root_url}/high-level-functions) for the recommended way to perform an equivalent circuit fit to generate a `FitResult` object.

"""
        + process_classes(
            classes_to_document=[
                pyimpspec.FitResult,
                pyimpspec.FittedParameter,
                pyimpspec.FittingError,
            ],
            module_name="pyimpspec",
        ),
    )
    # DRT results
    write_file(
        join(root_folder, "drt.md"),
        jekyll_header("drt", "drt")
        + f"""
Check the page for [high-level functions]({root_url}/high-level-functions) for the recommended way to calculate the distribution of relaxation times to generate a `DRTResult` object.

"""
        + process_classes(
            classes_to_document=[
                pyimpspec.DRTResult,
                pyimpspec.DRTError,
            ],
            module_name="pyimpspec",
        ),
    )
    # Circuits
    write_file(
        join(root_folder, "circuit.md"),
        jekyll_header("circuit", "circuit")
        + """
Circuits can be generated in one of two ways:
- by parsing a circuit description code (CDC)
- by using the `CircuitBuilder` class

The basic syntax for CDCs is fairly straighforward:

```python
# A resistor connected in series with a resistor and a capacitor connected in parallel
circuit: pyimpspec.Circuit = pyimpspec.parse_cdc("[R(RC)]")
```

An extended syntax, which allows for defining initial values, lower/upper limits, and labels, is also supported:

```python
circuit: pyimpspec.Circuit = pyimpspec.parse_cdc("[R{R=50:sol}(R{R=250f:ct}C{C=1.5e-6/1e-6/2e-6:dl})]")
```

Alternatively, the `CircuitBuilder` class can be used:

```python
with pyimpspec.CircuitBuilder() as builder:
    builder += (
        pyimpspec.Resistor(R=50)
        .set_label("sol")
    )
    with builder.parallel() as parallel:
        parallel += (
            pyimpspec.Resistor(R=250)
            .set_fixed("R", True)
        )
        parallel += (
            pyimpspec.Capacitor(C=1.5e-6)
            .set_label("dl")
            .set_lower_limit("C", 1e-6)
            .set_upper_limit("C", 2e-6)
        )
circuit: pyimpspec.Circuit = builder.to_circuit()
```

"""
        + f"Information about the supported circuit elements can be found [here]({root_url}/elements).\n\n"
        + process_classes(
            classes_to_document=[
                pyimpspec.Circuit,
                pyimpspec.CircuitBuilder,
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
These functions are for basic visualization of various objects (e.g., `DataSet`, `TestResult`, and `FitResult`) using the [matplotlib](https://matplotlib.org/) package.
            """,
        ),
    )
