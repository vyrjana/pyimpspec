---
layout: documentation
title: API documentation
permalink: /api/
---

## API documentation

Check out [this Jupyter notebook](https://github.com/vyrjana/pyimpspec/blob/main/examples/examples.ipynb) for examples of how to use the various functions and classes.
A single Markdown file of the API reference is available [here](https://raw.githubusercontent.com/vyrjana/pyimpspec/gh-pages/documentation/API.md).

- [High-level functions](https://vyrjana.github.io/pyimpspec/api/high-level-functions)
- [Data set](https://vyrjana.github.io/pyimpspec/api/data-set)
- [Kramers-Kronig testing](https://vyrjana.github.io/pyimpspec/api/kramers-kronig)
- [Distribution of relaxation times](https://vyrjana.github.io/pyimpspec/api/drt)
- [Circuit](https://vyrjana.github.io/pyimpspec/api/circuit)
    - [Elements](https://vyrjana.github.io/pyimpspec/api/elements)
    - [Base element](https://vyrjana.github.io/pyimpspec/api/base-element)
    - [Connections](https://vyrjana.github.io/pyimpspec/api/connections)
- [Fitting](https://vyrjana.github.io/pyimpspec/api/fitting)
- Plotting
  - [matplotlib](https://vyrjana.github.io/pyimpspec/api/plot-mpl)


**NOTE!** The API makes use of multiple processes where possible to perform tasks in parallel. Functions that implement this parallelization have a `num_procs` keyword argument that can be used to override the maximum number of processes allowed. Using this keyword argument should not be necessary for most users under most circumstances.

If NumPy is linked against a multithreaded linear algebra library like OpenBLAS or MKL, then this may in some circumstances result in unusually poor performance despite heavy CPU utilization. It may be possible to remedy the issue by specifying a lower number of processes via the `num_procs` keyword argument and/or limiting the number of threads that, e.g., OpenBLAS should use by setting the appropriate environment variable (e.g., `OPENBLAS_NUM_THREADS`). Again, this should not be necessary for most users and reporting this as an issue to the pyimpspec or DearEIS repository on GitHub would be preferred.


