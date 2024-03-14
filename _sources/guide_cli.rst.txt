.. include:: ./substitutions.rst

Command-line interface (CLI)
============================

The included CLI can be used to, e.g., convert experimental data from one file format to character-separated values (CSV), to plot data, or to perform Kramers-Kronig tests.
The CLI is intended primarily as a single-line alternative to writing a Python script or creating a Jupyter notebook when the goal is just to quickly check something.
The command itself is called ``pyimpspec`` and it supports several subcommands such as ``parse`` and ``plot``.
The CLI can also be accessed by running pyimpspec as a module.

.. code:: bash

   # Replace "python3" with whatever is applicable for your system
   # (e.g., "py" on Windows).
   python3 -m pyimpspec

These subcommands have a number of options that affect, e.g., the output or settings applied to the subcommand.
Below are some examples using the Bash shell (most options also have abbreviated forms such as ``-of`` instead of ``--output-format``).
These examples have been split across multiple lines here for the sake of formatting.

.. code:: bash

   # Parse some file for impedance spectra and output the result(s) to a CSV
   # file in a specific directory.
   pyimpspec parse "path to some file" \
     --output-to \
     --output-format csv \
     --output-dir "path to some directory"

   # Parse some file for impedance spectra and output a Bode plot as an SVG file
   # in the current working directory.
   pyimpspec plot "path to some file" \
     --type bode \
     --output-to \
     --plot-format svg

   # Parse some file for impedance spectra and perform Kramers-Kronig tests
   # with a series capacitance and a mu-criterion of 0.7.
   # The perform_exploratory_tests function is used by default unless the
   # number of parallel RC elements is specified or the --automatic option
   # is used.
   pyimpspec test "path to some file" \
     --add-capacitance \
     --mu-criterion 0.7

The path to an input file can be replaced with special placeholder values:

- ``"<EXAMPLE>"``: test circuit 1 from `Boukamp (1995)`_
- ``"<RANDLES>"``: a (simplified) Randles circuit
- ``"<DRIFTING>"``: a (simplified) Randles circuit with drift

.. _`Boukamp (1995)`: https://doi.org/10.1149/1.2044210

.. code:: bash

   pyimpspec zhit "<DRIFTING>"


A config file can be saved and then used to override the defaults arguments so that the arguments don't have to be explicitly specified in the terminal.
See the ``config`` subcommand for more information about how to generate the config file and where the file is stored:

.. code:: bash

   pyimpspec config -h

.. raw:: latex

    \clearpage
