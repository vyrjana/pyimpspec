.. include:: ./substitutions.rst

Equivalent circuits
===================

A collection of functions and classes for creating equivalent circuits and user-defined elements.

.. _sympify: https://docs.sympy.org/latest/modules/core.html#sympy.core.sympify.sympify


Functions
---------

.. automodule:: pyimpspec
   :members: get_elements, parse_cdc, simulate_spectrum, register_element

.. automodule:: pyimpspec.circuit.registry
   :members: remove_elements, reset, reset_default_parameter_values


Base classes
------------
.. automodule:: pyimpspec
   :members: Element, Container, Connection


Connection classes
------------------
.. automodule:: pyimpspec
   :members: Series, Parallel


Circuit classes
---------------
.. automodule:: pyimpspec
   :members: Circuit

.. automodule:: pyimpspec
   :members: CircuitBuilder


Element classes
---------------
.. automodule:: pyimpspec.circuit.elements
   :members:
   :imported-members:


Element registration classes
----------------------------
.. automodule:: pyimpspec
   :members: ElementDefinition, ContainerDefinition, ParameterDefinition, SubcircuitDefinition

.. raw:: latex

    \clearpage
