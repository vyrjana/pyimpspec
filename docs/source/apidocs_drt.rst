.. include:: ./substitutions.rst

Distribution of relaxation times
================================

A collection of functions and classes for calculating the distribution of relaxation times in data  sets.

Wrapper function and base class
-------------------------------
.. automodule:: pyimpspec
   :members: calculate_drt, DRTResult


Method functions and classes
----------------------------

BHT method
~~~~~~~~~~
.. automodule:: pyimpspec.analysis.drt
   :members: calculate_drt_bht

.. automodule:: pyimpspec.analysis.drt
   :members: BHTResult


m(RQ)fit method
~~~~~~~~~~~~~~~
.. automodule:: pyimpspec.analysis.drt
   :members: calculate_drt_mrq_fit

.. automodule:: pyimpspec.analysis.drt
   :members: MRQFitResult


TR-NNLS method
~~~~~~~~~~~~~~
.. automodule:: pyimpspec.analysis.drt
   :members: calculate_drt_tr_nnls

.. automodule:: pyimpspec.analysis.drt
   :members: TRNNLSResult


TR-RBF method
~~~~~~~~~~~~~
.. automodule:: pyimpspec.analysis.drt
   :members: calculate_drt_tr_rbf

.. automodule:: pyimpspec.analysis.drt
   :members: TRRBFResult

.. raw:: latex

    \clearpage