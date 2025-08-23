📖 API
======

.. currentmodule:: drlearn

.. autosummary::
   :toctree: api
   :nosignatures:
   
   Ridge
   BinaryLogisticRegression
   MultinomialLogisticRegression
   make_superquantile_spectrum
   make_extremile_spectrum
   make_esrm_spectrum

Estimators
----------

.. autoclass:: drlearn.Ridge
   :members: fit, predict
   :inherited-members:
   :show-inheritance:

.. autoclass:: drlearn.BinaryLogisticRegression
   :members: fit, predict, predict_proba
   :inherited-members:
   :show-inheritance:

.. autoclass:: drlearn.MultinomialLogisticRegression
   :members: fit, predict, predict_proba
   :inherited-members:
   :show-inheritance:

Risk Spectra
------------
.. autofunction:: make_superquantile_spectrum
.. autofunction:: make_extremile_spectrum
.. autofunction:: make_esrm_spectrum