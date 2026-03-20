Time Series
===========

.. code-block:: python

   import numpy as np
   from shgeomag.io.reader import load_model
   from shgeomag.core.field import compute_fdi

   model = load_model("models/WMM2025.COF")
   years = np.linspace(2025.0, 2030.0, 6)
