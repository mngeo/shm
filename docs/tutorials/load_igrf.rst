Load IGRF
=========

.. code-block:: python

   import numpy as np
   from shgeomag.io.reader import load_model
   from shgeomag.core.design_matrix import build_ned_jacobian
   model = load_model("models/igrf14coeffs.txt")

   # Coefficient Jacobian: d[X,Y,Z]/d[g,h] at one geocentric point.
   j = build_ned_jacobian(
       gc_lat_rad=np.deg2rad(np.array([10.0])),
       lon_rad=np.deg2rad(np.array([20.0])),
       r_km=np.array([6371.2]),
       n_max=model.n_max,
   )
   print(j.shape)
