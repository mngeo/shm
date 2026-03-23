Invert Gauss Coefficients (CG)
==============================

Use conjugate gradient to invert geodetic NED observations for Gauss
coefficients.

.. code-block:: python

   import numpy as np
   from shgeomag.inversion import invert_gauss_coefficients_cg

   # Observation matrix columns:
   # [MJD2000, gc_lat_deg, lon_deg, r_km, X_nT, Y_nT, Z_nT]
   data = np.array(
       [
           [-1000.0, 10.0, 20.0, 6371.2, 30000.0, 500.0, 1000.0],
           [-1000.0, -5.0, 60.0, 6371.2, 28000.0, 300.0, -2000.0],
       ],
       dtype=float,
   )

   result = invert_gauss_coefficients_cg(
       data=data,
       n_max=8,
       max_iter=200,
       tol=1e-8,
       damping=0.0,
       use_cache=True,
   )

   print(result.converged)
   print(result.coefficient_vector.shape)
   print(result.predicted_xyz.shape)

Notes
-----

- The inversion solves ``(J^T J + lambda I) c = J^T d`` with CG.
- ``J`` is ``d[X,Y,Z]/d[g,h]`` in coefficient-vector ordering
  ``[g10, g11, h11, g20, ...]``.
- ``use_cache=True`` reuses Jacobian blocks for repeated
  ``(gc_lat, lon, r)`` geometry.

