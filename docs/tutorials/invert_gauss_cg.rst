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

Regularized API
---------------

Use the explicit Tikhonov wrapper when you want ``lambda`` as a direct
user parameter:

.. code-block:: python

   from shgeomag.inversion import invert_gauss_coefficients_cg_tikhonov

   result_reg = invert_gauss_coefficients_cg_tikhonov(
       data=data,
       n_max=8,
       lambda_reg=1000.0,
        regularization="identity",  # or "Manojs_scheme"
        max_iter=200,
        tol=1e-8,
        use_cache=True,
   )

   print(result_reg.converged)
   print(result_reg.final_relative_residual)

``regularization="Manojs_scheme"`` uses
``R = L^T L`` with ``L = diag((1:n_coeff)^2)`` in
``(J^T J + lambda_reg * R) c = J^T d``.

L-curve
-------

Compute residual/solution norms over user-provided lambdas, with optional
plotting:

.. code-block:: python

   from shgeomag.inversion import compute_l_curve_tikhonov

   solution_norm, residual_norm = compute_l_curve_tikhonov(
       data=data,
       n_max=8,
       lambda_values=np.array([0.0, 1.0, 10.0, 100.0, 1000.0]),
       regularization="identity",
       plot=False,  # set True to generate log-log L-curve plot
   )

   print(solution_norm)
   print(residual_norm)

Practical Pipeline Notes
------------------------

- For the repository's current observatory pipeline (epoch ``2020.0``),
  apply mapping/filtering/averaging/bias-correction before inversion.
- Keep preprocessing fixed while sweeping lambda values for an interpretable
  L-curve.
- Use ``plot=False`` in headless or CI environments and persist the returned
  norm arrays to text.

Synthetic Recovery Tests
------------------------

The inversion test suite includes synthetic coefficient-recovery checks at
epoch ``2020.0`` for both IGRF and WMM models, using RMS coefficient error:

- clean synthetic data (expect near-zero RMS)
- 5% Gaussian component noise
- 10% Gaussian component noise
