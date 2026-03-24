Inversion API
=============

Main entry points:

- ``invert_gauss_coefficients_cg`` (CG inversion; optional damping argument)
- ``invert_gauss_coefficients_cg_tikhonov`` (explicit Tikhonov API with
  ``lambda_reg``)
- ``compute_l_curve_tikhonov`` (L-curve norms across lambda array with optional plotting)
- ``forward_ned_from_coefficients``

Notes:

- ``compute_l_curve_tikhonov(..., plot=True)`` requires ``matplotlib``.
- Use ``plot=False`` to return norms only without creating figures.

.. automodule:: shgeomag.inversion.cg
   :members:
