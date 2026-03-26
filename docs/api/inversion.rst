Inversion API
=============

Main entry points:

- ``invert_gauss_coefficients_cg`` (CG inversion; optional damping argument)
- ``invert_gauss_coefficients_cg_tikhonov`` (explicit Tikhonov API with
  ``lambda_reg`` and selectable regularization matrix)
- ``compute_l_curve_tikhonov`` (L-curve norms across lambda array with optional plotting)
- ``forward_ned_from_coefficients``

Notes:

- ``compute_l_curve_tikhonov(..., plot=True)`` requires ``matplotlib``.
- Use ``plot=False`` to return norms only without creating figures.
- ``invert_gauss_coefficients_cg_tikhonov`` supports
  ``regularization="identity"`` and
  ``regularization="Manojs_scheme"`` (where
  ``L = diag((1:n_coeff)^2)``, ``R = L^T L``), and also accepts a
  custom diagonal via ``reg_diag``.

.. automodule:: shgeomag.inversion.cg
   :members:
