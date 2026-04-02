Inversion API
=============

Main entry points:

- ``invert_gauss_coefficients_cg`` (CG inversion; optional damping argument)
- ``invert_gauss_coefficients_cg_tikhonov`` (explicit Tikhonov API with
  ``lambda_reg`` and selectable regularization matrix)
- ``compute_l_curve_tikhonov`` (L-curve norms across lambda array with optional plotting)
- ``forward_ned_from_coefficients``
- ``run_global_grid_inversion`` (CLI/programmatic synthetic global-grid runner)

Notes:

- ``compute_l_curve_tikhonov(..., plot=True)`` requires ``matplotlib``.
- Use ``plot=False`` to return norms only without creating figures.
- ``invert_gauss_coefficients_cg_tikhonov`` supports
  ``regularization="identity"`` and
  ``regularization="Manojs_scheme"`` (where
  ``L = diag((1:n_coeff)^2)``, ``R = L^T L``),
  ``regularization="Ohmic_heating"`` with
  ``diag_i = 4*pi*(Re/Rcmb)^(2*n+3)*(n+1)*(2*n+1)*(2*n+3)/n``
  (non-inverted degree-weighted Ohmic diagonal),
  and also accepts a custom diagonal via ``reg_diag``.

.. automodule:: shgeomag.inversion.cg
   :members:

.. automodule:: shgeomag.inversion.run_global_grid_inversion
   :members:
