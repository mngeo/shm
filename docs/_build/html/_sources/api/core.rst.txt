Core API
========

Design-matrix utilities include:

- ``build_design_matrix`` for spherical components ``[Br, Btheta, Bphi]``.
- ``build_ned_jacobian`` for coefficient Jacobian
  ``d[X,Y,Z]/d[g,h]`` in the coefficient-vector order
  ``[g10, g11, h11, g20, ...]``.

.. automodule:: shgeomag.core.design_matrix
   :members:

.. automodule:: shgeomag.core.field
   :members:

.. automodule:: shgeomag.core.rotation
   :members:
