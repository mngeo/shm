Compute Global Grid
===================

.. code-block:: python

   from shgeomag.io.reader import load_model
   from shgeomag.utils.grid import global_grid

   model = load_model("models/WMM2025.COF")
   ds = global_grid(model, 2026.0, -180, 180, -90, 90, 5.0, 5.0, 0.0)
