"""Core numerical routines for design matrices, field components, and rotations."""

from .design_matrix import build_design_matrix
from .field import compute_fdi, compute_geo, compute_multi_epoch, compute_sph, compute_sph_cached
from .rotation import rotate_sph_to_geodetic_ned

__all__ = [
    "build_design_matrix",
    "compute_sph",
    "compute_sph_cached",
    "compute_geo",
    "compute_fdi",
    "compute_multi_epoch",
    "rotate_sph_to_geodetic_ned",
]
