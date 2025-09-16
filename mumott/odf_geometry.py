# --- odf_geometry.py ---

from dataclasses import dataclass
import numpy as np
from typing import Optional, Sequence
from mumott.geometry import Geometry
from scipy.spatial.transform import Rotation as R

# import your own classes
from mumott.probed_coordinates import ProbedCoordinates

@dataclass
class ODFGeometry:
    """
    Skeleton geometry describing the ODF sampling for a given experiment.

    You should fill in: how to construct ProbedCoordinates from
    (rotations, two-theta ring(s), azimuthal bins, etc).

    Attributes
    ----------
    detector_angles : array-like (M,)
        Azimuthal bin centers in radians.
    two_theta : array-like (R,)
        Scattering angles (radians). If you use multiple rings, M_total = M * R.
    ell_max : int
        Spherical-harmonics band-limit.
    enforce_friedel_symmetry : bool
        If True, restricts ℓ to even values.
    """
    azimuthal_angles: Sequence[float] = (0.0,)
    two_theta: Sequence[float] = (0.0,)
    ell_max: int = 0
    enforce_friedel_symmetry: bool = False

    # Optional: anything you need to compute the probed directions
    # e.g. rotation matrices per projection, beam & detector axes, etc.
    rotation: Sequence[float] = (0.0,)  # (N, 3, 3) or None (fill later)


    def build_probed_coordinates(self) -> ProbedCoordinates:
        """
        Return ProbedCoordinates with vectors shaped (N, M_total, I, 3)
        and optional great_circle_offset of same broadcastable shape.

        TODO: Replace this stub with your actual construction (you can
        port the logic from your mumott Geometry._get_probed_coordinates()).
        """
        az = np.asarray(self.azimuthal_angles, dtype=float)   # (M,)
        tt = np.asarray(self.two_theta, dtype=float)         # (R,)
        M = az.size * tt.size
        geom = Geometry()
        geom.detector_angles = self.azimuthal_angles
        geom.two_theta = self.two_theta
        geom.full_circle_covered  = True   
        Rz = [R.from_rotvec(np.array([0.,0.,1.]) * self.rotation[i]).as_matrix() for i in range(len(self.rotation))]
        geom.rotations = Rz

        self.probed_coordinates = geom._get_probed_coordinates()
        return self.probed_coordinates

    # Helpers
    @property
    def n_segments(self) -> int:
        return int(np.size(self.azimuthal_angles) * np.size(self.two_theta))
