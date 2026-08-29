# autonomy/perception_state/geoloc.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import uuid

from .detector import Detection
from .camera import PiCameraModule3Wide
from .ekf import EKFState

@dataclass
class MineObservation:
    obs_id: str
    t: float
    drone_id: int
    label: str
    confidence: float
    pos2: np.ndarray           # [x, y] in local world frame
    cov2: np.ndarray           # 2x2 covariance
    size_m: float
    source: str = "VISION"

class MineGeolocalizer:
    def __init__(self, camera: PiCameraModule3Wide):
        self.camera = camera

    def observation_from_detection(
        self,
        det: Detection,
        ekf_state: EKFState,
        tf_luna_valid: bool,
        tf_luna_vertical_m: Optional[float]
    ) -> Optional[MineObservation]:
        """
        Uses estimated state only.
        """
        if det.label not in ("MINE_SURFACE", "MINE_SURFACE_CUE"):
            return None

        # Choose altitude source.
        # Prefer TF-Luna if valid, otherwise EKF altitude.
        if tf_luna_valid and tf_luna_vertical_m is not None:
            alt_source = float(tf_luna_vertical_m)
            alt_std = 0.08
        else:
            alt_source = float(ekf_state.pos[2])
            alt_std = max(0.10, float(np.sqrt(ekf_state.P[2, 2])))

        ground, cov = self.camera.estimate_ground_position(
            xyxy=det.bbox_xyxy,
            quat_wb=ekf_state.quat,
            drone_pos_world=ekf_state.pos,
            altitude_source=alt_source,
            altitude_std=alt_std,
            pixel_std=None
        )

        if ground is None or cov is None:
            return None

        size_m = self.camera.estimate_physical_size_m(
            xyxy=det.bbox_xyxy,
            ground_pos=ground,
            drone_pos_world=ekf_state.pos
        )

        return MineObservation(
            obs_id=str(uuid.uuid4()),
            t=det.t,
            drone_id=det.drone_id,
            label=det.label,
            confidence=float(det.confidence),
            pos2=ground[:2].copy(),
            cov2=cov.copy(),
            size_m=float(size_m),
            source="VISION"
        )