# autonomy/perception_state/camera.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np
import math

from .common_math import quat_to_rot, quat_to_euler, euler_to_quat
from .config import CameraConfig

@dataclass
class CameraFrame:
    t: float
    drone_id: int
    image: Optional[np.ndarray] = None
    synthetic_objects: list = field(default_factory=list)
    texture_score: float = 1.0
    blur_score: float = 0.0
    exposure_ok: bool = True

class PiCameraModule3Wide:
    """
    Raspberry Pi Camera Module 3 NoIR Wide.

    Verified/supplied properties:
      - Sony IMX708
      - 12 MP
      - approx 120 deg viewing angle (Wide)
      - autofocus

    Important:
      This does NOT require processing at 12 MP.
      Inference resolution is configurable.
    """

    def __init__(self, cfg: CameraConfig, drone_id: int):
        self.cfg = cfg
        self.drone_id = drone_id

        # Default downward-looking mounting matrix.
        #
        # Camera optical frame:
        #   x = right
        #   y = down
        #   z = forward
        #
        # Body frame used here:
        #   x = forward
        #   y = left
        #   z = up
        #
        # Downward-looking camera:
        #   optical z points to -body z
        #   optical x points to -body y
        #   optical y points to -body x
        self.R_body_cam = np.array([
            [0.0, -1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0]
        ], dtype=float)

        self.set_inference_size(cfg.active_inference_size)

    def set_inference_size(self, size: int):
        """
        Configure square inference resolution:
          640, 512, 416, etc.
        """
        self.infer_w = int(size)
        self.infer_h = int(size)

        native_w = float(self.cfg.native_width)
        native_h = float(self.cfg.native_height)
        native_diag = math.hypot(native_w, native_h)

        # Native focal length in pixels from diagonal FOV.
        fov_rad = math.radians(self.cfg.diagonal_fov_deg)
        self.native_f = (native_diag * 0.5) / math.tan(fov_rad * 0.5)

        # Center square crop then resize to inference square.
        crop_side = min(native_w, native_h)
        scale = float(self.infer_w) / float(crop_side)

        self.fx = self.native_f * scale
        self.fy = self.native_f * scale
        self.cx = self.infer_w * 0.5
        self.cy = self.infer_h * 0.5

        # Derived FOV at inference crop.
        self.hfov_rad = 2.0 * math.atan((self.infer_w * 0.5) / self.fx)
        self.vfov_rad = 2.0 * math.atan((self.infer_h * 0.5) / self.fy)

    def intrinsic_matrix(self) -> np.ndarray:
        return np.array([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0]
        ], dtype=float)

    def pixel_to_ray(self, u: float, v: float) -> np.ndarray:
        """
        Back-project an image pixel into camera optical frame.
        """
        x = (float(u) - self.cx) / self.fx
        y = (float(v) - self.cy) / self.fy
        ray = np.array([x, y, 1.0], dtype=float)
        return ray / max(np.linalg.norm(ray), 1e-9)

    def camera_ray_to_world(self, ray_cam: np.ndarray, quat_wb: np.ndarray) -> np.ndarray:
        R_wb = quat_to_rot(quat_wb)
        ray_body = self.R_body_cam @ ray_cam
        ray_world = R_wb @ ray_body
        norm = np.linalg.norm(ray_world)
        if norm < 1e-9:
            return np.array([0.0, 0.0, -1.0])
        return ray_world / norm

    def ground_intersection(
        self,
        ray_world: np.ndarray,
        drone_pos_world: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Intersect camera ray with z=0 ground plane.

        Important:
          This does NOT blindly convert bbox -> world.
          It uses ray geometry and altitude.
        """
        if ray_world[2] >= -1e-4:
            return None

        z = float(drone_pos_world[2])
        if z <= 0.02:
            z = 0.02

        t = z / (-ray_world[2])
        if t <= 0.0 or t > 60.0:
            return None

        p = drone_pos_world + ray_world * t
        return np.array([p[0], p[1], 0.0], dtype=float)

    def bbox_center(self, xyxy: Tuple[float, float, float, float]) -> Tuple[float, float]:
        x0, y0, x1, y1 = xyxy
        return ((x0 + x1) * 0.5, (y0 + y1) * 0.5)

    def estimate_ground_position(
        self,
        xyxy: Tuple[float, float, float, float],
        quat_wb: np.ndarray,
        drone_pos_world: np.ndarray,
        altitude_source: Optional[float] = None,
        altitude_std: float = 0.20,
        pixel_std: Optional[float] = None
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Returns:
          ground_pos [x, y, 0]
          covariance 2x2 in x,y

        Uses:
          - intrinsics
          - FOV
          - camera pose
          - altitude / range
          - attitude uncertainty
          - ground plane
        """
        u, v = self.bbox_center(xyxy)
        pix_std = pixel_std if pixel_std is not None else self.cfg.pixel_noise_std_px

        # Use range-derived altitude if supplied, otherwise EKF altitude.
        pos = np.array(drone_pos_world, dtype=float).copy()
        if altitude_source is not None:
            pos[2] = float(altitude_source)

        ray_cam = self.pixel_to_ray(u, v)
        ray_world = self.camera_ray_to_world(ray_cam, quat_wb)
        ground = self.ground_intersection(ray_world, pos)
        if ground is None:
            return None, None

        cov = self._ground_covariance(
            u=u,
            v=v,
            quat_wb=quat_wb,
            pos=pos,
            pix_std=pix_std,
            alt_std=altitude_std
        )

        return ground, cov

    def estimate_physical_size_m(
        self,
        xyxy: Tuple[float, float, float, float],
        ground_pos: np.ndarray,
        drone_pos_world: np.ndarray
    ) -> float:
        """
        Approximate object size from pixel extent and slant range.
        """
        x0, y0, x1, y1 = xyxy
        wpx = max(1.0, abs(x1 - x0))
        hpx = max(1.0, abs(y1 - y0))

        distance = max(0.25, float(np.linalg.norm(np.asarray(ground_pos) - np.asarray(drone_pos_world))))
        gsd = distance / self.fx

        w_m = wpx * gsd
        h_m = hpx * gsd

        size = 0.5 * (w_m + h_m)
        return float(np.clip(size, 0.03, 3.0))

    def _ground_covariance(
        self,
        u: float,
        v: float,
        quat_wb: np.ndarray,
        pos: np.ndarray,
        pix_std: float,
        alt_std: float
    ) -> np.ndarray:
        """
        First-order covariance by finite differences.

        Parameter vector:
          [u, v, roll, pitch, yaw, x, y, z]
        """
        rpy = quat_to_euler(quat_wb)
        base = np.array([
            u,
            v,
            rpy[0],
            rpy[1],
            rpy[2],
            pos[0],
            pos[1],
            pos[2]
        ], dtype=float)

        stds = np.array([
            pix_std,
            pix_std,
            self.cfg.attitude_std_rad,
            self.cfg.attitude_std_rad,
            self.cfg.attitude_std_rad,
            0.12,
            0.12,
            max(0.03, alt_std)
        ], dtype=float)

        f0 = self._ground_from_params(base)
        if f0 is None:
            return np.eye(2) * 1.5

        J = np.zeros((2, 8), dtype=float)

        for i in range(8):
            eps = max(1e-4, abs(stds[i]) * 1e-2)
            if i < 2:
                eps = max(0.05, eps)

            p1 = base.copy()
            p2 = base.copy()
            p1[i] += eps
            p2[i] -= eps

            g1 = self._ground_from_params(p1)
            g2 = self._ground_from_params(p2)

            if g1 is None or g2 is None:
                continue

            J[:, i] = (g1[:2] - g2[:2]) / (2.0 * eps)

        Sigma_params = np.diag(stds * stds)
        cov = J @ Sigma_params @ J.T

        # Numerical safety
        cov = 0.5 * (cov + cov.T)
        cov += np.eye(2) * 1e-5
        return cov

    def _ground_from_params(self, p: np.ndarray) -> Optional[np.ndarray]:
        u, v = p[0], p[1]
        rpy = p[2:5]
        pos = p[5:8]

        q = euler_to_quat(rpy)
        ray_cam = self.pixel_to_ray(u, v)
        ray_world = self.camera_ray_to_world(ray_cam, q)
        return self.ground_intersection(ray_world, pos)