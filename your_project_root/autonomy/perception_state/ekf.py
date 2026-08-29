# autonomy/perception_state/ekf.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np

from .common_math import (
    quat_multiply,
    quat_normalize,
    quat_to_rot,
    euler_to_quat,
    yaw_from_quat,
    wrap_angle,
    skew
)

GRAVITY_WORLD = np.array([0.0, 0.0, -9.81], dtype=float)

@dataclass
class EKFState:
    t: float
    pos: np.ndarray
    vel: np.ndarray
    quat: np.ndarray
    accel_bias: np.ndarray
    gyro_bias: np.ndarray
    P: np.ndarray

class GPSDeniedEKF:
    """
    Error-state EKF for GPS-denied localization.

    Error state:
      [pos_err(3), vel_err(3), att_err(3), accel_bias_err(3), gyro_bias_err(3)]

    Nominal state:
      pos, vel, quat, accel_bias, gyro_bias

    GPS is intentionally absent.
    """

    def __init__(self, initial_pos: Optional[np.ndarray] = None):
        self.pos = np.array(initial_pos if initial_pos is not None else [0.0, 0.0, 1.0], dtype=float)
        self.vel = np.zeros(3, dtype=float)
        self.quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        self.accel_bias = np.zeros(3, dtype=float)
        self.gyro_bias = np.zeros(3, dtype=float)

        self.P = np.eye(15, dtype=float)
        self.P[0:3, 0:3] *= 0.75       # position
        self.P[3:6, 3:6] *= 0.50       # velocity
        self.P[6:9, 6:9] *= 0.05       # attitude
        self.P[9:12, 9:12] *= 1e-4     # accel bias
        self.P[12:15, 12:15] *= 1e-4   # gyro bias

        self.t = 0.0

    def get_state(self) -> EKFState:
        return EKFState(
            t=self.t,
            pos=self.pos.copy(),
            vel=self.vel.copy(),
            quat=self.quat.copy(),
            accel_bias=self.accel_bias.copy(),
            gyro_bias=self.gyro_bias.copy(),
            P=self.P.copy()
        )

    def position_cov(self) -> np.ndarray:
        return self.P[0:3, 0:3].copy()

    def position_uncertainty(self) -> float:
        return float(np.sqrt(np.trace(self.P[0:3, 0:3])))

    def confidence(self) -> float:
        u = self.position_uncertainty()
        return float(1.0 / (1.0 + u))

    def predict(self, accel_body: np.ndarray, gyro_body: np.ndarray, dt: float):
        if dt <= 0.0:
            return

        self.t += dt

        gyro = gyro_body - self.gyro_bias
        accel = accel_body - self.accel_bias

        R = quat_to_rot(self.quat)

        # Nominal state propagation
        accel_world = R @ accel + GRAVITY_WORLD

        self.pos += self.vel * dt + 0.5 * accel_world * dt * dt
        self.vel += accel_world * dt

        dq = euler_to_quat(gyro * dt)
        self.quat = quat_normalize(quat_multiply(self.quat, dq))

        # Bias random walk
        self.accel_bias += np.random.randn(3) * 1e-5
        self.gyro_bias += np.random.randn(3) * 1e-6

        # Error-state covariance propagation
        F = np.eye(15, dtype=float)
        F[0:3, 3:6] = np.eye(3) * dt
        F[3:6, 6:9] = -R @ skew(accel) * dt
        F[3:6, 9:12] = -R * dt
        F[6:9, 12:15] = -np.eye(3) * dt

        Q = np.zeros((15, 15), dtype=float)
        Q[0:3, 0:3] = np.eye(3) * (0.010 * dt * dt)
        Q[3:6, 3:6] = np.eye(3) * (0.050 * dt * dt)
        Q[6:9, 6:9] = np.eye(3) * (0.006 * dt * dt)
        Q[9:12, 9:12] = np.eye(3) * (1.5e-5 * dt)
        Q[12:15, 12:15] = np.eye(3) * (1.5e-5 * dt)

        self.P = F @ self.P @ F.T + Q
        self.P = 0.5 * (self.P + self.P.T)

    def update_baro(self, altitude_m: float, R_meas: float, weight: float):
        if weight <= 0.05:
            return

        H = np.zeros((1, 15), dtype=float)
        H[0, 2] = 1.0

        z = np.array([altitude_m - self.pos[2]], dtype=float)
        Rmat = np.array([[R_meas]], dtype=float)

        self._update(z, H, Rmat, weight)

    def update_range(self, vertical_range_m: float, R_meas: float, weight: float):
        if weight <= 0.05:
            return

        H = np.zeros((1, 15), dtype=float)
        H[0, 2] = 1.0

        z = np.array([vertical_range_m - self.pos[2]], dtype=float)
        Rmat = np.array([[R_meas]], dtype=float)

        self._update(z, H, Rmat, weight)

    def update_optical_flow(self, vx_body: float, vy_body: float, quality: float, R_base: float, weight: float):
        if weight <= 0.05 or quality <= 0.05:
            return

        R_wb = quat_to_rot(self.quat)     # body -> world
        R_bw = R_wb.T                     # world -> body

        v_body_est = R_bw @ self.vel
        z = np.array([
            vx_body - v_body_est[0],
            vy_body - v_body_est[1]
        ], dtype=float)

        H = np.zeros((2, 15), dtype=float)
        H[:, 3:6] = R_bw[:2, :]

        # Inflate noise when quality is poor
        quality_scale = 1.0 / max(0.10, quality)
        Rmat = np.eye(2, dtype=float) * (R_base * quality_scale * quality_scale)

        self._update(z, H, Rmat, weight)

    def update_mag_yaw(self, yaw_meas_rad: float, R_meas: float, weight: float):
        if weight <= 0.05:
            return

        yaw_est = yaw_from_quat(self.quat)
        innov = np.array([wrap_angle(yaw_meas_rad - yaw_est)], dtype=float)

        H = np.zeros((1, 15), dtype=float)
        H[0, 8] = 1.0  # yaw attitude error approximation

        Rmat = np.array([[R_meas]], dtype=float)
        self._update(innov, H, Rmat, weight)

    def update_position_landmark(self, pos_meas: np.ndarray, R_diag: np.ndarray, weight: float):
        """
        Optional visual/map constraint.
        Still GPS-denied: this is a relative/local landmark update.
        """
        if weight <= 0.05:
            return

        H = np.zeros((3, 15), dtype=float)
        H[:3, :3] = np.eye(3)

        z = pos_meas - self.pos
        Rmat = np.diag(R_diag)

        self._update(z, H, Rmat, weight)

    def _update(self, z: np.ndarray, H: np.ndarray, R: np.ndarray, weight: float):
        weight = max(0.0, min(1.0, float(weight)))
        if weight <= 0.05:
            return

        # Sensor-health weighting: lower weight => larger measurement noise
        R_eff = R / max(weight, 1e-4)

        S = H @ self.P @ H.T + R_eff
        K = self.P @ H.T @ np.linalg.pinv(S)

        dx = K @ z

        self._apply_error_state(dx)

        I_KH = np.eye(15) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_eff @ K.T
        self.P = 0.5 * (self.P + self.P.T)

    def _apply_error_state(self, dx: np.ndarray):
        self.pos += dx[0:3]
        self.vel += dx[3:6]

        theta = dx[6:9]
        dq = euler_to_quat(theta)
        self.quat = quat_normalize(quat_multiply(self.quat, dq))

        self.accel_bias += dx[9:12]
        self.gyro_bias += dx[12:15]