# autonomy/perception_state/sensors.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np
import random
import math

from .common_math import quat_to_rot, quat_to_euler, yaw_from_quat, wrap_angle, clamp

@dataclass
class IMUSample:
    t: float
    accel: np.ndarray
    gyro: np.ndarray
    valid: bool = True

@dataclass
class BaroSample:
    t: float
    altitude_m: float
    drift_m: float
    valid: bool = True

@dataclass
class MagSample:
    t: float
    yaw_rad: float
    disturbance_rad: float
    valid: bool = True

@dataclass
class OpticalFlowSample:
    t: float
    vx_body_mps: float
    vy_body_mps: float
    quality: float
    valid: bool = True

@dataclass
class TFLunaSample:
    t: float
    range_m: float
    vertical_m: float
    valid: bool = True
    outlier: bool = False

@dataclass
class VL53Zone:
    range_m: float
    confidence: float
    valid: bool

@dataclass
class VL53Sample:
    t: float
    zones: list
    valid: bool = True

class SensorHealth:
    """
    Health weights used by EKF and planner.

    If a sensor degrades, its update confidence is reduced.
    If invalid, it is rejected.
    """

    def __init__(self):
        self.imu_ok = True
        self.baro_ok = True
        self.mag_ok = True
        self.optical_flow_ok = True
        self.tf_luna_ok = True
        self.vl53_ok = True
        self.camera_ok = True

        self.optical_flow_quality = 1.0
        self.mag_disturbance = 0.0
        self.tf_luna_valid = True

    def ekf_weight_baro(self) -> float:
        return 1.0 if self.baro_ok else 0.0

    def ekf_weight_mag(self) -> float:
        if not self.mag_ok:
            return 0.0
        # Downweight if disturbed
        if self.mag_disturbance > 0.35:
            return 0.0
        return clamp(1.0 - self.mag_disturbance / 0.35, 0.0, 1.0)

    def ekf_weight_optical_flow(self) -> float:
        if not self.optical_flow_ok:
            return 0.0
        q = clamp(self.optical_flow_quality, 0.0, 1.0)
        if q < 0.25:
            return 0.0
        return q

    def ekf_weight_range(self) -> float:
        return 1.0 if (self.tf_luna_ok and self.tf_luna_valid) else 0.0


class IMUModel:
    def __init__(self, cfg, drone_id: int):
        self.cfg = cfg
        self.drone_id = drone_id

        self.accel_bias = np.random.uniform(-cfg.accel_bias_init, cfg.accel_bias_init, size=3)
        self.gyro_bias = np.random.uniform(-cfg.gyro_bias_init, cfg.gyro_bias_init, size=3)

    def sample(self, t: float, true_accel_body: np.ndarray, true_gyro_body: np.ndarray) -> IMUSample:
        # Bias random walk
        self.accel_bias += np.random.randn(3) * self.cfg.accel_bias_walk
        self.gyro_bias += np.random.randn(3) * self.cfg.gyro_bias_walk

        accel = true_accel_body + self.accel_bias + np.random.randn(3) * self.cfg.accel_white_noise
        gyro = true_gyro_body + self.gyro_bias + np.random.randn(3) * self.cfg.gyro_white_noise

        return IMUSample(t=t, accel=accel, gyro=gyro, valid=True)


class BarometerModel:
    def __init__(self, cfg):
        self.cfg = cfg
        self.drift = 0.0

    def sample(self, t: float, true_altitude_m: float, dt: float) -> BaroSample:
        # Slow drift
        self.drift += (self.cfg.drift_m_per_min / 60.0) * dt * random.uniform(-1.0, 1.0)
        z = true_altitude_m + self.drift + np.random.randn() * self.cfg.noise_m
        return BaroSample(t=t, altitude_m=float(z), drift_m=self.drift, valid=True)


class MagnetometerModel:
    def __init__(self, cfg):
        self.cfg = cfg
        self.disturbance = 0.0

    def sample(self, t: float, true_yaw_rad: float, magnetic_disturbance_rad: float) -> MagSample:
        self.disturbance = magnetic_disturbance_rad
        yaw = true_yaw_rad + magnetic_disturbance_rad + np.random.randn() * self.cfg.noise_rad
        return MagSample(
            t=t,
            yaw_rad=float(wrap_angle(yaw)),
            disturbance_rad=float(abs(magnetic_disturbance_rad)),
            valid=True
        )


class OpticalFlowModel:
    def __init__(self, cfg):
        self.cfg = cfg

    def sample(
        self,
        t: float,
        true_v_body: np.ndarray,
        texture_score: float,
        motion_magnitude: float,
        forced_failure: bool = False
    ) -> OpticalFlowSample:
        quality = clamp(float(texture_score), 0.0, 1.0)

        # Excessive motion reduces quality
        if motion_magnitude > self.cfg.motion_failure_mps:
            quality *= 0.35

        if forced_failure:
            quality = 0.0

        quality = clamp(quality + np.random.randn() * 0.05, 0.0, 1.0)

        vx = true_v_body[0] + np.random.randn() * self.cfg.velocity_noise_mps
        vy = true_v_body[1] + np.random.randn() * self.cfg.velocity_noise_mps

        valid = quality >= self.cfg.min_quality

        return OpticalFlowSample(
            t=t,
            vx_body_mps=float(vx),
            vy_body_mps=float(vy),
            quality=float(quality),
            valid=valid
        )


class TFLunaModel:
    """
    TF-Luna LiDAR ranging module.

    Verified/supplied values used:
      - 0.2 m to 8 m range
      - ~2 deg FOV
      - 1 to 250 Hz frame rate
      - default 100 Hz
      - <= 5 g
      - <= 0.35 W
    """

    def __init__(self, cfg):
        self.cfg = cfg

    def sample(self, t: float, true_vertical_m: float, tilt_factor: float = 1.0) -> TFLunaSample:
        cfg = self.cfg

        invalid = random.random() < cfg.invalid_prob
        outlier = random.random() < cfg.outlier_prob

        z = clamp(true_vertical_m, 0.0, 20.0)

        # FOV / tilt effect: if tilted too much, return may degrade.
        if tilt_factor < 0.65:
            invalid = invalid or random.random() < 0.25

        if invalid:
            return TFLunaSample(t=t, range_m=-1.0, vertical_m=-1.0, valid=False, outlier=False)

        r = z + np.random.randn() * cfg.range_noise_m
        if outlier:
            r += np.random.randn() * cfg.range_noise_m * cfg.outlier_scale

        valid = cfg.min_range_m <= r <= cfg.max_range_m
        return TFLunaSample(
            t=t,
            range_m=float(r),
            vertical_m=float(r),
            valid=valid,
            outlier=outlier
        )


class VL53L5CXModel:
    def __init__(self, cfg):
        self.cfg = cfg

    def sample(self, t: float, true_zone_ranges: np.ndarray, forced_invalid: bool = False) -> VL53Sample:
        zones = []
        cfg = self.cfg

        for r_true in true_zone_ranges.flatten():
            invalid = forced_invalid or random.random() < cfg.low_conf_prob

            if invalid:
                zones.append(VL53Zone(range_m=-1.0, confidence=0.0, valid=False))
                continue

            r = float(r_true + np.random.randn() * cfg.range_noise_m)
            r = clamp(r, cfg.min_range_m, cfg.max_range_m)

            conf = clamp(random.uniform(0.55, 0.98), 0.0, 1.0)
            valid = True

            zones.append(VL53Zone(range_m=r, confidence=conf, valid=valid))

        return VL53Sample(t=t, zones=zones, valid=not forced_invalid)