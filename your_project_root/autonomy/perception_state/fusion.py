# autonomy/perception_state/fusion.py
from __future__ import annotations
from .ekf import GPSDeniedEKF
from .sensors import SensorHealth

class SensorFusionBridge:
    """
    Routes sensor samples into the GPS-denied EKF.

    Important:
      This uses health weights.
      It does NOT use GPS.
    """

    def __init__(self, ekf: GPSDeniedEKF, health: SensorHealth):
        self.ekf = ekf
        self.health = health

    def predict_imu(self, imu_sample, dt: float):
        if imu_sample.valid and self.health.imu_ok:
            self.ekf.predict(imu_sample.accel, imu_sample.gyro, dt)

    def update_baro(self, baro_sample):
        if not baro_sample.valid:
            self.health.baro_ok = False
            return
        self.health.baro_ok = True
        self.ekf.update_baro(
            altitude_m=baro_sample.altitude_m,
            R_meas=0.35**2,
            weight=self.health.ekf_weight_baro()
        )

    def update_mag(self, mag_sample):
        self.health.mag_disturbance = mag_sample.disturbance_rad

        if not mag_sample.valid:
            self.health.mag_ok = False
            return

        self.health.mag_ok = True
        self.ekf.update_mag_yaw(
            yaw_meas_rad=mag_sample.yaw_rad,
            R_meas=0.08**2,
            weight=self.health.ekf_weight_mag()
        )

    def update_optical_flow(self, flow_sample):
        self.health.optical_flow_quality = flow_sample.quality

        if not flow_sample.valid:
            self.health.optical_flow_ok = False
            return

        self.health.optical_flow_ok = True
        self.ekf.update_optical_flow(
            vx_body=flow_sample.vx_body_mps,
            vy_body=flow_sample.vy_body_mps,
            quality=flow_sample.quality,
            R_base=0.12**2,
            weight=self.health.ekf_weight_optical_flow()
        )

    def update_tf_luna(self, tf_sample):
        self.health.tf_luna_valid = tf_sample.valid

        if not tf_sample.valid:
            self.health.tf_luna_ok = False
            return

        self.health.tf_luna_ok = True
        self.ekf.update_range(
            vertical_range_m=tf_sample.vertical_m,
            R_meas=0.05**2,
            weight=self.health.ekf_weight_range()
        )