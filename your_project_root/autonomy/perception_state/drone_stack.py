# autonomy/perception_state/drone_stack.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List
import numpy as np
# add at top of drone_stack.py
import time
from .common_math import quat_to_euler
from .config import PerceptionStateConfig
from .camera import PiCameraModule3Wide, CameraFrame
from .detector import create_detector, Detection
from .adaptive_inference import AdaptiveInferencePolicy
from .sensors import (
    IMUModel, BarometerModel, MagnetometerModel, OpticalFlowModel,
    TFLunaModel, VL53L5CXModel, SensorHealth
)
from .ekf import GPSDeniedEKF
from .fusion import SensorFusionBridge
from .geoloc import MineGeolocalizer, MineObservation
from .mine_fusion import LocalMineConfirmationEngine, GlobalMineMap
from .occupancy import OccupancyGrid
from .swarm_map import DroneMapSync, SwarmChannel
from .human_tracker import HumanTracker, HumanObservation
from .gesture import create_gesture_interface
from .perf import PerformanceMonitor
from .runtime import MultiRateScheduler, default_rate_table

@dataclass
class DroneOutputs:
    est_pos: np.ndarray
    est_vel: np.ndarray
    est_quat: np.ndarray
    yaw_rad: float
    pos_uncertainty: float
    confidence: float
    detections: List[Detection]
    local_mine_events: List[str]
    global_mine_ids: List[str]
    human_state: str
    ai_mode: str

class DroneAutonomyStack:
    """
    Simulated onboard compute for one drone.

    This object processes only its own sensor data.
    No ground-truth is consumed by autonomy.
    """

    def __init__(self, cfg: PerceptionStateConfig, swarm_channel: SwarmChannel):
        self.cfg = cfg
        self.drone_id = cfg.drone_id

        # Perception
        self.camera = PiCameraModule3Wide(cfg.camera, self.drone_id)
        self.detector = create_detector(cfg.yolo, cfg.failures, self.drone_id)
        self.inference_policy = AdaptiveInferencePolicy(cfg.adaptive, self.drone_id)

        # Sensors
        self.imu = IMUModel(cfg.imu, self.drone_id)
        self.baro = BarometerModel(cfg.baro)
        self.mag = MagnetometerModel(cfg.mag)
        self.flow = OpticalFlowModel(cfg.optical_flow)
        self.tf = TFLunaModel(cfg.tf_luna)
        self.vl53 = VL53L5CXModel(cfg.vl53)
        self.health = SensorHealth()

        # State estimation
        self.ekf = GPSDeniedEKF(initial_pos=np.array([0.0, 0.0, 1.0]))
        self.fusion = SensorFusionBridge(self.ekf, self.health)

        # Mine perception/fusion
        self.geoloc = MineGeolocalizer(self.camera)
        self.local_mines = LocalMineConfirmationEngine(cfg.mine, self.drone_id)
        self.global_mines = GlobalMineMap(cfg.mine)   # fused swarm view can be mirrored locally

        # Mapping
        self.grid = OccupancyGrid(cfg.occupancy)

        # Swarm
        self.map_sync = DroneMapSync(self.drone_id, cfg.swarm, swarm_channel)

        # Human/gesture
        self.human_tracker = HumanTracker()
        self.gesture = create_gesture_interface()

        # Performance
        self.perf = PerformanceMonitor(self.drone_id)

        # Runtime
        self.scheduler = MultiRateScheduler()
        rates = default_rate_table(cfg.rates)

        self.scheduler.add_loop("IMU", rates["IMU"], self._loop_imu, priority=10)
        self.scheduler.add_loop("OPTICAL_FLOW", rates["OPTICAL_FLOW"], self._loop_optical_flow, priority=20)
        self.scheduler.add_loop("TF_LUNA", rates["TF_LUNA"], self._loop_tf_luna, priority=20)
        self.scheduler.add_loop("VL53L5CX", rates["VL53L5CX"], self._loop_vl53, priority=30)
        self.scheduler.add_loop("CAMERA", rates["CAMERA"], self._loop_camera, priority=40)
        self.scheduler.add_loop("YOLO", rates["YOLO"], self._loop_yolo, priority=50)
        self.scheduler.add_loop("MAPPING", rates["MAPPING"], self._loop_mapping, priority=60)
        self.scheduler.add_loop("SWARM", rates["SWARM"], self._loop_swarm, priority=70)

        # Latest data caches
        self.latest_frame: Optional[CameraFrame] = None
        self.latest_detections: List[Detection] = []
        self.latest_tf_valid = False
        self.latest_tf_vertical = None
        self.latest_human_obs: List[HumanObservation] = []
        self.latest_local_mine_events: List[str] = []
        self.latest_global_mine_ids: List[str] = []

        self.current_yolo_hz = cfg.adaptive.nominal_yolo_hz

    def step(self, t: float, dt: float):
        self.scheduler.step(t)

    # -----------------------------
    # Sensor simulation hooks
    # -----------------------------
    # In Part 1, the world simulator should provide these true values
    # to the *sensor models*, not to autonomy directly.

    def _loop_imu(self, t: float):
        # The true acceleration/gyro must come from Part-1 flight sim.
        # Here we show the interface.
        true_accel_body = np.array([0.0, 0.0, 9.81])
        true_gyro_body = np.zeros(3)

        imu_sample = self.imu.sample(t, true_accel_body, true_gyro_body)
        self.fusion.predict_imu(imu_sample, dt=1.0 / self.cfg.rates.imu_hz)

    def _loop_optical_flow(self, t: float):
        # True body velocity and texture must come from world simulation.
        true_v_body = np.zeros(3)
        texture_score = 1.0
        motion_mag = float(np.linalg.norm(true_v_body))

        flow_sample = self.flow.sample(
            t=t,
            true_v_body=true_v_body,
            texture_score=texture_score,
            motion_magnitude=motion_mag,
            forced_failure=False
        )

        self.fusion.update_optical_flow(flow_sample)

    def _loop_tf_luna(self, t: float):
        true_altitude = float(self.ekf.pos[2])  # in full sim, use true altitude only in sensor model
        tilt_factor = 1.0

        tf_sample = self.tf.sample(t, true_altitude, tilt_factor)

        self.latest_tf_valid = tf_sample.valid
        self.latest_tf_vertical = tf_sample.vertical_m if tf_sample.valid else None

        self.fusion.update_tf_luna(tf_sample)

    def _loop_vl53(self, t: float):
        # 8x8 zones; Part-1 world sim supplies true local obstacle ranges.
        true_zone_ranges = np.full((8, 8), 3.5)
        vl_sample = self.vl53.sample(t, true_zone_ranges, forced_invalid=False)

        # Use for local obstacle awareness.
        if vl_sample.valid:
            pass

    def _loop_camera(self, t: float):
        # Part-1 world sim creates a CameraFrame with synthetic_objects.
        frame = CameraFrame(
            t=t,
            drone_id=self.drone_id,
            image=None,
            synthetic_objects=[],
            texture_score=1.0,
            blur_score=0.0,
            exposure_ok=True
        )

        self.latest_frame = frame
        self.perf.record_camera_frame(t)

    def _loop_yolo(self, t: float):
        if self.latest_frame is None:
            return

        t0 = time.time()
        dets = self.detector.detect(self.latest_frame)
        latency_ms = (time.time() - t0) * 1000.0

        self.latest_detections = dets
        self.perf.record_yolo(t, latency_ms)

        # Mine and human observations
        ekf_state = self.ekf.get_state()
        self.latest_local_mine_events = []
        self.latest_global_mine_ids = []

        mine_like = []

        for det in dets:
            if det.label in ("MINE_SURFACE", "MINE_SURFACE_CUE"):
                obs = self.geoloc.observation_from_detection(
                    det=det,
                    ekf_state=ekf_state,
                    tf_luna_valid=self.latest_tf_valid,
                    tf_luna_vertical_m=self.latest_tf_vertical
                )

                if obs is not None:
                    mine_like.append(obs)
                    confirmed_track_id = self.local_mines.add_observation(obs)

                    if confirmed_track_id is not None:
                        # Confirmed locally -> publish to global fused map
                        gmid, is_duplicate = self.global_mines.add_confirmed_observation(obs)

                        self.latest_local_mine_events.append(confirmed_track_id)
                        self.latest_global_mine_ids.append(gmid)

                        # Map inflation
                        mine = self.global_mines.mines[gmid]
                        self.grid.inflate_mine(
                            pos2=mine.pos2,
                            mine_radius_m=mine.radius_m,
                            clearance_m=mine.clearance_m
                        )

                        # Swarm delta
                        self.map_sync.publish_mine(
                            t=t,
                            mine_id=gmid,
                            pos2=mine.pos2,
                            cov2=mine.cov2,
                            confidence=mine.confidence,
                            label=mine.label
                        )

            elif det.label == "HUMAN":
                # Convert detection to local human observation.
                ground, cov = self.camera.estimate_ground_position(
                    xyxy=det.bbox_xyxy,
                    quat_wb=ekf_state.quat,
                    drone_pos_world=ekf_state.pos,
                    altitude_source=self.latest_tf_vertical if self.latest_tf_valid else None,
                    altitude_std=0.15
                )

                if ground is not None and cov is not None:
                    self.latest_human_obs.append(HumanObservation(
                        t=t,
                        drone_id=self.drone_id,
                        pos2=ground[:2],
                        cov2=cov,
                        confidence=det.confidence
                    ))

        # Adaptive inference policy
        cpu_pct = self.perf.cpu_pct
        speed_mps = float(np.linalg.norm(self.ekf.vel))
        interesting = len(dets)
        texture_score = self.latest_frame.texture_score if self.latest_frame else 1.0
        mine_candidate = len(mine_like) > 0

        self.current_yolo_hz = self.inference_policy.update(
            t=t,
            cpu_pct=cpu_pct,
            speed_mps=speed_mps,
            interesting_objects=interesting,
            texture_score=texture_score,
            mine_candidate=mine_candidate
        )

        self.scheduler.set_rate("YOLO", self.current_yolo_hz)

    def _loop_mapping(self, t: float):
        # Update local occupancy using latest sensors / detections.
        pass

    def _loop_swarm(self, t: float):
        deltas = self.map_sync.receive(t)

        for d in deltas:
            if d.kind == "MINE":
                # Apply remote mine event into local fused grid.
                pos2 = np.array(d.payload["pos2"], dtype=float)
                cov2 = np.array(d.payload["cov2"], dtype=float)

                # Simple local insertion; in full implementation associate with global map.
                self.grid.inflate_mine(
                    pos2=pos2,
                    mine_radius_m=self.cfg.mine.mine_default_radius_m,
                    clearance_m=self.cfg.mine.required_clearance_m
                )

    def outputs(self, t: float) -> DroneOutputs:
        st = self.ekf.get_state()
        yaw = float(quat_to_euler(st.quat)[2])

        return DroneOutputs(
            est_pos=st.pos.copy(),
            est_vel=st.vel.copy(),
            est_quat=st.quat.copy(),
            yaw_rad=yaw,
            pos_uncertainty=self.ekf.position_uncertainty(),
            confidence=self.ekf.confidence(),
            detections=self.latest_detections.copy(),
            local_mine_events=self.latest_local_mine_events.copy(),
            global_mine_ids=self.latest_global_mine_ids.copy(),
            human_state=self.human_tracker.state,
            ai_mode=self.detector.ai_mode
        )