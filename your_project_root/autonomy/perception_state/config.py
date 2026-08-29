# autonomy/perception_state/config.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple

@dataclass
class CameraConfig:
    # Verified / supplied component properties
    sensor_name: str = "Sony IMX708"
    module_name: str = "Raspberry Pi Camera Module 3 NoIR Wide"
    native_width: int = 4608          # 12 MP 16:9 mode
    native_height: int = 2592
    diagonal_fov_deg: float = 120.0   # Wide version approximate viewing angle
    autofocus: bool = True

    # Do NOT force full 12 MP inference
    inference_resolutions: Tuple[int, ...] = (640, 512, 416)
    active_inference_size: int = 640

    # Center-crop square for square neural inference input
    crop_mode: str = "center_square"

    # Downward-looking mine inspection camera by default
    # This can be overridden per drone.
    mount: str = "DOWNWARD"

    pixel_noise_std_px: float = 1.2
    attitude_std_rad: float = 0.008
    focus_uncertainty_m: float = 0.12

@dataclass
class YoloConfig:
    model_path: str = ""              # local .pt or .onnx only
    class_names: Tuple[str, ...] = (
        "MINE_SURFACE",
        "MINE_SURFACE_CUE",
        "HUMAN",
        "OBSTACLE",
        "OPTIONAL_UNKNOWN_OBJECT"
    )
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    device: str = "cpu"               # onboard local CPU only, no cloud

@dataclass
class AdaptiveInferenceConfig:
    min_yolo_hz: float = 1.0
    nominal_yolo_hz: float = 6.0
    max_yolo_hz: float = 15.0

    cpu_high_pct: float = 82.0
    cpu_low_pct: float = 55.0

    fast_speed_mps: float = 3.5
    low_texture_score: float = 0.25
    mine_inspect_hz: float = 12.0
    mine_inspect_hold_s: float = 3.0

@dataclass
class LoopRatesConfig:
    flight_state_hz: float = 200.0
    imu_hz: float = 200.0
    optical_flow_hz: float = 80.0
    tf_luna_hz: float = 100.0
    vl53l5cx_hz: float = 25.0
    camera_hz: float = 30.0
    mapping_hz: float = 20.0
    planning_periodic_hz: float = 5.0
    swarm_heartbeat_hz: float = 2.0
    visualization_hz: float = 10.0

@dataclass
class IMUConfig:
    accel_white_noise: float = 0.06
    gyro_white_noise: float = 0.006
    accel_bias_init: float = 0.08
    gyro_bias_init: float = 0.004
    accel_bias_walk: float = 0.0025
    gyro_bias_walk: float = 0.00035

@dataclass
class BaroConfig:
    noise_m: float = 0.35
    drift_m_per_min: float = 0.25
    update_hz: float = 50.0

@dataclass
class MagConfig:
    noise_rad: float = 0.05
    disturbance_reject_rad: float = 0.35
    update_hz: float = 40.0

@dataclass
class OpticalFlowConfig:
    velocity_noise_mps: float = 0.10
    min_quality: float = 0.25
    motion_failure_mps: float = 5.0
    update_hz: float = 80.0

@dataclass
class TFLunaConfig:
    # Supplied verified specification
    min_range_m: float = 0.2
    max_range_m: float = 8.0
    fov_deg: float = 2.0
    default_update_hz: float = 100.0
    max_update_hz: float = 250.0
    mass_g_max: float = 5.0
    power_w_max: float = 0.35

    range_noise_m: float = 0.03
    outlier_prob: float = 0.006
    outlier_scale: float = 4.0
    invalid_prob: float = 0.003

@dataclass
class VL53L5CXConfig:
    zones_x: int = 8
    zones_y: int = 8
    update_hz: float = 15.0
    min_range_m: float = 0.05
    max_range_m: float = 4.0
    range_noise_m: float = 0.04
    low_conf_prob: float = 0.05

@dataclass
class OccupancyConfig:
    resolution_m: float = 0.10     # configurable: 0.05, 0.10, 0.20
    width_m: float = 60.0
    height_m: float = 60.0
    unknown_log_odds: float = 0.0
    free_increment: float = -0.20
    obstacle_increment: float = 0.55
    mine_increment: float = 0.95
    human_increment: float = 0.80
    clamp_min: float = -3.0
    clamp_max: float = 4.5

@dataclass
class MineFusionConfig:
    candidate_gate_m: float = 1.10
    confirmation_min_obs: int = 2
    confirmation_conf: float = 0.55
    confirmation_cov_trace_max: float = 0.85
    duplicate_gate_m: float = 1.25
    duplicate_mahalanobis_gate: float = 2.8
    mine_default_radius_m: float = 0.15
    required_clearance_m: float = 1.0

@dataclass
class SwarmCommsConfig:
    heartbeat_hz: float = 2.0
    base_latency_ms: float = 25.0
    jitter_ms: float = 35.0
    packet_loss_prob: float = 0.02
    max_delta_queue: int = 512

@dataclass
class FailureConfig:
    yolo_false_positive_rate: float = 0.010
    yolo_false_negative_rate: float = 0.035
    camera_dropout_prob: float = 0.001
    optical_flow_failure_prob: float = 0.003
    magnetometer_disturbance_prob: float = 0.002
    tf_luna_failure_prob: float = 0.002
    vl53_invalid_prob: float = 0.004
    imu_bias_spike_prob: float = 0.0008
    comm_dropout_prob: float = 0.002
    high_latency_prob: float = 0.004
    human_occlusion_prob: float = 0.010
    dynamic_obstacle_prob: float = 0.008

@dataclass
class PerceptionStateConfig:
    drone_id: int = 0
    camera: CameraConfig = field(default_factory=CameraConfig)
    yolo: YoloConfig = field(default_factory=YoloConfig)
    adaptive: AdaptiveInferenceConfig = field(default_factory=AdaptiveInferenceConfig)
    rates: LoopRatesConfig = field(default_factory=LoopRatesConfig)
    imu: IMUConfig = field(default_factory=IMUConfig)
    baro: BaroConfig = field(default_factory=BaroConfig)
    mag: MagConfig = field(default_factory=MagConfig)
    optical_flow: OpticalFlowConfig = field(default_factory=OpticalFlowConfig)
    tf_luna: TFLunaConfig = field(default_factory=TFLunaConfig)
    vl53: VL53L5CXConfig = field(default_factory=VL53L5CXConfig)
    occupancy: OccupancyConfig = field(default_factory=OccupancyConfig)
    mine: MineFusionConfig = field(default_factory=MineFusionConfig)
    swarm: SwarmCommsConfig = field(default_factory=SwarmCommsConfig)
    failures: FailureConfig = field(default_factory=FailureConfig)