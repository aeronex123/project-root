# autonomy/gui/core/state.py
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


Vec3 = Tuple[float, float, float]
Vec4 = Tuple[float, float, float, float]
BBox = Tuple[float, float, float, float]


class DataSource(str, Enum):
    SIMULATION = "SIMULATION"
    LIVE = "LIVE"
    HYBRID = "HYBRID"
    OFFLINE = "OFFLINE"


class MissionPhase(str, Enum):
    IDLE = "IDLE"
    INITIALIZING = "INITIALIZING"
    TAKEOFF = "TAKEOFF"
    SEARCH = "SEARCH"
    MAPPING = "MAPPING"
    TARGET_TRACKING = "TARGET_TRACKING"
    PATH_PLANNING = "PATH_PLANNING"
    ESCORT = "ESCORT"
    LANDING = "LANDING"
    COMPLETE = "COMPLETE"
    ABORTED = "ABORTED"
    FAILSAFE = "FAILSAFE"


class ComponentStatus(str, Enum):
    OK = "OK"
    WARNING = "WARNING"
    ERROR = "ERROR"
    OFFLINE = "OFFLINE"


@dataclass(frozen=True)
class EventView:
    t: float
    level: str
    subsystem: str
    message: str


@dataclass
class DetectionView:
    label: str
    confidence: float
    bbox: BBox
    status: str


@dataclass
class MineView:
    mine_id: str
    position: Vec3
    confidence: float
    status: str
    source: str
    radius: float
    clearance: float


@dataclass
class HumanView:
    position: Vec3
    velocity: Vec3
    heading: float
    confidence: float
    state: str
    gesture: str
    gesture_confidence: float


@dataclass
class PerformanceView:
    cpu_pct: float = 0.0
    ram_pct: float = 0.0
    temp_c: float = 0.0
    yolo_fps: float = 0.0
    yolo_latency_ms: float = 0.0
    gui_fps: float = 0.0


@dataclass
class CommLinkView:
    a: int
    b: int
    connected: bool
    quality: float
    latency_ms: float
    packet_loss: float


@dataclass
class DroneState:
    drone_id: int

    position: Vec3
    velocity: Vec3
    acceleration: Vec3

    attitude_quat: Vec4
    yaw: float
    altitude: float

    battery: float
    flight_mode: str
    armed: bool

    localization_confidence: float
    position_uncertainty: float

    ai_mode: str
    path_status: str

    path: List[Vec3] = field(default_factory=list)
    trail: List[Vec3] = field(default_factory=list)
    detections: List[DetectionView] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    sensor_health: Dict[str, ComponentStatus] = field(default_factory=dict)
    perf: PerformanceView = field(default_factory=PerformanceView)

    comm_quality: float = 1.0
    safety_radius: float = 1.2


@dataclass
class MissionState:
    phase: MissionPhase
    elapsed: float
    remaining: float
    progress: float
    objective: str
    start_position: Vec3
    target_position: Vec3
    result: str


@dataclass
class EnvironmentState:
    length_m: float
    width_m: float

    mines: List[MineView] = field(default_factory=list)
    obstacles: List[Vec3] = field(default_factory=list)

    human: Optional[HumanView] = None

    # Optional RGBA image for minimap / ground texture.
    # This can be a NumPy array, but Any is used here to avoid
    # forcing a NumPy dependency in the state schema itself.
    occupancy_rgba: Optional[Any] = None

    safe_path: List[Vec3] = field(default_factory=list)
    explored_fraction: float = 0.0


@dataclass
class SwarmState:
    t: float
    source: DataSource

    mission: MissionState
    environment: EnvironmentState

    drones: List[DroneState] = field(default_factory=list)
    links: List[CommLinkView] = field(default_factory=list)
    events: List[EventView] = field(default_factory=list)

    message: str = ""

    @classmethod
    def error_state(cls, message: str) -> "SwarmState":
        mission = MissionState(
            phase=MissionPhase.FAILSAFE,
            elapsed=0.0,
            remaining=0.0,
            progress=0.0,
            objective="ERROR",
            start_position=(0.0, 0.0, 0.0),
            target_position=(0.0, 0.0, 0.0),
            result="ERROR",
        )

        environment = EnvironmentState(
            length_m=60.0,
            width_m=15.0,
        )

        return cls(
            t=0.0,
            source=DataSource.OFFLINE,
            mission=mission,
            environment=environment,
            message=message,
        )