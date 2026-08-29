# autonomy/gui/simulation/demo_state.py
from __future__ import annotations

import math
from collections import deque
from typing import Dict, List, Tuple

import numpy as np

from ..config import GuiConfig
from ..core.state import (
    ComponentStatus,
    CommLinkView,
    DataSource,
    DetectionView,
    DroneState,
    EnvironmentState,
    EventView,
    HumanView,
    MissionPhase,
    MissionState,
    MineView,
    PerformanceView,
    SwarmState,
)


Vec3 = Tuple[float, float, float]


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


class DemoStateProvider:
    """
    Deterministic demonstration scenario.

    IMPORTANT:
      The whole timeline automatically scales to cfg.mission_duration_s.

      If duration = 180 s, the entire mission completes in under 3 minutes.
      If duration = 600 s, it behaves like the original long demo.
    """

    def __init__(self, cfg: GuiConfig):
        self.cfg = cfg

        self.length = float(cfg.field_length_m)
        self.width = float(cfg.field_width_m)
        self.duration = float(cfg.mission_duration_s)

        self.occ_w = 300
        self.occ_h = 75

        xs = np.linspace(0.0, self.length, self.occ_w, dtype=np.float32)
        ys = np.linspace(-self.width / 2.0, self.width / 2.0, self.occ_h, dtype=np.float32)
        self.X, self.Y = np.meshgrid(xs, ys, indexing="xy")

        s = self._s

        self.mine_defs = [
            {"mine_id": "MINE_001", "x": 18.0, "y": -2.0, "appear": s(180), "confirm": s(210), "source": "DRONE-01"},
            {"mine_id": "MINE_002", "x": 28.0, "y": 1.5, "appear": s(200), "confirm": s(235), "source": "DRONE-02"},
            {"mine_id": "MINE_003", "x": 38.0, "y": -3.0, "appear": s(230), "confirm": s(265), "source": "DRONE-03"},
            {"mine_id": "MINE_004", "x": 48.0, "y": 2.0, "appear": s(260), "confirm": s(295), "source": "DRONE-01"},
        ]

        base_wp: Dict[int, List[Tuple[float, float, float, float]]] = {
            1: [
                (0, 0, -3, 0), (10, 0, -3, 0), (20, 2, -3, 1.2), (60, 12, -3.5, 1.5),
                (180, 22, -5, 1.5), (240, 28, -4, 1.6), (300, 34, -2, 1.5),
                (360, 40, -1, 1.5), (420, 46, -0.5, 1.5), (480, 52, 0, 1.5),
                (540, 58, 0, 1.2), (580, 60, 0, 0.8), (600, 60, 0, 0),
            ],
            2: [
                (0, 0, 0, 0), (10, 0, 0, 0), (20, 2, 0, 1.2), (60, 12, 0, 1.5),
                (180, 22, 3, 1.5), (240, 28, 2, 1.6), (300, 34, 1, 1.5),
                (360, 40, 0.5, 1.5), (420, 46, 0, 1.5), (480, 52, 0, 1.5),
                (540, 58, 0, 1.2), (580, 60, 0, 0.8), (600, 60, 0, 0),
            ],
            3: [
                (0, 0, 3, 0), (10, 0, 3, 0), (20, 2, 3, 1.2), (60, 12, 3.5, 1.5),
                (180, 22, 5, 1.5), (240, 28, 4, 1.6), (300, 34, 2, 1.5),
                (360, 40, 1, 1.5), (420, 46, 0.5, 1.5), (480, 52, 0, 1.5),
                (540, 58, 0, 1.2), (580, 60, 0, 0.8), (600, 60, 0, 0),
            ],
        }

        self.waypoints = {
            k: [(s(t), x, y, z) for (t, x, y, z) in v]
            for k, v in base_wp.items()
        }

        self.reset()

    def _s(self, x: float) -> float:
        """Scale a 600-second design time to the configured mission duration."""
        return x * self.duration / 600.0

    def reset(self) -> None:
        self._last_t = 0.0
        self._freeze_t = 0.0
        self.emergency = False

        self._prev_pos: Dict[int, Vec3] = {}
        self._prev_vel: Dict[int, Vec3] = {1: (0, 0, 0), 2: (0, 0, 0), 3: (0, 0, 0)}

        self._trails: Dict[int, deque] = {
            1: deque(maxlen=700),
            2: deque(maxlen=700),
            3: deque(maxlen=700),
        }

        self._events: List[EventView] = []
        self._event_keys: set = set()

        self._last_occ_sec = -1
        self._occ_image = None

    def emergency_stop(self) -> None:
        self.emergency = True
        self._freeze_t = self._last_t

    def update(self, t: float, dt: float) -> SwarmState:
        self._last_t = float(t)
        sim_t = self._freeze_t if self.emergency else float(t)
        s = self._s

        phase = self._phase(sim_t)
        self._add_phase_events(sim_t, phase)

        pos_dict: Dict[int, Vec3] = {}
        vel_dict: Dict[int, Vec3] = {}
        yaw_dict: Dict[int, float] = {}
        trail_dict: Dict[int, List[Vec3]] = {}

        for drone_id in (1, 2, 3):
            pos = self._interp_waypoints(self.waypoints[drone_id], sim_t)

            prev_pos = self._prev_pos.get(drone_id, pos)
            raw_vel = (
                (pos[0] - prev_pos[0]) / dt if dt > 1e-6 else 0.0,
                (pos[1] - prev_pos[1]) / dt if dt > 1e-6 else 0.0,
                (pos[2] - prev_pos[2]) / dt if dt > 1e-6 else 0.0,
            )

            old_vel = self._prev_vel[drone_id]
            vel = (
                0.70 * old_vel[0] + 0.30 * raw_vel[0],
                0.70 * old_vel[1] + 0.30 * raw_vel[1],
                0.70 * old_vel[2] + 0.30 * raw_vel[2],
            )

            speed = math.hypot(vel[0], vel[1])
            yaw = math.atan2(vel[1], vel[0]) if speed > 0.15 else 0.0

            self._prev_pos[drone_id] = pos
            self._prev_vel[drone_id] = vel
            self._update_trail(drone_id, pos)

            pos_dict[drone_id] = pos
            vel_dict[drone_id] = vel
            yaw_dict[drone_id] = yaw
            trail_dict[drone_id] = list(self._trails[drone_id])

        mines = self._mines(sim_t)
        human = self._human(sim_t)
        links = self._links(sim_t, pos_dict)
        safe_path = self._safe_path(sim_t)
        occupancy = self._occupancy(sim_t, pos_dict, mines, human, safe_path)

        warnings = self._warnings(sim_t, pos_dict, mines)
        detections = self._detections(sim_t, pos_dict, mines, human)

        drones: List[DroneState] = []
        max_x = 0.0

        for drone_id in (1, 2, 3):
            pos = pos_dict[drone_id]
            vel = vel_dict[drone_id]
            max_x = max(max_x, pos[0])

            battery = clamp(100.0 - 40.0 * (sim_t / self.duration) - drone_id * 1.5, 18.0, 100.0)
            loc_conf = clamp(0.94 - 0.08 * abs(math.sin(sim_t / 11.0 + drone_id)), 0.55, 0.99)
            pos_uncertainty = 0.12 + 0.05 * abs(math.sin(sim_t / 7.0 + drone_id))

            armed = sim_t >= s(10) and not self.emergency

            flight_mode = "STANDBY"
            if phase == MissionPhase.TAKEOFF:
                flight_mode = "TAKEOFF"
            elif phase in (
                MissionPhase.SEARCH,
                MissionPhase.MAPPING,
                MissionPhase.TARGET_TRACKING,
                MissionPhase.PATH_PLANNING,
                MissionPhase.ESCORT,
            ):
                flight_mode = "AUTO"
            elif phase == MissionPhase.LANDING:
                flight_mode = "LANDING"
            elif phase == MissionPhase.COMPLETE:
                flight_mode = "DISARMED"
            elif phase == MissionPhase.ABORTED:
                flight_mode = "FAILSAFE"

            if sim_t < s(25):
                path_status = "PLANNING"
            elif s(240) <= sim_t < s(260):
                path_status = "REPLANNING"
            elif sim_t >= s(560):
                path_status = "COMPLETE"
            elif self.emergency:
                path_status = "BLOCKED"
            else:
                path_status = "ACTIVE"

            comm_quality = 1.0
            for link in links:
                if link.a == drone_id or link.b == drone_id:
                    comm_quality = max(comm_quality, link.quality)

            drones.append(
                DroneState(
                    drone_id=drone_id,
                    position=pos,
                    velocity=vel,
                    acceleration=(0.0, 0.0, 0.0),
                    attitude_quat=(1.0, 0.0, 0.0, 0.0),
                    yaw=yaw_dict[drone_id],
                    altitude=pos[2],
                    battery=battery,
                    flight_mode=flight_mode,
                    armed=armed,
                    localization_confidence=loc_conf,
                    position_uncertainty=pos_uncertainty,
                    ai_mode="SIMULATION FALLBACK",
                    path_status=path_status,
                    path=self._drone_path(drone_id),
                    trail=trail_dict[drone_id],
                    detections=detections[drone_id],
                    warnings=warnings[drone_id],
                    sensor_health=self._sensor_health(sim_t, drone_id),
                    perf=self._perf(sim_t, drone_id),
                    comm_quality=comm_quality,
                    safety_radius=1.2,
                )
            )

        environment = EnvironmentState(
            length_m=self.length,
            width_m=self.width,
            mines=mines,
            obstacles=[],
            human=human,
            occupancy_rgba=occupancy,
            safe_path=safe_path,
            explored_fraction=clamp(max_x / self.length, 0.0, 1.0),
        )

        mission = MissionState(
            phase=phase,
            elapsed=float(t),
            remaining=max(0.0, self.duration - float(t)),
            progress=clamp(float(t) / self.duration, 0.0, 1.0),
            objective=self._objective(phase),
            start_position=(0.0, 0.0, 0.0),
            target_position=(self.length, 0.0, 0.0),
            result=self._mission_result(phase),
        )

        return SwarmState(
            t=float(t),
            source=DataSource.SIMULATION,
            mission=mission,
            environment=environment,
            drones=drones,
            links=links,
            events=list(self._events),
            message="",
        )

    def _phase(self, t: float) -> MissionPhase:
        s = self._s

        if self.emergency:
            return MissionPhase.ABORTED
        if t < s(10):
            return MissionPhase.INITIALIZING
        if t < s(25):
            return MissionPhase.TAKEOFF
        if t < s(60):
            return MissionPhase.SEARCH
        if t < s(240):
            return MissionPhase.MAPPING
        if t < s(330):
            return MissionPhase.TARGET_TRACKING
        if t < s(420):
            return MissionPhase.PATH_PLANNING
        if t < s(540):
            return MissionPhase.ESCORT
        if t < s(580):
            return MissionPhase.LANDING

        return MissionPhase.COMPLETE

    def _objective(self, phase: MissionPhase) -> str:
        return {
            MissionPhase.IDLE: "Idle",
            MissionPhase.INITIALIZING: "Initialize swarm and self-tests",
            MissionPhase.TAKEOFF: "Autonomous takeoff and formation",
            MissionPhase.SEARCH: "Begin GPS-denied search pattern",
            MissionPhase.MAPPING: "Build fused occupancy/mine map",
            MissionPhase.TARGET_TRACKING: "Acquire and track person-at-risk",
            MissionPhase.PATH_PLANNING: "Plan safe escort corridor",
            MissionPhase.ESCORT: "Escort person through safe corridor",
            MissionPhase.LANDING: "Return and land",
            MissionPhase.COMPLETE: "Mission complete",
            MissionPhase.ABORTED: "Mission aborted",
            MissionPhase.FAILSAFE: "Failsafe condition",
        }.get(phase, "Unknown")

    def _mission_result(self, phase: MissionPhase) -> str:
        if phase == MissionPhase.COMPLETE:
            return "SUCCESS"
        if phase == MissionPhase.ABORTED:
            return "ABORTED"
        if phase == MissionPhase.FAILSAFE:
            return "FAILSAFE"
        return "IN_PROGRESS"

    def _interp_waypoints(self, waypoints, t: float) -> Vec3:
        if t <= waypoints[0][0]:
            _, x, y, z = waypoints[0]
            return (x, y, z)

        if t >= waypoints[-1][0]:
            _, x, y, z = waypoints[-1]
            return (x, y, z)

        for i in range(len(waypoints) - 1):
            t0, x0, y0, z0 = waypoints[i]
            t1, x1, y1, z1 = waypoints[i + 1]

            if t0 <= t <= t1:
                alpha = (t - t0) / max(1e-6, t1 - t0)
                return (
                    x0 + alpha * (x1 - x0),
                    y0 + alpha * (y1 - y0),
                    z0 + alpha * (z1 - z0),
                )

        _, x, y, z = waypoints[-1]
        return (x, y, z)

    def _drone_path(self, drone_id: int) -> List[Vec3]:
        return [(x, y, z) for (_, x, y, z) in self.waypoints[drone_id]]

    def _update_trail(self, drone_id: int, pos: Vec3) -> None:
        trail = self._trails[drone_id]

        if not trail:
            trail.append(pos)
            return

        last = trail[-1]
        dist = math.sqrt(
            (pos[0] - last[0]) ** 2 + (pos[1] - last[1]) ** 2 + (pos[2] - last[2]) ** 2
        )

        if dist > 0.10:
            trail.append(pos)

    def _mines(self, t: float) -> List[MineView]:
        mines: List[MineView] = []

        for m in self.mine_defs:
            if t < m["appear"]:
                continue

            if t >= m["confirm"]:
                status = "CONFIRMED"
                confidence = clamp(0.82 + 0.02 * math.sin(t / 3.0), 0.75, 0.97)
            elif t >= m["appear"] + self._s(30):
                status = "PROBABLE"
                confidence = clamp(0.58 + 0.04 * math.sin(t / 2.3), 0.45, 0.80)
            else:
                status = "SUSPECTED"
                confidence = clamp(0.38 + 0.03 * math.sin(t / 1.7), 0.30, 0.55)

            mines.append(
                MineView(
                    mine_id=m["mine_id"],
                    position=(m["x"], m["y"], 0.0),
                    confidence=confidence,
                    status=status,
                    source=m["source"],
                    radius=0.16,
                    clearance=1.0,
                )
            )

        return mines

    def _human(self, t: float):
        appear = self._s(300)
        escort_start = self._s(420)
        escort_span = max(1e-6, self._s(180))

        if t < appear:
            return None

        if t < escort_start:
            x, y, vx, vy = 36.0, 2.0, 0.0, 0.0
            state = "HUMAN_TRACKING"
        else:
            p = clamp((t - escort_start) / escort_span, 0.0, 1.0)
            x = 36.0 + p * 20.0
            y = 2.0 - p * 2.0
            vx, vy = 0.12, -0.012
            state = "HUMAN_ESCORT"

        heading = math.atan2(vy, vx) if abs(vx) > 1e-3 or abs(vy) > 1e-3 else 0.0

        gesture = "NONE"
        gesture_conf = 0.0

        if self._s(360) <= t < self._s(390):
            gesture = "FORWARD"
            gesture_conf = 0.87
        elif self._s(390) <= t < self._s(420):
            gesture = "PAUSE"
            gesture_conf = 0.78

        return HumanView(
            position=(x, y, 0.0),
            velocity=(vx, vy, 0.0),
            heading=heading,
            confidence=clamp(0.80 + 0.08 * math.sin(t / 4.0), 0.55, 0.98),
            state=state,
            gesture=gesture,
            gesture_confidence=gesture_conf,
        )

    def _links(self, t: float, pos_dict: Dict[int, Vec3]) -> List[CommLinkView]:
        pairs = [(1, 2), (2, 3), (1, 3)]
        links: List[CommLinkView] = []

        dropout = self._s(250) <= t < self._s(262)

        for a, b in pairs:
            pa = pos_dict[a]
            pb = pos_dict[b]

            dist = math.sqrt(
                (pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2 + (pa[2] - pb[2]) ** 2
            )

            quality = clamp(0.96 - dist / 90.0, 0.0, 1.0)

            if dropout:
                quality = 0.04

            connected = quality > 0.25

            latency = 18.0 + dist * 1.35 + 6.0 * abs(math.sin(t / 1.0 + a))
            packet_loss = clamp(0.004 + 0.020 * (1.0 - quality), 0.0, 0.35)

            links.append(
                CommLinkView(
                    a=a,
                    b=b,
                    connected=connected,
                    quality=quality,
                    latency_ms=latency,
                    packet_loss=packet_loss,
                )
            )

        if dropout:
            self._add_event(t, "comm_dropout", "WARNING", "COMM", "Swarm communication degraded")

        return links

    def _safe_path(self, t: float) -> List[Vec3]:
        if t < self._s(420):
            return []

        return [
            (5.0, 0.0, 0.0), (15.0, 0.0, 0.0), (25.0, 0.0, 0.0),
            (35.0, 0.0, 0.0), (45.0, 0.0, 0.0), (55.0, 0.0, 0.0),
        ]

    def _sensor_health(self, t: float, drone_id: int) -> Dict[str, ComponentStatus]:
        s = self._s

        health = {
            "CAMERA": ComponentStatus.OK,
            "IMU": ComponentStatus.OK,
            "BARO": ComponentStatus.OK,
            "MAG": ComponentStatus.OK,
            "OPTICAL_FLOW": ComponentStatus.OK,
            "TF_LUNA": ComponentStatus.OK,
            "VL53L5CX": ComponentStatus.OK,
            "EKF": ComponentStatus.OK,
            "DETECTOR": ComponentStatus.OK,
            "COMM": ComponentStatus.OK,
        }

        if drone_id == 2 and s(120) <= t < s(140):
            health["OPTICAL_FLOW"] = ComponentStatus.WARNING

        if drone_id == 3 and s(220) <= t < s(228):
            health["TF_LUNA"] = ComponentStatus.ERROR

        if s(250) <= t < s(262):
            health["COMM"] = ComponentStatus.WARNING

        return health

    def _perf(self, t: float, drone_id: int) -> PerformanceView:
        cpu = clamp(36.0 + 18.0 * abs(math.sin(t / 2.3 + drone_id)), 5.0, 98.0)
        ram = clamp(34.0 + 9.0 * abs(math.sin(t / 5.7 + drone_id)), 10.0, 90.0)
        temp = clamp(46.0 + cpu * 0.22, 35.0, 88.0)

        return PerformanceView(
            cpu_pct=cpu,
            ram_pct=ram,
            temp_c=temp,
            yolo_fps=clamp(6.0 + 1.5 * math.sin(t / 1.7 + drone_id), 1.0, 15.0),
            yolo_latency_ms=clamp(92.0 + 24.0 * math.sin(t / 2.0 + drone_id), 35.0, 220.0),
            gui_fps=self.cfg.render_hz,
        )

    def _warnings(self, t: float, pos_dict: Dict[int, Vec3], mines: List[MineView]) -> Dict[int, List[str]]:
        warnings: Dict[int, List[str]] = {1: [], 2: [], 3: []}

        for a, b in [(1, 2), (2, 3), (1, 3)]:
            pa = pos_dict[a]
            pb = pos_dict[b]
            dist = math.hypot(pa[0] - pb[0], pa[1] - pb[1])

            if dist < 2.0:
                if f"COLLISION RISK with DRONE-{b:02d}" not in warnings[a]:
                    warnings[a].append(f"COLLISION RISK with DRONE-{b:02d}")
                if f"COLLISION RISK with DRONE-{a:02d}" not in warnings[b]:
                    warnings[b].append(f"COLLISION RISK with DRONE-{a:02d}")

        for drone_id, pos in pos_dict.items():
            for mine in mines:
                mx, my, _ = mine.position
                dist = math.hypot(pos[0] - mx, pos[1] - my)

                if dist < mine.clearance + 0.65:
                    warnings[drone_id].append(f"NEAR EXCLUSION ZONE {mine.mine_id}")

            battery = clamp(100.0 - 40.0 * (t / self.duration) - drone_id * 1.5, 18.0, 100.0)
            if battery < 30.0:
                warnings[drone_id].append("LOW BATTERY")

        return warnings

    def _detections(self, t: float, pos_dict: Dict[int, Vec3], mines: List[MineView], human) -> Dict[int, List[DetectionView]]:
        detections: Dict[int, List[DetectionView]] = {1: [], 2: [], 3: []}

        for drone_id, pos in pos_dict.items():
            for mine in mines:
                mx, my, _ = mine.position
                if math.hypot(pos[0] - mx, pos[1] - my) < 14.0:
                    detections[drone_id].append(
                        DetectionView(
                            label="MINE_SURFACE_CUE",
                            confidence=mine.confidence,
                            bbox=(120.0, 120.0, 190.0, 190.0),
                            status=mine.status,
                        )
                    )

            if human is not None:
                hx, hy, _ = human.position
                if math.hypot(pos[0] - hx, pos[1] - hy) < 16.0:
                    detections[drone_id].append(
                        DetectionView(
                            label="HUMAN",
                            confidence=human.confidence,
                            bbox=(240.0, 100.0, 320.0, 300.0),
                            status="TRACKED",
                        )
                    )

        return detections

    def _occupancy(self, t: float, pos_dict, mines, human, safe_path):
        sec = int(t)

        if sec == self._last_occ_sec and self._occ_image is not None:
            return self._occ_image

        self._last_occ_sec = sec

        img = np.zeros((self.occ_h, self.occ_w, 4), dtype=np.uint8)

        img[:, :, 0] = 32
        img[:, :, 1] = 34
        img[:, :, 2] = 40
        img[:, :, 3] = 255

        max_x = max(pos[0] for pos in pos_dict.values())

        explored_mask = self.X <= max_x + 5.0
        img[explored_mask, 0] = 18
        img[explored_mask, 1] = 42
        img[explored_mask, 2] = 28
        img[explored_mask, 3] = 255

        for mine in mines:
            mx, my, _ = mine.position

            exclusion_mask = (self.X - mx) ** 2 + (self.Y - my) ** 2 <= (mine.radius + mine.clearance) ** 2
            img[exclusion_mask, 0] = 120
            img[exclusion_mask, 1] = 22
            img[exclusion_mask, 2] = 22
            img[exclusion_mask, 3] = 255

            core_mask = (self.X - mx) ** 2 + (self.Y - my) ** 2 <= mine.radius ** 2
            img[core_mask, 0] = 255
            img[core_mask, 1] = 45
            img[core_mask, 2] = 45
            img[core_mask, 3] = 255

        if safe_path:
            path_mask = (self.X >= safe_path[0][0]) & (self.X <= safe_path[-1][0]) & (abs(self.Y) <= 0.8)
            img[path_mask, 0] = 20
            img[path_mask, 1] = 110
            img[path_mask, 2] = 120
            img[path_mask, 3] = 255

        if human is not None:
            hx, hy, _ = human.position
            human_mask = (self.X - hx) ** 2 + (self.Y - hy) ** 2 <= 1.2 ** 2
            img[human_mask, 0] = 35
            img[human_mask, 1] = 90
            img[human_mask, 2] = 255
            img[human_mask, 3] = 255

        self._occ_image = img
        return img

    def _add_event(self, t: float, key: str, level: str, subsystem: str, message: str) -> None:
        if key in self._event_keys:
            return

        self._event_keys.add(key)
        self._events.append(EventView(t=t, level=level, subsystem=subsystem, message=message))

    def _add_phase_events(self, t: float, phase: MissionPhase) -> None:
        s = self._s

        self._add_event(t, f"phase:{phase.value}", "INFO", "MISSION", f"Mission phase changed to {phase.value}")

        for mine in self.mine_defs:
            if t >= mine["appear"]:
                self._add_event(t, f"mine:{mine['mine_id']}", "WARNING", "PERCEPTION", f"{mine['mine_id']} detected by {mine['source']}")

            if t >= mine["confirm"]:
                self._add_event(t, f"mine_confirmed:{mine['mine_id']}", "WARNING", "MINE_FUSION", f"{mine['mine_id']} confirmed and exclusion zone created")

        if t >= s(300):
            self._add_event(t, "human_acquired", "INFO", "HUMAN_TRACKER", "Person-at-risk track acquired")

        if t >= s(360):
            self._add_event(t, "gesture_forward", "INFO", "GESTURE", "Gesture recognized: FORWARD")

        if t >= s(420):
            self._add_event(t, "escort_path", "INFO", "PLANNER", "Safe escort corridor generated")

        if t >= s(580):
            self._add_event(t, "mission_complete", "INFO", "MISSION", "Mission complete")

        if self.emergency:
            self._add_event(t, "emergency_stop", "CRITICAL", "SAFETY", "Emergency stop commanded")