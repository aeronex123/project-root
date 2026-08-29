# autonomy/gui/adapters/perception_adapter.py
from __future__ import annotations

from typing import Optional

from ..config import GuiConfig
from ..core.state import (
    ComponentStatus,
    DataSource,
    DetectionView,
    DroneState,
    MissionPhase,
    MissionState,
    MineView,
    PerformanceView,
    SwarmState,
)
from ..simulation.demo_state import DemoStateProvider


class PerceptionAdapter:
    """
    Normalizes autonomy/system output into SwarmState.

    Default mode is deterministic SIMULATION.

    If mode == "live", this adapter attempts to bridge the existing
    autonomy.perception_state.system.ThreeDroneSwarmSim.

    Important:
      The existing ThreeDroneSwarmSim is still a simulated autonomy stack
      unless you connect it to your Part-1 world/physics simulator and
      sensor generation layer.

      Therefore the GUI clearly labels data as SIMULATION or HYBRID.
    """

    def __init__(self, cfg: GuiConfig):
        self.cfg = cfg
        self.demo = DemoStateProvider(cfg)

        self.mode = DataSource.SIMULATION
        self.live_sim = None
        self.live_error: Optional[str] = None

        if cfg.mode == "live":
            self._try_init_live()

    def _try_init_live(self) -> None:
        try:
            from autonomy.perception_state.system import ThreeDroneSwarmSim

            self.live_sim = ThreeDroneSwarmSim()
            self.mode = DataSource.HYBRID
            self.live_error = None

        except Exception as exc:
            self.live_sim = None
            self.mode = DataSource.SIMULATION
            self.live_error = str(exc)

    def reset(self) -> None:
        self.demo.reset()

        if self.cfg.mode == "live":
            self._try_init_live()

    def emergency_stop(self) -> None:
        self.demo.emergency_stop()

    def update(self, t: float, dt: float) -> SwarmState:
        if self.live_sim is not None:
            try:
                self.live_sim.step(t, dt)
                return self._map_live(t, dt)
            except Exception as exc:
                self.live_sim = None
                self.mode = DataSource.SIMULATION
                self.live_error = str(exc)

        return self.demo.update(t, dt)

    def _map_live(self, t: float, dt: float) -> SwarmState:
        """
        Maps the existing autonomy stack into GUI state.

        This uses the existing drone outputs where available.
        Mission/environment context is still provided by the demo provider
        unless a full Part-1 world bridge is available.
        """

        base = self.demo.update(t, dt)

        drones = []

        for drone_id, drone in self.live_sim.drones.items():
            out = drone.outputs(t)

            perf_snap = drone.perf.snapshot(t, drone.current_yolo_hz)

            sensor_health = {
                "CAMERA": ComponentStatus.OK if drone.health.camera_ok else ComponentStatus.ERROR,
                "IMU": ComponentStatus.OK if drone.health.imu_ok else ComponentStatus.ERROR,
                "BARO": ComponentStatus.OK if drone.health.baro_ok else ComponentStatus.WARNING,
                "MAG": ComponentStatus.OK if drone.health.mag_ok else ComponentStatus.WARNING,
                "OPTICAL_FLOW": ComponentStatus.OK if drone.health.optical_flow_ok else ComponentStatus.WARNING,
                "TF_LUNA": ComponentStatus.OK if drone.health.tf_luna_ok else ComponentStatus.ERROR,
                "VL53L5CX": ComponentStatus.OK if drone.health.vl53_ok else ComponentStatus.WARNING,
                "EKF": ComponentStatus.OK,
                "DETECTOR": ComponentStatus.OK,
                "COMM": ComponentStatus.OK,
            }

            detections = [
                DetectionView(
                    label=det.label,
                    confidence=det.confidence,
                    bbox=det.bbox_xyxy,
                    status="DETECTION",
                )
                for det in out.detections
            ]

            drones.append(
                DroneState(
                    drone_id=drone_id,
                    position=(
                        float(out.est_pos[0]),
                        float(out.est_pos[1]),
                        float(out.est_pos[2]),
                    ),
                    velocity=(
                        float(out.est_vel[0]),
                        float(out.est_vel[1]),
                        float(out.est_vel[2]),
                    ),
                    acceleration=(0.0, 0.0, 0.0),
                    attitude_quat=(
                        float(out.est_quat[0]),
                        float(out.est_quat[1]),
                        float(out.est_quat[2]),
                        float(out.est_quat[3]),
                    ),
                    yaw=float(out.yaw_rad),
                    altitude=float(out.est_pos[2]),
                    battery=100.0,
                    flight_mode="AUTO",
                    armed=True,
                    localization_confidence=float(out.confidence),
                    position_uncertainty=float(out.pos_uncertainty),
                    ai_mode=out.ai_mode,
                    path_status="ACTIVE",
                    path=[],
                    trail=[],
                    detections=detections,
                    warnings=[],
                    sensor_health=sensor_health,
                    perf=PerformanceView(
                        cpu_pct=float(perf_snap.cpu_pct),
                        ram_pct=float(perf_snap.ram_pct),
                        temp_c=float(perf_snap.temp_c),
                        yolo_fps=float(perf_snap.yolo_fps),
                        yolo_latency_ms=float(perf_snap.yolo_latency_ms),
                        gui_fps=self.cfg.render_hz,
                    ),
                    comm_quality=1.0,
                    safety_radius=1.2,
                )
            )

        mines = []

        first_drone = next(iter(self.live_sim.drones.values()), None)

        if first_drone is not None:
            for mine_id, mine in first_drone.global_mines.mines.items():
                mines.append(
                    MineView(
                        mine_id=mine_id,
                        position=(
                            float(mine.pos2[0]),
                            float(mine.pos2[1]),
                            0.0,
                        ),
                        confidence=float(mine.confidence),
                        status="CONFIRMED" if mine.obs_count >= 2 else "PROBABLE",
                        source=",".join(str(s) for s in mine.sources),
                        radius=float(mine.radius_m),
                        clearance=float(mine.clearance_m),
                    )
                )

        environment = base.environment
        environment.mines = mines

        return SwarmState(
            t=t,
            source=DataSource.HYBRID,
            mission=base.mission,
            environment=environment,
            drones=drones,
            links=base.links,
            events=base.events,
            message=self.live_error or "",
        )