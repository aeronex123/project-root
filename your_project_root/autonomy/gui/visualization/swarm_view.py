# autonomy/gui/visualization/swarm_view.py
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from PySide6.QtWidgets import QVBoxLayout, QWidget

try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
    PYVISTA_AVAILABLE = True
except Exception:
    pv = None
    QtInteractor = None
    PYVISTA_AVAILABLE = False

from ..config import GuiConfig
from ..core.state import MineView, SwarmState


Vec3 = Tuple[float, float, float]


class SwarmView3D(QWidget):
    """
    Main 3D tactical visualization.

    Uses PyVista/VTK for GPU-accelerated rendering.
    """

    DRONE_COLORS = {
        1: "#00e5ff",
        2: "#69ff69",
        3: "#ff69ff",
    }

    def __init__(self, cfg: GuiConfig, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.cfg = cfg

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        if not PYVISTA_AVAILABLE:
            raise RuntimeError(
                "PyVista or pyvistaqt is unavailable. "
                "Install with: python -m pip install pyvista pyvistaqt"
            )

        self.plotter = QtInteractor(self)
        self.layout.addWidget(self.plotter)

        self.state: Optional[SwarmState] = None

        self.show_paths = cfg.show_paths
        self.show_trails = cfg.show_trails
        self.show_sensor_fov = cfg.show_sensor_fov
        self.show_mines = cfg.show_mines
        self.show_human = cfg.show_human
        self.show_comm_links = cfg.show_comm_links
        self.show_labels = cfg.show_labels
        self.show_occupancy = cfg.show_occupancy
        self.show_safety_zones = cfg.show_safety_zones

        self.follow_drone_id = cfg.follow_drone_id

        self.drone_actors: Dict[int, object] = {}
        self.heading_datasets: Dict[int, object] = {}
        self.heading_actors: Dict[int, object] = {}
        self.safety_actors: Dict[int, object] = {}
        self.fov_actors: Dict[int, object] = {}

        self.path_actors: Dict[int, object] = {}
        self.path_hashes: Dict[int, int] = {}

        self.trail_actors: Dict[int, object] = {}
        self.last_trail_update = -1.0

        self.link_datasets: Dict[Tuple[int, int], object] = {}
        self.link_actors: Dict[Tuple[int, int], object] = {}

        self.mine_marker_actors: Dict[str, object] = {}
        self.mine_exclusion_actors: Dict[str, object] = {}
        self.mine_hashes: Dict[str, str] = {}

        self.label_actor = None
        self.last_label_update = -1.0

        self.occ_actor = None
        self.last_occ_sec = -1

        self.human_actor = None
        self.human_heading_dataset = None
        self.human_heading_actor = None

        self.rotor_dataset = None
        self.rotor_actor = None

        self._init_scene()

    def _init_scene(self) -> None:
        self.plotter.background_color = "#05070c"

        try:
            self.plotter.add_axes()
        except Exception:
            pass

        length = self.cfg.field_length_m
        width = self.cfg.field_width_m

        ground = pv.Plane(
            center=(length / 2.0, 0.0, 0.0),
            direction=(0.0, 0.0, 1.0),
            i_size=length,
            j_size=width,
            i_resolution=60,
            j_resolution=15,
        )

        self.plotter.add_mesh(
            ground,
            color="#0b1118",
            show_edges=True,
            opacity=1.0,
        )

        start_zone = pv.Plane(
            center=(2.0, 0.0, 0.01),
            direction=(0.0, 0.0, 1.0),
            i_size=4.0,
            j_size=width,
        )

        self.plotter.add_mesh(
            start_zone,
            color="#00ff88",
            opacity=0.06,
        )

        target_zone = pv.Plane(
            center=(length - 2.0, 0.0, 0.01),
            direction=(0.0, 0.0, 1.0),
            i_size=4.0,
            j_size=width,
        )

        self.plotter.add_mesh(
            target_zone,
            color="#00aaff",
            opacity=0.06,
        )

        for drone_id, color in self.DRONE_COLORS.items():
            sphere = pv.Sphere(radius=0.35, center=(0.0, 0.0, 0.0))
            actor = self.plotter.add_mesh(sphere, color=color)
            self.drone_actors[drone_id] = actor

            heading_line = pv.Line((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
            self.heading_datasets[drone_id] = heading_line
            self.heading_actors[drone_id] = self.plotter.add_mesh(
                heading_line,
                color=color,
                line_width=3,
            )

            safety_circle = self._make_circle(1.2, (0.0, 0.0, 0.02))
            self.safety_actors[drone_id] = self.plotter.add_mesh(
                safety_circle,
                color=color,
                opacity=0.08,
            )

            fov_circle = self._make_circle(1.0, (0.0, 0.0, 0.03))
            self.fov_actors[drone_id] = self.plotter.add_mesh(
                fov_circle,
                color=color,
                opacity=0.05,
            )

        self.rotor_dataset = pv.PolyData(np.zeros((12, 3), dtype=np.float32))
        self.rotor_actor = self.plotter.add_mesh(
            self.rotor_dataset,
            color="#e8f4ff",
            point_size=5,
            render_points_as_spheres=True,
        )

        for pair in [(1, 2), (2, 3), (1, 3)]:
            line = pv.Line((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
            self.link_datasets[pair] = line
            self.link_actors[pair] = self.plotter.add_mesh(
                line,
                color="#3affc3",
                opacity=0.35,
                line_width=2,
            )

        human_body = pv.Cylinder(
            radius=0.30,
            height=1.60,
            center=(0.0, 0.0, 0.80),
            direction=(0.0, 0.0, 1.0),
        )

        self.human_actor = self.plotter.add_mesh(
            human_body,
            color="#3f7dff",
            opacity=0.95,
        )

        self.human_heading_dataset = pv.Line((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
        self.human_heading_actor = self.plotter.add_mesh(
            self.human_heading_dataset,
            color="#3f7dff",
            line_width=3,
        )

        self.set_isometric_view()

    def _set_visibility(self, actor, visible: bool) -> None:
        if actor is None:
            return

        try:
            actor.SetVisibility(bool(visible))
        except Exception:
            pass

    def _remove_actor(self, actor) -> None:
        if actor is None:
            return

        try:
            self.plotter.remove_actor(actor)
        except Exception:
            pass

    def update_state(self, state: SwarmState) -> None:
        self.state = state

        self._update_drones(state)
        self._update_human(state)
        self._update_mines(state.environment.mines)
        self._update_links(state)
        self._update_paths(state)
        self._update_labels(state)
        self._update_occupancy(state)
        self._update_follow(state)

        try:
            self.plotter.render()
        except Exception:
            pass

    def _rotor_positions(self, pos: Vec3, yaw: float) -> List[Vec3]:
        offsets = [
            (0.38, 0.38),
            (-0.38, 0.38),
            (-0.38, -0.38),
            (0.38, -0.38),
        ]

        ca = math.cos(yaw)
        sa = math.sin(yaw)

        out: List[Vec3] = []

        for dx, dy in offsets:
            rx = dx * ca - dy * sa
            ry = dx * sa + dy * ca
            out.append((pos[0] + rx, pos[1] + ry, pos[2] + 0.06))

        return out

    def _update_drones(self, state: SwarmState) -> None:
        rotor_points: List[Vec3] = []

        for drone in state.drones:
            pos = np.array(drone.position, dtype=float)

            actor = self.drone_actors.get(drone.drone_id)
            if actor is not None:
                try:
                    actor.SetPosition(float(pos[0]), float(pos[1]), float(pos[2]))
                except Exception:
                    pass

            yaw = float(drone.yaw)
            direction = np.array([math.cos(yaw), math.sin(yaw), 0.0], dtype=float)

            heading_dataset = self.heading_datasets.get(drone.drone_id)
            if heading_dataset is not None:
                heading_dataset.points = np.array(
                    [
                        pos,
                        pos + direction * 1.4,
                    ],
                    dtype=float,
                )

                try:
                    heading_dataset.modified()
                except Exception:
                    pass

            safety_actor = self.safety_actors.get(drone.drone_id)
            if safety_actor is not None:
                try:
                    safety_actor.SetPosition(float(pos[0]), float(pos[1]), 0.02)
                except Exception:
                    pass

                self._set_visibility(safety_actor, self.show_safety_zones)

            fov_actor = self.fov_actors.get(drone.drone_id)
            if fov_actor is not None:
                fov_radius = max(0.35, float(pos[2]) * math.tan(math.radians(55.0)))

                try:
                    fov_actor.SetPosition(float(pos[0]), float(pos[1]), 0.03)
                    fov_actor.SetScale(fov_radius, fov_radius, 1.0)
                except Exception:
                    pass

                self._set_visibility(fov_actor, self.show_sensor_fov)

            rotor_points.extend(self._rotor_positions(drone.position, yaw))

        if self.rotor_dataset is not None and len(rotor_points) == 12:
            self.rotor_dataset.points = np.array(rotor_points, dtype=float)

            try:
                self.rotor_dataset.modified()
            except Exception:
                pass

    def _update_human(self, state: SwarmState) -> None:
        human = state.environment.human

        if human is None or not self.show_human:
            self._set_visibility(self.human_actor, False)
            self._set_visibility(self.human_heading_actor, False)
            return

        self._set_visibility(self.human_actor, True)
        self._set_visibility(self.human_heading_actor, True)

        x, y, z = human.position

        try:
            self.human_actor.SetPosition(float(x), float(y), float(z) + 0.80)
        except Exception:
            pass

        heading = np.array(
            [math.cos(human.heading), math.sin(human.heading), 0.0],
            dtype=float,
        )

        start = np.array([x, y, z + 0.1], dtype=float)
        end = start + heading * 1.1

        if self.human_heading_dataset is not None:
            self.human_heading_dataset.points = np.array([start, end], dtype=float)

            try:
                self.human_heading_dataset.modified()
            except Exception:
                pass

    def _update_mines(self, mines: List[MineView]) -> None:
        if not self.show_mines:
            for mine_id in list(self.mine_marker_actors.keys()):
                self._remove_actor(self.mine_marker_actors[mine_id])
                self._remove_actor(self.mine_exclusion_actors[mine_id])

                del self.mine_marker_actors[mine_id]
                del self.mine_exclusion_actors[mine_id]
                del self.mine_hashes[mine_id]

            return

        current_ids = {mine.mine_id for mine in mines}

        for mine_id in list(self.mine_marker_actors.keys()):
            if mine_id not in current_ids:
                self._remove_actor(self.mine_marker_actors[mine_id])
                self._remove_actor(self.mine_exclusion_actors[mine_id])

                del self.mine_marker_actors[mine_id]
                del self.mine_exclusion_actors[mine_id]
                del self.mine_hashes[mine_id]

        for mine in mines:
            h = f"{mine.status}:{round(mine.confidence, 2)}"

            if mine.mine_id in self.mine_hashes and self.mine_hashes[mine.mine_id] == h:
                continue

            if mine.mine_id in self.mine_marker_actors:
                self._remove_actor(self.mine_marker_actors[mine.mine_id])
                self._remove_actor(self.mine_exclusion_actors[mine.mine_id])

            color = {
                "SUSPECTED": "#ffd24a",
                "PROBABLE": "#ff9c2e",
                "CONFIRMED": "#ff3b3b",
                "REJECTED": "#8a97a8",
            }.get(mine.status, "#ff3b3b")

            marker = pv.Sphere(
                radius=0.18 + 0.12 * mine.confidence,
                center=(mine.position[0], mine.position[1], 0.14),
            )
            

            exclusion = self._make_circle(
                mine.radius + mine.clearance,
                (mine.position[0], mine.position[1], 0.02),
            )
            

            self.mine_marker_actors[mine.mine_id] = self.plotter.add_mesh(
                marker,
                color=color,
                opacity=0.95,
            )

            self.mine_exclusion_actors[mine.mine_id] = self.plotter.add_mesh(
                exclusion,
                color=color,
                opacity=0.12,
            )

            self.mine_hashes[mine.mine_id] = h

    def _update_links(self, state: SwarmState) -> None:
        pos_by_id = {drone.drone_id: drone.position for drone in state.drones}

        for link in state.links:
            pair = (link.a, link.b)

            dataset = self.link_datasets.get(pair)
            actor = self.link_actors.get(pair)

            if dataset is None or actor is None:
                continue

            visible = self.show_comm_links and link.connected

            if not visible:
                self._set_visibility(actor, False)
                continue

            if link.a not in pos_by_id or link.b not in pos_by_id:
                self._set_visibility(actor, False)
                continue

            p1 = np.array(pos_by_id[link.a], dtype=float)
            p2 = np.array(pos_by_id[link.b], dtype=float)

            dataset.points = np.array([p1, p2], dtype=float)

            try:
                dataset.modified()
            except Exception:
                pass

            self._set_visibility(actor, True)

    def _update_paths(self, state: SwarmState) -> None:
        if not self.show_paths:
            for drone_id, actor in list(self.path_actors.items()):
                self._remove_actor(actor)
                del self.path_actors[drone_id]
                del self.path_hashes[drone_id]

            for drone_id, actor in list(self.trail_actors.items()):
                self._remove_actor(actor)
                del self.trail_actors[drone_id]

            return

        for drone in state.drones:
            path = drone.path

            if len(path) >= 2:
                h = hash(tuple(path))

                if self.path_hashes.get(drone.drone_id) != h:
                    if drone.drone_id in self.path_actors:
                        self._remove_actor(self.path_actors[drone.drone_id])

                    spline = pv.Spline(np.array(path, dtype=float), n_points=120)

                    self.path_actors[drone.drone_id] = self.plotter.add_mesh(
                        spline,
                        color=self.DRONE_COLORS[drone.drone_id],
                        opacity=0.45,
                        line_width=2,
                    )

                    self.path_hashes[drone.drone_id] = h

        if state.t - self.last_trail_update > 0.5 and self.show_trails:
            self.last_trail_update = state.t

            for drone in state.drones:
                if drone.drone_id in self.trail_actors:
                    self._remove_actor(self.trail_actors[drone.drone_id])

                if len(drone.trail) >= 2:
                    trail = pv.Spline(
                        np.array(drone.trail, dtype=float),
                        n_points=max(2, len(drone.trail)),
                    )

                    self.trail_actors[drone.drone_id] = self.plotter.add_mesh(
                        trail,
                        color=self.DRONE_COLORS[drone.drone_id],
                        opacity=0.18,
                        line_width=2,
                    )

    def _update_labels(self, state: SwarmState) -> None:
        if not self.show_labels:
            if self.label_actor is not None:
                self._remove_actor(self.label_actor)
                self.label_actor = None

            return

        if state.t - self.last_label_update < 0.5:
            return

        self.last_label_update = state.t

        if self.label_actor is not None:
            self._remove_actor(self.label_actor)

        points = []
        labels = []

        for drone in state.drones:
            speed = math.hypot(drone.velocity[0], drone.velocity[1])

            points.append(drone.position)
            labels.append(
                f"DRONE-{drone.drone_id:02d}\n"
                f"ALT {drone.altitude:0.1f} m\n"
                f"SPD {speed:0.1f} m/s\n"
                f"LOC {drone.localization_confidence:0.2f}"
            )

        if not points:
            return

        try:
            self.label_actor = self.plotter.add_point_labels(
                np.array(points, dtype=float),
                labels,
                font_size=11,
                text_color="white",
                point_size=0,
                render_points_as_labels=False,
                show_points=False,
            )
        except Exception:
            self.label_actor = None

    def _update_occupancy(self, state: SwarmState) -> None:
        if not self.show_occupancy or state.environment.occupancy_rgba is None:
            if self.occ_actor is not None:
                self._remove_actor(self.occ_actor)
                self.occ_actor = None
                self.last_occ_sec = -1

            return

        sec = int(state.t)

        if sec == self.last_occ_sec:
            return

        self.last_occ_sec = sec

        try:
            rgba = np.ascontiguousarray(state.environment.occupancy_rgba)

            texture = pv.numpy_to_texture(rgba)

            plane = pv.Plane(
                center=(self.cfg.field_length_m / 2.0, 0.0, 0.015),
                direction=(0.0, 0.0, 1.0),
                i_size=self.cfg.field_length_m,
                j_size=self.cfg.field_width_m,
                i_resolution=rgba.shape[1] - 1,
                j_resolution=rgba.shape[0] - 1,
            )

            if self.occ_actor is not None:
                self._remove_actor(self.occ_actor)

            self.occ_actor = self.plotter.add_mesh(
                plane,
                texture=texture,
                opacity=0.42,
            )

        except Exception:
            self.occ_actor = None

    def _update_follow(self, state: SwarmState) -> None:
        if self.follow_drone_id <= 0:
            return

        drone = next(
            (d for d in state.drones if d.drone_id == self.follow_drone_id),
            None,
        )

        if drone is None:
            return

        x, y, z = drone.position

        try:
            self.plotter.camera_position = [
                (x - 8.0, y - 8.0, z + 6.0),
                (x, y, z),
                (0.0, 0.0, 1.0),
            ]
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Camera presets
    # ------------------------------------------------------------------

    def set_top_view(self) -> None:
        self.follow_drone_id = 0

        try:
            self.plotter.camera_position = [
                (self.cfg.field_length_m / 2.0, 0.0, 70.0),
                (self.cfg.field_length_m / 2.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
            ]
        except Exception:
            pass

    def set_side_view(self) -> None:
        self.follow_drone_id = 0

        try:
            self.plotter.camera_position = [
                (self.cfg.field_length_m / 2.0, -55.0, 12.0),
                (self.cfg.field_length_m / 2.0, 0.0, 1.0),
                (0.0, 0.0, 1.0),
            ]
        except Exception:
            pass

    def set_front_view(self) -> None:
        self.follow_drone_id = 0

        try:
            self.plotter.camera_position = [
                (-20.0, 0.0, 8.0),
                (self.cfg.field_length_m / 2.0, 0.0, 1.0),
                (0.0, 0.0, 1.0),
            ]
        except Exception:
            pass

    def set_isometric_view(self) -> None:
        self.follow_drone_id = 0

        try:
            self.plotter.camera_position = [
                (85.0, -55.0, 45.0),
                (self.cfg.field_length_m / 2.0, 0.0, 0.0),
                (0.0, 0.0, 1.0),
            ]
        except Exception:
            pass

    def follow_drone(self, drone_id: int) -> None:
        self.follow_drone_id = int(drone_id)

    # ------------------------------------------------------------------
    # Toggles
    # ------------------------------------------------------------------

    def toggle_paths(self, enabled: bool) -> None:
        self.show_paths = bool(enabled)

        if not enabled and self.state is not None:
            self._update_paths(self.state)

    def toggle_trails(self, enabled: bool) -> None:
        self.show_trails = bool(enabled)

        if not enabled and self.state is not None:
            self._update_paths(self.state)

    def toggle_fov(self, enabled: bool) -> None:
        self.show_sensor_fov = bool(enabled)

    def toggle_mines(self, enabled: bool) -> None:
        self.show_mines = bool(enabled)

        if self.state is not None:
            self._update_mines(self.state.environment.mines)

    def toggle_human(self, enabled: bool) -> None:
        self.show_human = bool(enabled)

        if self.state is not None:
            self._update_human(self.state)

    def toggle_comm_links(self, enabled: bool) -> None:
        self.show_comm_links = bool(enabled)

    def toggle_labels(self, enabled: bool) -> None:
        self.show_labels = bool(enabled)

        if not enabled:
            self.last_label_update = -1.0

    def toggle_occupancy(self, enabled: bool) -> None:
        self.show_occupancy = bool(enabled)
        self.last_occ_sec = -1

    def toggle_safety_zones(self, enabled: bool) -> None:
        self.show_safety_zones = bool(enabled)
        
    def _make_circle(self, radius: float, center) -> object:
      
        circle = pv.Circle(radius=float(radius))
        circle.translate([float(center[0]), float(center[1]), float(center[2])])
        return circle