# autonomy/gui/widgets/panels.py
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QBrush
from PySide6.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

from ..core.state import (
    ComponentStatus,
    DroneState,
    MissionPhase,
    SwarmState,
)


def fmt_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


def status_color(status: ComponentStatus) -> str:
    return {
        ComponentStatus.OK: "#4dff7c",
        ComponentStatus.WARNING: "#ffd24a",
        ComponentStatus.ERROR: "#ff4d4d",
        ComponentStatus.OFFLINE: "#8a97a8",
    }.get(status, "#d7e0ea")


class MissionHeader(QWidget):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.setLayout(QHBoxLayout())

        self.title = QLabel("AUTONOMOUS SWARM COMMAND CENTER")
        self.title.setStyleSheet("font-size: 18px; font-weight: 700; color: #7fd4ff;")

        self.phase = QLabel("PHASE: --")
        self.timer = QLabel("T+ --:-- | REM --:--")
        self.source = QLabel("SOURCE: --")
        self.gps = QLabel("GPS: DENIED")
        self.localization = QLabel("LOCALIZATION: ACTIVE")
        self.system = QLabel("SYSTEM: --")

        self.phase.setStyleSheet("font-weight: 600;")
        self.timer.setStyleSheet("font-weight: 700; color: #ffffff;")
        self.source.setStyleSheet("color: #ffd24a;")
        self.gps.setStyleSheet("color: #ff6b6b; font-weight: 700;")
        self.localization.setStyleSheet("color: #4dff7c; font-weight: 600;")

        self.layout().addWidget(self.title)
        self.layout().addStretch(1)
        self.layout().addWidget(self.phase)
        self.layout().addWidget(self.timer)
        self.layout().addWidget(self.source)
        self.layout().addWidget(self.gps)
        self.layout().addWidget(self.localization)
        self.layout().addWidget(self.system)

    def update_state(self, state: SwarmState) -> None:
        self.phase.setText(f"PHASE: {state.mission.phase.value}")
        self.timer.setText(
            f"T+ {fmt_time(state.mission.elapsed)} | REM {fmt_time(state.mission.remaining)}"
        )
        self.source.setText(f"SOURCE: {state.source.value}")

        has_error = False
        has_warning = False

        for drone in state.drones:
            for status in drone.sensor_health.values():
                if status == ComponentStatus.ERROR:
                    has_error = True
                elif status == ComponentStatus.WARNING:
                    has_warning = True

        if state.mission.phase == MissionPhase.ABORTED:
            self.system.setText("SYSTEM: ABORTED")
            self.system.setStyleSheet("color: #ff4d4d; font-weight: 700;")
        elif has_error:
            self.system.setText("SYSTEM: ERROR")
            self.system.setStyleSheet("color: #ff4d4d; font-weight: 700;")
        elif has_warning:
            self.system.setText("SYSTEM: WARNING")
            self.system.setStyleSheet("color: #ffd24a; font-weight: 700;")
        else:
            self.system.setText("SYSTEM: NOMINAL")
            self.system.setStyleSheet("color: #4dff7c; font-weight: 700;")


class DroneCard(QGroupBox):
    def __init__(self, drone_id: int, parent: Optional[QWidget] = None):
        super().__init__(f"DRONE-{drone_id:02d}", parent)

        self.drone_id = drone_id

        self.grid = QGridLayout(self)

        self.labels: Dict[str, QLabel] = {}

        rows = [
            "armed",
            "mode",
            "altitude",
            "speed",
            "battery",
            "localization",
            "link",
            "cpu",
            "inference",
            "warnings",
        ]

        for i, name in enumerate(rows):
            key = QLabel(name.upper())
            value = QLabel("--")
            value.setWordWrap(True)

            key.setStyleSheet("color: #7f8ea3;")
            value.setStyleSheet("font-weight: 600;")

            self.grid.addWidget(key, i, 0)
            self.grid.addWidget(value, i, 1)

            self.labels[name] = value

    def update_state(self, drone: DroneState) -> None:
        speed = math.hypot(drone.velocity[0], drone.velocity[1])

        self.labels["armed"].setText("ARMED" if drone.armed else "DISARMED")
        self.labels["armed"].setStyleSheet(
            "color: #4dff7c; font-weight: 700;" if drone.armed else "color: #ff4d4d; font-weight: 700;"
        )

        self.labels["mode"].setText(drone.flight_mode)
        self.labels["altitude"].setText(f"{drone.altitude:0.2f} m")
        self.labels["speed"].setText(f"{speed:0.2f} m/s")
        self.labels["battery"].setText(f"{drone.battery:0.0f} %")

        self.labels["localization"].setText(
            f"{drone.localization_confidence:0.2f} | σ {drone.position_uncertainty:0.2f} m"
        )

        self.labels["link"].setText(f"{drone.comm_quality:0.2f}")

        self.labels["cpu"].setText(f"{drone.perf.cpu_pct:0.0f} %")
        self.labels["inference"].setText(
            f"{drone.perf.yolo_fps:0.1f} FPS | {drone.perf.yolo_latency_ms:0.0f} ms"
        )

        if drone.warnings:
            self.labels["warnings"].setText("; ".join(drone.warnings[:2]))
            self.labels["warnings"].setStyleSheet("color: #ffd24a;")
        else:
            self.labels["warnings"].setText("NONE")
            self.labels["warnings"].setStyleSheet("color: #4dff7c;")


class SystemHealthPanel(QGroupBox):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__("SYSTEM HEALTH", parent)

        self.grid = QGridLayout(self)
        self.labels: Dict[str, QLabel] = {}

        components = [
            "CAMERA",
            "IMU",
            "BARO",
            "MAG",
            "OPTICAL_FLOW",
            "TF_LUNA",
            "VL53L5CX",
            "EKF",
            "DETECTOR",
            "COMM",
        ]

        for i, comp in enumerate(components):
            name = QLabel(comp)
            value = QLabel("OFFLINE")

            name.setStyleSheet("color: #7f8ea3;")
            value.setStyleSheet("font-weight: 700;")

            self.grid.addWidget(name, i, 0)
            self.grid.addWidget(value, i, 1)

            self.labels[comp] = value

    def update_state(self, state: SwarmState) -> None:
        aggregate: Dict[str, ComponentStatus] = {}

        for drone in state.drones:
            for comp, status in drone.sensor_health.items():
                current = aggregate.get(comp, ComponentStatus.OK)

                if status == ComponentStatus.ERROR:
                    aggregate[comp] = ComponentStatus.ERROR
                elif status == ComponentStatus.WARNING and current != ComponentStatus.ERROR:
                    aggregate[comp] = ComponentStatus.WARNING
                elif comp not in aggregate:
                    aggregate[comp] = status

        for comp, label in self.labels.items():
            status = aggregate.get(comp, ComponentStatus.OFFLINE)
            label.setText(status.value)
            label.setStyleSheet(f"color: {status_color(status)};")


class PerceptionPanel(QPlainTextEdit):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.setReadOnly(True)
        self._last_text_hash = None

    def update_state(self, state: SwarmState) -> None:
        lines: List[str] = []

        for drone in state.drones:
            lines.append(f"DRONE-{drone.drone_id:02d} | AI MODE: {drone.ai_mode}")

            if not drone.detections:
                lines.append("  NO DETECTIONS")

            for det in drone.detections[:8]:
                lines.append(
                    f"  {det.label:<18} "
                    f"conf={det.confidence:0.2f} "
                    f"status={det.status}"
                )

            lines.append("")

        human = state.environment.human

        if human is not None:
            lines.append("HUMAN TRACK")
            lines.append(
                f"  pos=({human.position[0]:0.1f}, {human.position[1]:0.1f}) "
                f"conf={human.confidence:0.2f} "
                f"state={human.state}"
            )

            if human.gesture != "NONE":
                lines.append(
                    f"  gesture={human.gesture} conf={human.gesture_confidence:0.2f}"
                )

        text = "\n".join(lines)
        h = hash(text)

        if h != self._last_text_hash:
            self._last_text_hash = h
            self.setPlainText(text)


class LocalizationPanel(QPlainTextEdit):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.setReadOnly(True)
        self._last_text_hash = None

    def update_state(self, state: SwarmState) -> None:
        lines = [
            "GPS: DENIED",
            "LOCALIZATION: EKF + OPTICAL FLOW + RANGE + MAG + VISION/MAP",
            "",
        ]

        for drone in state.drones:
            lines.append(f"DRONE-{drone.drone_id:02d}")
            lines.append(
                f"  pos=({drone.position[0]:6.2f}, {drone.position[1]:6.2f}, {drone.position[2]:4.2f})"
            )
            lines.append(
                f"  vel=({drone.velocity[0]:5.2f}, {drone.velocity[1]:5.2f}, {drone.velocity[2]:5.2f})"
            )
            lines.append(
                f"  yaw={math.degrees(drone.yaw):6.1f} deg | "
                f"conf={drone.localization_confidence:0.2f} | "
                f"unc={drone.position_uncertainty:0.2f} m"
            )
            lines.append("")

        text = "\n".join(lines)
        h = hash(text)

        if h != self._last_text_hash:
            self._last_text_hash = h
            self.setPlainText(text)


class PerformancePanel(QWidget):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.layout = QVBoxLayout(self)

        self.cpu_bar = QProgressBar()
        self.ram_bar = QProgressBar()

        self.fps_label = QLabel("GUI FPS: -- | YOLO FPS: --")
        self.latency_label = QLabel("YOLO LATENCY: -- ms")
        self.temp_label = QLabel("TEMP: -- C")

        self.layout.addWidget(QLabel("CPU"))
        self.layout.addWidget(self.cpu_bar)

        self.layout.addWidget(QLabel("RAM"))
        self.layout.addWidget(self.ram_bar)

        self.layout.addWidget(self.fps_label)
        self.layout.addWidget(self.latency_label)
        self.layout.addWidget(self.temp_label)

    def update_state(self, state: SwarmState) -> None:
        if not state.drones:
            return

        cpu = sum(d.perf.cpu_pct for d in state.drones) / len(state.drones)
        ram = sum(d.perf.ram_pct for d in state.drones) / len(state.drones)
        temp = sum(d.perf.temp_c for d in state.drones) / len(state.drones)
        yolo_fps = sum(d.perf.yolo_fps for d in state.drones) / len(state.drones)
        latency = sum(d.perf.yolo_latency_ms for d in state.drones) / len(state.drones)

        self.cpu_bar.setValue(int(cpu))
        self.ram_bar.setValue(int(ram))

        self.fps_label.setText(
            f"GUI FPS: {state.drones[0].perf.gui_fps:0.1f} | YOLO FPS: {yolo_fps:0.1f}"
        )

        self.latency_label.setText(f"YOLO LATENCY: {latency:0.0f} ms")
        self.temp_label.setText(f"TEMP: {temp:0.1f} C")


class EventLogWidget(QPlainTextEdit):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.setReadOnly(True)
        self._seen_events = 0

    def update_state(self, state: SwarmState) -> None:
        if len(state.events) < self._seen_events:
            self.clear()
            self._seen_events = 0

        if len(state.events) == self._seen_events:
            return

        new_events = state.events[self._seen_events:]

        for event in new_events:
            self.appendPlainText(
                f"[{fmt_time(event.t)}] "
                f"{event.level:<8} "
                f"{event.subsystem:<14} "
                f"{event.message}"
            )

        self._seen_events = len(state.events)


class MiniMapWidget(QWidget):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.state: Optional[SwarmState] = None
        self.image: Optional[QImage] = None

        self.setMinimumHeight(220)

    def update_state(self, state: SwarmState) -> None:
        self.state = state

        arr = state.environment.occupancy_rgba

        if arr is not None:
            arr = np.ascontiguousarray(arr)
            h, w, _ = arr.shape

            self.image = QImage(
                arr.data,
                w,
                h,
                w * 4,
                QImage.Format_RGBA8888,
            ).copy()
        else:
            self.image = None

        self.update()

    def _world_to_widget(self, x: float, y: float, rect):
        length = self.state.environment.length_m if self.state else 60.0
        width = self.state.environment.width_m if self.state else 15.0

        px = rect.left() + (x / length) * rect.width()
        py = rect.bottom() - ((y + width / 2.0) / width) * rect.height()

        return px, py

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#05070c"))

        rect = self.rect().adjusted(6, 6, -6, -6)

        if self.state is None:
            painter.end()
            return

        if self.image is not None:
            painter.drawImage(rect, self.image)
        else:
            painter.setPen(QPen(QColor("#1d2836")))
            painter.drawRect(rect)

        env = self.state.environment

        # Mines
        for mine in env.mines:
            x, y, _ = mine.position
            px, py = self._world_to_widget(x, y, rect)

            r = max(3, int((mine.radius + mine.clearance) / env.length_m * rect.width() * 6))

            painter.setPen(QPen(QColor("#ff4d4d")))
            painter.setBrush(QBrush(QColor(255, 60, 60, 45)))
            painter.drawEllipse(int(px - r / 2), int(py - r / 2), r, r)

            painter.setBrush(QBrush(QColor("#ff4d4d")))
            painter.drawEllipse(int(px - 2), int(py - 2), 4, 4)

        # Safe path
        if env.safe_path:
            painter.setPen(QPen(QColor("#00d0ff"), 2))

            for i in range(len(env.safe_path) - 1):
                x1, y1, _ = env.safe_path[i]
                x2, y2, _ = env.safe_path[i + 1]

                p1 = self._world_to_widget(x1, y1, rect)
                p2 = self._world_to_widget(x2, y2, rect)

                painter.drawLine(int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]))

        # Human
        if env.human is not None:
            x, y, _ = env.human.position
            px, py = self._world_to_widget(x, y, rect)

            painter.setPen(QPen(QColor("#3f7dff")))
            painter.setBrush(QBrush(QColor("#3f7dff")))
            painter.drawEllipse(int(px - 4), int(py - 4), 8, 8)

        # Drones
        colors = {
            1: QColor("#00e5ff"),
            2: QColor("#69ff69"),
            3: QColor("#ff69ff"),
        }

        for drone in self.state.drones:
            color = colors.get(drone.drone_id, QColor("white"))

            # Trail
            if len(drone.trail) > 1:
                painter.setPen(QPen(color, 1))

                for i in range(len(drone.trail) - 1):
                    x1, y1, _ = drone.trail[i]
                    x2, y2, _ = drone.trail[i + 1]

                    p1 = self._world_to_widget(x1, y1, rect)
                    p2 = self._world_to_widget(x2, y2, rect)

                    painter.drawLine(int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]))

            x, y, _ = drone.position
            px, py = self._world_to_widget(x, y, rect)

            painter.setPen(QPen(color))
            painter.setBrush(QBrush(color))
            painter.drawEllipse(int(px - 4), int(py - 4), 8, 8)

        painter.end()