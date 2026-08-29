# autonomy/gui/app.py
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from .config import GuiConfig
from .core.controller import Controller
from .core.state import SwarmState
from .visualization.swarm_view import SwarmView3D
from .widgets.panels import (
    DroneCard,
    EventLogWidget,
    LocalizationPanel,
    MiniMapWidget,
    MissionHeader,
    PerceptionPanel,
    PerformancePanel,
    SystemHealthPanel,
)


class MainWindow(QMainWindow):
    def __init__(self, cfg: GuiConfig):
        super().__init__()

        self.cfg = cfg
        self.setWindowTitle(cfg.window_title)
        self.resize(cfg.window_width, cfg.window_height)

        self.controller = Controller(cfg)
        self.controller.state_changed.connect(self.on_state)

        self.latest_state: Optional[SwarmState] = None

        self._build_ui()
        self._build_toolbar()

        QTimer.singleShot(150, self.controller.start)

    def _build_ui(self) -> None:
        central = QWidget(self)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(6, 6, 6, 6)
        root_layout.setSpacing(6)

        self.header = MissionHeader()

        self.drone_cards = {
            1: DroneCard(1),
            2: DroneCard(2),
            3: DroneCard(3),
        }

        self.minimap = MiniMapWidget()
        self.system_health = SystemHealthPanel()
        self.perception_panel = PerceptionPanel()
        self.localization_panel = LocalizationPanel()
        self.performance_panel = PerformancePanel()
        self.event_log = EventLogWidget()

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(self.drone_cards[1])
        left_layout.addWidget(self.drone_cards[2])
        left_layout.addWidget(self.drone_cards[3])
        left_layout.addWidget(self.minimap)
        left_layout.addStretch(1)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(self.system_health)
        right_layout.addWidget(self.perception_panel)
        right_layout.addStretch(1)

        try:
            self.view3d = SwarmView3D(self.cfg)
            center_widget = self.view3d
        except Exception as exc:
            self.view3d = None
            center_widget = QLabel(
                "3D VIEW UNAVAILABLE\n\n"
                f"{exc}\n\n"
                "Install PyVista and pyvistaqt:\n"
                "python -m pip install pyvista pyvistaqt"
            )
            center_widget.setAlignment(Qt.AlignCenter)

        middle_splitter = QSplitter(Qt.Horizontal)
        middle_splitter.addWidget(left)
        middle_splitter.addWidget(center_widget)
        middle_splitter.addWidget(right)

        middle_splitter.setStretchFactor(0, 0)
        middle_splitter.setStretchFactor(1, 1)
        middle_splitter.setStretchFactor(2, 0)

        middle_splitter.setSizes([330, 1100, 330])

        self.bottom_tabs = QTabWidget()
        self.bottom_tabs.addTab(self.event_log, "EVENTS")
        self.bottom_tabs.addTab(self.localization_panel, "LOCALIZATION")
        self.bottom_tabs.addTab(self.performance_panel, "PERFORMANCE")

        root_layout.addWidget(self.header)
        root_layout.addWidget(middle_splitter, stretch=1)
        root_layout.addWidget(self.bottom_tabs, stretch=0)

        self.setCentralWidget(central)

        self.statusBar().showMessage("Initializing...")

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("Main Controls")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        self.start_action = QAction("START", self)
        self.start_action.triggered.connect(self.controller.start)
        toolbar.addAction(self.start_action)

        self.pause_action = QAction("PAUSE", self)
        self.pause_action.setCheckable(True)
        self.pause_action.toggled.connect(self._on_pause_toggled)
        toolbar.addAction(self.pause_action)

        self.reset_action = QAction("RESET", self)
        self.reset_action.triggered.connect(self.controller.reset)
        toolbar.addAction(self.reset_action)

        self.emergency_action = QAction("EMERGENCY STOP", self)
        self.emergency_action.triggered.connect(self._on_emergency)
        toolbar.addAction(self.emergency_action)

        toolbar.addSeparator()

        toolbar.addWidget(QLabel("VIEW:"))

        self.view_combo = QComboBox()
        self.view_combo.addItems(
            [
                "ISOMETRIC",
                "TOP",
                "SIDE",
                "FRONT",
            ]
        )
        self.view_combo.currentIndexChanged.connect(self._on_view_changed)
        toolbar.addWidget(self.view_combo)

        toolbar.addWidget(QLabel("FOLLOW:"))

        self.follow_combo = QComboBox()
        self.follow_combo.addItems(
            [
                "FREE",
                "DRONE-01",
                "DRONE-02",
                "DRONE-03",
            ]
        )
        self.follow_combo.currentIndexChanged.connect(self._on_follow_changed)
        toolbar.addWidget(self.follow_combo)

        toolbar.addSeparator()

        self.paths_action = QAction("PATHS", self)
        self.paths_action.setCheckable(True)
        self.paths_action.setChecked(self.cfg.show_paths)
        self.paths_action.toggled.connect(self._toggle_paths)
        toolbar.addAction(self.paths_action)

        self.trails_action = QAction("TRAILS", self)
        self.trails_action.setCheckable(True)
        self.trails_action.setChecked(self.cfg.show_trails)
        self.trails_action.toggled.connect(self._toggle_trails)
        toolbar.addAction(self.trails_action)

        self.fov_action = QAction("FOV", self)
        self.fov_action.setCheckable(True)
        self.fov_action.setChecked(self.cfg.show_sensor_fov)
        self.fov_action.toggled.connect(self._toggle_fov)
        toolbar.addAction(self.fov_action)

        self.mines_action = QAction("MINES", self)
        self.mines_action.setCheckable(True)
        self.mines_action.setChecked(self.cfg.show_mines)
        self.mines_action.toggled.connect(self._toggle_mines)
        toolbar.addAction(self.mines_action)

        self.human_action = QAction("HUMAN", self)
        self.human_action.setCheckable(True)
        self.human_action.setChecked(self.cfg.show_human)
        self.human_action.toggled.connect(self._toggle_human)
        toolbar.addAction(self.human_action)

        self.links_action = QAction("LINKS", self)
        self.links_action.setCheckable(True)
        self.links_action.setChecked(self.cfg.show_comm_links)
        self.links_action.toggled.connect(self._toggle_links)
        toolbar.addAction(self.links_action)

        self.labels_action = QAction("LABELS", self)
        self.labels_action.setCheckable(True)
        self.labels_action.setChecked(self.cfg.show_labels)
        self.labels_action.toggled.connect(self._toggle_labels)
        toolbar.addAction(self.labels_action)

        self.occupancy_action = QAction("OCCUPANCY", self)
        self.occupancy_action.setCheckable(True)
        self.occupancy_action.setChecked(self.cfg.show_occupancy)
        self.occupancy_action.toggled.connect(self._toggle_occupancy)
        toolbar.addAction(self.occupancy_action)

    def _on_pause_toggled(self, checked: bool) -> None:
        if checked:
            self.controller.pause()
        else:
            self.controller.resume()

    def _on_emergency(self) -> None:
        self.controller.emergency_stop()
        self.statusBar().showMessage("EMERGENCY STOP COMMANDED", 5000)

    def _on_view_changed(self, index: int) -> None:
        if self.view3d is None:
            return

        if index == 0:
            self.view3d.set_isometric_view()
        elif index == 1:
            self.view3d.set_top_view()
        elif index == 2:
            self.view3d.set_side_view()
        elif index == 3:
            self.view3d.set_front_view()

    def _on_follow_changed(self, index: int) -> None:
        if self.view3d is None:
            return

        if index == 0:
            self.view3d.follow_drone(0)
            self.view3d.set_isometric_view()
        else:
            self.view3d.follow_drone(index)

    def _toggle_paths(self, checked: bool) -> None:
        if self.view3d is not None:
            self.view3d.toggle_paths(checked)

    def _toggle_trails(self, checked: bool) -> None:
        if self.view3d is not None:
            self.view3d.toggle_trails(checked)

    def _toggle_fov(self, checked: bool) -> None:
        if self.view3d is not None:
            self.view3d.toggle_fov(checked)

    def _toggle_mines(self, checked: bool) -> None:
        if self.view3d is not None:
            self.view3d.toggle_mines(checked)

    def _toggle_human(self, checked: bool) -> None:
        if self.view3d is not None:
            self.view3d.toggle_human(checked)

    def _toggle_links(self, checked: bool) -> None:
        if self.view3d is not None:
            self.view3d.toggle_comm_links(checked)

    def _toggle_labels(self, checked: bool) -> None:
        if self.view3d is not None:
            self.view3d.toggle_labels(checked)

    def _toggle_occupancy(self, checked: bool) -> None:
        if self.view3d is not None:
            self.view3d.toggle_occupancy(checked)

    def on_state(self, state: SwarmState) -> None:
        self.latest_state = state

        self.header.update_state(state)

        for drone in state.drones:
            card = self.drone_cards.get(drone.drone_id)
            if card is not None:
                card.update_state(drone)

        self.minimap.update_state(state)
        self.system_health.update_state(state)
        self.perception_panel.update_state(state)
        self.localization_panel.update_state(state)
        self.performance_panel.update_state(state)
        self.event_log.update_state(state)

        if self.view3d is not None:
            try:
                self.view3d.update_state(state)
            except Exception as exc:
                self.statusBar().showMessage(f"3D render warning: {exc}", 3000)

        if state.message:
            self.statusBar().showMessage(state.message, 5000)
        else:
            self.statusBar().showMessage(
                f"T+{state.mission.elapsed:06.1f}s | "
                f"{state.mission.phase.value} | "
                f"{state.source.value}"
            )

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_F11:
            if self.isFullScreen():
                self.showNormal()
            else:
                self.showFullScreen()

        elif event.key() == Qt.Key_Space:
            self.pause_action.toggle()

        else:
            super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        self.controller.stop()
        event.accept()