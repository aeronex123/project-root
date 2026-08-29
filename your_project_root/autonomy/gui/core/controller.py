# autonomy/gui/core/controller.py
from __future__ import annotations

import threading
import time

from PySide6.QtCore import QObject, QThread, Signal, Slot

from ..adapters.perception_adapter import PerceptionAdapter
from ..config import GuiConfig
from ..core.state import SwarmState


class StateWorker(QObject):
    """
    Background state-generation worker.

    The GUI thread must not do expensive simulation or autonomy work.
    This worker emits SwarmState snapshots that are consumed by the GUI.
    """

    state_ready = Signal(object)
    finished = Signal()

    def __init__(self, adapter: PerceptionAdapter, hz: float):
        super().__init__()

        self.adapter = adapter
        self.dt = 1.0 / max(1.0, float(hz))

        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._reset_event = threading.Event()
        self._emergency_event = threading.Event()

        self._sim_time = 0.0

    def request_pause(self) -> None:
        self._pause_event.set()

    def request_resume(self) -> None:
        self._pause_event.clear()

    def request_reset(self) -> None:
        self._reset_event.set()

    def request_emergency_stop(self) -> None:
        self._emergency_event.set()

    def stop(self) -> None:
        self._stop_event.set()

    @Slot()
    def run(self) -> None:
        while not self._stop_event.is_set():
            if self._reset_event.is_set():
                self._sim_time = 0.0
                self.adapter.reset()
                self._reset_event.clear()

            if self._emergency_event.is_set():
                self.adapter.emergency_stop()
                self._emergency_event.clear()

            if self._pause_event.is_set():
                time.sleep(0.03)
                continue

            self._sim_time += self.dt

            try:
                state = self.adapter.update(self._sim_time, self.dt)
            except Exception as exc:
                state = SwarmState.error_state(str(exc))

            self.state_ready.emit(state)

            time.sleep(self.dt)

        self.finished.emit()


class Controller(QObject):
    """
    Application controller.

    Owns the worker thread and emits normalized GUI state.
    """

    state_changed = Signal(object)

    def __init__(self, cfg: GuiConfig):
        super().__init__()

        self.cfg = cfg
        self.adapter = PerceptionAdapter(cfg)

        self.thread = None
        self.worker = None
        self.running = False
        self.paused = False

        self.latest_state = None

    def start(self) -> None:
        if self.running:
            return

        self.worker = StateWorker(self.adapter, self.cfg.update_hz)
        self.thread = QThread(self)

        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.state_ready.connect(self._on_worker_state)
        self.worker.finished.connect(self.thread.quit)
        self.thread.finished.connect(self.worker.deleteLater)

        self.thread.start()
        self.running = True
        self.paused = False

    def _on_worker_state(self, state: SwarmState) -> None:
        self.latest_state = state
        self.state_changed.emit(state)

    def pause(self) -> None:
        if self.worker is not None:
            self.worker.request_pause()
            self.paused = True

    def resume(self) -> None:
        if self.worker is not None:
            self.worker.request_resume()
            self.paused = False

    def reset(self) -> None:
        if self.worker is not None:
            self.worker.request_reset()

    def emergency_stop(self) -> None:
        if self.worker is not None:
            self.worker.request_emergency_stop()

    def stop(self) -> None:
        if self.worker is not None:
            self.worker.stop()

        if self.thread is not None:
            self.thread.quit()
            self.thread.wait(2000)

        self.running = False