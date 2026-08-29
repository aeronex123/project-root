# autonomy/perception_state/runtime.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List
import time

from .config import LoopRatesConfig

@dataclass
class LoopSpec:
    name: str
    hz: float
    callback: Callable[[float], None]
    last_run: float = -1e9
    enabled: bool = True
    priority: int = 50

class MultiRateScheduler:
    """
    Deterministic virtual-time multi-rate scheduler.

    This intentionally does NOT force every algorithm to run at the same rate.
    """

    def __init__(self):
        self.loops: List[LoopSpec] = []

    def add_loop(self, name: str, hz: float, callback: Callable[[float], None], priority: int = 50):
        self.loops.append(LoopSpec(name=name, hz=hz, callback=callback, priority=priority))

    def set_rate(self, name: str, hz: float):
        for loop in self.loops:
            if loop.name == name:
                loop.hz = max(0.1, float(hz))

    def step(self, t: float):
        # Sort by priority for deterministic execution order.
        due = []
        for loop in self.loops:
            if not loop.enabled or loop.hz <= 0.0:
                continue
            period = 1.0 / loop.hz
            if t - loop.last_run >= period - 1e-6:
                due.append(loop)

        due.sort(key=lambda x: x.priority)

        for loop in due:
            t0 = time.perf_counter()
            loop.callback(t)
            loop.last_run = t
            _ = time.perf_counter() - t0

def default_rate_table(cfg: LoopRatesConfig) -> Dict[str, float]:
    return {
        "FLIGHT_STATE": cfg.flight_state_hz,
        "IMU": cfg.imu_hz,
        "OPTICAL_FLOW": cfg.optical_flow_hz,
        "TF_LUNA": cfg.tf_luna_hz,
        "VL53L5CX": cfg.vl53l5cx_hz,
        "CAMERA": cfg.camera_hz,
        "YOLO": 6.0,  # adaptive
        "MAPPING": cfg.mapping_hz,
        "PATH_PLANNING": cfg.planning_periodic_hz,
        "SWARM": cfg.swarm_heartbeat_hz,
        "VISUALIZATION": cfg.visualization_hz
    }