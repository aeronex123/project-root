# autonomy/perception_state/adaptive_inference.py
from __future__ import annotations
from dataclasses import dataclass
from .config import AdaptiveInferenceConfig
from .common_math import clamp

class AdaptiveInferencePolicy:
    def __init__(self, cfg: AdaptiveInferenceConfig, drone_id: int):
        self.cfg = cfg
        self.drone_id = drone_id
        self.current_hz = cfg.nominal_yolo_hz
        self.mine_inspect_until = -1.0
        self.state = "NOMINAL"

    def update(
        self,
        t: float,
        cpu_pct: float,
        speed_mps: float,
        interesting_objects: int,
        texture_score: float,
        mine_candidate: bool
    ) -> float:
        cfg = self.cfg

        hz = self.current_hz

        # Computational protection
        if cpu_pct > cfg.cpu_high_pct:
            hz *= 0.80
            self.state = "CPU_PROTECTED"
        elif cpu_pct < cfg.cpu_low_pct:
            hz *= 1.05
            if self.state == "CPU_PROTECTED":
                self.state = "NOMINAL"

        # Motion priority
        if speed_mps > cfg.fast_speed_mps:
            hz = max(hz, cfg.nominal_yolo_hz)
            self.state = "HIGH_SPEED"

        # Boring scene
        if interesting_objects == 0 and texture_score < cfg.low_texture_score:
            hz = min(hz, cfg.min_yolo_hz * 1.5)
            self.state = "LOW_FEATURE"

        # Mine inspection boost
        if mine_candidate:
            self.mine_inspect_until = t + cfg.mine_inspect_hold_s
            hz = max(hz, cfg.mine_inspect_hz)
            self.state = "MINE_INSPECT"
        elif t < self.mine_inspect_until:
            hz = max(hz, cfg.mine_inspect_hz)
            self.state = "MINE_INSPECT"

        self.current_hz = clamp(hz, cfg.min_yolo_hz, cfg.max_yolo_hz)
        return self.current_hz