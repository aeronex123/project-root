# autonomy/perception_state/human_tracker.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List
import numpy as np
import math

from .detector import Detection

@dataclass
class HumanObservation:
    t: float
    drone_id: int
    pos2: np.ndarray
    cov2: np.ndarray
    confidence: float

class HumanTrack:
    def __init__(self, obs: HumanObservation):
        self.pos = obs.pos2.copy()
        self.vel = np.zeros(2, dtype=float)
        self.cov = obs.cov2.copy()
        self.confidence = obs.confidence
        self.last_seen = obs.t
        self.miss_count = 0

    @property
    def heading_rad(self) -> float:
        if np.linalg.norm(self.vel) < 1e-3:
            return 0.0
        return float(math.atan2(self.vel[1], self.vel[0]))

    def predict(self, dt: float):
        if dt <= 0:
            return
        self.pos += self.vel * dt
        self.cov += np.eye(2) * (0.05 * dt)

    def update(self, obs: HumanObservation):
        P1_inv = np.linalg.pinv(self.cov + np.eye(2)*1e-6)
        P2_inv = np.linalg.pinv(obs.cov2 + np.eye(2)*1e-6)
        P_fused_inv = P1_inv + P2_inv
        P_fused = np.linalg.pinv(P_fused_inv)

        new_pos = P_fused @ (P1_inv @ self.pos + P2_inv @ obs.pos2)

        if obs.t > self.last_seen:
            dt = obs.t - self.last_seen
            measured_vel = (obs.pos2 - self.pos) / max(dt, 1e-3)
            self.vel = 0.7 * self.vel + 0.3 * measured_vel

        self.pos = new_pos
        self.cov = P_fused
        self.confidence = max(self.confidence * 0.9, obs.confidence)
        self.last_seen = obs.t
        self.miss_count = 0


class HumanTracker:
    def __init__(self, max_misses: int = 8, timeout_s: float = 2.5):
        self.track: Optional[HumanTrack] = None
        self.max_misses = max_misses
        self.timeout_s = timeout_s
        self.state = "NO_HUMAN"

    def update(self, t: float, observations: List[HumanObservation], dt: float):
        if self.track is not None:
            self.track.predict(dt)

        if not observations:
            if self.track is not None:
                self.track.miss_count += 1
                if self.track.miss_count > self.max_misses or (t - self.track.last_seen) > self.timeout_s:
                    self.state = "HUMAN_SEARCH"
                else:
                    self.state = "HUMAN_TRACKING_COAST"
            else:
                self.state = "NO_HUMAN"
            return

        best = max(observations, key=lambda o: o.confidence)

        if self.track is None:
            self.track = HumanTrack(best)
            self.state = "HUMAN_TRACKING"
        else:
            self.track.update(best)
            self.state = "HUMAN_TRACKING"