# autonomy/perception_state/failure.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict

@dataclass
class FailureInjection:
    name: str
    drone_id: int
    start_t: float
    duration_s: float
    magnitude: float = 1.0

class FailureInjector:
    def __init__(self):
        self.active: Dict[str, FailureInjection] = {}

    def enable(self, name: str, drone_id: int, t: float, duration_s: float, magnitude: float = 1.0):
        self.active[f"{name}:{drone_id}"] = FailureInjection(
            name=name,
            drone_id=drone_id,
            start_t=t,
            duration_s=duration_s,
            magnitude=magnitude
        )

    def is_active(self, name: str, drone_id: int, t: float) -> bool:
        key = f"{name}:{drone_id}"
        if key not in self.active:
            return False

        f = self.active[key]
        if t < f.start_t or t > f.start_t + f.duration_s:
            return False

        return True