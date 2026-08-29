# autonomy/perception_state/gesture.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import random

from .common_math import clamp

GESTURES = ("FORWARD", "PAUSE", "SCAN_LEFT", "SCAN_RIGHT")

@dataclass
class GestureObservation:
    t: float
    drone_id: int
    gesture: str
    confidence: float
    source: str

class RealGestureModel:
    source = "REAL_GESTURE_MODEL"

    def __init__(self, model_path: str):
        self.model_path = model_path
        # Real model integration would go here.
        # Must run locally only.

    def infer(self, t: float, drone_id: int, frame=None) -> Optional[GestureObservation]:
        # Placeholder interface.
        return None


class SimulatedGestureInput:
    source = "SIMULATED_GESTURE_INPUT"

    def __init__(self, noise_prob: float = 0.10):
        self.noise_prob = noise_prob

    def infer(self, t: float, drone_id: int, intended_gesture: Optional[str] = None) -> Optional[GestureObservation]:
        if intended_gesture is None:
            return None

        if random.random() < self.noise_prob:
            intended_gesture = random.choice(GESTURES)

        return GestureObservation(
            t=t,
            drone_id=drone_id,
            gesture=intended_gesture,
            confidence=clamp(random.gauss(0.80, 0.08), 0.3, 0.99),
            source=self.source
        )


def create_gesture_interface(model_path: str = ""):
    if model_path:
        print("GESTURE MODE: REAL_GESTURE_MODEL")
        return RealGestureModel(model_path)

    print("GESTURE MODE: SIMULATED_GESTURE_INPUT")
    return SimulatedGestureInput()