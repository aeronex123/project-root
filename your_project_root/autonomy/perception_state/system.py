# autonomy/perception_state/system.py
from __future__ import annotations
from typing import Dict

from .config import PerceptionStateConfig
from .swarm_map import SwarmChannel
from .drone_stack import DroneAutonomyStack

class ThreeDroneSwarmSim:
    """
    Simulates actual onboard computation of all three drones.

    Each drone has:
      - its own sensors,
      - its own EKF,
      - its own detector,
      - its own local map,
      - its own performance monitor.
    """

    def __init__(self):
        self.channel = SwarmChannel(PerceptionStateConfig().swarm)
        self.drones: Dict[int, DroneAutonomyStack] = {}

        for drone_id in (1, 2, 3):
            cfg = PerceptionStateConfig(drone_id=drone_id)
            self.drones[drone_id] = DroneAutonomyStack(cfg, self.channel)

    def step(self, t: float, dt: float):
        for drone_id, drone in self.drones.items():
            drone.step(t, dt)