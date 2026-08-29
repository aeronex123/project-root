# autonomy/gui/config.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GuiConfig:
    """
    GUI-level configuration.

    This does not replace autonomy/perception_state/config.py.
    It only configures visualization, demo timing, and UI behavior.
    """

    window_title: str = "Autonomous Swarm Command Center"

    window_width: int = 1800
    window_height: int = 1000

    # Simulation / state update rate
    update_hz: float = 30.0

    # Rendering target
    render_hz: float = 30.0

    # Mission
    mission_duration_s: float = 180.0

    # Competition field approximation
    field_length_m: float = 60.0
    field_width_m: float = 15.0

    # Visualization defaults
    show_paths: bool = True
    show_trails: bool = True
    show_sensor_fov: bool = True
    show_mines: bool = True
    show_human: bool = True
    show_comm_links: bool = True
    show_labels: bool = True
    show_occupancy: bool = True
    show_safety_zones: bool = True

    # Camera
    follow_drone_id: int = 0  # 0 = free camera

    # Data source:
    #   simulation : deterministic demo state
    #   live       : attempt to bridge existing autonomy stack
    mode: str = "simulation"

    # Logging
    log_level: str = "INFO"