# Autonomous Swarm Command Center GUI

This GUI subsystem is a separate visualization and command layer.

It does NOT modify the existing autonomy/perception_state code.

## Architecture

```text
AUTONOMY
  perception_state/
    detector
    EKF
    fusion
    mine fusion
    occupancy
    swarm map
    human tracker
    drone stack
    system
        ↓
ADAPTER
  PerceptionAdapter
        ↓
GUI STATE
  SwarmState
  DroneState
  MissionState
  EnvironmentState
        ↓
GUI
  3D view
  telemetry panels
  event log
  minimap