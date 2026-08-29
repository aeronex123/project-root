

# 17. New file: `tests/test_gui_state.py`


# tests/test_gui_state.py
from autonomy.gui.config import GuiConfig
from autonomy.gui.core.state import MissionPhase
from autonomy.gui.simulation.demo_state import DemoStateProvider


def test_demo_state_creation():
    cfg = GuiConfig()
    provider = DemoStateProvider(cfg)

    state = provider.update(0.1, 0.033)

    assert len(state.drones) == 3
    assert state.mission.phase == MissionPhase.INITIALIZING
    assert state.source.value == "SIMULATION"


def test_demo_mission_progresses():
    cfg = GuiConfig()
    provider = DemoStateProvider(cfg)

    state = provider.update(100.0, 0.033)

    assert state.mission.elapsed > 0.0
    assert state.mission.progress > 0.0
    assert len(state.environment.mines) >= 0


def test_demo_mines_appear_and_confirm():
    cfg = GuiConfig()
    provider = DemoStateProvider(cfg)

    state_early = provider.update(100.0, 0.033)
    state_late = provider.update(300.0, 0.033)

    assert len(state_late.environment.mines) >= len(state_early.environment.mines)

    if state_late.environment.mines:
        assert state_late.environment.mines[0].status in {
            "SUSPECTED",
            "PROBABLE",
            "CONFIRMED",
        }


def test_demo_human_appears():
    cfg = GuiConfig()
    provider = DemoStateProvider(cfg)

    state_before = provider.update(200.0, 0.033)
    state_after = provider.update(330.0, 0.033)

    assert state_before.environment.human is None
    assert state_after.environment.human is not None


def test_demo_deterministic_positions():
    cfg = GuiConfig()

    provider_a = DemoStateProvider(cfg)
    provider_b = DemoStateProvider(cfg)

    t = 0.0
    dt = 0.1

    while t < 120.0:
        state_a = provider_a.update(t, dt)
        state_b = provider_b.update(t, dt)
        t += dt

    for da, db in zip(state_a.drones, state_b.drones):
        assert da.position == db.position