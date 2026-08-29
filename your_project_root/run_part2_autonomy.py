# run_part2_autonomy.py
from __future__ import annotations

import time
import argparse

from autonomy.perception_state.system import ThreeDroneSwarmSim


def main():
    parser = argparse.ArgumentParser(
        description="Run Part 2 three-drone autonomy/perception/state-estimation simulation."
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=300.0,
        help="Mission duration in seconds."
    )

    parser.add_argument(
        "--dt",
        type=float,
        default=0.01,
        help="Simulation step size in seconds."
    )

    parser.add_argument(
        "--print-rate",
        type=float,
        default=1.0,
        help="Print status every N seconds."
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Part 2: Three-Drone Autonomous Swarm Perception/Estimation Layer")
    print("=" * 70)

    sim = ThreeDroneSwarmSim()

    t = 0.0
    dt = args.dt
    next_print = 0.0

    try:
        while t < args.duration:
            # ------------------------------------------------------------
            # In the full competition simulator, Part 1 world/physics sim
            # should advance here and generate simulated sensor truth.
            #
            # Important:
            #   Ground truth may be used only to generate sensor data
            #   and for optional evaluation overlay.
            #
            #   The autonomy stack itself must NOT receive ground truth.
            # ------------------------------------------------------------

            sim.step(t, dt)

            if t >= next_print:
                next_print += args.print_rate

                print()
                print(f"[SIM TIME] {t:8.2f} s")
                print("-" * 70)

                for drone_id, drone in sim.drones.items():
                    out = drone.outputs(t)

                    print(
                        f"[DRONE {drone_id}] "
                        f"pos={out.est_pos.round(2).tolist()} "
                        f"unc={out.pos_uncertainty:0.3f} "
                        f"conf={out.confidence:0.2f} "
                        f"mines={out.global_mine_ids} "
                        f"human={out.human_state} "
                        f"AI={out.ai_mode}"
                    )

            t += dt

    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")

    print()
    print("=" * 70)
    print("Part 2 autonomy smoke run complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()