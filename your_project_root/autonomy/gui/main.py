# autonomy/gui/main.py
from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Autonomous Swarm Command Center GUI"
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="simulation",
        choices=["simulation", "live"],
        help="simulation = deterministic demo, live = attempt existing autonomy bridge",
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=180.0,
        help="Mission duration in seconds",
    )

    parser.add_argument(
        "--hz",
        type=float,
        default=30.0,
        help="State update rate in Hz",
    )

    parser.add_argument(
        "--fullscreen",
        action="store_true",
        help="Start in full-screen presentation mode",
    )

    args = parser.parse_args()

    try:
        from PySide6.QtWidgets import QApplication
    except Exception as exc:
        print("GUI dependencies missing.")
        print("Install with:")
        print("python -m pip install PySide6 pyvista pyvistaqt pyqtgraph numpy")
        print()
        print(f"Details: {exc}")
        return 1

    from .app import MainWindow
    from .config import GuiConfig
    from .themes.dark_theme import DARK_THEME_QSS

    cfg = GuiConfig()
    cfg.mode = args.mode
    cfg.mission_duration_s = args.duration
    cfg.update_hz = args.hz

    app = QApplication(sys.argv)
    app.setApplicationName(cfg.window_title)
    app.setStyleSheet(DARK_THEME_QSS)

    window = MainWindow(cfg)

    if args.fullscreen:
        window.showFullScreen()
    else:
        window.show()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())