# autonomy/gui/themes/dark_theme.py

DARK_THEME_QSS = """
QMainWindow {
    background-color: #05070c;
    color: #d7e0ea;
}

QWidget {
    background-color: #05070c;
    color: #d7e0ea;
    font-family: "Segoe UI", "Roboto", "Inter", sans-serif;
    font-size: 12px;
}

QGroupBox {
    border: 1px solid #1d2836;
    border-radius: 4px;
    margin-top: 12px;
    background-color: #090d14;
    font-weight: 600;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 4px;
    color: #7fd4ff;
}

QLabel {
    background: transparent;
    color: #d7e0ea;
}

QToolBar {
    background: #070a10;
    border-bottom: 1px solid #182433;
    padding: 4px;
    spacing: 6px;
}

QToolButton {
    background: #0d1420;
    border: 1px solid #223349;
    border-radius: 3px;
    padding: 5px 9px;
    color: #d7e0ea;
}

QToolButton:hover {
    background: #152030;
}

QToolButton:pressed {
    background: #0a1018;
}

QToolButton:checked {
    background: #173042;
    border-color: #4080a8;
}

QComboBox {
    background: #0d1420;
    border: 1px solid #223349;
    border-radius: 3px;
    padding: 4px;
}

QComboBox QAbstractItemView {
    background: #0d1420;
    selection-background-color: #173042;
}

QTabWidget::pane {
    border: 1px solid #182433;
    background: #070a10;
}

QTabBar::tab {
    background: #0a0f16;
    border: 1px solid #182433;
    padding: 6px 12px;
}

QTabBar::tab:selected {
    background: #102030;
    color: #8fe1ff;
}

QPlainTextEdit, QTextEdit {
    background: #06090f;
    border: 1px solid #182433;
    color: #c9d6e4;
    font-family: Consolas, "JetBrains Mono", monospace;
}

QScrollBar:vertical {
    background: #070a10;
    width: 12px;
}

QScrollBar::handle:vertical {
    background: #223349;
    border-radius: 3px;
    min-height: 24px;
}

QScrollBar:horizontal {
    background: #070a10;
    height: 12px;
}

QScrollBar::handle:horizontal {
    background: #223349;
    border-radius: 3px;
    min-width: 24px;
}

QProgressBar {
    border: 1px solid #223349;
    background: #090d14;
    text-align: center;
    color: #d7e0ea;
    height: 16px;
}

QProgressBar::chunk {
    background: #2a7ca8;
}
"""