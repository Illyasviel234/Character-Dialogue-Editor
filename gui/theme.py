QSS = """
* {
    font-family: "Microsoft YaHei UI", "Segoe UI", sans-serif;
    font-size: 13px;
}
QMainWindow, QWidget { background: #1b1b19; color: #e8e6e0; }

QGroupBox {
    border: 1px solid #31312d;
    border-radius: 8px;
    margin-top: 16px;
    padding: 10px 10px 8px 10px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
    color: #918f86;
    font-size: 12px;
}

QPushButton {
    background: #2a2a27;
    border: 1px solid #3a3a35;
    border-radius: 6px;
    padding: 7px 14px;
}
QPushButton:hover { background: #343430; }
QPushButton:pressed { background: #262623; }
QPushButton:disabled { color: #65645e; background: #242422; }

QPushButton#primary {
    background: #0f6e56;
    border: 1px solid #1d9e75;
    color: #eafaf3;
}
QPushButton#primary:hover { background: #12866a; }
QPushButton#primary:disabled { background: #2a3a35; border-color: #33453f; color: #6d8d83; }

QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox {
    background: #232320;
    border: 1px solid #3a3a35;
    border-radius: 6px;
    padding: 6px 8px;
    selection-background-color: #0f6e56;
}
QLineEdit:focus, QComboBox:focus, QDoubleSpinBox:focus, QSpinBox:focus {
    border-color: #1d9e75;
}
QComboBox::drop-down { border: none; width: 20px; }
QComboBox QAbstractItemView {
    background: #232320;
    border: 1px solid #3a3a35;
    selection-background-color: #0f6e56;
    outline: none;
}

QPlainTextEdit {
    background: #131312;
    border: 1px solid #2c2c28;
    border-radius: 8px;
    color: #c6c4bc;
    font-family: "Cascadia Mono", "Consolas", monospace;
    font-size: 12px;
    padding: 6px;
}

QTableWidget {
    background: #191917;
    alternate-background-color: #1d1d1b;
    border: 1px solid #2c2c28;
    border-radius: 8px;
    gridline-color: #262622;
}
QHeaderView::section {
    background: #232320;
    color: #918f86;
    border: none;
    border-bottom: 1px solid #31312d;
    padding: 6px 4px;
    font-size: 12px;
}
QTableWidget::item { padding: 3px 4px; }
QTableWidget::item:selected { background: #14453a; color: #eafaf3; }

QProgressBar {
    background: #232320;
    border: none;
    border-radius: 3px;
    height: 6px;
    text-align: center;
    color: transparent;
}
QProgressBar::chunk { background: #1d9e75; border-radius: 3px; }

QScrollBar:vertical { background: transparent; width: 10px; margin: 0; }
QScrollBar::handle:vertical { background: #3a3a35; border-radius: 5px; min-height: 26px; }
QScrollBar::handle:vertical:hover { background: #4a4a44; }
QScrollBar:horizontal { background: transparent; height: 10px; margin: 0; }
QScrollBar::handle:horizontal { background: #3a3a35; border-radius: 5px; min-width: 26px; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; width: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }

QSplitter::handle { background: #262622; }
QSplitter::handle:horizontal { width: 1px; }
QSplitter::handle:vertical { height: 1px; }

QLabel#hint { color: #85837a; font-size: 12px; }
QLabel#stat { color: #9c9a91; font-size: 12px; }
QLabel#stage { color: #5dcaa5; font-size: 12px; }
"""
