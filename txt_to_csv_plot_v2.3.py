import csv
import sys
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.ticker import AutoLocator
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTabWidget,
    QVBoxLayout,
    QWidget,
    QScrollArea,
    QToolButton,
    QFrame,
)


PLOT_TITLES = {
    "rpm_torque": "RPM and Torque Data",
    "rpm_current": "RPM and Current Data",
    "rpm_only": "RPM Data",
}

COL = {
    "time": 0,
    "w1_cmd_rpm": 3,
    "w1_rpm": 4,
    "w1_cmd_torque": 5,
    "w1_torque": 6,
    "w1_cur": 7,
    "w2_cmd_rpm": 10,
    "w2_rpm": 11,
    "w2_cmd_torque": 12,
    "w2_torque": 13,
    "w2_cur": 14,
    "w3_cmd_rpm": 17,
    "w3_rpm": 18,
    "w3_cmd_torque": 19,
    "w3_torque": 20,
    "w3_cur": 21,
}

RPM_YLIM = (-8000, 8000)
RPM_YTICKS = np.arange(-8000, 8001, 2000)
WHEEL3_COLOR = (0.3, 0.0, 0.4)
BASE_FIGURE_HEIGHT_IN = 5.0
FIG_DPI = 100
DEFAULT_RATIO_W = 2.0
DEFAULT_RATIO_H = 1.0
DEFAULT_TITLE_PAD = 24.0
LEGEND_FONT_SCALE = 0.75
ENCODINGS = ["utf-8-sig", "utf-8", "cp949", "latin-1"]

FIG_SERIES_ORDER = {
    1: [
        "fig1_cmd_rpm",
        "fig1_cmd_torque",
        "fig1_w1_rpm",
        "fig1_w2_rpm",
        "fig1_w3_rpm",
        "fig1_w1_torque",
        "fig1_w2_torque",
        "fig1_w3_torque",
    ],
    2: [
        "fig2_cmd_rpm",
        "fig2_w1_rpm",
        "fig2_w2_rpm",
        "fig2_w3_rpm",
        "fig2_w1_current",
        "fig2_w2_current",
        "fig2_w3_current",
    ],
    3: [
        "fig3_cmd_rpm",
        "fig3_cmd_torque",
        "fig3_w1_rpm",
        "fig3_w2_rpm",
        "fig3_w3_rpm",
    ],
}

FIG_SERIES_META = {
    "fig1_cmd_rpm": {"axis": "ax1", "series": "wCMD", "label": "CMD RPM", "style": {"color": "tab:blue","linewidth": "cmd_lw"}},
    "fig1_cmd_torque": {"axis": "ax2", "series": "wtqCMD", "label": "CMD Torque", "style": {"color": "tab:orange","linewidth": "cmd_lw"}},
    "fig1_w1_rpm": {"axis": "ax1", "series": "w1", "label": "Wheel1 RPM", "style": {"color": "r", "linewidth": "rpm_lw"}},
    "fig1_w2_rpm": {"axis": "ax1", "series": "w2", "label": "Wheel2 RPM", "style": {"color": "b", "linewidth": "rpm_lw"}},
    "fig1_w3_rpm": {"axis": "ax1", "series": "w3", "label": "Wheel3 RPM", "style": {"color": WHEEL3_COLOR, "linewidth": "rpm_lw"}},
    "fig1_w1_torque": {"axis": "ax2", "series": "wtq1", "label": "Wheel1 Torque", "style": {"color": "tab:green","linewidth": "secondary_lw"}},
    "fig1_w2_torque": {"axis": "ax2", "series": "wtq2", "label": "Wheel2 Torque", "style": {"color": "tab:purple","linewidth": "secondary_lw"}},
    "fig1_w3_torque": {"axis": "ax2", "series": "wtq3", "label": "Wheel3 Torque", "style": {"color": "tab:olive","linewidth": "secondary_lw"}},
    "fig2_cmd_rpm": {"axis": "ax1", "series": "wCMD", "label": "CMD RPM", "style": {"color": "tab:blue","linewidth": "cmd_lw"}},
    "fig2_w1_rpm": {"axis": "ax1", "series": "w1", "label": "Wheel1 RPM", "style": {"color": "r", "linewidth": "rpm_lw"}},
    "fig2_w2_rpm": {"axis": "ax1", "series": "w2", "label": "Wheel2 RPM", "style": {"color": "b", "linewidth": "rpm_lw"}},
    "fig2_w3_rpm": {"axis": "ax1", "series": "w3", "label": "Wheel3 RPM", "style": {"color": WHEEL3_COLOR, "linewidth": "rpm_lw"}},
    "fig2_w1_current": {"axis": "ax2", "series": "cur1", "label": "Wheel1 Current", "style": {"color": "tab:green","linewidth": "secondary_lw"}},
    "fig2_w2_current": {"axis": "ax2", "series": "cur2", "label": "Wheel2 Current", "style": {"color": "tab:purple","linewidth": "secondary_lw"}},
    "fig2_w3_current": {"axis": "ax2", "series": "cur3", "label": "Wheel3 Current", "style": {"color": "tab:olive","linewidth": "secondary_lw"}},
    "fig3_cmd_rpm": {"axis": "ax1", "series": "wCMD", "label": "CMD RPM", "style": {"color": "tab:blue", "linewidth": "cmd_lw"}},
    "fig3_cmd_torque": {"axis": "ax2", "series": "wtqCMD", "label": "CMD Torque", "style": {"color": "tab:orange", "linewidth": "cmd_lw"}},
    "fig3_w1_rpm": {"axis": "ax1", "series": "w1", "label": "Wheel1 RPM", "style": {"color": "r", "linewidth": "rpm_lw"}},
    "fig3_w2_rpm": {"axis": "ax1", "series": "w2", "label": "Wheel2 RPM", "style": {"color": "b", "linewidth": "rpm_lw"}},
    "fig3_w3_rpm": {"axis": "ax1", "series": "w3", "label": "Wheel3 RPM", "style": {"color": WHEEL3_COLOR, "linewidth": "rpm_lw"}},
}


PLOT_KEY_TO_SERIES = {key: meta["series"] for key, meta in FIG_SERIES_META.items()}


def make_safe_filename(name: str) -> str:
    invalid = '<>:"/\\|?*'
    return "".join("_" if ch in invalid else ch for ch in name).strip()


def read_delimited_rows(path: Path, delimiter: str) -> list[list[str]]:
    last_error = None
    for enc in ENCODINGS:
        try:
            with path.open("r", encoding=enc, newline="") as f:
                return [row for row in csv.reader(f, delimiter=delimiter)]
        except UnicodeDecodeError as exc:
            last_error = exc
    raise UnicodeDecodeError(
        "unknown",
        b"",
        0,
        1,
        f"Unable to decode file with supported encodings. Last error: {last_error}",
    )


def normalize_rows(rows: list[list[str]]) -> list[list[str]]:
    max_len = max((len(row) for row in rows), default=0)
    normalized = []
    for row in rows:
        cleaned = [cell.strip() for cell in row]
        if len(cleaned) < max_len:
            cleaned.extend([""] * (max_len - len(cleaned)))
        normalized.append(cleaned)
    return normalized


def convert_txt_to_csv(txt_path: str, csv_path: str) -> str:
    txt_file = Path(txt_path)
    csv_file = Path(csv_path)
    rows = normalize_rows(read_delimited_rows(txt_file, "\t"))
    csv_file.parent.mkdir(parents=True, exist_ok=True)

    with csv_file.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    return str(csv_file)


def _parse_float(cell: str) -> float:
    cell = str(cell).strip()
    if cell == "":
        return np.nan
    try:
        return float(cell)
    except ValueError:
        return np.nan


def detect_numeric_start(rows: list[list[str]]) -> int:
    time_idx = COL["time"]

    actual_candidates = [
        COL["w1_rpm"], COL["w1_torque"], COL["w1_cur"],
        COL["w2_rpm"], COL["w2_torque"], COL["w2_cur"],
        COL["w3_rpm"], COL["w3_torque"], COL["w3_cur"],
    ]

    for i, row in enumerate(rows):
        if len(row) <= time_idx:
            continue

        time_value = _parse_float(row[time_idx])
        if not np.isfinite(time_value):
            continue

        actual_count = 0
        for idx in actual_candidates:
            if idx < len(row) and np.isfinite(_parse_float(row[idx])):
                actual_count += 1

        if actual_count >= 1:
            return i

    raise ValueError("Can't find Numeric data row. Check CSV file data!!")


def load_numeric_csv_auto(csv_path: str) -> tuple[pd.DataFrame, int]:
    rows = normalize_rows(read_delimited_rows(Path(csv_path), ","))
    start_row = detect_numeric_start(rows)
    df = pd.DataFrame(rows[start_row:])
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=0, how="all")
    return df, start_row


def extract_series(df: pd.DataFrame) -> dict[str, np.ndarray]:
    data = df.to_numpy(dtype=float)
    if data.size == 0:
        raise ValueError("CSV numeric data is empty.")

    nrows, ncols = data.shape

    def get_col(name: str) -> np.ndarray:
        idx = COL[name]
        if idx < ncols:
            return data[:, idx]
        return np.full(nrows, np.nan, dtype=float)

    def has_data(values: np.ndarray) -> bool:
        return bool(np.isfinite(values).any())

    essential = ["time"]
    missing = [name for name in essential if COL[name] >= ncols]
    if missing:
        raise ValueError(
            "CSV numeric data is missing required column(s): " + ", ".join(missing)
        )

    wheel_data = {
        1: {
            "cmd_rpm": get_col("w1_cmd_rpm"),
            "cmd_torque": get_col("w1_cmd_torque"),
            "rpm": get_col("w1_rpm"),
            "torque": get_col("w1_torque"),
            "current": get_col("w1_cur"),
        },
        2: {
            "cmd_rpm": get_col("w2_cmd_rpm"),
            "cmd_torque": get_col("w2_cmd_torque"),
            "rpm": get_col("w2_rpm"),
            "torque": get_col("w2_torque"),
            "current": get_col("w2_cur"),
        },
        3: {
            "cmd_rpm": get_col("w3_cmd_rpm"),
            "cmd_torque": get_col("w3_cmd_torque"),
            "rpm": get_col("w3_rpm"),
            "torque": get_col("w3_torque"),
            "current": get_col("w3_cur"),
        },
    }

    active_wheels = []
    for wheel_no, wd in wheel_data.items():
        actual_exists = any(
            has_data(arr) for arr in [wd["rpm"], wd["torque"], wd["current"]]
        )
        if actual_exists:
            active_wheels.append(wheel_no)

    command_wheel = min(active_wheels) if active_wheels else 1

    return {
        "t": get_col("time"),

        "wCMD": wheel_data[command_wheel]["cmd_rpm"],
        "wtqCMD": wheel_data[command_wheel]["cmd_torque"],

        "w1": wheel_data[1]["rpm"],
        "w2": wheel_data[2]["rpm"],
        "w3": wheel_data[3]["rpm"],
        "wtq1": wheel_data[1]["torque"],
        "wtq2": wheel_data[2]["torque"],
        "wtq3": wheel_data[3]["torque"],
        "cur1": wheel_data[1]["current"],
        "cur2": wheel_data[2]["current"],
        "cur3": wheel_data[3]["current"],
    }


def finite_ylim(values: np.ndarray, pad_ratio: float, fallback=(-1.0, 1.0)) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return fallback

    value_range = float(np.max(finite) - np.min(finite))
    if value_range == 0:
        value_range = max(1.0, abs(float(finite[0])))

    pad = pad_ratio * value_range
    return float(np.min(finite) - pad), float(np.max(finite) + pad)


@dataclass
class PlotSettings:
    cmd_lw: float
    rpm_lw: float
    secondary_lw: float
    ratio_w: float
    ratio_h: float
    title_pad: float
    show_grid: bool
    x_min: Optional[float]
    x_max: Optional[float]
    y1_min: Optional[float]
    y1_max: Optional[float]
    y2_min: Optional[float]
    y2_max: Optional[float]


class CollapsibleBox(QWidget):
    def __init__(self, title: str = "", parent=None, checked: bool = True):
        super().__init__(parent)

        self.toggle_button = QToolButton()
        self.toggle_button.setText(title)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(checked)
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle_button.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        self.toggle_button.clicked.connect(self._on_toggled)

        self.header_line = QFrame()
        self.header_line.setFrameShape(QFrame.HLine)
        self.header_line.setFrameShadow(QFrame.Sunken)

        self.content_area = QWidget()
        self.content_layout = QVBoxLayout(self.content_area)
        self.content_layout.setContentsMargins(20, 6, 0, 6)
        self.content_layout.setSpacing(4)
        self.content_area.setVisible(checked)

        top_layout = QVBoxLayout(self)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(4)

        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(6)
        header_layout.addWidget(self.toggle_button)
        header_layout.addWidget(self.header_line, 1)

        top_layout.addLayout(header_layout)
        top_layout.addWidget(self.content_area)

    def _on_toggled(self):
        expanded = self.toggle_button.isChecked()
        self.toggle_button.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self.content_area.setVisible(expanded)

    def add_widget(self, widget):
        self.content_layout.addWidget(widget)

    def add_layout(self, layout):
        self.content_layout.addLayout(layout)


class PlotCanvas(QWidget):
    def __init__(self, parent=None, ratio_w: float = DEFAULT_RATIO_W, ratio_h: float = DEFAULT_RATIO_H):
        super().__init__(parent)
        self.ratio_w = ratio_w
        self.ratio_h = ratio_h
        self.figure = Figure(dpi=FIG_DPI, facecolor="white")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.save_button = QPushButton("Save This Figure")
        save_font = QFont(self.save_button.font())
        save_font.setPointSize(max(1, save_font.pointSize() * 2))
        self.save_button.setFont(save_font)
        self.save_button.setMinimumSize(280, 72)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        layout.addStretch(1)
        layout.addWidget(self.save_button, alignment=Qt.AlignLeft | Qt.AlignBottom)
        self._apply_ratio_geometry()

    def _apply_ratio_geometry(self):
        available_width = max(700, self.width() - 24)
        target_height = max(300, int(available_width * (self.ratio_h / self.ratio_w)))
        self.canvas.setFixedHeight(target_height)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply_ratio_geometry()

    def set_ratio(self, ratio_w: float, ratio_h: float):
        self.ratio_w = max(0.1, ratio_w)
        self.ratio_h = max(0.1, ratio_h)
        self._apply_ratio_geometry()
        width_in = BASE_FIGURE_HEIGHT_IN * (self.ratio_w / self.ratio_h)
        self.figure.set_size_inches(width_in, BASE_FIGURE_HEIGHT_IN, forward=True)
        self.canvas.draw_idle()

    def clear(self):
        self.figure.clear()
        self.canvas.draw_idle()


class MainWindow(QMainWindow):
    def clear_axis_ranges(self):
        self.x_from_edit.clear()
        self.x_to_edit.clear()
        self.y1_from_edit.clear()
        self.y1_to_edit.clear()
        self.y2_from_edit.clear()
        self.y2_to_edit.clear()

        if self.last_series is not None:
            try:
                self.refresh_plots_from_cache()
                if self.last_csv_path:
                    self.set_status(f"Axis range cleared.\nCSV used: {self.last_csv_path}")
                else:
                    self.set_status("Axis range cleared.")
            except Exception as exc:
                self._show_critical("Axis Range Error", str(exc))
                self.set_status(f"Error: {exc}")
        else:
            self.set_status("Axis range cleared.")
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RAW to CSV Auto Plot Maker v2.1")
        self.resize(1200, 800)

        self.figures = {1: None, 2: None, 3: None}
        self.last_csv_path = None
        self.last_series = None
        self.last_base_title = None
        self.plot_checks = {}
        self.plot_draw_order = {fig_no: [] for fig_no in FIG_SERIES_ORDER}

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        top_layout = QHBoxLayout()
        top_layout.addWidget(self._build_left_panel(), 0)
        top_layout.addWidget(self._build_tabs(), 1)
        main_layout.addLayout(top_layout, 1)
        main_layout.addWidget(self._build_status_group())

        self._initialize_plot_draw_order()
        self._connect_signals()

    def _build_left_panel(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(420)
        scroll.setMaximumWidth(560)
        scroll.setFrameShape(QFrame.NoFrame)

        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        layout.addWidget(self._build_file_group())
        layout.addWidget(self._build_plot_group())
        layout.addStretch(1)

        scroll.setWidget(panel)
        return scroll

    def _build_file_group(self) -> QGroupBox:
        box = QGroupBox("Files and Basic Settings")
        layout = QGridLayout(box)

        self.txt_edit = QLineEdit()
        self.csv_load_edit = QLineEdit()
        self.csv_save_edit = QLineEdit()
        self.title_edit = QLineEdit()
        self.title_edit.setPlaceholderText("Plot title")

        self.txt_btn = QPushButton("Browse TXT")
        self.csv_load_btn = QPushButton("Browse CSV")
        self.csv_save_btn = QPushButton("Set Save Path")
        self.run_txt_btn = QPushButton("Convert TXT → CSV and Plot")
        self.run_csv_btn = QPushButton("Load CSV and Plot")
        self.clear_btn = QPushButton("Clear Figures")

        row = 0
        layout.addWidget(QLabel("Raw Data file (.txt)"), row, 0)
        layout.addWidget(self.txt_edit, row, 1)
        layout.addWidget(self.txt_btn, row, 2)
        row += 1

        layout.addWidget(QLabel("Existing CSV file"), row, 0)
        layout.addWidget(self.csv_load_edit, row, 1)
        layout.addWidget(self.csv_load_btn, row, 2)
        row += 1

        layout.addWidget(QLabel("CSV Save Path"), row, 0)
        layout.addWidget(self.csv_save_edit, row, 1)
        layout.addWidget(self.csv_save_btn, row, 2)
        row += 1

        layout.addWidget(QLabel("Plot Title"), row, 0)
        layout.addWidget(self.title_edit, row, 1, 1, 2)
        row += 1

        button_col = QVBoxLayout()
        button_col.addWidget(self.run_txt_btn)
        button_col.addWidget(self.run_csv_btn)
        button_col.addWidget(self.clear_btn)
        layout.addLayout(button_col, row, 0, 1, 3)

        layout.setColumnStretch(1, 1)
        return box

    def _make_series_checkbox(self, text: str, checked: bool = True) -> QCheckBox:
        cb = QCheckBox(text)
        cb.setChecked(checked)
        return cb

    def _make_double_spin(self, minimum: float, maximum: float, step: float, value: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setValue(value)
        return spin

    def _build_plot_group(self) -> QGroupBox:
        box = QGroupBox("Plot Options")
        outer = QVBoxLayout(box)

        form = QFormLayout()

        self.cmd_lw_spin = self._make_double_spin(0.1, 10.0, 0.1, 1.5)
        self.rpm_lw_spin = self._make_double_spin(0.1, 10.0, 0.1, 1.2)
        self.secondary_lw_spin = self._make_double_spin(0.1, 10.0, 0.1, 0.9)
        self.ratio_w_spin = self._make_double_spin(0.5, 10.0, 0.1, DEFAULT_RATIO_W)
        self.ratio_h_spin = self._make_double_spin(0.5, 10.0, 0.1, DEFAULT_RATIO_H)
        self.title_pad_spin = self._make_double_spin(0.0, 60.0, 1.0, DEFAULT_TITLE_PAD)

        self.show_grid_check = QCheckBox("Enable grid")
        self.show_grid_check.setChecked(True)

        form.addRow("CMD line width", self.cmd_lw_spin)
        form.addRow("Wheel RPM line width", self.rpm_lw_spin)
        form.addRow("Torque / Current line width", self.secondary_lw_spin)
        form.addRow("Figure ratio width", self.ratio_w_spin)
        form.addRow("Figure ratio height", self.ratio_h_spin)
        form.addRow("Title top margin", self.title_pad_spin)
        form.addRow(self.show_grid_check)

        axis_box = QGroupBox("Axis Range (blank = auto)")
        axis_layout = QGridLayout(axis_box)

        self.x_from_edit = QLineEdit()
        self.x_to_edit = QLineEdit()
        self.y1_from_edit = QLineEdit()
        self.y1_to_edit = QLineEdit()
        self.y2_from_edit = QLineEdit()
        self.y2_to_edit = QLineEdit()

        axis_edits = [
            self.x_from_edit,
            self.x_to_edit,
            self.y1_from_edit,
            self.y1_to_edit,
            self.y2_from_edit,
            self.y2_to_edit,
        ]
        for edit in axis_edits:
            edit.setPlaceholderText("Auto")

        axis_layout.addWidget(QLabel("Axis"), 0, 0)
        axis_layout.addWidget(QLabel("From"), 0, 1)
        axis_layout.addWidget(QLabel("To"), 0, 2)

        axis_layout.addWidget(QLabel("X"), 1, 0)
        axis_layout.addWidget(self.x_from_edit, 1, 1)
        axis_layout.addWidget(self.x_to_edit, 1, 2)

        axis_layout.addWidget(QLabel("Left Y"), 2, 0)
        axis_layout.addWidget(self.y1_from_edit, 2, 1)
        axis_layout.addWidget(self.y1_to_edit, 2, 2)

        axis_layout.addWidget(QLabel("Right Y"), 3, 0)
        axis_layout.addWidget(self.y2_from_edit, 3, 1)
        axis_layout.addWidget(self.y2_to_edit, 3, 2)

        self.apply_axis_btn = QPushButton("Apply Axis Range")
        self.clear_axis_btn = QPushButton("All Clear")

        axis_btn_layout = QHBoxLayout()
        axis_btn_layout.addWidget(self.apply_axis_btn)
        axis_btn_layout.addWidget(self.clear_axis_btn)

        axis_layout.addLayout(axis_btn_layout, 4, 0, 1, 3)

        outer.addLayout(form)
        outer.addWidget(axis_box)

        fig1_box = CollapsibleBox("Figure 1 visible series", checked=True)
        self.plot_checks["fig1_cmd_rpm"] = self._make_series_checkbox("CMD RPM")
        self.plot_checks["fig1_cmd_torque"] = self._make_series_checkbox("CMD Torque")
        self.plot_checks["fig1_w1_rpm"] = self._make_series_checkbox("Wheel1 RPM")
        self.plot_checks["fig1_w2_rpm"] = self._make_series_checkbox("Wheel2 RPM")
        self.plot_checks["fig1_w3_rpm"] = self._make_series_checkbox("Wheel3 RPM")
        self.plot_checks["fig1_w1_torque"] = self._make_series_checkbox("Wheel1 Torque")
        self.plot_checks["fig1_w2_torque"] = self._make_series_checkbox("Wheel2 Torque")
        self.plot_checks["fig1_w3_torque"] = self._make_series_checkbox("Wheel3 Torque")

        for key in FIG_SERIES_ORDER[1]:
            fig1_box.add_widget(self.plot_checks[key])

        outer.addWidget(fig1_box)

        fig2_box = CollapsibleBox("Figure 2 visible series", checked=False)
        self.plot_checks["fig2_cmd_rpm"] = self._make_series_checkbox("CMD RPM")
        self.plot_checks["fig2_w1_rpm"] = self._make_series_checkbox("Wheel1 RPM")
        self.plot_checks["fig2_w2_rpm"] = self._make_series_checkbox("Wheel2 RPM")
        self.plot_checks["fig2_w3_rpm"] = self._make_series_checkbox("Wheel3 RPM")
        self.plot_checks["fig2_w1_current"] = self._make_series_checkbox("Wheel1 Current")
        self.plot_checks["fig2_w2_current"] = self._make_series_checkbox("Wheel2 Current")
        self.plot_checks["fig2_w3_current"] = self._make_series_checkbox("Wheel3 Current")

        for key in FIG_SERIES_ORDER[2]:
            fig2_box.add_widget(self.plot_checks[key])

        outer.addWidget(fig2_box)

        fig3_box = CollapsibleBox("Figure 3 visible series", checked=False)
        self.plot_checks["fig3_cmd_rpm"] = self._make_series_checkbox("CMD RPM")
        self.plot_checks["fig3_cmd_torque"] = self._make_series_checkbox("CMD Torque")
        self.plot_checks["fig3_w1_rpm"] = self._make_series_checkbox("Wheel1 RPM")
        self.plot_checks["fig3_w2_rpm"] = self._make_series_checkbox("Wheel2 RPM")
        self.plot_checks["fig3_w3_rpm"] = self._make_series_checkbox("Wheel3 RPM")

        for key in FIG_SERIES_ORDER[3]:
            fig3_box.add_widget(self.plot_checks[key])

        outer.addWidget(fig3_box)

        outer.addStretch(1)
        return box

    def _build_tabs(self) -> QTabWidget:
        self.tabs = QTabWidget()
        self.plot1 = PlotCanvas(ratio_w=DEFAULT_RATIO_W, ratio_h=DEFAULT_RATIO_H)
        self.plot2 = PlotCanvas(ratio_w=DEFAULT_RATIO_W, ratio_h=DEFAULT_RATIO_H)
        self.plot3 = PlotCanvas(ratio_w=DEFAULT_RATIO_W, ratio_h=DEFAULT_RATIO_H)
        self.tabs.addTab(self.plot1, "Figure 1 - RPM + Torque")
        self.tabs.addTab(self.plot2, "Figure 2 - RPM + Current")
        self.tabs.addTab(self.plot3, "Figure 3 - RPM")
        return self.tabs

    def _build_status_group(self) -> QGroupBox:
        box = QGroupBox("Status")
        layout = QVBoxLayout(box)
        self.status_label = QLabel("Ready")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        return box

    def _make_message_box(self, icon: QMessageBox.Icon, title: str, text: str):
        box = QMessageBox(self)
        box.setIcon(icon)
        box.setWindowTitle(title)
        box.setText(text)
        box.setStyleSheet("QLabel { font-size: 24px; font-weight: 700; }")
        box.setMinimumSize(900, 500)
        box.setStandardButtons(QMessageBox.Ok)
        box.exec()

    def _show_warning(self, title: str, text: str):
        self._make_message_box(QMessageBox.Warning, title, text)

    def _show_critical(self, title: str, text: str):
        self._make_message_box(QMessageBox.Critical, title, text)

    def _show_info(self, title: str, text: str):
        self._make_message_box(QMessageBox.Information, title, text)

    def _connect_signals(self):
        self.txt_btn.clicked.connect(self.select_txt_file)
        self.csv_load_btn.clicked.connect(self.select_existing_csv_file)
        self.csv_save_btn.clicked.connect(self.select_csv_save_path)
        self.run_txt_btn.clicked.connect(self.run_from_txt)
        self.run_csv_btn.clicked.connect(self.run_from_csv)
        self.clear_btn.clicked.connect(self.clear_figures)
        self.apply_axis_btn.clicked.connect(self.apply_axis_ranges)
        self.clear_axis_btn.clicked.connect(self.clear_axis_ranges)
        self.plot1.save_button.clicked.connect(lambda: self.save_single_figure(1))
        self.plot2.save_button.clicked.connect(lambda: self.save_single_figure(2))
        self.plot3.save_button.clicked.connect(lambda: self.save_single_figure(3))

        for key, cb in self.plot_checks.items():
            cb.stateChanged.connect(lambda state, plot_key=key: self.handle_plot_check_change(plot_key, state))

    def _initialize_plot_draw_order(self):
        for fig_no, keys in FIG_SERIES_ORDER.items():
            active_keys = [key for key in keys if self.plot_checks[key].isChecked()]
            self.plot_draw_order[fig_no] = active_keys

    def _fig_no_from_key(self, key: str) -> int:
        return int(key[3])

    def handle_plot_check_change(self, key: str, state: int):
        fig_no = self._fig_no_from_key(key)
        draw_order = self.plot_draw_order[fig_no]

        if state == Qt.Checked.value:
            if key in draw_order:
                draw_order.remove(key)
            draw_order.append(key)
        else:
            if key in draw_order:
                draw_order.remove(key)

        try:
            self.refresh_plots_from_cache()
        except Exception as exc:
            self._show_critical("Error", str(exc))
            self.set_status(f"Error: {exc}")

    def _parse_optional_limit(self, edit: QLineEdit, label: str) -> Optional[float]:
        text = edit.text().strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError as exc:
            raise ValueError(f"{label} must be numeric or blank.") from exc

    def _validate_limit_pair(self, minimum: Optional[float], maximum: Optional[float], axis_name: str):
        if minimum is not None and maximum is not None and minimum >= maximum:
            raise ValueError(f"{axis_name} range must satisfy From < To.")

    def current_plot_settings(self) -> PlotSettings:
        x_min = self._parse_optional_limit(self.x_from_edit, "X axis From")
        x_max = self._parse_optional_limit(self.x_to_edit, "X axis To")
        y1_min = self._parse_optional_limit(self.y1_from_edit, "Left Y axis From")
        y1_max = self._parse_optional_limit(self.y1_to_edit, "Left Y axis To")
        y2_min = self._parse_optional_limit(self.y2_from_edit, "Right Y axis From")
        y2_max = self._parse_optional_limit(self.y2_to_edit, "Right Y axis To")

        self._validate_limit_pair(x_min, x_max, "X axis")
        self._validate_limit_pair(y1_min, y1_max, "Left Y axis")
        self._validate_limit_pair(y2_min, y2_max, "Right Y axis")

        return PlotSettings(
            cmd_lw=self.cmd_lw_spin.value(),
            rpm_lw=self.rpm_lw_spin.value(),
            secondary_lw=self.secondary_lw_spin.value(),
            ratio_w=self.ratio_w_spin.value(),
            ratio_h=self.ratio_h_spin.value(),
            title_pad=self.title_pad_spin.value(),
            show_grid=self.show_grid_check.isChecked(),
            x_min=x_min,
            x_max=x_max,
            y1_min=y1_min,
            y1_max=y1_max,
            y2_min=y2_min,
            y2_max=y2_max,
        )

    def set_status(self, text: str):
        self.status_label.setText(text)

    def _resolve_axis_limits(
        self,
        auto_min: float,
        auto_max: float,
        user_min: Optional[float],
        user_max: Optional[float],
    ) -> tuple[float, float]:
        lower = auto_min if user_min is None else user_min
        upper = auto_max if user_max is None else user_max
        if lower >= upper:
            raise ValueError("Axis range must satisfy From < To.")
        return float(lower), float(upper)

    def apply_axis_ranges(self):
        if self.last_series is None:
            self.set_status("Load data first, then apply axis range.")
            return

        try:
            self.refresh_plots_from_cache()
            if self.last_csv_path:
                self.set_status(f"Axis range applied.\nCSV used: {self.last_csv_path}")
        except Exception as exc:
            self._show_critical("Axis Range Error", str(exc))
            self.set_status(f"Error: {exc}")

    def _is_checked(self, key: str) -> bool:
        return self.plot_checks[key].isChecked()

    def _active_plot_order(self, fig_no: int) -> list[str]:
        default_order = FIG_SERIES_ORDER[fig_no]
        active_keys = [key for key in self.plot_draw_order[fig_no] if self._is_checked(key)]
        for key in default_order:
            if self._is_checked(key) and key not in active_keys:
                active_keys.append(key)
        self.plot_draw_order[fig_no] = active_keys
        return active_keys

    def _resolve_plot_kwargs(self, meta: dict, settings: PlotSettings) -> dict:
        kwargs = {}
        for name, value in meta["style"].items():
            if isinstance(value, str) and hasattr(settings, value):
                kwargs[name] = getattr(settings, value)
            else:
                kwargs[name] = value
        return kwargs

    def _series_has_data(self, values: np.ndarray) -> bool:
        return bool(np.isfinite(values).any())

    def _reset_checkbox_availability(self):
        for cb in self.plot_checks.values():
            cb.setEnabled(True)

    def update_checkbox_availability(self, series: dict[str, np.ndarray]):
        for key, cb in self.plot_checks.items():
            series_key = PLOT_KEY_TO_SERIES[key]
            has_data = bool(self._series_has_data(series[series_key]))

            cb.blockSignals(True)
            cb.setEnabled(has_data)
            if not has_data:
                cb.setChecked(False)
            cb.blockSignals(False)

        for fig_no, keys in FIG_SERIES_ORDER.items():
            self.plot_draw_order[fig_no] = [
                key for key in self.plot_draw_order[fig_no]
                if bool(self.plot_checks[key].isChecked())
            ]

    def _plot_series_in_order(self, fig_no: int, ax1, ax2, series: dict[str, np.ndarray], settings: PlotSettings):
        t = series["t"]
        handles = []
        y2_values = []
        zorder = 1

        for key in self._active_plot_order(fig_no):
            meta = FIG_SERIES_META[key]
            values = series[meta["series"]]
            if not self._series_has_data(values):
                continue

            axis = ax1 if meta["axis"] == "ax1" else ax2
            kwargs = self._resolve_plot_kwargs(meta, settings)
            kwargs["label"] = meta["label"]
            kwargs["zorder"] = zorder
            handle, = axis.plot(t, values, **kwargs)
            handles.append(handle)
            if meta["axis"] == "ax2":
                y2_values.append(values)
            zorder += 1

        return handles, y2_values

    def select_txt_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Raw Data file (.txt)",
            "",
            "Text Files (*.txt);;All Files (*.*)",
        )
        if not path:
            return
        self.txt_edit.setText(path)
        txt_path = Path(path)
        if not self.csv_save_edit.text().strip():
            self.csv_save_edit.setText(str(txt_path.with_suffix(".csv")))
        if not self.title_edit.text().strip():
            self.title_edit.setText(txt_path.stem)

    def select_existing_csv_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Existing CSV file",
            "",
            "CSV Files (*.csv);;All Files (*.*)",
        )
        if not path:
            return
        self.csv_load_edit.setText(path)
        if not self.title_edit.text().strip():
            self.title_edit.setText(Path(path).stem)

    def select_csv_save_path(self):
        suggested = self.csv_save_edit.text().strip() or "converted_plot_data.csv"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Select CSV Save Path",
            suggested,
            "CSV Files (*.csv)",
        )
        if path:
            self.csv_save_edit.setText(path)

    def clear_figures(self):
        self.plot1.clear()
        self.plot2.clear()
        self.plot3.clear()
        self.figures = {1: None, 2: None, 3: None}
        self.last_csv_path = None
        self.last_series = None
        self.last_base_title = None
        self._reset_checkbox_availability()
        self._initialize_plot_draw_order()
        self.set_status("Figures cleared")

    def apply_plot_ratio(self, settings: PlotSettings):
        self.plot1.set_ratio(settings.ratio_w, settings.ratio_h)
        self.plot2.set_ratio(settings.ratio_w, settings.ratio_h)
        self.plot3.set_ratio(settings.ratio_w, settings.ratio_h)

    def _effective_title(self, preferred_stem: str) -> str:
        return self.title_edit.text().strip() or preferred_stem

    def run_from_txt(self):
        txt_path = self.txt_edit.text().strip()
        csv_save_path = self.csv_save_edit.text().strip()
        if not txt_path:
            self._show_warning("Missing TXT", "Select Data File!!")
            return
        if not Path(txt_path).is_file():
            self._show_warning("Invalid TXT", "Invalid Data File!!")
            return
        if not csv_save_path:
            self._show_warning("Missing CSV path", "Selec CSV Save Path!!")
            return

        try:
            saved_csv = convert_txt_to_csv(txt_path, csv_save_path)
            self.csv_load_edit.setText(saved_csv)
            base_title = self._effective_title(Path(txt_path).stem)
            self.generate_plots_from_csv(saved_csv, base_title, converted_from_txt=True)
        except Exception as exc:
            self._show_critical("Error", f"Error Occur During data converting!!\n\n{exc}")
            self.set_status(f"Error: {exc}")

    def run_from_csv(self):
        csv_path = self.csv_load_edit.text().strip()
        if not csv_path:
            self._show_warning("Missing CSV", "Select CSV File!!")
            return
        if not Path(csv_path).is_file():
            self._show_warning("Invalid CSV", "Invalid CSV File!!")
            return

        try:
            base_title = self._effective_title(Path(csv_path).stem)
            self.generate_plots_from_csv(csv_path, base_title, converted_from_txt=False)
        except Exception as exc:
            self._show_critical("Error", f"Error Occur During Plot Data!! \n\n{exc}")
            self.set_status(f"Error: {exc}")

    def generate_plots_from_csv(self, csv_path: str, base_title: str, converted_from_txt: bool):
        numeric_df, start_row = load_numeric_csv_auto(csv_path)
        series = extract_series(numeric_df)

        self.update_checkbox_availability(series)

        self.last_csv_path = csv_path
        self.last_series = series
        self.last_base_title = base_title

        self._render_all_plots()
        self.tabs.setCurrentIndex(0)

        available_series = sorted(
            key for key, values in series.items()
            if bool(self._series_has_data(values))
        )
        source_msg = "TXT converted to CSV and plots generated" if converted_from_txt else "CSV loaded and plots generated"
        self.set_status(
            f"{source_msg} successfully.\n"
            f"CSV used: {csv_path}\n"
            f"Detected numeric data start row: {start_row + 1}\n"
            f"Available data series: {', '.join(available_series)}"
        )

    def refresh_plots_from_cache(self):
        if self.last_series is None or self.last_base_title is None:
            return

        self._render_all_plots()

        if self.last_csv_path:
            self.set_status(f"Plot visibility updated.\nCSV used: {self.last_csv_path}")

    def _render_all_plots(self):
        settings = self.current_plot_settings()
        self.apply_plot_ratio(settings)

        fig1 = self.build_fig1(self.last_series, f"{self.last_base_title} {PLOT_TITLES['rpm_torque']}", settings)
        fig2 = self.build_fig2(self.last_series, f"{self.last_base_title} {PLOT_TITLES['rpm_current']}", settings)
        fig3 = self.build_fig3(self.last_series, f"{self.last_base_title} {PLOT_TITLES['rpm_only']}", settings)

        self.figures[1] = fig1
        self.figures[2] = fig2
        self.figures[3] = fig3

        self.plot1.canvas.draw_idle()
        self.plot2.canvas.draw_idle()
        self.plot3.canvas.draw_idle()

    def save_single_figure(self, fig_no: int):
        fig = self.figures.get(fig_no)
        if fig is None:
            self._show_info("No Figure", "Draw Figure First!!")
            return

        base_title = self.title_edit.text().strip() or "Plot"
        suffix_map = {
            1: PLOT_TITLES["rpm_torque"],
            2: PLOT_TITLES["rpm_current"],
            3: PLOT_TITLES["rpm_only"],
        }
        default_name = make_safe_filename(f"{base_title}_{suffix_map[fig_no]}.png")
        path, _ = QFileDialog.getSaveFileName(
            self,
            f"Save Figure {fig_no}",
            default_name,
            "PNG Files (*.png);;All Files (*.*)",
        )
        if not path:
            return

        fig.savefig(path, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.04)
        self.set_status(f"Figure {fig_no} saved: {path}")

    def _prepare_figure(self, plot_widget: PlotCanvas, settings: PlotSettings) -> Figure:
        fig = plot_widget.figure
        fig.clear()
        fig.set_facecolor("white")
        width_in = BASE_FIGURE_HEIGHT_IN * (settings.ratio_w / settings.ratio_h)
        fig.set_size_inches(width_in, BASE_FIGURE_HEIGHT_IN, forward=True)
        fig.subplots_adjust(left=0.09, right=0.65, top=0.86, bottom=0.15)
        return fig

    def _place_legend(self, fig: Figure, handles, labels):
        fig.legends.clear()
        if not handles:
            return

        base_fontsize = fig.get_size_inches()[1] * 2.0
        legend_fontsize = max(6.0, base_fontsize * LEGEND_FONT_SCALE)
        fig.legend(
            handles,
            labels,
            loc="upper right",
            bbox_to_anchor=(0.795, 0.87),
            bbox_transform=fig.transFigure,
            frameon=True,
            borderaxespad=0.0,
            fontsize=legend_fontsize,
            labelspacing=0.35,
            borderpad=0.45,
            handlelength=1.8,
        )

    def _apply_common_axes(self, ax1, ax2, t: np.ndarray, title: str, y2_label: str, y2_ylim, settings: PlotSettings):
        x_auto = (float(t[0]), float(t[-1]))
        y1_auto = RPM_YLIM

        x_lim = self._resolve_axis_limits(x_auto[0], x_auto[1], settings.x_min, settings.x_max)
        y1_lim = self._resolve_axis_limits(y1_auto[0], y1_auto[1], settings.y1_min, settings.y1_max)
        y2_lim = self._resolve_axis_limits(y2_ylim[0], y2_ylim[1], settings.y2_min, settings.y2_max)

        ax1.set_xlim(x_lim)
        ax1.set_ylim(y1_lim)
        if settings.y1_min is None and settings.y1_max is None:
            ax1.set_yticks(RPM_YTICKS)
        else:
            ax1.yaxis.set_major_locator(AutoLocator())

        ax2.set_ylim(y2_lim)
        ax1.set_xlabel("Time(sec)")
        ax1.set_ylabel("Speed(RPM)", labelpad=6)
        ax2.set_ylabel(y2_label, labelpad=10)
        ax1.set_title(title, pad=settings.title_pad)
        if settings.show_grid:
            ax1.grid(True)

    def build_fig1(self, series: dict[str, np.ndarray], title: str, settings: PlotSettings) -> Figure:
        fig = self._prepare_figure(self.plot1, settings)
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()

        handles, y2_values = self._plot_series_in_order(1, ax1, ax2, series, settings)
        torque_ylim = finite_ylim(np.concatenate(y2_values), pad_ratio=0.10) if y2_values else (-1.0, 1.0)
        self._apply_common_axes(ax1, ax2, series["t"], title, "Torque(mNm)", torque_ylim, settings)
        self._place_legend(fig, handles, [h.get_label() for h in handles])
        return fig

    def build_fig2(self, series: dict[str, np.ndarray], title: str, settings: PlotSettings) -> Figure:
        fig = self._prepare_figure(self.plot2, settings)
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()

        handles, y2_values = self._plot_series_in_order(2, ax1, ax2, series, settings)
        current_ylim = finite_ylim(np.concatenate(y2_values), pad_ratio=0.20) if y2_values else (-1.0, 1.0)
        self._apply_common_axes(ax1, ax2, series["t"], title, "Current(mA)", current_ylim, settings)
        self._place_legend(fig, handles, [h.get_label() for h in handles])
        return fig

    def build_fig3(self, series: dict[str, np.ndarray], title: str, settings: PlotSettings) -> Figure:
        fig = self._prepare_figure(self.plot3, settings)
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()

        handles, y2_values = self._plot_series_in_order(3, ax1, ax2, series, settings)
        torque_cmd_ylim = finite_ylim(np.concatenate(y2_values), pad_ratio=0.50) if y2_values else (-1.0, 1.0)
        self._apply_common_axes(ax1, ax2, series["t"], title, "Torque(mNm)", torque_cmd_ylim, settings)
        self._place_legend(fig, handles, [h.get_label() for h in handles])
        return fig


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
