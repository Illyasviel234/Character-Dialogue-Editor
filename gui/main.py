"""角色台词剪辑器主窗口。

界面只做三件事：把分析结果摆成可核对的表、让人听到并确认每一句、
把确认过的片段导出去。所有判定逻辑都在 work/core/ 里，GUI 和 CLI 共用。
"""

import json
import os
import shutil
import sys
from datetime import datetime

from PySide6.QtCore import QEvent, Qt, QTimer
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFrame,
    QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow,
    QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QScrollArea,
    QSpinBox, QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.path.join(ROOT, "work")
sys.path.insert(0, WORK)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import audio, manifest, media, naming, paths, scoring, segmenter  # noqa: E402
from audition import Player  # noqa: E402
from runner import Runner, Step  # noqa: E402
from theme import QSS  # noqa: E402

SCALES = {"保持源规格": "native", "1080p": "1920:1080", "720p": "1280:720"}
FRAME_MODES = {"转恒定帧率": "cfr", "保留源时间戳": "preserve"}
FILTERS = ("全部", "阈值以上", "阈值附近", "仅勾选")
SORTS = ("时间", "相似度", "时长", "勾选")
VIDEO_EXT = (".mp4", ".mkv", ".mov", ".webm", ".m4v", ".avi")


def local_models():
    found = []
    if os.path.isdir(paths.MODELS):
        prefix = "models--Systran--faster-whisper-"
        for name in os.listdir(paths.MODELS):
            if name.startswith(prefix):
                found.append(name[len(prefix):])
    return sorted(found) or ["large-v3"]


def fmt_ts(t):
    return f"{int(t // 60):02d}:{t % 60:05.2f}"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("角色台词剪辑器")
        self.resize(1320, 900)
        self.setMinimumSize(1060, 700)

        self.video = ""
        self.sid = ""
        self.source = {}
        self.info = {}
        self.duration = 0.0
        self.segments = []
        self.emb = None
        self.emb_ids = []
        self.scores = {}
        self.checked = set()
        self.seed_ids = []
        self.mode = None
        self.run_id = ""
        self.last_output = ""
        self.view = []
        self.row_ids = []
        self._out_manual = False
        self._filling = False
        self._scroll_pending = False

        self.player = Player(self)
        self.player.failed.connect(self.append_log)

        self._sum_timer = QTimer(self)
        self._sum_timer.setSingleShot(True)
        self._sum_timer.timeout.connect(self.refresh_summary)

        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.timeout.connect(self.preview_current)

        self._build_ui()

        self.runner = Runner(ROOT, self)
        self.runner.log.connect(self.append_log)
        self.runner.stage.connect(self.set_stage)
        self.runner.progress.connect(self.progress.setValue)
        self.runner.finished.connect(self.on_finished)

        self._load_default_video()

    # ---------- 界面 ----------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        row = QHBoxLayout(central)
        row.setContentsMargins(14, 14, 14, 12)
        row.setSpacing(14)

        row.addWidget(self._build_left())

        right = QVBoxLayout()
        right.setSpacing(8)
        right.addLayout(self._build_toolbar())

        split = QSplitter(Qt.Vertical)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(3000)
        self.log_view.setPlaceholderText("运行日志会显示在这里")
        split.addWidget(self.log_view)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["", "时间码", "时长", "台词", "相似度"])
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(False)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.itemChanged.connect(self.on_item_changed)
        self.table.itemSelectionChanged.connect(self.on_selection_changed)
        self.table.doubleClicked.connect(lambda _i: self.play_current())
        self.table.installEventFilter(self)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.Fixed)
        hh.setSectionResizeMode(1, QHeaderView.Fixed)
        hh.setSectionResizeMode(2, QHeaderView.Fixed)
        hh.setSectionResizeMode(3, QHeaderView.Stretch)
        hh.setSectionResizeMode(4, QHeaderView.Fixed)
        self.table.setColumnWidth(0, 34)
        self.table.setColumnWidth(1, 136)
        self.table.setColumnWidth(2, 66)
        self.table.setColumnWidth(4, 76)
        split.addWidget(self.table)
        split.setSizes([190, 500])
        right.addWidget(split, 1)
        right.addWidget(self._build_player())

        holder = QWidget()
        holder.setLayout(right)
        row.addWidget(holder, 1)

    def _build_left(self):
        content = QWidget()
        lv = QVBoxLayout(content)
        lv.setContentsMargins(0, 0, 8, 0)
        lv.setSpacing(7)

        title = QLabel("角色台词剪辑器")
        title.setStyleSheet("font-size:17px; font-weight:500; color:#eae8e2;")
        lv.addWidget(title)
        sub = QLabel("按角色声纹筛选动画片段 · 全程本地离线")
        sub.setObjectName("hint")
        lv.addWidget(sub)

        g = QGroupBox("输入")
        v = QVBoxLayout(g)
        v.setSpacing(6)
        hr = QHBoxLayout()
        self.video_edit = QLineEdit()
        self.video_edit.setPlaceholderText("选择视频文件")
        self.video_edit.setReadOnly(True)
        hr.addWidget(self.video_edit, 1)
        btn = QPushButton("选择")
        btn.setFixedWidth(58)
        btn.clicked.connect(self.pick_video)
        hr.addWidget(btn)
        v.addLayout(hr)

        hr2 = QHBoxLayout()
        hr2.setSpacing(7)
        self.model_combo = QComboBox()
        self.model_combo.addItems(local_models())
        self.device_combo = QComboBox()
        self.device_combo.addItems(["cuda", "cpu"])
        hr2.addWidget(QLabel("模型"))
        hr2.addWidget(self.model_combo, 1)
        hr2.addWidget(QLabel("设备"))
        hr2.addWidget(self.device_combo, 1)
        v.addLayout(hr2)

        hr_track = QHBoxLayout()
        hr_track.setSpacing(7)
        self.track_combo = QComboBox()
        self.track_combo.setEnabled(False)
        self.track_combo.currentIndexChanged.connect(self._on_track_changed)
        hr_track.addWidget(QLabel("音轨"))
        hr_track.addWidget(self.track_combo, 1)
        v.addLayout(hr_track)

        self.dur_label = QLabel("时长 —")
        self.dur_label.setObjectName("stat")
        self.dur_label.setWordWrap(True)
        v.addWidget(self.dur_label)
        lv.addWidget(g)

        g2 = QGroupBox("角色识别")
        v2 = QVBoxLayout(g2)
        v2.setSpacing(6)
        hr_role = QHBoxLayout()
        hr_role.addWidget(QLabel("角色名"))
        self.role_edit = QLineEdit("碧翠丝")
        self.role_edit.setPlaceholderText("用于报告与输出文件名")
        hr_role.addWidget(self.role_edit, 1)
        v2.addLayout(hr_role)

        self.kw_edit = QLineEdit("かしら,ベティ")
        self.kw_edit.setPlaceholderText("台词特征词，逗号分隔")
        self.kw_edit.setToolTip("用于自动挑选种子台词，词越独特越好")
        v2.addWidget(self.kw_edit)

        hr3 = QHBoxLayout()
        hr3.addWidget(QLabel("相似度阈值"))
        self.thr_spin = QDoubleSpinBox()
        self.thr_spin.setRange(0.0, 1.0)
        self.thr_spin.setSingleStep(0.01)
        self.thr_spin.setDecimals(2)
        self.thr_spin.setValue(0.66)
        self.thr_spin.valueChanged.connect(self._debounce_summary)
        hr3.addWidget(self.thr_spin, 1)
        v2.addLayout(hr3)

        hr4 = QHBoxLayout()
        b2 = QPushButton("按关键词重算")
        b2.clicked.connect(self.rescore_by_keyword)
        b3 = QPushButton("用勾选项作种子")
        b3.clicked.connect(self.rescore_by_selection)
        hr4.addWidget(b2)
        hr4.addWidget(b3)
        v2.addLayout(hr4)
        lv.addWidget(g2)

        g3 = QGroupBox("剪辑参数")
        v3 = QVBoxLayout(g3)
        v3.setSpacing(6)
        grid = QHBoxLayout()
        grid.setSpacing(7)
        self.gap_spin = self._spin(0.0, 10.0, 0.1, 2.0, 1)
        self.pad_h = self._spin(0.0, 2.0, 0.05, 0.25, 2)
        self.pad_t = self._spin(0.0, 2.0, 0.05, 0.30, 2)
        for label, widget in (("合并间隔", self.gap_spin), ("前留白", self.pad_h),
                              ("后留白", self.pad_t)):
            box = QVBoxLayout()
            lab = QLabel(label)
            lab.setObjectName("hint")
            box.addWidget(lab)
            box.addWidget(widget)
            grid.addLayout(box)
        v3.addLayout(grid)
        for sp in (self.gap_spin, self.pad_h, self.pad_t):
            sp.valueChanged.connect(self._debounce_summary)

        hr5 = QHBoxLayout()
        hr5.setSpacing(7)
        self.scale_combo = QComboBox()
        self.scale_combo.addItems(list(SCALES))
        self.crf_spin = QSpinBox()
        self.crf_spin.setRange(14, 30)
        self.crf_spin.setValue(20)
        hr5.addWidget(QLabel("输出"))
        hr5.addWidget(self.scale_combo, 1)
        hr5.addWidget(QLabel("CRF"))
        hr5.addWidget(self.crf_spin)
        v3.addLayout(hr5)

        hr_frame = QHBoxLayout()
        hr_frame.setSpacing(7)
        self.frame_combo = QComboBox()
        self.frame_combo.addItems(list(FRAME_MODES))
        hr_frame.addWidget(QLabel("帧率"))
        hr_frame.addWidget(self.frame_combo, 1)
        v3.addLayout(hr_frame)
        self.vfr_hint = self._hint("")
        self.vfr_hint.setVisible(False)
        v3.addWidget(self.vfr_hint)

        hr7 = QHBoxLayout()
        hr7.setSpacing(7)
        self.out_edit = QLineEdit()
        self.out_edit.setPlaceholderText("默认 output/<源名>_<角色>_clip.mp4")
        self.out_edit.setToolTip("成片保存位置，可直接编辑或点「浏览」另存")
        self.out_edit.textEdited.connect(self._on_output_edited)
        hr7.addWidget(self.out_edit, 1)
        out_btn = QPushButton("浏览")
        out_btn.setFixedWidth(58)
        out_btn.clicked.connect(self.pick_output)
        hr7.addWidget(out_btn)
        v3.addLayout(hr7)
        self.role_edit.textChanged.connect(lambda _t: self._refresh_output())
        lv.addWidget(g3)

        lv.addStretch(1)

        self.btn_analyze = QPushButton("开始分析")
        self.btn_analyze.setObjectName("primary")
        self.btn_analyze.setMinimumHeight(34)
        self.btn_analyze.clicked.connect(self.start_analysis)
        lv.addWidget(self.btn_analyze)

        hr6 = QHBoxLayout()
        hr6.setSpacing(7)
        self.btn_render = QPushButton("生成成片")
        self.btn_render.setMinimumHeight(32)
        self.btn_render.setEnabled(False)
        self.btn_render.clicked.connect(self.start_render)
        self.btn_open = QPushButton("打开目录")
        self.btn_open.setFixedWidth(76)
        self.btn_open.setMinimumHeight(32)
        self.btn_open.clicked.connect(self.open_output)
        self.btn_stop = QPushButton("停止")
        self.btn_stop.setFixedWidth(58)
        self.btn_stop.setMinimumHeight(32)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(lambda: self.runner.cancel())
        hr6.addWidget(self.btn_render, 1)
        hr6.addWidget(self.btn_open)
        hr6.addWidget(self.btn_stop)
        lv.addLayout(hr6)

        self.btn_labels = QPushButton("保存人工样本")
        self.btn_labels.setToolTip("把当前勾选与临界未勾选存成标注，供 eval_threshold.py 评估")
        self.btn_labels.clicked.connect(self.save_labels)
        lv.addWidget(self.btn_labels)

        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(6)
        lv.addWidget(self.progress)

        self.stage_label = QLabel("就绪")
        self.stage_label.setObjectName("stage")
        lv.addWidget(self.stage_label)

        self.summary_label = QLabel("—")
        self.summary_label.setObjectName("stat")
        self.summary_label.setWordWrap(True)
        lv.addWidget(self.summary_label)

        scroll = QScrollArea()
        scroll.setWidget(content)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setFixedWidth(368)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        return scroll

    def _build_toolbar(self):
        bar = QHBoxLayout()
        bar.setSpacing(7)
        bar.addWidget(QLabel("显示"))
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(list(FILTERS))
        self.filter_combo.currentIndexChanged.connect(lambda _i: self.fill_table())
        bar.addWidget(self.filter_combo)

        bar.addWidget(QLabel("排序"))
        self.sort_combo = QComboBox()
        self.sort_combo.addItems(list(SORTS))
        self.sort_combo.currentIndexChanged.connect(lambda _i: self.fill_table())
        bar.addWidget(self.sort_combo)

        self.order_btn = QPushButton("降序")
        self.order_btn.setCheckable(True)
        self.order_btn.setFixedWidth(58)
        self.order_btn.toggled.connect(self._on_order_toggled)
        bar.addWidget(self.order_btn)

        bar.addStretch(1)
        self.count_label = QLabel("")
        self.count_label.setObjectName("hint")
        bar.addWidget(self.count_label)
        return bar

    def _build_player(self):
        box = QFrame()
        box.setObjectName("player")
        h = QHBoxLayout(box)
        h.setContentsMargins(8, 8, 8, 8)
        h.setSpacing(10)

        self.thumb = QLabel("—")
        self.thumb.setFixedSize(168, 95)
        self.thumb.setAlignment(Qt.AlignCenter)
        self.thumb.setStyleSheet(
            "background:#151513; border:1px solid #33322e; border-radius:6px;"
            "color:#6b6a65; font-size:12px;")
        h.addWidget(self.thumb)

        info = QVBoxLayout()
        info.setSpacing(4)
        self.audio_time = QLabel("未选择台词")
        self.audio_time.setStyleSheet("color:#5dcaa5; font-size:12.5px;")
        info.addWidget(self.audio_time)
        self.audio_text = QLabel("在表格里选一行，按空格试听")
        self.audio_text.setWordWrap(True)
        self.audio_text.setStyleSheet("color:#c9c7c0; font-size:13px;")
        info.addWidget(self.audio_text, 1)
        self.audio_hint = QLabel("空格 试听 · C 勾选 · L 循环 · Esc 停止")
        self.audio_hint.setObjectName("hint")
        info.addWidget(self.audio_hint)
        h.addLayout(info, 1)

        btns = QVBoxLayout()
        btns.setSpacing(5)
        self.btn_play = QPushButton("试听")
        self.btn_play.setFixedWidth(76)
        self.btn_play.clicked.connect(self.toggle_play)
        btns.addWidget(self.btn_play)
        self.loop_check = QCheckBox("循环")
        self.loop_check.setToolTip("循环播放当前句，按 L 切换")
        btns.addWidget(self.loop_check)
        self.btn_audio_stop = QPushButton("停止")
        self.btn_audio_stop.setFixedWidth(76)
        self.btn_audio_stop.clicked.connect(self.stop_audio)
        btns.addWidget(self.btn_audio_stop)
        btns.addStretch(1)
        h.addLayout(btns)

        if not self.player.available:
            self.btn_play.setEnabled(False)
            self.audio_hint.setText("当前系统不支持试听（缺少 winsound）")
        return box

    def _spin(self, lo, hi, step, val, dec):
        sp = QDoubleSpinBox()
        sp.setRange(lo, hi)
        sp.setSingleStep(step)
        sp.setDecimals(dec)
        sp.setValue(val)
        # QDoubleSpinBox 的默认最小宽度会撑破左栏，固定住让它跟着列宽走
        sp.setFixedWidth(96)
        return sp

    def _hint(self, text):
        lab = QLabel(text)
        lab.setObjectName("hint")
        lab.setWordWrap(True)
        return lab

    # ---------- 输出路径 ----------

    def _default_output(self):
        if not self.video:
            return ""
        stem = naming.safe_stem(self.video)
        role = naming.safe_filename(self.role_edit.text() or "目标角色", fallback="role")
        return os.path.join(paths.OUTPUT, f"{stem}_{role}_clip.mp4")

    @staticmethod
    def _shown_path(path):
        """工作区内的路径显示成相对路径，和输入框保持一致；区外仍显示绝对路径。"""
        try:
            rel = os.path.relpath(path, ROOT)
            if not rel.startswith(".."):
                return rel
        except ValueError:
            pass
        return path

    def _set_output(self, path):
        self.out_edit.setText(self._shown_path(path))
        self.out_edit.setToolTip(path)

    def _refresh_output(self, force=False):
        """角色名或视频变化时跟着更新默认路径；用户手动改过就不再覆盖。"""
        if not force and self._out_manual:
            return
        path = self._default_output()
        if path:
            self._set_output(path)
        self._out_manual = False

    def _on_output_edited(self, _text):
        self._out_manual = True

    def pick_output(self):
        start = self.out_edit.text().strip() or self._default_output() or paths.OUTPUT
        if not os.path.isabs(start):
            start = os.path.join(ROOT, start)
        path, _ = QFileDialog.getSaveFileName(
            self, "选择成片保存位置", start, "MP4 视频 (*.mp4)")
        if not path:
            return
        if not path.lower().endswith(".mp4"):
            path += ".mp4"
        self._set_output(os.path.abspath(path))
        self._out_manual = True

    def resolve_output_path(self):
        """返回可用的绝对输出路径；用户取消或路径非法时返回空串。

        相对路径按项目根目录解析，而不是当前工作目录——界面里显示的就是相对路径，
        两种解析方式混用会让人以为写进了 output/ 实际却落到了别处。
        """
        text = self.out_edit.text().strip()
        if text:
            path = text if os.path.isabs(text) else os.path.join(ROOT, text)
            path = os.path.abspath(os.path.expanduser(path))
        else:
            path = self._default_output()
        if not path:
            QMessageBox.warning(self, "提示", "请先选择视频。")
            return ""
        if not path.lower().endswith(".mp4"):
            path += ".mp4"

        if os.path.normcase(path) == os.path.normcase(os.path.abspath(self.video)):
            QMessageBox.warning(self, "提示",
                                "输出路径不能与源视频相同，否则会覆盖原片。请换一个位置。")
            return ""

        parent = os.path.dirname(path)
        try:
            os.makedirs(parent, exist_ok=True)
        except OSError as e:
            QMessageBox.warning(self, "提示", f"无法创建输出目录：{e}")
            return ""

        if os.path.exists(path):
            r = QMessageBox.question(
                self, "确认覆盖", f"文件已存在，是否覆盖？\n{path}",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if r != QMessageBox.Yes:
                return ""
        return path

    # ---------- 任务与缓存 ----------

    def _load_default_video(self):
        if not os.path.isdir(paths.INPUT):
            return
        for name in sorted(os.listdir(paths.INPUT)):
            if name.lower().endswith(VIDEO_EXT):
                self.set_video(os.path.join(paths.INPUT, name))
                return

    def set_video(self, path):
        """切换视频时立即清空旧状态，避免用上一部片子的时间码切新片子。"""
        self._reset_state()
        self.stop_audio()

        self.video = path
        try:
            shown = os.path.relpath(path, ROOT)
            if shown.startswith(".."):
                shown = os.path.basename(path)
        except Exception:
            shown = os.path.basename(path)
        self.video_edit.setText(shown)
        self.video_edit.setToolTip(path)
        self._refresh_output(force=True)

        self.info = media.probe(path)
        if not self.info:
            self.dur_label.setText("无法读取视频信息")
            self.track_combo.clear()
            self.track_combo.setEnabled(False)
            return

        self.duration = float(self.info.get("duration") or 0)
        self.sid = manifest.source_id(path)
        self.source = manifest.make_source(path, self.info)
        self._fill_tracks()

        bits = [fmt_ts(self.duration)]
        if self.info.get("width"):
            bits.append(f'{self.info["width"]}×{self.info["height"]}')
        if self.info.get("fps"):
            bits.append(f'{self.info["fps"]:.3f} fps')
        if self.info.get("audio_streams"):
            bits.append(f'{self.info["audio_streams"]} 音轨')
        self.dur_label.setText(" · ".join(bits))

        if self.info.get("is_vfr"):
            self.vfr_hint.setText(
                f'源为可变帧率（r {self.info.get("fps"):.3f} / avg '
                f'{self.info.get("avg_fps"):.3f}）。选「保留源时间戳」可维持原节奏，'
                f'但时长偏差可能超过一帧。')
            self.vfr_hint.setVisible(True)
        else:
            self.vfr_hint.setVisible(False)

        self._load_job()

    def _fill_tracks(self):
        self.track_combo.blockSignals(True)
        self.track_combo.clear()
        tracks = self.info.get("audio_tracks") or []
        for i, t in enumerate(tracks):
            label = f"第 {i + 1} 条 · {t.get('codec') or '?'} {t.get('channels')}ch"
            if t.get("language"):
                label += f" · {t['language']}"
            self.track_combo.addItem(label, i)
        used = manifest.audio_track(self.sid) if self.sid else 0
        if 0 <= used < self.track_combo.count():
            self.track_combo.setCurrentIndex(used)
        self.track_combo.setEnabled(len(tracks) > 1)
        self.track_combo.blockSignals(False)

    def _on_track_changed(self, index):
        if index < 0:
            return
        have = manifest.audio_track(self.sid) if self.sid else 0
        if index != have and self.run_id:
            self.append_log(
                f"音轨已切到第 {index + 1} 条，当前分析结果来自第 {have + 1} 条，"
                f"需要重新点「开始分析」才能与画面一致。")

    def audio_track(self):
        idx = self.track_combo.currentIndex()
        return idx if idx >= 0 else 0

    def pick_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择视频", paths.INPUT,
            "视频文件 (*.mp4 *.mkv *.mov *.webm *.m4v *.avi)")
        if path:
            self.set_video(path)

    def _reset_state(self):
        self.segments = []
        self.run_id = ""
        self.emb = None
        self.emb_ids = []
        self.scores = {}
        self.checked = set()
        self.seed_ids = []
        self.view = []
        self.row_ids = []
        self._filling = True
        self.table.setRowCount(0)
        self._filling = False
        self.btn_render.setEnabled(False)
        self.summary_label.setText("—")
        self.count_label.setText("")
        self.thumb.setText("—")
        self.audio_time.setText("未选择台词")
        self.audio_text.setText("在表格里选一行，按空格试听")
        self._clear_audition_cache()

    def _clear_audition_cache(self):
        shutil.rmtree(os.path.join(paths.CACHE, "audition"), ignore_errors=True)

    def _load_job(self):
        if not self.sid:
            return
        man = manifest.load_manifest(self.sid)
        ok, why = manifest.matches(man, self.source)
        if not ok:
            self.append_log(f"该视频尚无可用分析结果（{why}），请点「开始分析」。")
            return

        rid = manifest.active_run(self.sid)
        if not rid:
            pending = manifest.pending_run(self.sid)
            if pending:
                self.append_log(f"检测到未完成的分析运行 {pending}（转写有、声纹缺失或校验未过），"
                                f"请重新执行分析。")
            else:
                self.append_log("该视频尚无可用分析结果，请点「开始分析」。")
            return

        rdir = manifest.run_dir(self.sid, rid)
        try:
            with open(os.path.join(rdir, manifest.TRANSCRIPT), encoding="utf-8") as f:
                tr = json.load(f)
            self.segments = self._clean(tr.get("segments") or [])
            self.emb, self.emb_ids = scoring.load_embeddings(
                os.path.join(rdir, manifest.EMBEDDINGS))
            self.run_id = rid
            self.append_log(f"已载入分析结果：{len(self.segments)} 个语段（运行 {rid}）")
            self.rescore_by_keyword(quiet=True)
        except Exception as e:
            self.append_log(f"载入失败：{e}")
            self._reset_state()

    def _clean(self, segs):
        """只剔除空文本和 initial_prompt 回声。

        no_speech_prob 在动画上不可靠：BGM 和音效会把有台词的段也推高，
        早期按它过滤会误删目标角色的台词，因此不再使用。
        """
        out = []
        for s in segs:
            text = (s.get("text") or "").strip()
            if not text or "日本語のアニメの会話" in text:
                continue
            out.append(s)
        return out

    # ---------- 打分 ----------

    def rescore_by_keyword(self, quiet=False):
        if not self.segments:
            if not quiet:
                self.append_log("还没有转写结果。")
            return
        keywords = self.kw_edit.text().replace("，", ",").split(",")
        seeds = scoring.keyword_seeds(self.segments, keywords)
        if not seeds:
            # 零命中时不能留一张空表，否则用户既看不到台词也无法手动选种子
            self.seed_ids = []
            self.scores = {}
            self.checked = set()
            self.fill_table()
            self.refresh_summary()
            if not quiet:
                self.append_log("特征词没有命中任何台词，已列出全部语段。"
                                "可手动勾选几句后点「用勾选项作种子」。")
            return
        self.seed_ids = seeds
        self._apply_scores(quiet)

    def rescore_by_selection(self):
        if not self.segments:
            return
        if not self.checked:
            self.append_log("先勾选几句确认属于目标角色的台词作为种子。")
            return
        self.seed_ids = sorted(self.checked)
        self._apply_scores(False)

    def _apply_scores(self, quiet):
        if self.emb is None:
            return
        arr = scoring.template_scores(self.emb, self.emb_ids, self.seed_ids)
        self.scores = {i: float(s) for i, s in zip(self.emb_ids, arr)}
        thr = self.thr_spin.value()
        valid = {s["id"] for s in self.segments}
        self.checked = {i for i, s in self.scores.items() if s >= thr and i in valid}
        self.fill_table()
        self.refresh_summary()
        if not quiet:
            stat = scoring.threshold_report(self.scores, thr)
            self.append_log(f"种子 {len(self.seed_ids)} 句 → 阈值 {thr:.2f} "
                            f"命中 {len(self.checked)} 句")
            self.append_log(f"提示：{thr:.2f}–{thr + 0.08:.2f} 之间的 {stat['near']} 句"
                            f"最易误判，建议逐条核对后再导出。")

    # ---------- 表格：过滤与排序 ----------

    def _visible_indices(self):
        idx = list(range(len(self.segments)))
        mode = self.filter_combo.currentText()
        thr = self.thr_spin.value()
        if mode == "阈值以上":
            idx = [i for i in idx if self.scores.get(self.segments[i]["id"], -1) >= thr]
        elif mode == "阈值附近":
            idx = [i for i in idx
                   if thr <= self.scores.get(self.segments[i]["id"], -1) < thr + 0.08]
        elif mode == "仅勾选":
            idx = [i for i in idx if self.segments[i]["id"] in self.checked]

        desc = self.order_btn.isChecked()
        key = self.sort_combo.currentText()
        if key == "相似度":
            idx.sort(key=lambda i: self.scores.get(self.segments[i]["id"], -1), reverse=desc)
        elif key == "时长":
            idx.sort(key=lambda i: self.segments[i]["end"] - self.segments[i]["start"],
                     reverse=desc)
        elif key == "勾选":
            idx.sort(key=lambda i: self.segments[i]["id"] in self.checked, reverse=desc)
        else:
            idx.sort(key=lambda i: self.segments[i]["start"], reverse=desc)
        return idx

    def _on_order_toggled(self, desc):
        self.order_btn.setText("降序" if desc else "升序")
        self.fill_table()

    def fill_table(self):
        self._filling = True
        t = self.table
        t.setUpdatesEnabled(False)
        self.view = self._visible_indices()
        self.row_ids = [self.segments[i]["id"] for i in self.view]
        t.setRowCount(len(self.view))
        for r, i in enumerate(self.view):
            s = self.segments[i]
            raw = self.scores.get(s["id"])
            sc = -1.0 if raw is None else raw

            chk = QTableWidgetItem()
            chk.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            chk.setCheckState(Qt.Checked if s["id"] in self.checked else Qt.Unchecked)
            t.setItem(r, 0, chk)

            tc = QTableWidgetItem(f"{fmt_ts(s['start'])} – {fmt_ts(s['end'])}")
            tc.setForeground(QColor("#5dcaa5"))
            t.setItem(r, 1, tc)

            t.setItem(r, 2, QTableWidgetItem(f"{s['end'] - s['start']:.1f}s"))

            tx = QTableWidgetItem(s["text"])
            tx.setToolTip(s["text"])
            if sc < 0.35:
                tx.setForeground(QColor("#77756d"))
            t.setItem(r, 3, tx)

            sv = QTableWidgetItem("—" if raw is None else f"{sc:.3f}")
            sv.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            if raw is None:
                sv.setForeground(QColor("#8a8880"))
            elif sc >= 0.6:
                sv.setForeground(QColor("#5dcaa5"))
            elif sc >= 0.4:
                sv.setForeground(QColor("#c9a227"))
            else:
                sv.setForeground(QColor("#8a8880"))
            t.setItem(r, 4, sv)
        t.setUpdatesEnabled(True)
        self._filling = False
        self._scroll_pending = True
        self.btn_render.setEnabled(bool(self.segments))
        self.count_label.setText(
            f"{len(self.view)}/{len(self.segments)} 行 · 已选 {len(self.checked)} 句")

    def _scroll_first(self):
        if not self._scroll_pending or not self.row_ids:
            return
        first = next((r for r, i in enumerate(self.row_ids) if i in self.checked), None)
        if first is not None:
            self.table.scrollToItem(self.table.item(first, 0), QTableWidget.PositionAtTop)
        self._scroll_pending = False

    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(0, self._scroll_first)

    def on_item_changed(self, item):
        if self._filling or item.column() != 0:
            return
        row = item.row()
        if not 0 <= row < len(self.row_ids):
            return
        sid = self.row_ids[row]
        if item.checkState() == Qt.Checked:
            self.checked.add(sid)
        else:
            self.checked.discard(sid)
        self.count_label.setText(
            f"{len(self.view)}/{len(self.segments)} 行 · 已选 {len(self.checked)} 句")
        self._debounce_summary()

    def _debounce_summary(self):
        self._sum_timer.start(180)

    def refresh_summary(self):
        if not self.segments:
            return
        segs = segmenter.build_segments(
            sorted(self.checked), self.segments, self.duration,
            self.gap_spin.value(), self.pad_h.value(), self.pad_t.value())
        total = sum(s["dur"] for s in segs)
        pct = total / self.duration * 100 if self.duration else 0
        self.summary_label.setText(
            f"已选 {len(self.checked)} 句 → {len(segs)} 个片段，"
            f"成片 {int(total // 60)}分{total % 60:04.1f}秒（占原片 {pct:.1f}%）")

    # ---------- 试听 ----------

    def current_segment(self):
        row = self.table.currentRow()
        if 0 <= row < len(self.row_ids):
            sid = self.row_ids[row]
            return next((s for s in self.segments if s["id"] == sid), None)
        return None

    def _audition_files(self, seg):
        """切出单句音频、取一帧画面，都缓存在 work/.cache/audition/。"""
        d = os.path.join(paths.CACHE, "audition", self.sid or "tmp")
        wav = os.path.join(d, f"{seg['id']:04d}.wav")
        png = os.path.join(d, f"{seg['id']:04d}.png")
        if not (os.path.exists(wav) and os.path.getsize(wav) > 44):
            src = os.path.join(manifest.job_dir(self.sid), "audio16k.wav")
            if os.path.exists(src):
                audio.slice_wav(src, seg["start"], seg["end"], wav, 0.05, 0.05)
            else:
                audio.extract_wav(self.video, seg["start"], seg["end"], wav,
                                  self.audio_track())
        if not os.path.exists(png):
            audio.grab_frame(self.video, (seg["start"] + seg["end"]) / 2, png)
        return wav, (png if os.path.exists(png) else None)

    def _show_frame(self, png):
        pm = QPixmap(png)
        if pm.isNull():
            return
        self.thumb.setPixmap(pm.scaled(self.thumb.size(), Qt.KeepAspectRatio,
                                       Qt.SmoothTransformation))

    def _describe(self, seg):
        span = f"{fmt_ts(seg['start'])} – {fmt_ts(seg['end'])}（{seg['end'] - seg['start']:.1f}s）"
        sc = self.scores.get(seg["id"])
        if sc is not None:
            span += f"　相似度 {sc:.3f}"
        self.audio_time.setText(span)
        self.audio_text.setText(seg["text"])

    def on_selection_changed(self):
        self._preview_timer.start(180)

    def preview_current(self):
        seg = self.current_segment()
        if not seg:
            return
        self._describe(seg)
        try:
            _wav, png = self._audition_files(seg)
        except Exception as e:
            self.append_log(f"预览失败：{e}")
            return
        if png:
            self._show_frame(png)

    def play_current(self):
        seg = self.current_segment()
        if not seg:
            self.append_log("先在表格里选一行。")
            return
        self.play_segment(seg)

    def play_segment(self, seg, loop=None):
        if loop is None:
            loop = self.loop_check.isChecked()
        try:
            wav, png = self._audition_files(seg)
        except Exception as e:
            self.append_log(f"试听失败：{e}")
            return
        if png:
            self._show_frame(png)
        self._describe(seg)
        if self.player.play(wav, loop):
            self.btn_play.setText("暂停")
            self.audio_hint.setText(
                f"正在播放{fmt_ts(seg['start'])}–{fmt_ts(seg['end'])}"
                + ("（循环）" if loop else ""))

    def toggle_play(self):
        if self.player.playing:
            self.stop_audio()
        else:
            self.play_current()

    def stop_audio(self):
        self.player.stop()
        self.btn_play.setText("试听")
        self.audio_hint.setText("空格 试听 · C 勾选 · L 循环 · Esc 停止")

    def toggle_current(self):
        row = self.table.currentRow()
        if row < 0:
            return
        item = self.table.item(row, 0)
        if item is None:
            return
        item.setCheckState(Qt.Unchecked if item.checkState() == Qt.Checked else Qt.Checked)

    def eventFilter(self, obj, event):
        if obj is self.table and event.type() == QEvent.KeyPress:
            key = event.key()
            if key == Qt.Key_Space:
                self.toggle_play()
                return True
            if key in (Qt.Key_Return, Qt.Key_Enter):
                self.play_current()
                return True
            if key == Qt.Key_C:
                self.toggle_current()
                return True
            if key == Qt.Key_L:
                self.loop_check.setChecked(not self.loop_check.isChecked())
                return True
            if key == Qt.Key_Escape:
                self.stop_audio()
                return True
        return super().eventFilter(obj, event)

    # ---------- 人工样本 ----------

    def save_labels(self):
        if not self.run_id or not self.scores:
            QMessageBox.warning(self, "提示", "需要先有分析结果和打分，才能保存标注。")
            return
        thr = self.thr_spin.value()
        positive = sorted(self.checked)
        # 负样本取"复核后仍判为不是目标角色"的临界句，全片其余低分句没有信息量
        negative = sorted(i for i, s in self.scores.items()
                          if i not in self.checked and s >= thr - 0.10)
        role = (self.role_edit.text() or "目标角色").strip()
        manifest.save_labels(self.sid, role, self.run_id, positive, negative, thr)
        self.append_log(f"已保存人工样本：正 {len(positive)} 句、负 {len(negative)} 句 "
                        f"→ {manifest.artifact(self.sid, manifest.LABELS)}")
        QMessageBox.information(
            self, "已保存",
            f"正样本 {len(positive)} 句，负样本 {len(negative)} 句。\n\n"
            f"可用 eval_threshold.py 基于这份标注重新评估阈值与负样本抑制效果。")

    # ---------- 运行 ----------

    def append_log(self, text):
        self.log_view.appendPlainText(text)

    def set_stage(self, text):
        self.stage_label.setText(text)

    def _busy(self, busy):
        self.btn_analyze.setEnabled(not busy)
        self.btn_render.setEnabled(not busy and bool(self.segments))
        self.btn_stop.setEnabled(busy)

    def start_analysis(self):
        if not self.video or not os.path.exists(self.video):
            QMessageBox.warning(self, "提示", "请先选择视频文件。")
            return
        if not self.info.get("audio_streams"):
            QMessageBox.warning(self, "提示", "该视频没有音轨。")
            return

        self.stop_audio()
        self.mode = "analysis"
        self.log_view.clear()
        self._busy(True)
        self.progress.setValue(0)
        steps = [
            Step("extract_audio.py", ["--video", self.video,
                                      "--audio-track", str(self.audio_track())],
                 5, "抽取音频"),
            Step("transcribe.py", ["--video", self.video,
                                   "--model", self.model_combo.currentText(),
                                   "--device", self.device_combo.currentText()],
                 55, "日语转写", r"t=(\d+)s"),
            Step("voiceprint.py", ["--video", self.video], 40, "声纹提取"),
        ]
        self.runner.start(steps, self.duration)

    def start_render(self):
        if not self.segments or not self.checked:
            QMessageBox.warning(self, "提示", "没有可导出的片段，请先分析并勾选台词。")
            return
        if not self.run_id:
            QMessageBox.warning(self, "提示", "当前没有可用的分析运行，请先点「开始分析」。")
            return

        segs = segmenter.build_segments(
            sorted(self.checked), self.segments, self.duration,
            self.gap_spin.value(), self.pad_h.value(), self.pad_t.value())
        if not segs:
            QMessageBox.warning(self, "提示", "当前勾选无法组成有效片段。")
            return

        for s in segs:
            vals = [self.scores[i] for i in s["ids"] if i in self.scores]
            s["score"] = round(sum(vals) / len(vals), 4) if vals else None

        out_path = self.resolve_output_path()
        if not out_path:
            return

        self.stop_audio()
        role = (self.role_edit.text() or "目标角色").strip()
        total = round(sum(s["dur"] for s in segs), 2)
        payload = {
            "version": manifest.VERSION,
            "source_id": self.sid,
            "run_id": self.run_id,
            "source": self.source,
            "role": role,
            "created": datetime.now().isoformat(timespec="seconds"),
            "threshold": self.thr_spin.value(),
            "model": self.model_combo.currentText(),
            "audio_track": self.audio_track(),
            "seed_ids": sorted(self.seed_ids),
            "merge_gap": self.gap_spin.value(),
            "pad_head": self.pad_h.value(),
            "pad_tail": self.pad_t.value(),
            "total_sec": total,
            "segments": segs,
        }
        manifest.atomic_write_json(manifest.artifact(self.sid, manifest.SELECTED), payload)

        self.last_output = out_path
        out_dir = os.path.dirname(out_path)
        frame_mode = FRAME_MODES[self.frame_combo.currentText()]

        self.mode = "render"
        self._busy(True)
        self.progress.setValue(0)
        self.append_log(f"导出 {len(segs)} 个片段，共 {total:.1f} 秒 → {out_path}")
        steps = [
            Step("cut.py", ["--video", self.video,
                            "--scale", SCALES[self.scale_combo.currentText()],
                            "--crf", str(self.crf_spin.value()),
                            "--audio-track", str(self.audio_track()),
                            "--vfr", frame_mode,
                            "--output", out_path], 88, "切割拼接"),
            Step("gen_report.py", ["--video", self.video,
                                   "--out-dir", out_dir], 12, "生成清单"),
        ]
        self.runner.start(steps, 100.0)

    def on_finished(self, ok, msg):
        self._busy(False)
        if not ok:
            self.set_stage(msg)
            self.append_log(f"[失败] {msg}")
            return
        if self.mode == "analysis":
            self.set_stage("分析完成")
            self._load_job()
        else:
            self.set_stage("成片已生成")
            self.append_log(f"[完成] {self.last_output}")

    def open_output(self):
        target = os.path.dirname(self.last_output) if self.last_output else paths.OUTPUT
        if not os.path.isdir(target):
            target = paths.OUTPUT if os.path.isdir(paths.OUTPUT) else ROOT
        os.startfile(target)

    def closeEvent(self, event):
        self.stop_audio()
        if self.runner.running():
            self.runner.cancel()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(QSS)
    app.setApplicationName("角色台词剪辑器")
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
