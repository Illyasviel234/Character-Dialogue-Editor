import json
import os
import subprocess
import sys
from datetime import datetime

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QApplication, QComboBox, QDoubleSpinBox, QFileDialog, QGroupBox,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow, QMessageBox,
    QPlainTextEdit, QProgressBar, QPushButton, QSpinBox, QSplitter,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.path.join(ROOT, "work")
sys.path.insert(0, WORK)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import manifest, media, paths, scoring, segmenter  # noqa: E402
from runner import Runner, Step  # noqa: E402
from theme import QSS  # noqa: E402

SCALES = {"1080p": "1920:1080", "720p": "1280:720", "保持源规格": "native"}


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
        self.resize(1240, 810)
        self.setMinimumSize(1020, 680)

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
        self.last_output = ""
        self._filling = False
        self._scroll_pending = False

        self._sum_timer = QTimer(self)
        self._sum_timer.setSingleShot(True)
        self._sum_timer.timeout.connect(self.refresh_summary)

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

        left = QWidget()
        left.setFixedWidth(350)
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(9)

        title = QLabel("角色台词剪辑器")
        title.setStyleSheet("font-size:17px; font-weight:500; color:#eae8e2;")
        lv.addWidget(title)
        sub = QLabel("按角色声纹筛选动画片段 · 全程本地离线")
        sub.setObjectName("hint")
        lv.addWidget(sub)

        g = QGroupBox("输入")
        v = QVBoxLayout(g)
        v.setSpacing(7)
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

        self.dur_label = QLabel("时长 —")
        self.dur_label.setObjectName("stat")
        self.dur_label.setWordWrap(True)
        v.addWidget(self.dur_label)
        lv.addWidget(g)

        g2 = QGroupBox("角色识别")
        v2 = QVBoxLayout(g2)
        v2.setSpacing(7)
        hr_role = QHBoxLayout()
        hr_role.addWidget(QLabel("角色名"))
        self.role_edit = QLineEdit("碧翠丝")
        self.role_edit.setPlaceholderText("用于报告与输出文件名")
        hr_role.addWidget(self.role_edit, 1)
        v2.addLayout(hr_role)

        v2.addWidget(QLabel("台词特征词（逗号分隔）"))
        self.kw_edit = QLineEdit("かしら,ベティ")
        self.kw_edit.setPlaceholderText("用于自动挑选种子台词")
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
        v3.setSpacing(7)
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
        self.btn_stop = QPushButton("停止")
        self.btn_stop.setFixedWidth(64)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(lambda: self.runner.cancel())
        hr6.addWidget(self.btn_render, 1)
        hr6.addWidget(self.btn_stop)
        lv.addLayout(hr6)

        self.btn_open = QPushButton("打开输出目录")
        self.btn_open.clicked.connect(self.open_output)
        lv.addWidget(self.btn_open)

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

        row.addWidget(left)

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
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.itemChanged.connect(self.on_item_changed)
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
        split.setSizes([250, 530])
        row.addWidget(split, 1)

    def _spin(self, lo, hi, step, val, dec):
        sp = QDoubleSpinBox()
        sp.setRange(lo, hi)
        sp.setSingleStep(step)
        sp.setDecimals(dec)
        sp.setValue(val)
        return sp

    # ---------- 任务与缓存 ----------

    def _load_default_video(self):
        if not os.path.isdir(paths.INPUT):
            return
        for name in sorted(os.listdir(paths.INPUT)):
            if name.lower().endswith((".mp4", ".mkv", ".mov", ".webm", ".m4v")):
                self.set_video(os.path.join(paths.INPUT, name))
                return

    def set_video(self, path):
        """切换视频时立即清空旧状态，避免用上一部片子的时间码切新片子。"""
        self._reset_state()

        self.video = path
        try:
            shown = os.path.relpath(path, ROOT)
            if shown.startswith(".."):
                shown = os.path.basename(path)
        except Exception:
            shown = os.path.basename(path)
        self.video_edit.setText(shown)
        self.video_edit.setToolTip(path)

        self.info = media.probe(path)
        if not self.info:
            self.dur_label.setText("无法读取视频信息")
            return

        self.duration = float(self.info.get("duration") or 0)
        self.sid = manifest.source_id(path)
        self.source = manifest.make_source(path, self.info)

        bits = [fmt_ts(self.duration)]
        if self.info.get("width"):
            bits.append(f'{self.info["width"]}×{self.info["height"]}')
        if self.info.get("fps"):
            bits.append(f'{self.info["fps"]:.3f} fps')
        if self.info.get("audio_streams"):
            bits.append(f'{self.info["audio_streams"]} 音轨')
        self.dur_label.setText(" · ".join(bits))

        self._load_job()

    def pick_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择视频", paths.INPUT, "视频文件 (*.mp4 *.mkv *.mov *.webm *.m4v)")
        if path:
            self.set_video(path)

    def _reset_state(self):
        self.segments = []
        self.emb = None
        self.emb_ids = []
        self.scores = {}
        self.checked = set()
        self.seed_ids = []
        self._filling = True
        self.table.setRowCount(0)
        self._filling = False
        self.btn_render.setEnabled(False)
        self.summary_label.setText("—")

    def _load_job(self):
        if not self.sid:
            return
        man = manifest.load_manifest(self.sid)
        ok, why = manifest.matches(man, self.source)

        if not ok and manifest.migrate_legacy(self.video, self.sid, self.source):
            man = manifest.load_manifest(self.sid)
            ok, why = manifest.matches(man, self.source)
            if ok:
                self.append_log("已把旧版缓存迁入当前任务目录。")

        if not ok:
            self.append_log(f"该视频尚无可用分析结果（{why}），请点「开始分析」。")
            return
        if not manifest.has_analysis(self.sid):
            self.append_log("分析结果不完整，请重新运行分析。")
            return

        try:
            with open(manifest.artifact(self.sid, manifest.TRANSCRIPT), encoding="utf-8") as f:
                tr = json.load(f)
            self.segments = self._clean(tr.get("segments") or [])
            self.emb, self.emb_ids = scoring.load_embeddings(
                manifest.artifact(self.sid, manifest.EMBEDDINGS))
            self.append_log(f"已载入分析结果：{len(self.segments)} 个语段。")
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
            if not quiet:
                self.append_log("特征词没有命中任何台词，请换几个词。")
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

    def fill_table(self):
        self._filling = True
        t = self.table
        t.setUpdatesEnabled(False)
        t.setRowCount(len(self.segments))
        for r, s in enumerate(self.segments):
            sc = self.scores.get(s["id"], -1.0)

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

            sv = QTableWidgetItem(f"{sc:.3f}" if sc >= 0 else "—")
            sv.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            if sc >= 0.6:
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

    def _scroll_first(self):
        if not self._scroll_pending or not self.segments:
            return
        first = next((r for r, s in enumerate(self.segments) if s["id"] in self.checked), None)
        if first is not None:
            self.table.scrollToItem(self.table.item(first, 0), QTableWidget.PositionAtTop)
        self._scroll_pending = False

    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(0, self._scroll_first)

    def on_item_changed(self, item):
        if self._filling or item.column() != 0:
            return
        sid = self.segments[item.row()]["id"]
        if item.checkState() == Qt.Checked:
            self.checked.add(sid)
        else:
            self.checked.discard(sid)
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

        self.mode = "analysis"
        self.log_view.clear()
        self._busy(True)
        self.progress.setValue(0)
        steps = [
            Step("extract_audio.py", ["--video", self.video], 5, "抽取音频"),
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

        segs = segmenter.build_segments(
            sorted(self.checked), self.segments, self.duration,
            self.gap_spin.value(), self.pad_h.value(), self.pad_t.value())
        if not segs:
            QMessageBox.warning(self, "提示", "当前勾选无法组成有效片段。")
            return

        for s in segs:
            vals = [self.scores[i] for i in s["ids"] if i in self.scores]
            s["score"] = round(sum(vals) / len(vals), 4) if vals else None

        role = (self.role_edit.text() or "目标角色").strip()
        total = round(sum(s["dur"] for s in segs), 2)
        payload = {
            "version": manifest.VERSION,
            "source_id": self.sid,
            "source": self.source,
            "role": role,
            "created": datetime.now().isoformat(timespec="seconds"),
            "threshold": self.thr_spin.value(),
            "model": self.model_combo.currentText(),
            "merge_gap": self.gap_spin.value(),
            "pad_head": self.pad_h.value(),
            "pad_tail": self.pad_t.value(),
            "total_sec": total,
            "segments": segs,
        }
        manifest.atomic_write_json(manifest.artifact(self.sid, manifest.SELECTED), payload)

        stem = os.path.splitext(os.path.basename(self.video))[0]
        self.last_output = os.path.join(paths.OUTPUT, f"{stem}_{role}_clip.mp4")

        self.mode = "render"
        self._busy(True)
        self.progress.setValue(0)
        self.append_log(f"导出 {len(segs)} 个片段，共 {total:.1f} 秒 → {self.last_output}")
        steps = [
            Step("cut.py", ["--video", self.video,
                            "--scale", SCALES[self.scale_combo.currentText()],
                            "--crf", str(self.crf_spin.value()),
                            "--output", self.last_output], 88, "切割拼接"),
            Step("gen_report.py", ["--video", self.video], 12, "生成清单"),
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
        os.startfile(paths.OUTPUT if os.path.isdir(paths.OUTPUT) else ROOT)

    def closeEvent(self, event):
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
