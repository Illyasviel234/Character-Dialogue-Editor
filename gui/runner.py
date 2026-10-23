import os
import re
import subprocess
import sys

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, Signal


class Step:
    """一个流水线阶段。progress_re 用于从日志里抠出该阶段的细分进度。"""

    def __init__(self, script, args, weight, label, progress_re=None):
        self.script = script
        self.args = list(args)
        self.weight = weight
        self.label = label
        self.progress_re = progress_re


class Runner(QObject):
    log = Signal(str)
    stage = Signal(str)
    progress = Signal(int)
    finished = Signal(bool, str)

    def __init__(self, root, parent=None):
        super().__init__(parent)
        self.root = root
        self.steps = []
        self.i = 0
        self.base = 0
        self.proc = None
        self.cancelled = False
        self.buf = b""
        self.duration = 1.0

    def running(self):
        return self.proc is not None

    def start(self, steps, duration):
        self.steps = steps
        self.duration = max(float(duration), 1.0)
        self.i = 0
        self.base = 0
        self.cancelled = False
        self.progress.emit(0)
        self._next()

    def cancel(self):
        """终止任务。Windows 上要连子进程一起杀，否则 cut.py 里的 ffmpeg 会继续编码。"""
        self.cancelled = True
        p = self.proc
        if p is None:
            self.finished.emit(False, "已取消")
            return
        pid = int(p.processId())
        if sys.platform == "win32" and pid:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                           capture_output=True)
        p.kill()

    def _next(self):
        if self.cancelled:
            self.finished.emit(False, "已取消")
            return
        if self.i >= len(self.steps):
            self.progress.emit(100)
            self.finished.emit(True, "完成")
            return

        st = self.steps[self.i]
        self.stage.emit(st.label)
        py = os.path.join(self.root, ".venv", "Scripts", "python.exe")

        p = QProcess(self)
        p.setWorkingDirectory(os.path.join(self.root, "work"))
        p.setProcessChannelMode(QProcess.MergedChannels)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONIOENCODING", "utf-8")
        env.insert("PYTHONUNBUFFERED", "1")
        env.insert("PYTHONUTF8", "1")
        p.setProcessEnvironment(env)
        p.readyReadStandardOutput.connect(lambda: self._read(p))
        p.finished.connect(lambda code, _s: self._done(code))

        self.proc = p
        self.buf = b""
        p.start(py, ["-u", st.script] + st.args)

    def _read(self, p):
        self.buf += bytes(p.readAllStandardOutput())
        while b"\n" in self.buf:
            line, self.buf = self.buf.split(b"\n", 1)
            text = line.decode("utf-8", "replace").rstrip("\r")
            if not text.strip():
                continue
            self.log.emit(text)
            if self.i < len(self.steps):
                st = self.steps[self.i]
                if st.progress_re:
                    m = re.search(st.progress_re, text)
                    if m:
                        frac = min(1.0, float(m.group(1)) / self.duration)
                        self.progress.emit(int(self.base + st.weight * frac))

    def _done(self, code):
        self.proc = None
        if self.cancelled:
            self.finished.emit(False, "已取消")
            return
        st = self.steps[self.i]
        self.base += st.weight
        if code != 0:
            self.finished.emit(False, f"{st.label}失败（退出码 {code}）")
            return
        self.progress.emit(int(self.base))
        self.i += 1
        self._next()
