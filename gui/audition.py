"""单句试听播放。

不依赖 QtMultimedia：PySide6-Essentials 没带这个模块，装 Addons 要多占
300 MB 以上，而 Windows 自带的 winsound 播 WAV 已经够用。
切片和取帧交给 core.audio，这里只管播放状态。

winsound 没有播放完成回调，所以按 WAV 实际时长挂一个定时器把状态收回来，
否则自然播完后 playing 仍是 True，用户要按两次空格才能重播。
"""

import wave

from PySide6.QtCore import QObject, QTimer, Signal

try:
    import winsound
except ImportError:  # 非 Windows 平台只禁用试听，不影响其他功能
    winsound = None


def wav_seconds(path):
    try:
        with wave.open(path, "rb") as r:
            rate = r.getframerate()
            return r.getnframes() / rate if rate else 0.0
    except Exception:
        return 0.0


class Player(QObject):
    failed = Signal(str)
    finished = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.playing = False
        self.loop = False
        self.path = ""
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._on_finished)

    @property
    def available(self):
        return winsound is not None

    def play(self, path, loop=False):
        if winsound is None:
            self.failed.emit("当前系统没有 winsound，无法试听")
            return False
        try:
            self.stop()
            flags = winsound.SND_FILENAME | winsound.SND_ASYNC
            if loop:
                flags |= winsound.SND_LOOP
            winsound.PlaySound(path, flags)
            self.path = path
            self.loop = bool(loop)
            self.playing = True
            if not loop:
                # 留 150 ms 余量，避免定时器比实际播放早触发
                self._timer.start(int(wav_seconds(path) * 1000) + 150)
            return True
        except Exception as e:
            self.failed.emit(f"播放失败：{e}")
            return False

    def stop(self):
        self._timer.stop()
        if winsound is not None and self.playing:
            try:
                winsound.PlaySound(None, 0)
            except Exception:
                pass
        self.playing = False

    def _on_finished(self):
        self.playing = False
        self.finished.emit()
