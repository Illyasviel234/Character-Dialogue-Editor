"""单句试听播放。

不依赖 QtMultimedia：PySide6-Essentials 没带这个模块，装 Addons 要多占
300 MB 以上，而 Windows 自带的 winsound 播 WAV 已经够用。
切片和取帧交给 core.audio，这里只管播放状态。
"""

from PySide6.QtCore import QObject, Signal

try:
    import winsound
except ImportError:  # 非 Windows 平台只禁用试听，不影响其他功能
    winsound = None


class Player(QObject):
    failed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.playing = False
        self.loop = False
        self.path = ""

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
            return True
        except Exception as e:
            self.failed.emit(f"播放失败：{e}")
            return False

    def stop(self):
        if winsound is not None and self.playing:
            try:
                winsound.PlaySound(None, 0)
            except Exception:
                pass
        self.playing = False
