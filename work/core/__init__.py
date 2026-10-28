"""项目核心层：路径、媒体探测、任务清单、切点合并、声纹打分、批量队列。

命令行脚本与图形界面都从这里取实现，避免两套逻辑分叉。
"""

from . import (audio, cli, manifest, media, naming, paths, queue,  # noqa: F401
               scoring, segmenter)

__all__ = ["audio", "cli", "manifest", "media", "naming", "paths", "queue",
           "scoring", "segmenter"]
