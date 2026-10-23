"""项目核心层：路径、媒体探测、任务清单、切点合并、声纹打分。

命令行脚本与图形界面都从这里取实现，避免两套逻辑分叉。
"""

from . import cli, manifest, media, paths, scoring, segmenter  # noqa: F401

__all__ = ["cli", "manifest", "media", "paths", "scoring", "segmenter"]
