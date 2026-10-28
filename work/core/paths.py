"""项目统一路径。所有模块只从这里取路径，不再各自推导。"""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORK = os.path.join(ROOT, "work")
CORE = os.path.join(WORK, "core")
JOBS = os.environ.get("CLIP_JOBS_DIR") or os.path.join(WORK, "jobs")
CACHE = os.environ.get("CLIP_CACHE_DIR") or os.path.join(WORK, ".cache")
INPUT = os.path.join(ROOT, "input")
OUTPUT = os.path.join(ROOT, "output")
MODELS = os.path.join(ROOT, "models")
TOOLS = os.path.join(ROOT, "tools")
TESTS = os.path.join(ROOT, "tests")

FFMPEG_DIR = os.path.join(TOOLS, "ffmpeg", "bin")
FFMPEG = os.path.join(FFMPEG_DIR, "ffmpeg.exe")
FFPROBE = os.path.join(FFMPEG_DIR, "ffprobe.exe")

VENV_PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")

if not os.path.exists(FFMPEG):
    FFMPEG = "ffmpeg"
if not os.path.exists(FFPROBE):
    FFPROBE = "ffprobe"


def ensure_dirs():
    for d in (JOBS, OUTPUT, INPUT, CACHE):
        os.makedirs(d, exist_ok=True)
