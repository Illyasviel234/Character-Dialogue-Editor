"""输出文件名净化。

角色名等用户输入会直接进入成片和报告的文件名，必须过滤 Windows 非法字符，
避免出现意外的子目录或写入失败。
"""

import re

_ILLEGAL = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED = ({"CON", "PRN", "AUX", "NUL"}
             | {f"COM{i}" for i in range(1, 10)}
             | {f"LPT{i}" for i in range(1, 10)})


def safe_filename(name, fallback="output", max_len=80):
    """把任意字符串变成可用的文件名片段。"""
    text = _ILLEGAL.sub("_", str(name or "")).strip(" .")
    if not text:
        return fallback
    if text.upper() in _RESERVED:
        text = "_" + text
    return text[:max_len]


def safe_stem(path):
    """从路径取不含扩展名的文件名，并净化。

    不用 os.path.basename：Windows 会把 "a:b" 里的 "a:" 当成盘符而丢掉前半段。
    """
    import os
    name = str(path or "").replace("\\", "/").split("/")[-1]
    return safe_filename(os.path.splitext(name)[0])
