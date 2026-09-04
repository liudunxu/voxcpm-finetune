"""运行日志：内存环形缓冲（页面实时展示）+ 文件落盘（logs/voxft.log）。"""
from __future__ import annotations

import threading
import time
from collections import deque
from pathlib import Path

from .paths import ROOT

LOG_DIR = ROOT / "logs"
_LOCK = threading.Lock()


def _write_file(line: str) -> None:
    LOG_DIR.mkdir(exist_ok=True)
    with _LOCK:
        with (LOG_DIR / "voxft.log").open("a", encoding="utf-8") as f:
            f.write(line + "\n")


class OpLog:
    """单操作日志缓冲；实例本身可直接作为 progress 回调。"""

    def __init__(self, name: str, maxlen: int = 500):
        self.name = name
        self.lines: deque[str] = deque(maxlen=maxlen)

    def __call__(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        self.lines.append(line)
        _write_file(f"[{self.name}] {line}")

    def text(self, tail: int = 100) -> str:
        return "\n".join(list(self.lines)[-tail:]) or "（暂无日志）"


_LOGS: dict[str, OpLog] = {}


def get_log(name: str) -> OpLog:
    if name not in _LOGS:
        _LOGS[name] = OpLog(name)
    return _LOGS[name]


def file_tail(lines: int = 200) -> str:
    p = LOG_DIR / "voxft.log"
    if not p.exists():
        return "（暂无日志）"
    return "".join(p.read_text(encoding="utf-8", errors="replace")
                   .splitlines(keepends=True)[-lines:])
