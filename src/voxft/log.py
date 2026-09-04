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


def _fmt_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024 or unit == "GB":
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}GB"


class LogBar:
    """tqdm 替身：把进度条转成日志行（供 hf_hub_download 的 tqdm_class 等）。"""

    def __init__(self, *args, **kwargs):
        self._total = kwargs.get("total") or 0
        self._desc = (kwargs.get("desc") or "").strip(" :")
        self._callback = kwargs.pop("log", None)
        self.n = kwargs.get("initial") or 0
        self._last_msg = ""
        self._last_emit = 0.0

    def update(self, n: int = 1) -> None:
        self.n += n
        now = time.monotonic()
        if self._total and now - self._last_emit < 3 and self.n < self._total:
            return  # 限速：每 3 秒一行，避免刷屏
        self._last_emit = now
        pct = 100 * self.n / self._total if self._total else 0
        msg = (f"{self._desc} " if self._desc else "") + (
            f"{_fmt_size(self.n)}/{_fmt_size(self._total)} ({pct:.0f}%)"
            if self._total else f"{self.n} 条")
        if msg != self._last_msg and self._callback:
            self._last_msg = msg
            self._callback(msg)

    def close(self) -> None:
        if self._total and self.n >= self._total:
            self._last_msg = ""
        self.update(0)

    def set_description(self, desc=None, refresh=True) -> None:
        self._desc = (desc or "").strip(" :")

    def set_postfix(self, ordered_dict=None, refresh=True, **kwargs) -> None:
        pass

    def set_postfix_str(self, s="", refresh=True) -> None:
        pass

    def set_transfer_postfix_str(self, s="", refresh=True) -> None:
        pass

    def update_transfer(self, n: int = 0) -> None:
        pass

    @property
    def total(self):
        return self._total

    @total.setter
    def total(self, value) -> None:
        self._total = value or 0

    @property
    def format_dict(self) -> dict:
        return {"rate": None}

    def refresh(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        self.close()
