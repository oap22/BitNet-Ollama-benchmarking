"""Monitor peak RAM usage of a process and its children."""

import threading
import time

import psutil


class RAMMonitor:
    """Tracks peak RSS of a process (by PID) in a background thread."""

    def __init__(self, pid: int, interval: float = 0.1):
        self.pid = pid
        self.interval = interval
        self.peak_mb: float = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)

    def _poll(self):
        while not self._stop.is_set():
            try:
                proc = psutil.Process(self.pid)
                rss = proc.memory_info().rss
                # Include children (bitnet.cpp spawns sub-processes)
                for child in proc.children(recursive=True):
                    try:
                        rss += child.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                mb = rss / (1024 * 1024)
                if mb > self.peak_mb:
                    self.peak_mb = mb
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            time.sleep(self.interval)

    def start(self):
        self._thread.start()

    def stop(self) -> float:
        self._stop.set()
        self._thread.join(timeout=2)
        return self.peak_mb


def get_process_ram_mb(pid: int) -> float:
    """Snapshot current RSS of a process + children in MB."""
    try:
        proc = psutil.Process(pid)
        rss = proc.memory_info().rss
        for child in proc.children(recursive=True):
            try:
                rss += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return rss / (1024 * 1024)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0.0
