from __future__ import annotations

import threading

import numpy as np
import psutil


class MemorySampler:
    def __init__(self) -> None:
        self.process = psutil.Process()
        self.baseline = self.process.memory_info().rss
        self.peak = self.baseline
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self.stop_event.wait(0.005):
            self.peak = max(self.peak, self.process.memory_info().rss)

    def __enter__(self) -> MemorySampler:
        self.thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop_event.set()
        self.thread.join()
        self.peak = max(self.peak, self.process.memory_info().rss)


def timing_summary(times_ms: list[float]) -> dict:
    return {
        "mean_ms": float(np.mean(times_ms)),
        "std_ms": float(np.std(times_ms, ddof=1)),
        "median_ms": float(np.median(times_ms)),
        "p95_ms": float(np.percentile(times_ms, 95)),
        "throughput_per_second": float(1000 / np.mean(times_ms)),
        "repetitions": len(times_ms),
    }
