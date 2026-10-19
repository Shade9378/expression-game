from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from constants import CLASSES


@dataclass(frozen=True)
class ChallengeSettings:
    threshold: float = 0.5
    hold_seconds: float = 1.2
    timeout_seconds: float = 25.0
    level: int = 0
    max_gap: float = 0.25
    max_frame_age: float = 0.45
    minimum_samples: int = 8
    timed: bool = True

    def __post_init__(self) -> None:
        values = (
            self.threshold,
            self.hold_seconds,
            self.timeout_seconds,
            self.max_gap,
            self.max_frame_age,
        )
        if (
            not all(math.isfinite(value) for value in values)
            or not 0.35 <= self.threshold <= 0.9
            or not 0.5 <= self.hold_seconds <= 5
            or not self.hold_seconds + 1 <= self.timeout_seconds <= 120
            or self.level not in range(3)
            or not 0 < self.max_gap <= 0.5
            or not 0 < self.max_frame_age <= 1
            or self.minimum_samples < 3
            or not isinstance(self.timed, bool)
        ):
            raise ValueError("Invalid imitation game settings")


@dataclass(frozen=True)
class ImitationResult:
    target: str
    outcome: str
    points: int
    elapsed: float
    held_seconds: float
    valid_samples: int


class ImitationRound:
    def __init__(self, target: str, now: float, settings: ChallengeSettings) -> None:
        if target not in CLASSES or not math.isfinite(now):
            raise ValueError("Invalid target or start timestamp")
        self.target, self.index, self.settings = target, CLASSES.index(target), settings
        self.started_at = now
        self.last_timestamp = -math.inf
        self.last_match: float | None = None
        self.track_id: int | None = None
        self.held_seconds = 0.0
        self.held_samples = 0
        self.valid_samples = 0
        self.probability = 0.0
        self.history: deque = deque(maxlen=15)
        self.status = "Waiting for one face"
        self.result: ImitationResult | None = None

    @property
    def stability(self) -> float | None:
        if len(self.history) < 2:
            return None
        values = np.asarray(self.history)
        return float(1 - np.abs(np.diff(values, axis=0)).sum(axis=1).mean() / 2)

    def reset_hold(self, status: str = "Hold reset") -> None:
        self.held_seconds = 0.0
        self.held_samples = 0
        self.last_match = None
        self.status = status

    def finish(self, outcome: str, now: float) -> ImitationResult:
        if self.result is not None:
            return self.result
        if outcome not in {"success", "timeout", "skipped", "cancelled", "unavailable"}:
            raise ValueError("Invalid imitation outcome")
        elapsed = max(0.0, now - self.started_at)
        points = 0
        if outcome == "success":
            points = 100 * (self.settings.level + 1)
            if self.settings.timed:
                points += round(25 * max(0.0, 1 - elapsed / self.settings.timeout_seconds))
        self.result = ImitationResult(
            self.target, outcome, points, elapsed, self.held_seconds, self.valid_samples
        )
        self.status = (
            "Target held - game complete" if outcome == "success" else outcome.capitalize()
        )
        return self.result

    def tick(self, now: float) -> ImitationResult | None:
        if self.result is not None:
            return self.result
        if self.settings.timed and now - self.started_at >= self.settings.timeout_seconds:
            return self.finish("timeout", now)
        if self.last_match is not None and now - self.last_match > self.settings.max_gap:
            self.reset_hold("Tracking interruption - hold reset")
            self.probability = 0.0
            self.history.clear()
        return None

    def observe(
        self,
        timestamp: float,
        now: float,
        track_id: int | None,
        raw: np.ndarray | None,
        smoothed: np.ndarray | None,
        valid: bool = True,
    ) -> ImitationResult | None:
        result = self.tick(now)
        if result is not None:
            return result
        if (
            not math.isfinite(timestamp)
            or timestamp <= self.last_timestamp
            or timestamp < self.started_at
            or not 0 <= now - timestamp <= self.settings.max_frame_age
        ):
            self.reset_hold("Stale or repeated frame - hold reset")
            self.probability = 0.0
            self.history.clear()
            return None
        self.last_timestamp = timestamp
        if not valid or track_id is None or raw is None or smoothed is None:
            self.probability = 0.0
            self.history.clear()
            self.reset_hold("")
            return None
        for values in (raw, smoothed):
            if (
                np.shape(values) != (7,)
                or not np.isfinite(values).all()
                or np.any(values < 0)
                or not np.isclose(np.sum(values), 1, atol=1e-3)
            ):
                self.reset_hold("Invalid model output - hold reset")
                self.probability = 0.0
                self.history.clear()
                return None
        if track_id != self.track_id:
            self.reset_hold("New face track - starting a fresh hold")
            self.history.clear()
            self.track_id = track_id
        self.valid_samples += 1
        self.history.append(np.asarray(raw).copy())
        self.probability = float(smoothed[self.index])
        matched = (
            int(np.argmax(raw)) == self.index
            and int(np.argmax(smoothed)) == self.index
            and raw[self.index] >= max(0.3, self.settings.threshold * 0.8)
            and self.probability >= self.settings.threshold
        )
        if not matched:
            self.reset_hold("Waiting for sustained target-class agreement")
            return None
        if self.last_match is not None:
            interval = timestamp - self.last_match
            if interval <= self.settings.max_gap:
                self.held_seconds += interval
            else:
                self.reset_hold("Tracking gap - start the hold again")
        self.last_match = timestamp
        self.held_samples += 1
        self.status = "Target detected - keep holding"
        if (
            self.held_seconds + 1e-9 >= self.settings.hold_seconds
            and self.held_samples >= self.settings.minimum_samples
        ):
            return self.finish("success", now)
        return None
