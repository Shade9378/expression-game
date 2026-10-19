from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


class EMASmoother:
    def __init__(self, alpha: float = 0.3) -> None:
        if not 0 < alpha <= 1:
            raise ValueError("EMA alpha must be in (0, 1]")
        self.alpha = alpha
        self.values: dict[int, np.ndarray] = {}

    def update(self, track_id: int, probabilities: np.ndarray) -> np.ndarray:
        values = np.asarray(probabilities, dtype=np.float32)
        if (
            values.shape != (7,)
            or not np.isfinite(values).all()
            or np.any(values < 0)
            or values.sum() <= 0
        ):
            raise ValueError("Expected seven finite, nonnegative class probabilities")
        values = values / values.sum()
        previous = self.values.get(track_id)
        smoothed = values if previous is None else self.alpha * values + (1 - self.alpha) * previous
        smoothed /= smoothed.sum()
        self.values[track_id] = smoothed.copy()
        return smoothed.copy()

    def retain(self, track_ids: set[int]) -> None:
        self.values = {key: value for key, value in self.values.items() if key in track_ids}

    def reset(self, track_id: int) -> None:
        self.values.pop(track_id, None)


def analyze_temporal(trace: Path, output: Path) -> dict:
    data = json.loads(trace.read_text())
    if data.get("source") != "live_webcam" or data.get("labels_available") is not False:
        raise ValueError(
            "Expected an actual local webcam probability trace, without inferred labels"
        )
    groups = defaultdict(list)
    for row in data["rows"]:
        groups[row["track_id"]].append(row)
    summaries = []
    valid_groups = []
    for track_id, rows in groups.items():
        if len(rows) < 2:
            continue
        timestamps = np.asarray([r["timestamp"] for r in rows])
        if not np.all(np.diff(timestamps) > 0):
            raise ValueError("Prediction timestamps must increase within each track")
        raw = np.asarray([r["raw"] for r in rows])
        smooth = np.asarray([r["smoothed"] for r in rows])
        if raw.shape != smooth.shape or raw.shape[1] != 7:
            raise ValueError("Invalid temporal prediction dimensions")
        duration = float(timestamps[-1] - timestamps[0])
        raw_switches = int(np.sum(np.diff(raw.argmax(axis=1)) != 0))
        smooth_switches = int(np.sum(np.diff(smooth.argmax(axis=1)) != 0))
        intervals = np.diff(timestamps)
        gap_threshold = max(0.1, float(np.median(intervals)) * 3)
        continuous = intervals <= gap_threshold
        summaries.append(
            {
                "track_id": track_id,
                "frames": len(rows),
                "seconds": duration,
                "raw_switches": raw_switches,
                "smoothed_switches": smooth_switches,
                "raw_switches_per_second": raw_switches / duration,
                "smoothed_switches_per_second": smooth_switches / duration,
                "raw_mean_probability_tv": float(
                    np.abs(np.diff(raw, axis=0)).sum(axis=1).mean() / 2
                ),
                "smoothed_mean_probability_tv": float(
                    np.abs(np.diff(smooth, axis=0)).sum(axis=1).mean() / 2
                ),
                "gap_threshold_seconds": gap_threshold,
                "gaps": int(np.sum(~continuous)),
                "within_segment_transitions": int(np.sum(continuous)),
                "raw_within_segment_switches": int(
                    np.sum((np.diff(raw.argmax(axis=1)) != 0) & continuous)
                ),
                "smoothed_within_segment_switches": int(
                    np.sum((np.diff(smooth.argmax(axis=1)) != 0) & continuous)
                ),
            }
        )
        valid_groups.append(rows)
    report = {
        "source_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
        "alpha": data["alpha"],
        "tracks": summaries,
        "status": "Measured unlabeled prediction stability"
        if summaries
        else "Unavailable: no valid face prediction sequences",
        "temporal_accuracy": None,
        "supervised_training": False,
        "interpretation": "Stability is not expression accuracy; smoothing may delay transitions",
        "transition_scope": "Consecutive valid observations; within-segment counts exclude gaps",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "stability.json").write_text(json.dumps(report, indent=2) + "\n")
    if valid_groups and max(map(len, valid_groups)) >= 10:
        from evaluation.plot_utils import temporal_plot

        temporal_plot(max(valid_groups, key=len), output)
    print(json.dumps(report, indent=2))
    return report
