from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

GEOMETRY_KEYS = ("mouth_open", "mouth_width", "eye_open", "brow_raise", "corner_raise")
BLENDSHAPE_GROUPS = {
    "smile": ("mouthSmileLeft", "mouthSmileRight"),
    "frown": ("mouthFrownLeft", "mouthFrownRight"),
    "inner_brow": ("browInnerUp",),
    "outer_brow": ("browOuterUpLeft", "browOuterUpRight"),
    "brow_down": ("browDownLeft", "browDownRight"),
    "eye_wide": ("eyeWideLeft", "eyeWideRight"),
    "eye_squint": ("eyeSquintLeft", "eyeSquintRight"),
    "nose_wrinkle": ("noseSneerLeft", "noseSneerRight"),
    "upper_lip": ("mouthUpperUpLeft", "mouthUpperUpRight"),
    "lip_press": ("mouthPressLeft", "mouthPressRight"),
}
CUES = {
    "happy": (("smile", "corner_raise", "Try gently raising the mouth corners."),),
    "surprise": (
        ("outer_brow", "brow_raise", "Try gently raising your eyebrows."),
        ("mouth_open", "mouth_open", "Try opening your mouth slightly."),
    ),
    "fear": (
        ("eye_wide", "eye_open", "Try widening your eyes slightly."),
        ("inner_brow", "brow_raise", "Try gently raising the inner eyebrows."),
    ),
    "sad": (
        ("inner_brow", "brow_raise", "Try gently raising the inner eyebrows."),
        ("frown", "frown", "Try a slight downward movement of the mouth corners."),
    ),
    "angry": (
        ("brow_down", "brow_down", "Try bringing your eyebrows slightly down and together."),
        ("lip_press", "lip_press", "Try gently pressing your lips together."),
    ),
    "disgust": (
        ("nose_wrinkle", "nose_wrinkle", "Try a slight nose wrinkle, if comfortable."),
        ("upper_lip", "upper_lip", "Try gently raising your upper lip."),
    ),
}


def movement_features(
    points: np.ndarray, blendshapes: dict[str, float], shape: tuple[int, ...] | None = None
) -> dict[str, float]:
    if points.shape != (478, 3) or not np.isfinite(points).all():
        raise ValueError("Expected finite MediaPipe face landmarks")
    xy = np.asarray(points[:, :2], dtype=float)
    if shape is not None:
        xy = xy * [shape[1], shape[0]]
    left = xy[[33, 133]].mean(axis=0)
    right = xy[[362, 263]].mean(axis=0)
    distance = float(np.linalg.norm(right - left))
    if distance < 1e-5:
        raise ValueError("Degenerate eye geometry")
    horizontal = (right - left) / distance
    vertical = np.array([-horizontal[1], horizontal[0]])
    projected = (xy - (left + right) / 2) @ vertical / distance
    values = {
        "mouth_open": abs(float(projected[14] - projected[13])),
        "mouth_width": abs(float((xy[291] - xy[61]) @ horizontal / distance)),
        "eye_open": float(
            (abs(projected[159] - projected[145]) + abs(projected[386] - projected[374])) / 2
        ),
        "brow_raise": float(projected[[159, 145, 386, 374]].mean() - projected[[105, 334]].mean()),
        "corner_raise": float(projected[[13, 14]].mean() - projected[[61, 291]].mean()),
    }
    for key, names in BLENDSHAPE_GROUPS.items():
        if all(name in blendshapes for name in names):
            coefficients = np.array([blendshapes[name] for name in names])
            if not np.isfinite(coefficients).all() or np.any(
                (coefficients < 0) | (coefficients > 1)
            ):
                raise ValueError("Invalid blendshape coefficients")
            values[key] = float(coefficients.mean())
    return values


@dataclass(frozen=True)
class NeutralBaseline:
    schema: int
    median: dict[str, float]
    noise: dict[str, float]
    samples: int
    seconds: float

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> NeutralBaseline:
        baseline = cls(**data)
        if (
            baseline.schema != 1
            or baseline.samples < 40
            or not 2 <= baseline.seconds <= 60
            or not set(GEOMETRY_KEYS) <= set(baseline.median)
            or not set(baseline.median) <= set(GEOMETRY_KEYS) | set(BLENDSHAPE_GROUPS)
            or set(baseline.median) != set(baseline.noise)
            or not all(np.isfinite(value) for value in baseline.median.values())
            or not all(np.isfinite(value) and value >= 0 for value in baseline.noise.values())
        ):
            raise ValueError("Invalid or incompatible neutral calibration")
        return baseline


class NeutralCalibrator:
    def __init__(self) -> None:
        self.samples: list[dict[str, float]] = []
        self.timestamps: list[float] = []
        self.track_id: int | None = None

    def reset(self) -> None:
        self.samples.clear()
        self.timestamps.clear()
        self.track_id = None

    def observe(
        self, timestamp: float, track_id: int, features: dict[str, float]
    ) -> NeutralBaseline | None:
        if not np.isfinite(timestamp) or not set(GEOMETRY_KEYS) <= features.keys():
            raise ValueError("Missing neutral-face geometry")
        if not all(np.isfinite(value) for value in features.values()):
            raise ValueError("Nonfinite movement features")
        if self.timestamps and timestamp <= self.timestamps[-1]:
            self.reset()
            return None
        if self.timestamps and (
            track_id != self.track_id
            or timestamp - self.timestamps[-1] > 0.25
            or set(features) != set(self.samples[0])
        ):
            self.reset()
        self.track_id = track_id
        self.samples.append(features.copy())
        self.timestamps.append(timestamp)
        self.samples = self.samples[-180:]
        self.timestamps = self.timestamps[-180:]
        duration = self.timestamps[-1] - self.timestamps[0]
        if len(self.samples) < 40 or duration < 2:
            return None
        keys = sorted(features)
        values = np.array([[row[key] for key in keys] for row in self.samples])
        median = np.median(values, axis=0)
        noise = np.median(np.abs(values - median), axis=0) * 1.4826
        span = np.percentile(values, 90, axis=0) - np.percentile(values, 10, axis=0)
        limits = np.array([0.07 if key in GEOMETRY_KEYS else 0.2 for key in keys])
        if np.any(span > limits):
            self.reset()
            raise ValueError("Too much movement during calibration; rest comfortably and try again")
        return NeutralBaseline(
            1,
            dict(zip(keys, median.tolist(), strict=True)),
            dict(zip(keys, noise.tolist(), strict=True)),
            len(self.samples),
            duration,
        )


def movement_hints(
    target: str, features: dict[str, float], baseline: NeutralBaseline, sensitivity: float = 1.0
) -> list[str]:
    if not 0.5 <= sensitivity <= 2 or set(features) != set(baseline.median):
        return ["Face tracking has changed. Set up movement hints again."]
    if not all(np.isfinite(value) for value in features.values()):
        return ["Can't follow your face right now. Face the camera in good lighting."]
    if target == "neutral":
        keys = [
            key for key in ("mouth_open", "brow_raise", "smile", "brow_down") if key in features
        ]
        relaxed = all(
            abs(features[key] - baseline.median[key])
            <= max(0.025 if key in GEOMETRY_KEYS else 0.08, 3 * baseline.noise[key])
            for key in keys
        )
        return [
            "Your face looks close to its resting position."
            if relaxed
            else "Try resting your jaw and eyebrows comfortably."
        ]
    hints = []
    available = 0
    for preferred, fallback, instruction in CUES.get(target, ()):
        key = preferred if preferred in features else fallback
        if key not in features:
            continue
        available += 1
        threshold = max(
            (0.012 if key in GEOMETRY_KEYS else 0.07) / sensitivity, 3 * baseline.noise[key]
        )
        if features[key] - baseline.median[key] < threshold:
            hints.append(instruction)
    if available == 0:
        return ["Target-specific movement cues unavailable; use the classifier indicator."]
    return hints[:2] or [
        "Face movement detected. Keep following the target indicator."
    ]
