import json
import os
from pathlib import Path

import mlflow

report = json.loads(Path("results/webcam/gui/gui.json").read_text())
stability = json.loads(Path("results/temporal/gui/stability.json").read_text())
tracking = Path(os.environ.get("FER_TRACKING_DIR", "mlruns")).resolve()
if str(tracking).startswith("\\\\"):
    raise ValueError("Historical MLflow Windows file tracking requires a local drive")
mlflow.set_tracking_uri(tracking.as_uri())
mlflow.set_experiment("FER2013-desktop-runtime")
with mlflow.start_run(run_name="ResNet18-FP32-camera0-native-GUI") as run:
    mlflow.log_params(
        {
            "camera": report["camera"],
            "model": report["model"],
            "alpha": report["alpha"],
            "warmup_seconds": report["warmup_seconds"],
            "training_seed": 42,
            "video_labels_available": False,
            "footage_saved": False,
        }
    )
    mlflow.log_metrics(
        {
            "display_fps": report["display_fps"],
            "classified_frames": report["frames_with_classification"],
            "inference_ms": report["mean_inference_ms_when_classified"],
            "capture_to_paint_ms": report["mean_capture_to_paint_ms"],
            "peak_rss_bytes": report["sampled_peak_rss_bytes"],
            "raw_switches": sum(r["raw_switches"] for r in stability["tracks"]),
            "ema_switches": sum(r["smoothed_switches"] for r in stability["tracks"]),
        }
    )
    for path in (
        Path("results/webcam/gui/gui.json"),
        Path("results/temporal/gui/stability.json"),
        Path("results/application/model_preflight.json"),
    ):
        mlflow.log_artifact(str(path))
    Path("results/application/mlflow_run.json").write_text(
        json.dumps(
            {
                "run_id": run.info.run_id,
                "experiment_id": run.info.experiment_id,
                "scope": (
                    "Actual saved native GUI trial and unlabeled stability; no video artifacts"
                ),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Tracked desktop trial locally: {run.info.run_id}")
