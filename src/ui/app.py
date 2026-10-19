from __future__ import annotations

import json
import os
import queue
import sys
import time
from collections import deque
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import numpy as np
import psutil
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QCloseEvent, QImage, QPaintEvent, QPixmap, QResizeEvent
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from constants import CLASSES
from inference.webcam import ProcessedFrame, WebcamSession
from ui.colors import EXPRESSION_COLORS
from ui.face_guide import render_guide_frame
from ui.game_hub import GameHub


def available_models(root: Path = Path("results/onnx")) -> list[tuple[str, Path]]:
    choices = []
    comparison = Path("results/optimization/comparison.json")
    scores = {}
    if comparison.exists():
        scores = {
            (row["name"], row["precision"]): row["val"]["macro_f1"]
            for row in json.loads(comparison.read_text())["results"]
            if row["status"] == "supported"
        }
    for sidecar in sorted(root.glob("*.onnx.json")):
        metadata = json.loads(sidecar.read_text())
        path = sidecar.with_suffix("")
        if metadata.get("supported") and path.is_file():
            population = "paired" if metadata["population"] == "paired" else "full"
            score = scores.get((metadata["name"], metadata["precision"]))
            suffix = f" - val F1 {score:.3f}" if score is not None else ""
            choices.append(
                (
                    f"{metadata['name']} - {metadata['precision'].upper()} - {population}{suffix}",
                    path,
                )
            )
    return choices


def rolling_fps(times: deque) -> float:
    return (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 else 0.0


class VideoLabel(QLabel):
    def __init__(self, callback: Callable[[ProcessedFrame, float], None]) -> None:
        super().__init__("Camera off")
        self.callback = callback
        self.pending: ProcessedFrame | None = None
        self.original: QPixmap | None = None
        self.coalesced = 0
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(480, 360)

    def present(self, frame: ProcessedFrame) -> None:
        if self.pending is not None:
            self.coalesced += 1
        self.pending = frame
        rgb = np.ascontiguousarray(frame.rgb)
        image = QImage(
            rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888
        )
        self.original = QPixmap.fromImage(image.copy())
        self.scale_image()

    def scale_image(self) -> None:
        if self.original is not None:
            self.setPixmap(
                self.original.scaled(
                    self.contentsRect().size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self.scale_image()

    def paintEvent(self, event: QPaintEvent) -> None:
        super().paintEvent(event)
        if self.pending is not None:
            frame, self.pending = self.pending, None
            self.callback(frame, time.perf_counter())

    def reset(self) -> None:
        self.pending = None
        self.original = None
        self.clear()
        self.setText("Camera off")


class ExpressionWindow(QMainWindow):
    def __init__(
        self,
        model: Path | None = None,
        asset: Path = Path("results/models/face_landmarker_v1.task"),
        camera: int = 0,
        benchmark_seconds: float | None = None,
        benchmark_output: Path = Path("results/webcam/gui"),
        timeline: Path | None = None,
        progress_database: Path | None = None,
        catalog_path: Path = Path("results/game/references.json"),
    ) -> None:
        super().__init__()
        self.asset = asset
        self.session: WebcamSession | None = None
        self.processed_times: deque = deque(maxlen=90)
        self.painted_times: deque = deque(maxlen=90)
        self.benchmark_seconds = benchmark_seconds
        self.benchmark_output, self.timeline = benchmark_output, timeline
        self.warmup_started: float | None = None
        self.benchmark_started: float | None = None
        self.launch_time = time.perf_counter()
        self.measurements: list[dict] = []
        self.trace_rows: list[dict] = []
        self.peak_rss = psutil.Process().memory_info().rss
        self.process = psutil.Process()
        self.process.cpu_percent()
        self.cpu_samples: list[float] = []
        self.cpu_sample_at = time.perf_counter()
        self.last_error: str | None = None
        self.benchmark_finished = False
        self.selected_model: Path | None = None
        self.setWindowTitle("Facial Expression Practice Game")
        self.resize(1120, 760)
        self.setMinimumSize(960, 680)
        content = QWidget()
        self.setCentralWidget(content)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(24, 20, 24, 20)
        self.models = QComboBox()
        choices = available_models()
        if model is not None and all(path != model for _, path in choices):
            choices.insert(0, (model.stem, model))
        for label, path in choices:
            self.models.addItem(label, str(path))
        if model is None and Path("results/optimization/comparison.json").exists():
            model = Path(
                json.loads(Path("results/optimization/comparison.json").read_text())["selected"][
                    "model_path"
                ]
            )
        for index in range(self.models.count()):
            if model is not None and Path(self.models.itemData(index)) == model:
                self.models.setCurrentIndex(index)
        self.camera = QSpinBox()
        self.camera.setRange(0, 16)
        self.camera.setValue(camera)
        self.start_button = QPushButton("Start camera")
        self.start_button.setObjectName("start")
        self.start_button.clicked.connect(self.start_capture)
        self.stop_button = QPushButton("Stop camera")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.request_stop)
        body = QHBoxLayout()
        self.video = VideoLabel(self.on_presented)
        body.addWidget(self.video, 1)
        side = QVBoxLayout()
        camera_actions = QHBoxLayout()
        camera_actions.addWidget(self.start_button)
        camera_actions.addWidget(self.stop_button)
        side.addLayout(camera_actions)
        self.face_choice = QComboBox()
        self.face_choice.setPlaceholderText("No face detected")
        side.addWidget(QLabel("Expression probabilities"))
        side.addWidget(self.face_choice)
        self.expression = QLabel("Awaiting camera")
        self.expression.setWordWrap(True)
        side.addWidget(self.expression)
        self.bars: list[QProgressBar] = []
        for label, color in zip(CLASSES, EXPRESSION_COLORS, strict=True):
            side.addWidget(QLabel(label.capitalize()))
            bar = QProgressBar()
            bar.setRange(0, 1000)
            bar.setValue(0)
            bar.setFormat("-")
            bar.setStyleSheet(f"QProgressBar::chunk {{background:{color}; border-radius:4px;}}")
            self.bars.append(bar)
            side.addWidget(bar)
        self.overlay = QCheckBox("Show landmarks")
        self.imitation_overlay = QCheckBox("Show detection overlay in imitation game mode")
        self.imitation_overlay.toggled.connect(self.update_settings)
        self.smoothing = QCheckBox("EMA smoothing")
        self.smoothing.setChecked(True)
        self.alpha_label = QLabel("EMA alpha: 0.30")
        self.alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self.alpha_slider.setRange(1, 100)
        self.alpha_slider.setValue(30)
        self.overlay.toggled.connect(self.update_settings)
        self.smoothing.toggled.connect(self.update_settings)
        self.alpha_slider.valueChanged.connect(self.update_settings)
        if benchmark_seconds is not None:
            self.smoothing.setEnabled(False)
            self.alpha_slider.setEnabled(False)
        side.addStretch()
        panel = QWidget()
        panel.setLayout(side)
        panel.setFixedWidth(290)
        body.addWidget(panel)
        live = QWidget()
        live_layout = QVBoxLayout(live)
        live_layout.setContentsMargins(0, 0, 0, 0)
        self.live_expression = QComboBox()
        self.live_expression.addItems([label.capitalize() for label in CLASSES])
        self.live_expression.setCurrentText("Happy")
        self.live_face_guide = QCheckBox("Face guide")
        self.live_face_guide.setToolTip(
            "Run Setup hint first, then follow the cyan expression outline."
        )
        self.live_face_guide.toggled.connect(self.update_settings)
        self.live_calibrate = QPushButton("Setup hint")
        self.live_calibrate.clicked.connect(self.begin_live_calibration)
        practice_controls = QHBoxLayout()
        practice_controls.addWidget(QLabel("Expression"))
        practice_controls.addWidget(self.live_expression, 1)
        practice_controls.addWidget(self.live_face_guide)
        practice_controls.addWidget(self.live_calibrate)
        live_layout.addLayout(practice_controls)
        live_layout.addLayout(body, 1)
        self.game_video = VideoLabel(self.on_presented)
        self.games = GameHub(live, self.game_video, asset, progress_database, catalog_path)
        self.games.model_provider = lambda: str(self.models.currentData() or "")
        self.games.imitation.camera_requested.connect(self.start_game_camera)
        self.games.imitation.camera_stop_requested.connect(self.request_stop)
        self.games.imitation.session_finished.connect(self.finish_game_capture)
        self.games.imitation.session_finished.connect(self.sync_live_hint)
        self.games.imitation.calibration_ready.connect(lambda _baseline: self.sync_live_hint())
        self.games.tabs.currentChanged.connect(self.update_settings)
        layout.addWidget(self.games, 1)
        self.performance_values = dict.fromkeys(
            ("Processed FPS", "Display FPS", "Inference", "Frame age"), "-"
        )
        self.performance_checks: dict[str, QCheckBox] = {}
        self.stats = QLabel()
        self.stats.setWordWrap(True)
        layout.addWidget(self.stats)
        self.build_settings()
        self.games.settings_requested.connect(self.open_settings)
        self.games.player_changing.connect(self.stop_capture)
        self.games.player_available.connect(self.sync_player_controls)
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.hide()
        layout.addWidget(self.status)
        self.timer = QTimer(self)
        self.timer.setInterval(15)
        self.timer.timeout.connect(self.poll)
        self.timer.start()
        self.sync_player_controls(self.games.profile is not None)

    def build_settings(self) -> None:
        self.settings_dialog = QDialog(self)
        self.settings_dialog.setWindowTitle("Settings")
        self.settings_dialog.resize(680, 650)
        layout = QVBoxLayout(self.settings_dialog)
        model_group = QGroupBox("Camera and model")
        model_form = QFormLayout(model_group)
        model_form.addRow("Model", self.models)
        model_form.addRow("Camera", self.camera)
        layout.addWidget(model_group)
        tracking_group = QGroupBox("Face tracking and smoothing")
        tracking_layout = QVBoxLayout(tracking_group)
        for widget in (
            self.overlay,
            self.imitation_overlay,
            self.smoothing,
            self.alpha_label,
            self.alpha_slider,
        ):
            tracking_layout.addWidget(widget)
        layout.addWidget(tracking_group)
        challenge_group = QGroupBox("Game options")
        challenge_layout = QVBoxLayout(challenge_group)
        challenge_layout.addWidget(self.games.adaptive)
        challenge_layout.addWidget(self.games.imitation.advanced)
        self.games.imitation.advanced.show()
        layout.addWidget(challenge_group)
        reference_group = QGroupBox("Game images")
        reference_layout = QVBoxLayout(reference_group)
        reference_layout.addWidget(self.games.notice)
        reference_layout.addWidget(self.games.reference_source)
        reference_layout.addWidget(self.games.import_images)
        reference_layout.addWidget(self.games.prepare)
        layout.addWidget(reference_group)
        performance_group = QGroupBox("Performance")
        performance_group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        performance_layout = QHBoxLayout(performance_group)
        for name in self.performance_values:
            checkbox = QCheckBox(name)
            checkbox.toggled.connect(self.render_performance)
            self.performance_checks[name] = checkbox
            performance_layout.addWidget(checkbox)
        layout.addWidget(performance_group)
        self.settings_sections = (
            model_group,
            tracking_group,
            challenge_group,
            reference_group,
            performance_group,
        )
        self.render_performance()
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.settings_dialog.hide)
        layout.addWidget(buttons)
        self.settings_dialog.resize(680, self.settings_dialog.sizeHint().height())

    def open_settings(self) -> None:
        self.settings_dialog.show()
        self.settings_dialog.raise_()
        self.settings_dialog.activateWindow()

    def render_performance(self) -> None:
        selected = [
            f"{name} {self.performance_values[name]}"
            for name, checkbox in self.performance_checks.items()
            if checkbox.isChecked()
        ]
        self.stats.setText("   ".join(selected))
        self.stats.setVisible(bool(selected))

    def sync_player_controls(self, active: bool) -> None:
        if not active:
            self.stop_capture()
        for section in self.settings_sections:
            section.setEnabled(active)
        running = self.session is not None
        self.models.setEnabled(active and not running)
        self.camera.setEnabled(active and not running)
        self.start_button.setEnabled(active and not running)
        self.stop_button.setEnabled(active and running)
        self.sync_live_hint()

    def begin_live_calibration(self) -> None:
        self.games.imitation.begin_calibration()
        self.games.imitation.set_active(self.games.imitation.calibrator is not None)
        self.sync_live_hint()
        self.update_settings()

    def sync_live_hint(self) -> None:
        source = self.games.imitation.calibrate
        self.live_calibrate.setText(source.text())
        self.live_calibrate.setIcon(source.icon())
        self.live_calibrate.setStyleSheet(source.styleSheet())
        self.live_calibrate.setEnabled(self.games.profile is not None)
        self.live_calibrate.setToolTip(source.toolTip())

    def update_settings(self) -> None:
        alpha = self.alpha_slider.value() / 100
        self.alpha_label.setText(f"EMA alpha: {alpha:.2f}")
        self.alpha_slider.setEnabled(self.smoothing.isChecked() and self.benchmark_seconds is None)
        if self.session is not None:
            self.session.overlays = self.overlay.isChecked()
            self.session.draw_detections = (
                self.games.tabs.currentIndex() != 2 or self.imitation_overlay.isChecked()
            )
            self.session.alpha = alpha if self.smoothing.isChecked() else 1.0
            self.session.game_feedback = (
                self.games.tabs.currentIndex() == 2
                or self.live_face_guide.isChecked()
                or self.games.imitation.calibrator is not None
            )
        if self.games.tabs.currentIndex() == 0 and self.games.imitation.calibrator is not None:
            self.games.imitation.set_active(True)

    def start_game_camera(self) -> None:
        self.start_capture()
        if self.session is None:
            self.games.imitation.camera_error(self.last_error or "Camera unavailable")
        self.update_settings()
        self.sync_live_hint()

    def finish_game_capture(self) -> None:
        if self.benchmark_seconds is not None:
            self.games.imitation.media_stack.setCurrentIndex(0)
            return
        session = self.session
        if self.games.tabs.currentIndex() != 2 or session is None:
            return

        def stop_finished_session() -> None:
            game = self.games.imitation
            if self.session is session and not game.challenge_active and game.calibrator is None:
                self.stop_capture()

        QTimer.singleShot(0, stop_finished_session)

    def start_capture(self) -> None:
        if self.session is not None:
            return
        if self.games.profile is None:
            self.last_error = "Add a player before starting the camera."
            self.set_status(self.last_error)
            return
        if self.models.currentData() is None:
            self.set_status(
                "No local trained models found. Run the experiment/export pipeline first."
            )
            self.last_error = "No local model available"
            return
        self.selected_model = Path(self.models.currentData())
        if not self.selected_model.is_file() or not self.asset.is_file():
            self.last_error = "Local model or MediaPipe asset is missing"
            self.set_status(self.last_error)
            return
        self.last_error = None
        self.processed_times.clear()
        self.painted_times.clear()
        self.video.coalesced = 0
        self.session = WebcamSession(
            self.selected_model,
            self.asset,
            self.camera.value(),
            self.overlay.isChecked(),
            self.alpha_slider.value() / 100 if self.smoothing.isChecked() else 1.0,
        )
        self.update_settings()
        self.models.setEnabled(False)
        self.camera.setEnabled(False)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.set_status("Starting local camera and model…")
        self.session.start()

    def clear_predictions(self) -> None:
        self.face_choice.clear()
        self.expression.setText("No expression estimate")
        for bar in self.bars:
            bar.setValue(0)
            bar.setFormat("-")

    def stop_capture(self) -> None:
        if self.session is not None:
            self.session.stop()
            self.session = None
        self.models.setEnabled(True)
        self.camera.setEnabled(True)
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.video.reset()
        self.game_video.reset()
        self.clear_predictions()
        self.performance_values = dict.fromkeys(self.performance_values, "-")
        self.render_performance()
        self.set_status(self.last_error or "")
        self.sync_live_hint()

    def set_status(self, text: str) -> None:
        self.status.setText(text)
        self.status.setToolTip(text)
        self.status.setVisible(bool(text))

    def request_stop(self) -> None:
        if self.benchmark_seconds is not None:
            self.last_error = "Benchmark stopped before completion"
            self.finish_benchmark()
        else:
            self.games.imitation.camera_error("Camera stopped")
            self.stop_capture()

    def poll(self) -> None:
        if self.session is not None:
            if self.session.error:
                self.last_error = self.session.error
                self.games.imitation.camera_error(self.last_error)
                if self.benchmark_seconds is not None:
                    self.finish_benchmark()
                else:
                    self.stop_capture()
                return
            try:
                frame = self.session.results.get(timeout=0)
            except queue.Empty:
                pass
            else:
                self.consume_frame(frame)
        if self.benchmark_seconds is not None:
            self.peak_rss = max(self.peak_rss, psutil.Process().memory_info().rss)
            now = time.perf_counter()
            if now - self.cpu_sample_at >= 0.5:
                sample = self.process.cpu_percent()
                if self.benchmark_started is not None:
                    self.cpu_samples.append(sample)
                self.cpu_sample_at = now
            if (
                self.benchmark_started is not None
                and now - self.benchmark_started >= self.benchmark_seconds
            ):
                self.finish_benchmark()
            elif self.benchmark_started is None and now - self.launch_time > 45:
                self.last_error = self.last_error or "No measured paints within startup deadline"
                self.finish_benchmark()

    def consume_frame(self, frame: ProcessedFrame) -> None:
        self.processed_times.append(frame.ready_at)
        if self.games.tabs.currentIndex() == 0:
            game = self.games.imitation
            preview_frame = (
                render_guide_frame(
                    frame,
                    self.live_expression.currentText().lower(),
                    game.calibration,
                    game.sensitivity.value(),
                )
                if self.live_face_guide.isChecked() and game.calibrator is None
                else frame
            )
            self.video.present(preview_frame)
            if game.calibrator is not None:
                game.observe_movements(frame)
        elif self.games.tabs.currentIndex() == 2:
            game_started = time.perf_counter()
            self.games.imitation.observe_frame(frame)
            frame.stages["game_ui_ms"] = (time.perf_counter() - game_started) * 1000
        ids = [face["track_id"] for face in frame.faces]
        existing = [self.face_choice.itemData(i) for i in range(self.face_choice.count())]
        if ids != existing:
            previous = self.face_choice.currentData()
            self.face_choice.clear()
            for track_id in ids:
                self.face_choice.addItem(f"Face {track_id}", track_id)
            if previous in ids:
                self.face_choice.setCurrentIndex(ids.index(previous))
            elif ids:
                self.face_choice.setCurrentIndex(0)
        selected = next(
            (face for face in frame.faces if face["track_id"] == self.face_choice.currentData()),
            None,
        )
        probabilities = selected["probabilities"] if selected is not None else None
        if probabilities is None:
            self.expression.setText(selected["status"] if selected else "No face detected")
            for bar in self.bars:
                bar.setValue(0)
                bar.setFormat("-")
        else:
            winner = int(np.argmax(probabilities))
            self.expression.setText(
                f"{CLASSES[winner].capitalize()} · {probabilities[winner]:.1%}"
                if probabilities[winner] >= 0.4
                else "Uncertain expression"
            )
            for bar, value in zip(self.bars, probabilities, strict=True):
                bar.setValue(round(float(value) * 1000))
                bar.setFormat(f"{value:.1%}")
        self.set_status("")
        if self.timeline is not None and self.benchmark_started is not None:
            for face in frame.faces:
                if face["raw"] is not None:
                    self.trace_rows.append(
                        {
                            "timestamp": frame.captured_at,
                            "track_id": face["track_id"],
                            "raw": face["raw"].tolist(),
                            "smoothed": face["probabilities"].tolist(),
                        }
                    )

    def on_presented(self, frame: ProcessedFrame, painted_at: float) -> None:
        self.painted_times.append(painted_at)
        classified = sum(face["probabilities"] is not None for face in frame.faces)
        inference = f"{frame.inference_ms:.1f} ms" if classified else "-"
        age = (painted_at - frame.captured_at) * 1000
        self.performance_values.update(
            {
                "Processed FPS": f"{rolling_fps(self.processed_times):.1f}",
                "Display FPS": f"{rolling_fps(self.painted_times):.1f}",
                "Inference": inference,
                "Frame age": f"{age:.1f} ms",
            }
        )
        self.render_performance()
        if self.benchmark_seconds is not None:
            if self.warmup_started is None:
                self.warmup_started = painted_at
            if self.benchmark_started is None and painted_at - self.warmup_started >= 5:
                self.benchmark_started = painted_at
            if self.benchmark_started is not None:
                self.measurements.append(
                    {
                        "elapsed": painted_at - self.benchmark_started,
                        "ready_at": frame.ready_at,
                        "painted_at": painted_at,
                        "faces": len(frame.faces),
                        "classified_faces": classified,
                        "movement_faces": sum(
                            face.get("movements") is not None for face in frame.faces
                        ),
                        "inference_ms": frame.inference_ms,
                        "capture_to_paint_ms": age,
                        "face_statuses": [face["status"] for face in frame.faces],
                        "stages": {
                            **frame.stages,
                            "ready_to_paint_ms": (painted_at - frame.ready_at) * 1000,
                        },
                    }
                )

    def finish_benchmark(self) -> None:
        if self.benchmark_finished:
            return
        self.benchmark_finished = True
        self.timer.stop()
        rows = self.measurements
        elapsed = rows[-1]["elapsed"] if rows else 0.0
        classified = [row for row in rows if row["classified_faces"]]
        result = {
            "model": str(self.selected_model),
            "camera": self.camera.value(),
            "frames_presented": len(rows),
            "elapsed_seconds": elapsed,
            "display_fps": (len(rows) - 1) / elapsed if elapsed else None,
            "processed_fps_for_presented_frames": rolling_fps(deque(r["ready_at"] for r in rows)),
            "frames_with_classification": len(classified),
            "faces_detected": sum(row["faces"] for row in rows),
            "faces_classified": sum(row["classified_faces"] for row in rows),
            "mean_inference_ms_when_classified": (
                float(np.mean([row["inference_ms"] for row in classified])) if classified else None
            ),
            "mean_capture_to_paint_ms": (
                float(np.mean([row["capture_to_paint_ms"] for row in rows])) if rows else None
            ),
            "input_frames_dropped": self.session.frames.dropped if self.session else None,
            "result_frames_dropped": self.session.results.dropped if self.session else None,
            "gui_frames_coalesced": self.video.coalesced + self.game_video.coalesced,
            "mode": self.games.tabs.tabText(self.games.tabs.currentIndex()),
            "game_results": [asdict(result) for result in self.games.imitation.results],
            "game_config": self.games.session_configs.get("imitation"),
            "neutral_calibration_available": self.games.imitation.calibration is not None,
            "game_waiting_for_usable_face": self.games.imitation.pending_at is not None,
            "sampled_peak_rss_bytes": self.peak_rss,
            "process_cpu_percent_samples": self.cpu_samples,
            "cpu_percent_scope": "Process percent; 100% equals one logical CPU",
            "memory_sampling_ms": 15,
            "warmup_seconds": 5,
            "landmark_overlay": self.overlay.isChecked(),
            "alpha": 0.3,
            "error": self.last_error,
            "footage_saved": False,
            "scope": "Capture through Qt paint completion; excludes exposure/driver buffering",
            "paint_completion_is_not_physical_screen_scanout": True,
        }
        self.benchmark_output.mkdir(parents=True, exist_ok=True)
        (self.benchmark_output / "gui.json").write_text(json.dumps(result, indent=2) + "\n")
        (self.benchmark_output / "measurements.json").write_text(json.dumps(rows, indent=2) + "\n")
        if self.timeline is not None:
            self.timeline.parent.mkdir(parents=True, exist_ok=True)
            self.timeline.write_text(
                json.dumps(
                    {
                        "source": "live_webcam",
                        "labels_available": False,
                        "alpha": 0.3,
                        "camera": self.camera.value(),
                        "rows": self.trace_rows,
                    },
                    indent=2,
                )
                + "\n"
            )
        print(json.dumps(result, indent=2), flush=True)
        self.stop_capture()
        QApplication.instance().quit()

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.benchmark_seconds is not None and not self.benchmark_finished:
            self.last_error = "Benchmark closed before completion"
            self.finish_benchmark()
        self.timer.stop()
        self.stop_capture()
        self.games.shutdown()
        event.accept()


def run_app(
    model: Path | None,
    asset: Path,
    camera: int,
    preview: Path | None,
    benchmark_seconds: float | None,
    benchmark_output: Path,
    timeline: Path | None,
    mode: str = "live",
    progress_database: Path | None = None,
) -> int:
    if preview is not None and os.name != "nt":
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication([sys.argv[0]])
    window = ExpressionWindow(
        model, asset, camera, benchmark_seconds, benchmark_output, timeline, progress_database
    )
    window.games.tabs.setCurrentIndex(("live", "recognition", "imitation", "progress").index(mode))
    window.show()
    if preview is not None:
        app.processEvents()
        preview.parent.mkdir(parents=True, exist_ok=True)
        window.grab().save(str(preview))
        window.close()
        return 0
    if benchmark_seconds is not None:
        QTimer.singleShot(
            500, window.games.imitation.begin if mode == "imitation" else window.start_capture
        )
    result = app.exec()
    window.close()
    return result
