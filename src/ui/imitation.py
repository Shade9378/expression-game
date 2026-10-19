from __future__ import annotations

import random
import secrets
import time
from dataclasses import asdict

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from constants import CLASSES
from games.catalog import Reference, ReferenceCatalog
from games.feedback import NeutralBaseline, NeutralCalibrator, movement_hints
from games.imitation import ChallengeSettings, ImitationResult, ImitationRound
from inference.webcam import ProcessedFrame
from ui.face_guide import render_guide_frame
from ui.game_style import (
    ScoreStrip,
    check_icon,
    game_controls,
    game_layout,
    round_progress_bar,
    style_game,
    style_image,
    tint,
    trailing_streak,
)
from ui.images import show_rgb
from ui.instructions import GameInstructions

HINT_SETUP_TEXT = "Setup hint"
HINT_SETUP_HELP = (
    "Get tips such as 'raise your eyebrows' while you play. "
    "Click here, then look at the camera with your face relaxed and still for about two seconds."
)


class ImitationWidget(QWidget):
    camera_requested = Signal()
    camera_stop_requested = Signal()
    session_started = Signal(object)
    session_finished = Signal()
    result_ready = Signal(object, object)
    calibration_ready = Signal(object)

    def __init__(self, preview: QLabel | None = None) -> None:
        super().__init__()
        style_game(self)
        self.catalog: ReferenceCatalog | None = None
        self.reference: Reference | None = None
        self.round: ImitationRound | None = None
        self.results: list[ImitationResult] = []
        self.total_rounds = 0
        self.current_round = 0
        self.challenge_active = False
        self.targets: list[str] = []
        self.used: set[str] = set()
        self.rng = random.Random()
        self.pending_at: float | None = None
        self.advance_at: float | None = None
        self.paused_at: float | None = None
        self.settings = ChallengeSettings()
        self.calibration: NeutralBaseline | None = None
        self.calibrator: NeutralCalibrator | None = None
        self.calibration_started: float | None = None
        self.target = QComboBox()
        self.target.addItems(["Mixed sequence", *[label.capitalize() for label in CLASSES]])
        self.face_guide = QCheckBox("Face guide")
        self.face_guide.setToolTip(
            "Use Setup hint first. Follow the cyan feature outlines and gold movement arrows. "
            "Suggested movements are approximate and do not change your score."
        )
        self.face_guide.toggled.connect(self.explain_face_guide)
        self.level = QComboBox()
        self.level.addItems(["Easy", "Standard", "Hard"])
        self.level.currentIndexChanged.connect(self.set_defaults)
        self.round_count = QSpinBox()
        self.round_count.setRange(1, 15)
        self.round_count.setValue(3)
        self.timed = QCheckBox("Timed rounds")
        self.timed.setChecked(True)
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.35, 0.9)
        self.threshold.setSingleStep(0.05)
        self.threshold.setValue(0.5)
        self.hold = QDoubleSpinBox()
        self.hold.setRange(0.5, 5)
        self.hold.setSingleStep(0.1)
        self.hold.setValue(1.2)
        self.deadline = QSpinBox()
        self.deadline.setRange(7, 120)
        self.deadline.setValue(25)
        self.timed.toggled.connect(self.update_timing_controls)
        timing = QFormLayout()
        timing.addRow("Classifier threshold", self.threshold)
        timing.addRow("Hold (seconds)", self.hold)
        timing.addRow("Round limit (seconds)", self.deadline)
        target_option = QWidget()
        target_layout = QHBoxLayout(target_option)
        target_layout.setContentsMargins(0, 0, 0, 0)
        target_layout.addWidget(self.timed)
        target_layout.addWidget(QLabel("Expression"))
        target_layout.addWidget(self.target, 1)
        target_layout.addWidget(self.face_guide)
        self.advanced = QWidget(self)
        advanced_layout = QHBoxLayout(self.advanced)
        advanced_layout.setContentsMargins(0, 0, 0, 0)
        advanced_layout.addLayout(timing)
        self.advanced.hide()
        self.start = QPushButton("Start game")
        self.start.setProperty("role", "primary")
        self.start.setEnabled(False)
        self.start.clicked.connect(self.toggle_challenge)
        self.skip = QPushButton("Skip round")
        self.refresh_skip_button()
        self.skip.clicked.connect(self.skip_or_new_game)
        self.instructions = GameInstructions(
            self, "imitation", self.set_active, lambda: self.paused_at is None
        )
        self.instructions_button = QPushButton("How to play")
        self.instructions_button.clicked.connect(self.instructions.open_for_game)
        self.reference_image = QLabel("Set up game images in Settings")
        self.reference_image.setMinimumSize(160, 180)
        self.reference_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        style_image(self.reference_image)
        self.preview = preview or QLabel("Camera stopped")
        self.preview.setMinimumSize(360, 260)
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        style_image(self.preview)
        visuals = QHBoxLayout()
        self.media_stack = QStackedWidget()
        media = QWidget()
        media_layout = QHBoxLayout(media)
        media_layout.setContentsMargins(0, 0, 0, 0)
        media_layout.addWidget(self.reference_image, 2)
        media_layout.addWidget(self.preview, 5)
        self.media_stack.addWidget(media)
        result_card = QFrame()
        style_image(result_card)
        result_layout = QVBoxLayout(result_card)
        result_layout.setContentsMargins(24, 20, 24, 20)
        result_layout.addStretch()
        self.end_title = QLabel()
        self.end_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.end_title.setStyleSheet("font-size:26px; font-weight:700; border:0;")
        self.end_details = QLabel()
        self.end_details.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.end_details.setWordWrap(True)
        self.end_details.setStyleSheet("font-size:18px; border:0; padding:10px;")
        result_layout.addWidget(self.end_title)
        result_layout.addWidget(self.end_details)
        result_layout.addStretch()
        self.media_stack.addWidget(result_card)
        visuals.addWidget(self.media_stack, 7)
        self.target_label = QLabel("Choose an expression or a mixed sequence.")
        self.target_label.setStyleSheet("font-size:16px; font-weight:600; padding:3px 0;")
        self.time_left = QLabel()
        self.time_left.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.time_left.setStyleSheet("font-size:16px; font-weight:600; padding:3px 0;")
        target_line = QHBoxLayout()
        target_line.addWidget(self.target_label, 1)
        target_line.addWidget(self.time_left)
        self.probability = QProgressBar()
        self.probability.setRange(0, 1000)
        self.probability.setFormat("Expression probability: -")
        self.hold_bar = QProgressBar()
        self.hold_bar.setRange(0, 1000)
        self.hold_bar.setFormat("Hold: -")
        self.hold_bar.setStyleSheet("QProgressBar::chunk {background:#2f9e75; border-radius:7px;}")
        self.movements = QLabel()
        self.movements.setWordWrap(True)
        self.movements.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.movements.setStyleSheet("font-size:18px; padding:4px 0; background:transparent;")
        self.calibrate = QPushButton(HINT_SETUP_TEXT)
        self.calibrate.setToolTip(HINT_SETUP_HELP)
        self.calibrate.clicked.connect(self.begin_calibration)
        self.refresh_calibration_button()
        self.sensitivity = QDoubleSpinBox()
        self.sensitivity.setRange(0.5, 2)
        self.sensitivity.setSingleStep(0.1)
        self.sensitivity.setValue(1)
        self.sensitivity.setToolTip("Hint sensitivity only; does not change game scoring")
        calibration_controls = QHBoxLayout()
        calibration_controls.addWidget(QLabel("Hint sensitivity"))
        calibration_controls.addWidget(self.sensitivity)
        advanced_layout.addLayout(calibration_controls)
        self.hint_panel = QFrame()
        self.hint_panel.setObjectName("movementHints")
        self.hint_panel.setMinimumWidth(250)
        self.hint_panel.setMaximumWidth(290)
        self.hint_panel.setStyleSheet(
            f"QFrame#movementHints {{background:{tint(self, '#8174df', 0.12)}; "
            "border:1px solid #8174df; border-radius:14px;}"
        )
        hint_layout = QVBoxLayout(self.hint_panel)
        hint_layout.setContentsMargins(16, 14, 16, 14)
        hint_layout.setSpacing(12)
        hint_title = QLabel("Movement hints")
        hint_title.setStyleSheet("font-size:19px; font-weight:700;")
        self.score_strip = ScoreStrip()
        self.score_strip.update_values("- / -", 0, 0)
        self.round_progress = round_progress_bar()
        hint_layout.addWidget(hint_title)
        hint_layout.addWidget(self.movements, 1)
        hint_layout.addWidget(self.calibrate)
        visuals.addWidget(self.hint_panel, 3)
        layout = game_layout(self)
        layout.addWidget(
            game_controls(
                self.level,
                self.round_count,
                target_option,
                self.start,
                self.skip,
                self.instructions_button,
            )
        )
        layout.addWidget(self.score_strip)
        layout.addWidget(self.round_progress)
        layout.addLayout(target_line)
        layout.addLayout(visuals, 1)
        layout.addWidget(self.probability)
        layout.addWidget(self.hold_bar)
        self.timer = QTimer(self)
        self.timer.setInterval(50)
        self.timer.timeout.connect(self.tick)
        self.timer.start()

    def set_defaults(self, level: int) -> None:
        self.threshold.setValue((0.5, 0.55, 0.65)[level])
        self.hold.setValue((1.2, 1.6, 2.0)[level])
        self.deadline.setValue((25, 22, 18)[level])

    def lock_settings(self, locked: bool) -> None:
        for control in (
            self.target,
            self.level,
            self.round_count,
            self.timed,
            self.threshold,
            self.hold,
            self.deadline,
        ):
            control.setEnabled(not locked)
        self.deadline.setEnabled(not locked and self.timed.isChecked())

    def update_timing_controls(self) -> None:
        self.deadline.setEnabled(not self.challenge_active and self.timed.isChecked())

    def set_catalog(self, catalog: ReferenceCatalog) -> None:
        self.catalog = catalog
        selection = self.target.currentText()
        self.target.clear()
        self.target.addItems(["Mixed sequence", *[label.capitalize() for label in catalog.labels]])
        if selection in [self.target.itemText(index) for index in range(self.target.count())]:
            self.target.setCurrentText(selection)
        self.start.setEnabled(True)
        self.refresh_skip_button()
        self.reference_image.setText("Your target image will appear here")

    def reset_player(self) -> None:
        self.cancel()
        self.calibrator = None
        self.calibration_started = None
        self.calibration = None
        self.reference = None
        self.results = []
        self.total_rounds = 0
        self.current_round = 0
        self.refresh_calibration_button()
        self.update_score_strip()
        self.movements.clear()
        self.reference_image.clear()
        self.reference_image.setText("Your target image will appear here")
        if hasattr(self.preview, "reset"):
            self.preview.reset()
        else:
            self.preview.clear()
        self.end_title.clear()
        self.end_details.clear()
        self.media_stack.setCurrentIndex(0)
        self.target_label.setText("Choose an expression or a mixed sequence.")
        self.target_label.show()
        self.time_left.clear()
        self.time_left.show()
        self.probability.setValue(0)
        self.hold_bar.setValue(0)
        self.probability.setFormat("Expression probability: -")
        self.hold_bar.setFormat("Hold: -")
        self.probability.show()
        self.hold_bar.show()

    def set_challenge_active(self, active: bool) -> None:
        self.challenge_active = active
        self.start.setText("Stop game" if active else "Start game")
        self.calibrate.setEnabled(not active)
        self.refresh_calibration_button()
        self.refresh_skip_button()

    def refresh_skip_button(self) -> None:
        initialized = self.total_rounds > 0
        self.skip.setVisible(initialized)
        if not initialized:
            self.skip.setText("")
            self.skip.setEnabled(False)
            return
        playing = self.challenge_active and self.paused_at is None
        self.skip.setText("Skip round" if playing else "New game")
        self.skip.setEnabled(
            (self.pending_at is not None or (self.round is not None and self.round.result is None))
            if playing
            else self.catalog is not None and self.calibrator is None
        )

    def skip_or_new_game(self) -> None:
        if not self.challenge_active or self.paused_at is not None:
            self.set_active(True)
            self.begin()
        else:
            self.skip_round()

    def update_score_strip(self) -> None:
        self.score_strip.update_values(
            f"{self.current_round} / {self.total_rounds}" if self.total_rounds else "- / -",
            sum(result.points for result in self.results),
            trailing_streak([result.outcome for result in self.results], "success"),
        )
        self.round_progress.setRange(0, max(1, self.total_rounds))
        self.round_progress.setValue(len(self.results))

    def refresh_calibration_button(self) -> None:
        self.refresh_skip_button()
        complete = self.calibration is not None and self.calibrator is None
        self.calibrate.setText("Cancel setup" if self.calibrator is not None else HINT_SETUP_TEXT)
        self.calibrate.setIcon(check_icon() if complete else QIcon())
        self.calibrate.setStyleSheet(
            "QPushButton {background:#237a58; color:white; border:1px solid #39a97e;}"
            "QPushButton:hover {background:#2b8c67;}"
            "QPushButton:disabled {background:#205f49; color:#d8ede5;}"
            if complete
            else ""
        )
        self.calibrate.setToolTip(
            "Movement hints are set up. Click to set up again." if complete else HINT_SETUP_HELP
        )

    def show_game_summary(self, title: str, message: str = "") -> None:
        matched = sum(result.outcome == "success" for result in self.results)
        score = sum(result.points for result in self.results)
        lines = [f"Expressions matched: {matched} / {self.total_rounds}", f"Score: {score}"]
        skipped = sum(result.outcome == "skipped" for result in self.results)
        timed_out = sum(result.outcome == "timeout" for result in self.results)
        if skipped or timed_out:
            lines.append(
                "   ".join(
                    f"{label}: {count}"
                    for label, count in (("Skipped", skipped), ("Timed out", timed_out))
                    if count
                )
            )
        if message:
            lines.append(message)
        self.end_title.setText(title)
        self.end_details.setText("\n".join(lines))
        self.reference_image.clear()
        self.reference_image.setText("Your target image will appear here")
        if hasattr(self.preview, "reset"):
            self.preview.reset()
        else:
            self.preview.clear()
        self.movements.clear()
        self.target_label.hide()
        self.time_left.clear()
        self.time_left.hide()
        self.media_stack.setCurrentIndex(1)
        self.probability.hide()
        self.hold_bar.hide()

    def finish_challenge(self) -> None:
        self.lock_settings(False)
        self.set_challenge_active(False)
        self.show_game_summary("Game complete")
        self.refresh_skip_button()
        self.session_finished.emit()

    def toggle_challenge(self) -> None:
        if self.challenge_active:
            self.cancel()
            self.camera_stop_requested.emit()
            self.show_game_summary("Game stopped")
        else:
            self.begin()

    def begin(self) -> None:
        if self.catalog is None:
            return
        self.cancel()
        self.calibrator = None
        self.calibration_started = None
        self.refresh_calibration_button()
        seed = secrets.randbits(32)
        self.rng.seed(seed)
        labels = list(self.catalog.labels)
        self.rng.shuffle(labels)
        self.targets = [
            labels[index % len(labels)]
            if self.target.currentIndex() == 0
            else self.target.currentText().lower()
            for index in range(self.round_count.value())
        ]
        self.results, self.used = [], set()
        self.total_rounds = len(self.targets)
        self.current_round = 0
        self.update_score_strip()
        self.settings = ChallengeSettings(
            self.threshold.value(),
            self.hold.value(),
            self.deadline.value(),
            self.level.currentIndex(),
            timed=self.timed.isChecked(),
        )
        self.lock_settings(True)
        self.set_challenge_active(True)
        self.media_stack.setCurrentIndex(0)
        self.probability.show()
        self.hold_bar.show()
        self.session_started.emit(
            {
                "mode": "imitation",
                "seed": seed,
                "targets": self.targets.copy(),
                **asdict(self.settings),
            }
        )
        self.prepare_round()
        if self.pending_at is not None:
            self.camera_requested.emit()

    def prepare_round(self) -> None:
        self.advance_at = None
        if not self.targets:
            self.finish_challenge()
            return
        if not self.total_rounds:
            self.total_rounds = len(self.targets) + len(self.results)
        try:
            target = self.targets.pop(0)
            self.reference = self.catalog.choose(target, self.settings.level, self.rng, self.used)
            self.used.add(self.reference.sample_id)
            show_rgb(self.reference_image, self.catalog.image(self.reference))
        except (OSError, ValueError) as error:
            self.camera_error(f"Local reference unavailable: {error}")
            return
        self.target_label.setText(
            f"Expression: {target.capitalize()} - "
            f"keep it steady for {self.settings.hold_seconds:.1f} s"
        )
        self.target_label.show()
        self.time_left.clear()
        self.time_left.show()
        self.current_round = len(self.results) + 1
        self.update_score_strip()
        self.round = None
        self.pending_at = time.perf_counter()
        self.refresh_skip_button()
        self.probability.setValue(0)
        self.hold_bar.setValue(0)

    def observe_frame(self, frame: ProcessedFrame) -> None:
        if self.paused_at is not None:
            return
        preview_frame = self.add_face_guide(frame)
        if self.media_stack.currentIndex() == 0:
            if hasattr(self.preview, "present"):
                self.preview.present(preview_frame)
            else:
                show_rgb(self.preview, preview_frame.rgb)
        self.observe_movements(frame)
        face = frame.faces[0] if len(frame.faces) == 1 else None
        valid = face is not None and face["raw"] is not None and face["status"] == "OK"
        now = time.perf_counter()
        if (
            self.pending_at is not None
            and valid
            and 0 <= now - frame.captured_at <= self.settings.max_frame_age
        ):
            self.round = ImitationRound(self.reference.label, now, self.settings)
            self.pending_at = None
        if self.round is None or self.round.result is not None:
            return
        result = self.round.observe(
            frame.captured_at,
            now,
            face["track_id"] if face else None,
            face["raw"] if face else None,
            face["probabilities"] if face else None,
            valid,
        )
        self.update_round(now)
        if result is not None:
            self.show_result(result)

    def explain_face_guide(self, checked: bool) -> None:
        if checked and self.calibration is None and self.calibrator is None:
            self.movements.setText("Use Setup hint to enable the face guide.")
        elif not checked and not self.challenge_active and self.calibrator is None:
            self.movements.clear()

    def add_face_guide(self, frame: ProcessedFrame) -> ProcessedFrame:
        if (
            not self.face_guide.isChecked()
            or not self.challenge_active
            or self.reference is None
            or (self.round is not None and self.round.result is not None)
        ):
            return frame
        return render_guide_frame(
            frame,
            self.reference.label,
            self.calibration,
            self.sensitivity.value(),
            self.settings.max_frame_age,
        )

    def observe_movements(self, frame: ProcessedFrame) -> None:
        if not self.challenge_active and self.calibrator is None:
            return
        face = frame.faces[0] if len(frame.faces) == 1 else None
        features = face.get("movements") if face is not None and face["status"] == "OK" else None
        if features is None or not 0 <= time.perf_counter() - frame.captured_at <= 0.45:
            if self.calibrator is not None:
                self.calibrator.reset()
                self.movements.setText("Look straight at the camera with only your face in view.")
            elif self.calibration is not None:
                self.movements.setText(
                    "Movement hints unavailable - check tracking, pose and lighting."
                )
            return
        if self.calibrator is not None:
            try:
                baseline = self.calibrator.observe(frame.captured_at, face["track_id"], features)
                self.movements.setText("Keep your face relaxed and still for a moment...")
                if baseline is not None:
                    self.calibration = baseline
                    self.calibrator = None
                    self.calibration_started = None
                    self.refresh_calibration_button()
                    self.movements.clear()
                    self.calibration_ready.emit(baseline)
                    self.camera_stop_requested.emit()
            except ValueError:
                self.movements.setText("Keep your face relaxed and still, then try again.")
            return
        if self.calibration is not None and self.round is not None and self.round.result is None:
            self.movements.setText(
                " ".join(
                    movement_hints(
                        self.round.target, features, self.calibration, self.sensitivity.value()
                    )
                )
            )

    def begin_calibration(self) -> None:
        if self.calibrator is not None:
            self.calibrator = None
            self.calibration_started = None
            self.refresh_calibration_button()
            self.camera_stop_requested.emit()
            self.movements.setText("Setup cancelled. Your previous settings are kept.")
            return
        self.cancel()
        self.calibrator = NeutralCalibrator()
        self.calibration_started = time.perf_counter()
        self.refresh_calibration_button()
        self.media_stack.setCurrentIndex(0)
        self.movements.setText(
            "Look at the camera and relax your face for 2 seconds. No need to smile or pose."
        )
        self.camera_requested.emit()

    def update_round(self, now: float) -> None:
        if self.round is None:
            return
        self.probability.setValue(round(self.round.probability * 1000))
        self.probability.setFormat(f"Expression probability: {self.round.probability:.1%}")
        fraction = min(1.0, self.round.held_seconds / self.settings.hold_seconds)
        self.hold_bar.setValue(round(fraction * 1000))
        self.hold_bar.setFormat(
            f"Hold: {self.round.held_seconds:.1f}/{self.settings.hold_seconds:.1f} s"
        )
        if self.settings.timed:
            remaining = max(0.0, self.settings.timeout_seconds - (now - self.round.started_at))
            self.time_left.setText(f"{remaining:.1f} s left")
        else:
            self.time_left.clear()

    def show_result(self, result: ImitationResult) -> None:
        if any(result is existing for existing in self.results):
            return
        self.results.append(result)
        self.result_ready.emit(result, self.reference)
        self.update_score_strip()
        self.refresh_skip_button()
        if self.targets:
            self.advance_at = time.perf_counter() + 1.2
        else:
            self.finish_challenge()

    def skip_round(self) -> None:
        if self.reference is None:
            return
        if self.round is None:
            self.round = ImitationRound(self.reference.label, time.perf_counter(), self.settings)
            self.pending_at = None
        self.show_result(self.round.finish("skipped", time.perf_counter()))

    def cancel(self) -> None:
        if self.round is not None and self.round.result is None:
            self.show_result(self.round.finish("cancelled", time.perf_counter()))
        self.targets = []
        self.pending_at = self.advance_at = None
        self.round = None
        self.refresh_skip_button()
        self.lock_settings(False)
        self.set_challenge_active(False)
        self.session_finished.emit()

    def camera_error(self, message: str) -> None:
        was_active = self.challenge_active
        self.calibrator = None
        self.calibration_started = None
        self.refresh_calibration_button()
        if self.round is not None and self.round.result is None:
            self.show_result(self.round.finish("unavailable", time.perf_counter()))
        self.targets = []
        self.pending_at = self.advance_at = None
        self.refresh_skip_button()
        self.lock_settings(False)
        self.set_challenge_active(False)
        if was_active:
            self.show_game_summary("Game ended", message)
        self.session_finished.emit()

    def set_active(self, active: bool) -> None:
        now = time.perf_counter()
        if not active and self.paused_at is None:
            self.paused_at = now
            if self.round is not None:
                self.round.reset_hold("Paused - start a fresh hold when returning")
            if self.calibrator is not None:
                self.calibrator.reset()
        elif active and self.paused_at is not None:
            delay = now - self.paused_at
            if self.round is not None:
                self.round.started_at += delay
            if self.pending_at is not None:
                self.pending_at += delay
            if self.advance_at is not None:
                self.advance_at += delay
            if self.calibration_started is not None:
                self.calibration_started += delay
            self.paused_at = None
        self.refresh_skip_button()

    def tick(self) -> None:
        if self.paused_at is not None:
            return
        now = time.perf_counter()
        if self.calibration_started is not None and now - self.calibration_started > 20:
            self.camera_error("Setup timed out. Face the camera in good lighting, then try again.")
        if self.pending_at is not None and now - self.pending_at > 35:
            self.camera_error(
                "No usable camera face within 35 seconds. Check hardware, lighting and pose."
            )
        if self.advance_at is not None and now >= self.advance_at:
            self.prepare_round()
        if self.round is not None and self.round.result is None:
            result = self.round.tick(now)
            self.update_round(now)
            if result is not None:
                self.show_result(result)
