from __future__ import annotations

import secrets
import time

from PySide6.QtCore import QSize, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QWidget,
)

from games.catalog import ReferenceCatalog
from games.recognition import RecognitionGame, RecognitionResult
from ui.game_style import (
    ScoreStrip,
    feedback_style,
    game_controls,
    game_layout,
    round_progress_bar,
    style_choice,
    style_game,
    style_image,
    trailing_streak,
)
from ui.images import show_rgb
from ui.instructions import GameInstructions


class RecognitionWidget(QWidget):
    result_ready = Signal(object)
    session_started = Signal(object)
    session_finished = Signal()

    def __init__(self) -> None:
        super().__init__()
        style_game(self)
        self.catalog: ReferenceCatalog | None = None
        self.game: RecognitionGame | None = None
        self.challenge_active = False
        self.paused_at: float | None = None
        self.level = QComboBox()
        self.level.addItems(["Easy", "Standard", "Hard"])
        self.round_count = QSpinBox()
        self.round_count.setRange(1, 30)
        self.round_count.setValue(7)
        self.timed = QCheckBox("Timed rounds")
        self.start = QPushButton("Start game")
        self.start.setProperty("role", "primary")
        self.start.setEnabled(False)
        self.start.clicked.connect(self.toggle_challenge)
        self.instructions = GameInstructions(
            self, "recognition", self.set_active, lambda: self.paused_at is None
        )
        self.instructions_button = QPushButton("How to play")
        self.instructions_button.clicked.connect(self.instructions.open_for_game)
        self.image = QLabel("Set up game images in Settings")
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image.setMinimumWidth(180)
        self.image.setFixedHeight(260)
        self.image.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        style_image(self.image)
        self.image.setToolTip(
            "Local FER2013 training image, automatically screened; labels can be ambiguous"
        )
        self.summary = QLabel("Which emotion is it?")
        self.summary.setStyleSheet("font-size:16px; font-weight:600; padding:3px 0;")
        self.summary.ensurePolished()
        self.summary.setFixedHeight(self.summary.sizeHint().height())
        summary_policy = self.summary.sizePolicy()
        summary_policy.setRetainSizeWhenHidden(True)
        self.summary.setSizePolicy(summary_policy)
        self.score_strip = ScoreStrip()
        self.score_strip.update_values("- / -", 0, 0)
        self.round_progress = round_progress_bar()
        self.feedback = QLabel("")
        self.feedback.setWordWrap(True)
        self.feedback.setFixedHeight(self.feedback.fontMetrics().lineSpacing() * 2 + 16)
        feedback_policy = self.feedback.sizePolicy()
        feedback_policy.setRetainSizeWhenHidden(True)
        self.feedback.setSizePolicy(feedback_policy)
        feedback_style(self.feedback)
        self.buttons: list[QPushButton] = []
        choices = QGridLayout()
        for index in range(4):
            button = QPushButton()
            button.setFixedHeight(50)
            choice_policy = button.sizePolicy()
            choice_policy.setRetainSizeWhenHidden(True)
            button.setSizePolicy(choice_policy)
            button.setIconSize(QSize(30, 30))
            button.hide()
            button.clicked.connect(lambda checked=False, i=index: self.choose(i))
            self.buttons.append(button)
            choices.addWidget(button, index // 4, index % 4)
        self.skip = QPushButton("Skip round")
        self.refresh_skip_button()
        self.skip.clicked.connect(self.skip_or_new_game)
        self.next = QPushButton("Next")
        self.next.setFixedHeight(36)
        self.next.setProperty("role", "primary")
        self.next.setEnabled(False)
        self.next.clicked.connect(self.next_round)
        layout = game_layout(self)
        layout.addWidget(
            game_controls(
                self.level,
                self.round_count,
                self.timed,
                self.start,
                self.skip,
                self.instructions_button,
            )
        )
        layout.addWidget(self.score_strip)
        layout.addWidget(self.round_progress)
        layout.addWidget(self.summary)
        layout.addWidget(self.image, 1)
        layout.addLayout(choices)
        layout.addWidget(self.feedback)
        layout.addWidget(self.next)
        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.tick)
        self.timer.start()

    def set_catalog(self, catalog: ReferenceCatalog) -> None:
        self.catalog = catalog
        self.image.setToolTip(
            "Imported image; the folder name supplies its expression label."
            if catalog.custom
            else "Local FER2013 training image, automatically screened; labels can be ambiguous"
        )
        self.start.setEnabled(True)
        self.refresh_skip_button()
        self.image.setText("Choose Start game to begin")

    def set_challenge_active(self, active: bool) -> None:
        self.challenge_active = active
        self.start.setText("Stop game" if active else "Start game")
        for control in (self.level, self.round_count, self.timed):
            control.setEnabled(not active)
        self.refresh_skip_button()

    def refresh_skip_button(self) -> None:
        initialized = self.game is not None
        self.skip.setVisible(initialized)
        if not initialized:
            self.skip.setText("")
            self.skip.setEnabled(False)
            return
        playing = self.challenge_active and self.paused_at is None
        self.skip.setText("Skip round" if playing else "New game")
        self.skip.setEnabled(
            self.game is not None and self.game.state == "playing"
            if playing
            else self.catalog is not None
        )

    def skip_or_new_game(self) -> None:
        if not self.challenge_active or self.paused_at is not None:
            self.set_active(True)
            self.begin()
        else:
            self.submit(None)

    def toggle_challenge(self) -> None:
        if self.challenge_active:
            self.finish_challenge("Game stopped")
        else:
            self.begin()

    def finish_challenge(self, title: str) -> None:
        if self.game is None:
            return
        self.game.state = "finished"
        self.set_challenge_active(False)
        for button in self.buttons:
            button.setEnabled(False)
            button.hide()
        self.refresh_skip_button()
        self.next.setEnabled(False)
        self.image.clear()
        self.image.setText(
            f"{title}\n\nScore: {self.game.score}\n"
            f"Rounds answered: {len(self.game.results)} / {self.game.rounds}"
        )
        self.feedback.clear()
        self.feedback.hide()
        self.summary.hide()
        self.session_finished.emit()

    def reset_player(self) -> None:
        self.game = None
        self.set_challenge_active(False)
        self.image.clear()
        self.image.setText("Choose Start game to begin")
        self.summary.setText("Which emotion is it?")
        self.summary.show()
        self.score_strip.update_values("- / -", 0, 0)
        self.round_progress.setRange(0, 1)
        self.round_progress.setValue(0)
        self.feedback.clear()
        self.feedback.hide()
        for button in self.buttons:
            button.hide()
        self.refresh_skip_button()
        self.next.setEnabled(False)

    def begin(self) -> None:
        if self.catalog is None:
            return
        if self.game is not None and self.game.state != "finished":
            self.finish_challenge("Game stopped")
        seed = secrets.randbits(32)
        self.game = RecognitionGame(
            self.catalog,
            self.level.currentIndex(),
            self.round_count.value(),
            seed,
            self.timed.isChecked(),
        )
        self.set_challenge_active(True)
        self.session_started.emit(
            {
                "mode": "recognition",
                "level": self.game.level,
                "seed": seed,
                "rounds": self.game.rounds,
                "timed": self.game.timed,
            }
        )
        self.next_round()

    def next_round(self) -> None:
        if self.game is None or self.game.state == "finished":
            return
        try:
            reference = self.game.next_round(time.perf_counter())
            if reference is None:
                self.finish_challenge("Game complete")
                return
            show_rgb(self.image, self.catalog.image(reference))
            self.game.started_at = time.perf_counter()
        except (OSError, ValueError) as error:
            self.finish_challenge("Game ended")
            self.feedback.setText(f"Local reference unavailable: {error}")
            self.feedback.show()
            return
        for index, button in enumerate(self.buttons):
            button.setVisible(index < len(self.game.choices))
            if index < len(self.game.choices):
                button.setText(self.game.choices[index].capitalize())
                style_choice(button, self.game.choices[index])
                button.setEnabled(True)
        self.feedback.setText("")
        self.feedback.hide()
        self.refresh_skip_button()
        self.next.setEnabled(False)
        self.update_summary(time.perf_counter())

    def choose(self, index: int) -> None:
        if self.game is not None and self.game.state == "playing":
            self.submit(self.game.choices[index])

    def submit(self, answer: str | None) -> None:
        if self.game is None or self.game.state != "playing":
            return
        self.show_result(self.game.answer(answer, time.perf_counter()))

    def show_result(self, result: RecognitionResult) -> None:
        for button in self.buttons:
            button.setEnabled(False)
        self.refresh_skip_button()
        self.next.setEnabled(True)
        self.feedback.setText(
            f"{result.outcome.capitalize()} - Emotion: {result.target.capitalize()} - "
            f"+{result.points} points"
        )
        self.feedback.show()
        feedback_style(self.feedback, result.outcome)
        self.update_summary(time.perf_counter())
        self.result_ready.emit(result)

    def update_summary(self, now: float) -> None:
        if self.game is None:
            return
        number = min(len(self.game.results) + (self.game.state == "playing"), self.game.rounds)
        remaining = max(0.0, self.game.time_limit - (now - self.game.started_at))
        clock = (
            f" - {remaining:.1f} s left" if self.game.timed and self.game.state == "playing" else ""
        )
        self.summary.setText(f"Which emotion is it?{clock}")
        self.summary.show()
        self.score_strip.update_values(
            f"{number} / {self.game.rounds}",
            self.game.score,
            trailing_streak([result.outcome for result in self.game.results], "correct"),
        )
        self.round_progress.setRange(0, self.game.rounds)
        self.round_progress.setValue(len(self.game.results))

    def set_active(self, active: bool) -> None:
        now = time.perf_counter()
        if not active and self.paused_at is None:
            self.paused_at = now
        elif active and self.paused_at is not None:
            if self.game is not None:
                self.game.started_at += now - self.paused_at
            self.paused_at = None
        self.refresh_skip_button()

    def tick(self) -> None:
        if self.game is None or self.paused_at is not None:
            return
        now = time.perf_counter()
        result = self.game.tick(now)
        if result is not None:
            self.show_result(result)
        elif self.game.state == "playing":
            self.update_summary(now)
