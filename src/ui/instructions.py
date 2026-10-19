from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QScrollArea, QVBoxLayout, QWidget

INSTRUCTIONS = {
    "recognition": (
        "<h2>Recognition</h2>"
        "<p>Look at the face and choose the expression that matches it.</p>"
        "<ol><li>Choose a difficulty and number of rounds, then press <b>Start game</b>.</li>"
        "<li>Pick an answer. The game shows the correct answer and your points.</li>"
        "<li>Press <b>Next</b> to continue.</li></ol>"
        "<p>Use <b>Skip round</b> if an image is hard to read. "
        "If <b>Timed rounds</b> is enabled, each round has a time limit; "
        "try to finish before the timer reaches 0. "
        "Press <b>Stop game</b> to finish early.</p>"
        "<p><b>Scoring</b></p><ul>"
        "<li>Correct answer: 100 points on Easy, 200 on Standard, 300 on Hard.</li>"
        "<li>With timed rounds, faster answers earn up to 25 bonus points.</li>"
        "<li>Wrong answers and skipped rounds earn 0 points.</li></ul>"
    ),
    "imitation": (
        "<h2>Imitation</h2>"
        "<p>Try the target expression while watching yourself on camera.</p>"
        "<ol><li>Choose one expression or a <b>Mixed sequence</b>, plus difficulty and rounds.</li>"
        "<li>Press <b>Start game</b> and face the camera. "
        "The round starts when one clear face is visible.</li>"
        "<li>Follow the target and keep it steady until the green hold bar fills.</li>"
        "<li>If <b>Timed rounds</b> is enabled, each round has a time limit; "
        "try to finish before the timer reaches 0.</li>"
        "<li>Use <b>Skip round</b> to move on, or <b>Stop game</b> to end "
        "the game and turn off the camera.</li></ol>"
        "<p><b>Movement hints</b> are tips shown beside the camera while you play, "
        "such as &ldquo;raise your eyebrows&rdquo; or &ldquo;open your mouth a little.&rdquo;</p>"
        "<p>To turn them on, click <b>Setup hint</b> beside the camera. "
        "Look straight at the camera and keep your face relaxed and still for about two seconds. "
        "When setup is done, the button turns green with a tick and the camera switches off. "
        "Start a game to see the tips. You can play without setting them up.</p>"
        "<p>Enable <b>Face guide</b> beside <b>Expression</b> for a cyan outline and gold "
        "movement arrows over your face. Run <b>Setup hint</b> first. "
        "The guide suggests approximate movements and does not change scoring.</p>"
        "<p><b>Scoring</b></p><ul>"
        "<li>Fill the green hold bar to earn 100 points on Easy, 200 on Standard, "
        "or 300 on Hard.</li>"
        "<li>The hold bar resets if the app stops recognizing the target expression "
        "or loses track of your face.</li>"
        "<li>With timed rounds, finishing faster earns up to 25 bonus points.</li>"
        "<li>Skipped rounds earn 0 points.</li></ul>"
    ),
}


class GameInstructions(QDialog):
    def __init__(
        self,
        owner: QWidget,
        mode: str,
        set_active: Callable[[bool], None],
        is_active: Callable[[], bool],
    ) -> None:
        super().__init__(owner)
        self.owner = owner
        self.set_game_active = set_active
        self.is_game_active = is_active
        self.resume_active = False
        self.setWindowTitle("How to play")
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self.resize(560, 400)
        text = QLabel(INSTRUCTIONS[mode])
        text.setWordWrap(True)
        text.setStyleSheet("font-size:14px;")
        text.setAlignment(Qt.AlignmentFlag.AlignTop)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(text)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Got it")
        buttons.accepted.connect(self.accept)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.addWidget(scroll)
        layout.addWidget(buttons)
        self.finished.connect(self.resume_game)

    def open_for_game(self) -> None:
        if not self.isVisible():
            self.resume_active = self.is_game_active()
            self.set_game_active(False)
        self.show()
        self.raise_()
        self.activateWindow()

    def resume_game(self) -> None:
        self.set_game_active(self.resume_active and self.owner.isVisible())
