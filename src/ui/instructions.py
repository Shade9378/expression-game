from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QVBoxLayout, QWidget

INSTRUCTIONS = {
    "recognition": (
        "<h2>Recognition</h2>"
        "<p>Look at the face and choose the expression that matches it.</p>"
        "<ol><li>Choose a difficulty and number of rounds, then press <b>Start game</b>.</li>"
        "<li>Pick an answer. The game shows the correct answer and your points.</li>"
        "<li>Press <b>Next</b> to continue.</li></ol>"
        "<p>Use <b>Skip round</b> if an image is hard to read. "
        "With <b>Timed rounds</b> enabled, answer before the timer runs out. "
        "Press <b>Stop game</b> to finish early.</p>"
    ),
    "imitation": (
        "<h2>Imitation</h2>"
        "<p>Try the target expression while watching yourself on camera.</p>"
        "<ol><li>Choose one expression or a <b>Mixed sequence</b>, plus difficulty and rounds.</li>"
        "<li>Press <b>Start game</b> and face the camera. "
        "The round starts when one clear face is visible.</li>"
        "<li>Follow the target and keep it steady until the green hold bar fills.</li>"
        "<li>Turn off <b>Timed rounds</b> to practise without a countdown or time bonus.</li>"
        "<li>Use <b>Skip round</b> to move on, or <b>Stop game</b> to end "
        "the game and turn off the camera.</li></ol>"
        "<p><b>Movement hints are optional.</b> Press <b>Setup hint</b> in the "
        "panel beside the camera, then relax your face for two seconds. "
        "The hints suggest movements such as raising your eyebrows or opening your mouth.</p>"
        "<p>Enable <b>Face guide</b> beside <b>Expression</b> for a cyan outline and gold "
        "movement arrows over your face. Run <b>Setup hint</b> first. "
        "The guide suggests approximate movements and does not change scoring.</p>"
        "<p>Scoring follows the app's recognition rules. Tracking can be wrong.</p>"
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
        pause_note = QLabel("The round pauses while these instructions are open.")
        pause_note.setWordWrap(True)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Got it")
        buttons.accepted.connect(self.accept)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.addWidget(text)
        layout.addWidget(pause_note)
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
