from __future__ import annotations

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPalette, QPen, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLayout,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from constants import CLASSES
from ui.colors import EXPRESSION_COLORS

RESULT_COLORS = {
    "correct": ("#15803d", "#bbf7d0", "#14532d"),
    "wrong": ("#b91c1c", "#fecaca", "#7f1d1d"),
}


def tint(widget: QWidget, color: str, amount: float = 0.15) -> str:
    base = widget.palette().color(QPalette.ColorRole.Window)
    accent = QColor(color)
    return QColor(
        *[
            round(a * amount + b * (1 - amount))
            for a, b in zip(accent.getRgb()[:3], base.getRgb()[:3], strict=True)
        ]
    ).name()


def expression_icon(label: str, color: str) -> QIcon:
    pixmap = QPixmap(48, 48)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(QPen(QColor(color).darker(250), 2.5))
    painter.setBrush(QColor(color))
    painter.drawEllipse(QRectF(3, 3, 42, 42))
    painter.setPen(QPen(QColor("#ffffff"), 2.4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
    for x in (17, 31):
        if label in ("fear", "surprise"):
            painter.drawEllipse(QRectF(x - 2, 16, 4, 6))
        else:
            painter.drawLine(x, 18, x, 21)
    if label == "angry":
        painter.drawLine(13, 13, 20, 16)
        painter.drawLine(28, 16, 35, 13)
    elif label in ("fear", "sad"):
        painter.drawLine(13, 14, 20, 11)
        painter.drawLine(28, 11, 35, 14)
    mouth = QPainterPath()
    if label in ("surprise", "fear"):
        painter.drawEllipse(QRectF(20, 28, 8, 10))
    elif label == "happy":
        mouth.moveTo(14, 28)
        mouth.quadTo(24, 42, 34, 28)
    elif label in ("sad", "angry"):
        mouth.moveTo(15, 35)
        mouth.quadTo(24, 25, 33, 35)
    elif label == "disgust":
        mouth.moveTo(15, 33)
        mouth.cubicTo(21, 27, 27, 38, 33, 30)
    else:
        mouth.moveTo(16, 32)
        mouth.lineTo(32, 32)
    painter.drawPath(mouth)
    painter.end()
    return QIcon(pixmap)


def style_choice(button: QPushButton, label: str) -> None:
    color = EXPRESSION_COLORS[CLASSES.index(label)]
    edge = QColor(color).darker(160).name()
    background = QColor(
        *[round(channel * 0.25 + 255 * 0.75) for channel in QColor(color).getRgb()[:3]]
    ).name()
    foreground = "#1e293b"
    icon = expression_icon(label, color)
    icon.addPixmap(icon.pixmap(48, 48), QIcon.Mode.Disabled)
    button.setIcon(icon)
    button.setStyleSheet(
        f"QPushButton {{background:{background}; color:{foreground}; border:2px solid {edge}; "
        "border-radius:12px; padding:8px 14px; font-weight:600; text-align:left;}"
        f"QPushButton:hover {{background:{background}; border:2px solid {edge};}}"
        f"QPushButton:disabled {{background:{background}; color:{foreground}; "
        f"border:2px solid {edge};}}"
    )


def check_icon() -> QIcon:
    pixmap = QPixmap(24, 24)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(QPen(QColor("white"), 2.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
    path = QPainterPath()
    path.moveTo(4, 12)
    path.lineTo(10, 18)
    path.lineTo(20, 6)
    painter.drawPath(path)
    painter.end()
    return QIcon(pixmap)


def feedback_style(label: QLabel, outcome: str = "") -> None:
    state = "correct" if outcome == "success" else outcome
    if state in RESULT_COLORS:
        color, background, foreground = RESULT_COLORS[state]
    else:
        color = "#bd852f" if outcome == "timeout" else "#8174df"
        background, foreground = tint(label, color), "palette(window-text)"
    label.setStyleSheet(
        f"background:{background}; color:{foreground}; border-left:4px solid {color}; "
        "border-radius:8px; padding:8px 12px;"
    )


def style_game(widget: QWidget) -> None:
    widget.setStyleSheet("""
        QPushButton {padding:5px 12px; min-height:24px; border-radius:9px;
                     border:1px solid palette(mid); background:palette(button);}
        QPushButton:hover {border-color:#9688ee;}
        QPushButton[role="primary"] {background:#7661d5; color:white;
                                      border:1px solid #7661d5; font-weight:600;}
        QPushButton[role="primary"]:hover {background:#8570e3;}
        QPushButton:disabled {color:palette(mid); background:palette(window);
                              border-color:palette(mid);}
        QProgressBar {border:1px solid palette(mid); border-radius:8px;
                      min-height:20px; text-align:center; background:palette(base);}
        QProgressBar::chunk {background:#8174df; border-radius:7px;}
    """)


def style_image(label: QWidget) -> None:
    label.setStyleSheet(
        f"background:{tint(label, '#8174df', 0.06)}; "
        "border:1px solid palette(mid); border-radius:14px; padding:8px;"
    )


def game_layout(widget: QWidget) -> QVBoxLayout:
    shell = QVBoxLayout(widget)
    shell.setContentsMargins(0, 0, 0, 0)
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.Shape.NoFrame)
    content = QWidget()
    layout = QVBoxLayout(content)
    layout.setAlignment(Qt.AlignmentFlag.AlignTop)
    layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
    scroll.setWidget(content)
    shell.addWidget(scroll)
    return layout


def game_controls(
    level: QWidget,
    rounds: QWidget,
    option: QWidget,
    start: QPushButton,
    skip: QPushButton,
    instructions: QPushButton,
) -> QWidget:
    controls = QWidget()
    controls.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    layout = QVBoxLayout(controls)
    layout.setContentsMargins(0, 0, 0, 0)
    skip_policy = skip.sizePolicy()
    skip_policy.setRetainSizeWhenHidden(True)
    skip.setSizePolicy(skip_policy)
    settings = QHBoxLayout()
    for title, control in (("Difficulty", level), ("Rounds", rounds)):
        settings.addWidget(QLabel(title))
        settings.addWidget(control, 1)
    settings.addWidget(option, 2)
    actions = QHBoxLayout()
    for button in (start, skip, instructions):
        actions.addWidget(button, 1)
    layout.addLayout(settings)
    layout.addLayout(actions)
    return controls


def round_progress_bar() -> QProgressBar:
    progress = QProgressBar()
    progress.setTextVisible(False)
    progress.setFixedHeight(8)
    progress.setStyleSheet(
        "QProgressBar {min-height:6px; border:0; border-radius:4px; "
        "background:palette(mid);} QProgressBar::chunk {background:#8174df; border-radius:4px;}"
    )
    progress.setRange(0, 1)
    progress.setValue(0)
    return progress


class ScoreStrip(QWidget):
    def __init__(self, first: str = "ROUND") -> None:
        super().__init__()
        self.setFixedHeight(64)
        self.values: list[QLabel] = []
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        for title, color in zip(
            (first, "POINTS", "STREAK"), ("#8174df", "#d59a35", "#2f9e75"), strict=True
        ):
            card = QFrame()
            card.setStyleSheet(
                f"QFrame {{background:{tint(self, color, 0.1)}; "
                f"border:1px solid {tint(self, color, 0.4)}; border-radius:12px;}}"
                "QLabel {border:none; background:transparent;}"
            )
            inner = QVBoxLayout(card)
            inner.setContentsMargins(14, 8, 14, 8)
            inner.setSpacing(2)
            caption = QLabel(title)
            caption.setStyleSheet("font-size:10px; font-weight:600;")
            value = QLabel("\N{FIRE} 0" if title == "STREAK" else "0")
            value.setStyleSheet("font-size:22px; font-weight:700;")
            inner.addWidget(caption)
            inner.addWidget(value)
            self.values.append(value)
            layout.addWidget(card, 1)

    def update_values(self, round_text: str, points: int, streak: int) -> None:
        for label, value in zip(
            self.values, (round_text, str(points), f"\N{FIRE} {streak}"), strict=True
        ):
            label.setText(value)


def trailing_streak(outcomes: list[str], success: str) -> int:
    count = 0
    for outcome in reversed(outcomes):
        if outcome != success:
            break
        count += 1
    return count
