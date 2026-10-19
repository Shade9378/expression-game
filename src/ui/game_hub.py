from __future__ import annotations

import hashlib
import json
import queue
import sqlite3
import threading
from dataclasses import asdict
from pathlib import Path

from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from constants import CLASSES
from data.fer2013 import audit
from games.catalog import ReferenceCatalog, prepare_catalog
from games.custom_references import import_reference_folder
from games.feedback import NeutralBaseline
from games.progress import DEFAULT_SETTINGS, ELIGIBLE, ProgressStore, progress_path
from ui.imitation import ImitationWidget
from ui.recognition import RecognitionWidget

STORAGE_ERRORS = (OSError, ValueError, sqlite3.Error, KeyError, TypeError)


class GameHub(QWidget):
    settings_requested = Signal()
    player_changing = Signal()
    player_available = Signal(bool)

    def __init__(
        self,
        live: QWidget,
        preview: QLabel,
        asset: Path,
        database: Path | None = None,
        catalog_path: Path = Path("results/game/references.json"),
    ) -> None:
        super().__init__()
        self.store: ProgressStore | None = None
        self.profile: int | None = None
        self.storage_failed = False
        self.active_sessions: dict[str, str] = {}
        self.session_configs: dict[str, dict] = {}
        self.round_numbers: dict[str, int] = {}
        self.asset_hash = hashlib.sha256(asset.read_bytes()).hexdigest() if asset.is_file() else ""
        self.catalog_path = catalog_path
        self.reference_library = (database or progress_path()).parent / "references"
        self.reference_selection = self.reference_library / "selection.json"
        self.custom_catalog_path: Path | None = None
        self.reference_source = QComboBox()
        self.reference_source.addItems(["FER2013", "My images"])
        self.reference_source.model().item(1).setEnabled(False)
        try:
            selection = json.loads(self.reference_selection.read_text(encoding="utf-8"))
            if not isinstance(selection, dict):
                raise ValueError("Invalid reference selection")
            self.custom_catalog_path = (
                Path(selection["catalog"]) if selection.get("catalog") else None
            )
            if self.custom_catalog_path is not None and self.custom_catalog_path.is_file():
                self.reference_source.model().item(1).setEnabled(True)
                self.reference_source.setCurrentIndex(1 if selection.get("custom") else 0)
        except (OSError, ValueError, KeyError, TypeError):
            pass
        self.worker: threading.Thread | None = None
        self.cancel_worker = threading.Event()
        self.messages: queue.Queue = queue.Queue()
        self.model_provider = lambda: ""
        self.settings = DEFAULT_SETTINGS.copy()
        self.aliases = QComboBox()
        self.alias = QLineEdit()
        self.alias.setMaxLength(40)
        self.alias.setPlaceholderText("New player alias")
        self.add = QPushButton("Add player")
        self.delete = QPushButton("Delete player")
        self.adaptive = QCheckBox("Adaptive difficulty")
        self.prepare = QPushButton("Prepare FER2013 images")
        self.import_images = QPushButton("Import my images")
        self.import_images.setToolTip("Choose a parent folder containing: " + ", ".join(CLASSES))
        self.settings_button = QPushButton("Settings")
        self.settings_button.clicked.connect(self.settings_requested.emit)
        self.notice = QLabel()
        self.notice.setWordWrap(True)
        self.reference_status = QLabel()
        self.reference_status.setWordWrap(True)
        self.reference_status.hide()
        self.storage_notice = QLabel()
        self.storage_notice.setWordWrap(True)
        self.player_notice = QLabel("Add a player to start playing.")
        self.player_notice.setFixedHeight(self.player_notice.sizeHint().height())
        player_notice_policy = self.player_notice.sizePolicy()
        player_notice_policy.setRetainSizeWhenHidden(True)
        self.player_notice.setSizePolicy(player_notice_policy)
        toolbar = QHBoxLayout()
        for widget in (
            QLabel("Player"),
            self.aliases,
            self.alias,
            self.add,
            self.delete,
            self.settings_button,
        ):
            toolbar.addWidget(widget)
        self.tabs = QTabWidget()
        self.recognition = RecognitionWidget()
        self.imitation = ImitationWidget(preview)
        self.progress = QWidget()
        self.progress_layout = QVBoxLayout(self.progress)
        self.progress_summary = QLabel("No recorded rounds yet")
        self.progress_layout.addWidget(self.progress_summary)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["Mode", "Target", "Outcome", "Level", "Points", "Time (s)"]
        )
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.progress_layout.addWidget(self.table)
        self.canvas = None
        for widget, name in (
            (live, "Live practise"),
            (self.recognition, "Recognition"),
            (self.imitation, "Imitation"),
            (self.progress, "Progress"),
        ):
            self.tabs.addTab(widget, name)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(toolbar)
        layout.addWidget(self.reference_status)
        layout.addWidget(self.storage_notice)
        layout.addWidget(self.player_notice)
        layout.addWidget(self.tabs, 1)
        try:
            self.store = ProgressStore(database)
            self.reload_profiles()
            self.load_profile()
        except STORAGE_ERRORS as error:
            self.storage_error(error)
        self.aliases.currentIndexChanged.connect(self.switch_profile)
        self.add.clicked.connect(self.add_profile)
        self.delete.clicked.connect(self.delete_player)
        self.adaptive.toggled.connect(self.save_settings)
        self.prepare.clicked.connect(self.prepare_catalog)
        self.import_images.clicked.connect(self.import_custom_images)
        self.reference_source.currentIndexChanged.connect(self.switch_reference_source)
        self.tabs.currentChanged.connect(self.changed_tab)
        for mode, widget in (("recognition", self.recognition), ("imitation", self.imitation)):
            widget.session_started.connect(self.start_session)
            widget.session_finished.connect(lambda mode=mode: self.end_session(mode))
        self.recognition.result_ready.connect(
            lambda result: self.record_result("recognition", result, result.reference_id)
        )
        self.imitation.result_ready.connect(
            lambda result, reference: self.record_result("imitation", result, reference.sample_id)
        )
        self.imitation.calibration_ready.connect(self.save_calibration)
        for control in (self.imitation.threshold, self.imitation.hold, self.imitation.sensitivity):
            control.valueChanged.connect(self.save_settings)
        self.recognition.level.currentIndexChanged.connect(self.save_settings)
        self.imitation.level.currentIndexChanged.connect(self.save_settings)
        self.load_catalog()
        self.changed_tab(0)
        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.poll_worker)
        self.timer.start()

    def storage_error(self, error: Exception) -> None:
        self.storage_failed = True
        self.profile = None
        self.storage_notice.setText(
            f"Local progress unavailable: {error}. "
            "For a shared checkout, set FER_GAME_DIR to a native local directory."
        )
        self.apply_player_state()

    def apply_player_state(self) -> None:
        active = self.profile is not None and not self.storage_failed
        self.tabs.setEnabled(active)
        self.adaptive.setEnabled(active)
        self.aliases.setEnabled(active)
        self.delete.setEnabled(active)
        self.add.setEnabled(self.store is not None and not self.storage_failed)
        self.alias.setEnabled(self.add.isEnabled())
        self.imitation.advanced.setEnabled(active)
        self.player_notice.setVisible(not active)
        self.changed_tab(self.tabs.currentIndex())
        self.player_available.emit(active)

    def reload_profiles(self, selected: int | None = None) -> None:
        self.aliases.blockSignals(True)
        self.aliases.clear()
        for profile in self.store.profiles():
            self.aliases.addItem(profile["name"], profile["id"])
        if selected is not None:
            self.aliases.setCurrentIndex(self.aliases.findData(selected))
        self.aliases.blockSignals(False)

    def clear_current_player(self) -> None:
        self.player_changing.emit()
        self.imitation.reset_player()
        self.recognition.reset_player()
        for mode in list(self.active_sessions):
            self.end_session(mode)
        self.session_configs.clear()
        self.round_numbers.clear()
        self.table.setRowCount(0)
        self.progress_summary.setText("No recorded rounds yet")
        if self.canvas is not None:
            self.canvas.figure.clear()
            self.canvas.draw_idle()

    def switch_profile(self) -> None:
        self.clear_current_player()
        self.load_profile()

    def load_profile(self) -> None:
        if self.store is None:
            return
        try:
            self.profile = self.aliases.currentData()
            if self.profile is None:
                self.apply_player_state()
                return
            self.settings = self.store.settings(self.profile)
            controls = (
                self.adaptive,
                self.recognition.level,
                self.imitation.level,
                self.imitation.threshold,
                self.imitation.hold,
                self.imitation.sensitivity,
            )
            for control in controls:
                control.blockSignals(True)
            self.adaptive.setChecked(self.settings["adaptive"])
            self.recognition.level.setCurrentIndex(self.settings["recognition_level"])
            self.imitation.level.setCurrentIndex(self.settings["imitation_level"])
            self.imitation.threshold.setValue(self.settings["threshold"])
            self.imitation.hold.setValue(self.settings["hold_seconds"])
            self.imitation.deadline.setValue((25, 22, 18)[self.settings["imitation_level"]])
            self.imitation.sensitivity.setValue(self.settings["sensitivity"])
            for control in controls:
                control.blockSignals(False)
            self.imitation.calibration = self.store.calibration(self.profile, self.asset_hash)
            self.imitation.refresh_calibration_button()
            self.imitation.movements.clear()
            self.refresh_progress()
            self.apply_player_state()
        except STORAGE_ERRORS as error:
            self.storage_error(error)

    def add_profile(self) -> None:
        if self.store is None:
            return
        try:
            identifier = self.store.add_profile(self.alias.text())
            self.clear_current_player()
            self.reload_profiles(identifier)
            self.load_profile()
            self.alias.clear()
            self.notice.setText("")
        except STORAGE_ERRORS as error:
            self.notice.setText(f"Cannot add player: {error}")

    def delete_player(self) -> None:
        if self.store is None or self.profile is None:
            return
        self.recognition.set_active(False)
        self.imitation.set_active(False)
        reply = QMessageBox.question(
            self,
            "Delete player",
            f"Delete {self.aliases.currentText()}? Their scores and movement-hint settings "
            "will be removed.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            self.changed_tab(self.tabs.currentIndex())
            return
        try:
            identifier = self.profile
            self.clear_current_player()
            self.store.delete_profile(identifier)
            self.reload_profiles()
            self.load_profile()
        except STORAGE_ERRORS as error:
            self.storage_notice.setText(f"Could not delete player: {error}")
            self.load_profile()

    def save_settings(self) -> None:
        if self.store is None or self.profile is None:
            return
        self.settings = {
            "adaptive": self.adaptive.isChecked(),
            "recognition_level": self.recognition.level.currentIndex(),
            "imitation_level": self.imitation.level.currentIndex(),
            "threshold": self.imitation.threshold.value(),
            "hold_seconds": self.imitation.hold.value(),
            "sensitivity": self.imitation.sensitivity.value(),
        }
        try:
            self.store.save_settings(self.profile, self.settings)
        except STORAGE_ERRORS as error:
            self.storage_error(error)

    def start_session(self, settings: dict) -> None:
        mode = settings["mode"]
        self.session_configs[mode] = settings.copy()
        self.end_session(mode)
        if self.store is None or self.profile is None:
            return
        try:
            self.active_sessions[mode] = self.store.start_session(
                self.profile,
                mode,
                settings["level"],
                self.recognition.catalog.screening_model
                if mode == "recognition"
                else self.model_provider(),
                settings,
            )
            self.round_numbers[mode] = 0
        except STORAGE_ERRORS as error:
            self.storage_error(error)

    def end_session(self, mode: str) -> None:
        identifier = self.active_sessions.pop(mode, None)
        if identifier is not None and self.store is not None:
            try:
                self.store.end_session(identifier)
                if self.adaptive.isChecked():
                    widget = self.recognition if mode == "recognition" else self.imitation
                    level = self.store.next_level(self.profile, mode, widget.level.currentIndex())
                    widget.level.setCurrentIndex(level)
                    self.save_settings()
                self.refresh_progress()
            except STORAGE_ERRORS as error:
                self.storage_error(error)

    def record_result(self, mode: str, result: object, reference: str) -> None:
        if mode not in self.active_sessions or self.store is None:
            return
        try:
            self.store.record_round(
                self.active_sessions[mode], self.round_numbers[mode], asdict(result), reference
            )
            self.round_numbers[mode] += 1
        except STORAGE_ERRORS as error:
            self.storage_error(error)

    def save_calibration(self, baseline: NeutralBaseline) -> None:
        if self.store is not None and self.profile is not None:
            try:
                self.store.save_calibration(self.profile, self.asset_hash, baseline)
            except STORAGE_ERRORS as error:
                self.storage_error(error)

    def load_catalog(self) -> None:
        try:
            path = (
                self.custom_catalog_path
                if self.reference_source.currentIndex() == 1
                else self.catalog_path
            )
            if path is None:
                raise ValueError("Import your labeled image folders first")
            catalog = ReferenceCatalog(path)
            self.recognition.set_catalog(catalog)
            self.imitation.set_catalog(catalog)
            self.notice.setText(f"{len(catalog.references)} local reference images ready")
            self.reference_status.hide()
        except (OSError, ValueError, KeyError, TypeError) as error:
            self.reference_status.setText("Game images aren't ready. Open Settings to set them up.")
            self.reference_status.show()
            self.notice.setText(
                f"References unavailable: {error}. "
                "Prepare FER2013 references or import your images."
            )

    def save_reference_selection(self) -> None:
        self.reference_library.mkdir(parents=True, exist_ok=True)
        temporary = self.reference_selection.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "catalog": str(self.custom_catalog_path) if self.custom_catalog_path else None,
                    "custom": self.reference_source.currentIndex() == 1,
                }
            ),
            encoding="utf-8",
        )
        temporary.replace(self.reference_selection)

    def switch_reference_source(self) -> None:
        if self.recognition.challenge_active:
            self.recognition.finish_challenge("Game stopped")
        self.imitation.cancel()
        self.player_changing.emit()
        self.recognition.reset_player()
        self.imitation.reset_player()
        self.recognition.catalog = self.imitation.catalog = None
        self.recognition.start.setEnabled(False)
        self.imitation.start.setEnabled(False)
        self.load_catalog()
        try:
            self.save_reference_selection()
        except OSError as error:
            self.notice.setText(f"Could not save reference selection: {error}")

    def set_reference_controls(self, enabled: bool) -> None:
        for control in (self.prepare, self.import_images, self.reference_source):
            control.setEnabled(enabled)

    def import_custom_images(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            return
        selected = QFileDialog.getExistingDirectory(
            self, "Choose a folder containing expression-named folders"
        )
        if not selected:
            return
        self.set_reference_controls(False)
        self.notice.setText("Importing your images...")
        self.cancel_worker.clear()

        def work() -> None:
            try:
                self.messages.put(
                    import_reference_folder(
                        Path(selected), self.reference_library, self.cancel_worker.is_set
                    )
                )
            except Exception as error:
                self.messages.put(f"Image import failed: {error}")

        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()

    def prepare_catalog(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            return
        self.set_reference_controls(False)
        self.notice.setText("Checking FER2013 images and preparing references...")
        self.cancel_worker.clear()

        def work() -> None:
            try:
                data = Path("data/FER2013")
                if not data.exists():
                    raise FileNotFoundError("Place FER2013 in data/FER2013/ first")
                audit(data, Path("results/data"))
                if self.cancel_worker.is_set():
                    raise InterruptedError("Reference preparation cancelled")
                prepare_catalog(
                    Path("results/data/manifest.json"),
                    Path("results/onnx/resnet18_fp32.onnx"),
                    self.catalog_path,
                    cancelled=self.cancel_worker.is_set,
                )
                self.messages.put(None)
            except Exception as error:
                self.messages.put(str(error))

        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()

    def poll_worker(self) -> None:
        try:
            error = self.messages.get_nowait()
        except queue.Empty:
            return
        self.set_reference_controls(True)
        if isinstance(error, dict):
            self.custom_catalog_path = Path(error["catalog"])
            self.reference_source.model().item(1).setEnabled(True)
            self.reference_source.blockSignals(True)
            self.reference_source.setCurrentIndex(1)
            self.reference_source.blockSignals(False)
            self.switch_reference_source()
            skipped = error["skipped"]
            count = error["imported"]
            noun = "image" if count == 1 else "images"
            self.notice.setText(f"Imported {count} {noun}; skipped {len(skipped)}.")
            if skipped:
                details = "\n".join(f"{row['file']}: {row['reason']}" for row in skipped)
                self.notice.setToolTip(details)
                dialog = QMessageBox(self)
                dialog.setWindowTitle("Image import")
                dialog.setText(self.notice.text())
                dialog.setDetailedText(details)
                dialog.exec()
            return
        if error is None:
            self.reference_source.blockSignals(True)
            self.reference_source.setCurrentIndex(0)
            self.reference_source.blockSignals(False)
            self.switch_reference_source()
        else:
            self.notice.setText(f"Reference preparation failed: {error}")

    def changed_tab(self, index: int) -> None:
        active = self.profile is not None and not self.storage_failed
        self.recognition.set_active(active and index == 1)
        self.imitation.set_active(active and index == 2)
        if index == 3:
            self.refresh_progress()

    def refresh_progress(self) -> None:
        if self.store is None or self.profile is None:
            return
        rows = self.store.history(self.profile)
        self.table.setRowCount(len(rows))
        for index, row in enumerate(rows):
            values = (
                row["mode"],
                row["target"],
                row["outcome"],
                row["level"] + 1,
                row["points"],
                f"{row['elapsed']:.1f}",
            )
            for column, value in enumerate(values):
                self.table.setItem(index, column, QTableWidgetItem(str(value)))
        summaries = []
        for mode, label in (
            ("recognition", "Recognition correct"),
            ("imitation", "Imitation completed"),
        ):
            eligible = [r for r in rows if r["mode"] == mode and r["outcome"] in ELIGIBLE]
            success = sum(r["outcome"] in ("correct", "success") for r in eligible)
            summaries.append(f"{label}: {success}/{len(eligible)}" if eligible else f"{label}: -")
        self.progress_summary.setText(" - ".join(summaries))
        if self.canvas is not None and not rows:
            self.canvas.figure.clear()
            self.canvas.draw_idle()
        if not rows or self.tabs.currentIndex() != 3:
            return
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        if self.canvas is None:
            self.canvas = FigureCanvasQTAgg(Figure(figsize=(5, 2), layout="constrained"))
            self.progress_layout.addWidget(self.canvas, 1)
        self.canvas.figure.clear()
        axis = self.canvas.figure.subplots()
        for mode, color in (("recognition", "#0072B2"), ("imitation", "#D55E00")):
            eligible = [r for r in rows if r["mode"] == mode and r["outcome"] in ELIGIBLE]
            if eligible:
                values = [r["outcome"] in ("correct", "success") for r in eligible]
                rates = [
                    100 * sum(values[max(0, i - 7) : i + 1]) / min(i + 1, 8)
                    for i in range(len(values))
                ]
                axis.plot(range(1, len(rates) + 1), rates, color=color, label=mode.capitalize())
        axis.set(xlabel="Round", ylabel="Last 8 rounds (%)", ylim=(-5, 105))
        if axis.lines:
            axis.legend()
        self.canvas.draw_idle()

    def shutdown(self) -> None:
        self.cancel_worker.set()
        self.timer.stop()
        self.recognition.timer.stop()
        self.imitation.timer.stop()
        self.imitation.cancel()
        for mode in list(self.active_sessions):
            self.end_session(mode)
        if self.store is not None:
            self.store.close()
            self.store = None
