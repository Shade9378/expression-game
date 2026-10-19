from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from constants import CLASSES
from data.fer2013 import load_manifest, read_image
from inference.backends import OnnxBackend


@dataclass(frozen=True)
class Reference:
    sample_id: str
    label: str
    probability: float | None
    margin: float | None
    image_sha256: str


class ReferenceCatalog:
    def __init__(self, path: Path) -> None:
        self.path = path
        data = json.loads(path.read_text())
        self.screening_model = str(data.get("model", "unknown"))
        self.custom = data.get("schema") == 2 and data.get("source") == "custom"
        if (data.get("schema") != 1 and not self.custom) or data.get("classes") != list(CLASSES):
            raise ValueError("Unsupported reference catalog")
        if self.custom:
            from data.fer2013 import Record

            records = [Record(**row) for row in data["records"]]
        else:
            manifest = Path(data["manifest"])
            if hashlib.sha256(manifest.read_bytes()).hexdigest() != data["manifest_sha256"]:
                raise ValueError("Reference manifest changed; prepare the catalog again")
            records = load_manifest(manifest)
        self.records = {record.sample_id: record for record in records}
        self.references = tuple(Reference(**row) for row in data["references"])
        ids = set()
        for reference in self.references:
            record = self.records.get(reference.sample_id)
            if (
                record is None
                or record.split != ("custom" if self.custom else "train")
                or record.label not in range(len(CLASSES))
                or CLASSES[record.label] != reference.label
                or reference.sample_id in ids
                or (
                    reference.probability is not None
                    if self.custom
                    else reference.probability is None or not 0 <= reference.probability <= 1
                )
            ):
                raise ValueError("Invalid or repeated training reference")
            ids.add(reference.sample_id)
        self.labels = tuple(
            label for label in CLASSES if any(r.label == label for r in self.references)
        )
        if not self.references:
            raise ValueError("Reference catalog is empty")
        if not self.custom and set(self.labels) != set(CLASSES):
            raise ValueError("Reference catalog must cover all seven labels")

    def image(self, reference: Reference) -> np.ndarray:
        image = np.asarray(read_image(self.records[reference.sample_id]))
        if hashlib.sha256(image.tobytes()).hexdigest() != reference.image_sha256:
            raise ValueError("Reference image changed; prepare the catalog again")
        return image

    def choose(self, label: str, level: int, rng: random.Random, used: set[str]) -> Reference:
        candidates = sorted(
            (r for r in self.references if r.label == label and r.sample_id not in used),
            key=lambda r: r.probability if r.probability is not None else 0.0,
            reverse=True,
        )
        if not candidates:
            candidates = [r for r in self.references if r.label == label]
        if not candidates:
            raise ValueError(f"No local references for {label}")
        if self.custom:
            return rng.choice(candidates)
        width = max(1, (len(candidates) + 2) // 3)
        start = min(level * width, len(candidates) - 1)
        return rng.choice(candidates[start : start + width])


def prepare_catalog(
    manifest: Path,
    model: Path,
    output: Path,
    per_class: int = 12,
    candidates_per_class: int = 80,
    cancelled: Callable[[], bool] | None = None,
) -> dict:
    if per_class < 3 or candidates_per_class < per_class:
        raise ValueError("At least three references per class and enough candidates are required")
    records = load_manifest(manifest)
    backend = OnnxBackend(model)
    if backend.metadata["input_kind"] != "image":
        raise ValueError("Reference screening requires an image-only ONNX model")
    rng = random.Random(42)
    references = []
    rejected = Counter()
    for index, label in enumerate(CLASSES):
        candidates = [r for r in records if r.split == "train" and r.label == index]
        rng.shuffle(candidates)
        accepted = []
        for record in candidates[:candidates_per_class]:
            if cancelled is not None and cancelled():
                raise InterruptedError("Reference preparation cancelled")
            try:
                image = np.asarray(read_image(record))
                if image.mean() < 25 or image.mean() > 230 or image.std() < 12:
                    rejected["lighting_or_contrast"] += 1
                    continue
                probabilities = backend.predict_rgb(image)
                order = np.argsort(probabilities)
                margin = float(probabilities[order[-1]] - probabilities[order[-2]])
                if int(order[-1]) != index or probabilities[index] < 0.5 or margin < 0.1:
                    rejected["classifier_disagreement_or_uncertainty"] += 1
                    continue
                accepted.append(
                    Reference(
                        record.sample_id,
                        label,
                        float(probabilities[index]),
                        margin,
                        hashlib.sha256(image.tobytes()).hexdigest(),
                    )
                )
            except (OSError, ValueError):
                rejected["missing_or_invalid_image"] += 1
        if len(accepted) < per_class:
            raise ValueError(
                f"Only {len(accepted)} reliable {label} references; need {per_class}. "
                "Increase candidates_per_class; no images were downloaded."
            )
        accepted.sort(key=lambda r: r.probability, reverse=True)
        positions = np.linspace(0, len(accepted) - 1, per_class, dtype=int)
        references.extend(accepted[i] for i in positions)
    report = {
        "schema": 1,
        "classes": list(CLASSES),
        "manifest": str(manifest),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "model": str(model),
        "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "seed": 42,
        "screening": "Training-only; contrast/lighting, correct FER label, p>=0.5, margin>=0.1",
        "human_reviewed": False,
        "images_copied": False,
        "rejected": dict(rejected),
        "references": [asdict(r) for r in references],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(output)
    return report
