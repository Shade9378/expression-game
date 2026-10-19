from __future__ import annotations

import hashlib
import json
import shutil
import uuid
import warnings
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from constants import CLASSES
from data.fer2013 import Record
from games.catalog import Reference, ReferenceCatalog

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def import_reference_folder(
    source: Path, library: Path, cancelled: Callable[[], bool] | None = None
) -> dict:
    source = source.resolve()
    library = library.resolve()
    if not source.is_dir():
        raise ValueError("Choose a folder containing expression-named subfolders")
    if library == source or source in library.parents:
        raise ValueError("Choose your image folder, not the application's reference library")
    destination = library / uuid.uuid4().hex
    destination.mkdir(parents=True)
    records = []
    references = []
    skipped = []
    seen: dict[str, tuple[str, str]] = {}
    try:
        for path in sorted(source.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if cancelled is not None and cancelled():
                raise InterruptedError("Image import cancelled")
            relative = path.relative_to(source)
            label = relative.parts[0].lower() if len(relative.parts) > 1 else ""
            if label not in CLASSES or path.is_symlink():
                skipped.append(
                    {"file": str(relative), "reason": "Unknown expression folder or link"}
                )
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", Image.DecompressionBombWarning)
                    with Image.open(path) as original:
                        image = ImageOps.exif_transpose(original).convert("RGB")
                        image.thumbnail((1024, 1024))
                if min(image.size) < 16:
                    raise ValueError("Image must be at least 16 pixels wide and tall")
            except (
                OSError,
                ValueError,
                Image.DecompressionBombWarning,
                Image.DecompressionBombError,
            ) as error:
                skipped.append({"file": str(relative), "reason": str(error)})
                continue
            pixels = np.asarray(image)
            digest = hashlib.sha256(pixels.tobytes()).hexdigest()
            identity = hashlib.sha256(pixels.tobytes() + str(image.size).encode()).hexdigest()
            if identity in seen:
                previous_label, previous_file = seen[identity]
                if previous_label != label:
                    raise ValueError(
                        f"Conflicting labels for identical images: {previous_file}, {relative}"
                    )
                skipped.append({"file": str(relative), "reason": f"Duplicate of {previous_file}"})
                continue
            seen[identity] = (label, str(relative))
            copied = destination / f"{identity}.png"
            image.save(copied)
            sample_id = f"custom/{destination.name}/{label}/{identity}"
            records.append(Record(sample_id, str(copied), CLASSES.index(label), "custom"))
            references.append(Reference(sample_id, label, None, None, digest))
        if not references:
            details = "; ".join(f"{row['file']}: {row['reason']}" for row in skipped[:3])
            raise ValueError(
                "No usable images found. Use folders named " + ", ".join(CLASSES) + ". " + details
            )
        report = {
            "schema": 2,
            "source": "custom",
            "classes": list(CLASSES),
            "model": "user-supplied folder labels; no model screening",
            "records": [asdict(record) for record in records],
            "references": [asdict(reference) for reference in references],
            "skipped": skipped,
        }
        catalog = destination / "catalog.json"
        catalog.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        ReferenceCatalog(catalog)
        return {"catalog": str(catalog), "imported": len(references), "skipped": skipped}
    except Exception:
        shutil.rmtree(destination)
        raise
