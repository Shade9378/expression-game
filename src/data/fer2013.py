from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from constants import CLASSES

USAGES = {"Training": "train", "PublicTest": "val", "PrivateTest": "test"}
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class Record:
    sample_id: str
    source: str
    label: int
    split: str
    csv_row: int | None = None


@lru_cache(maxsize=2)
def _csv_pixels(path: str) -> tuple[str, ...]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return tuple(row["pixels"] for row in csv.DictReader(handle))


def read_image(record: Record) -> Image.Image:
    if record.csv_row is None:
        with Image.open(record.source) as image:
            return image.convert("RGB")
    rows = _csv_pixels(record.source)
    if record.csv_row >= len(rows):
        raise ValueError(f"Missing CSV row: {record.sample_id}")
    pixels = np.fromstring(rows[record.csv_row], sep=" ", dtype=np.int64)
    if pixels.size != 2304 or pixels.min() < 0 or pixels.max() > 255:
        raise ValueError(f"Invalid pixels in {record.sample_id}")
    return Image.fromarray(pixels.astype(np.uint8).reshape(48, 48)).convert("RGB")


def discover(root: Path, seed: int = 42) -> tuple[list[Record], dict]:
    root = root.resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    records = []
    ignored = Counter()
    unknown = []
    files = (
        [Path(base) / name for base, _, names in os.walk(root) for name in names]
        if root.is_dir()
        else [root]
    )
    csv_files = sorted(path for path in files if path.suffix.lower() == ".csv")
    fer_csvs = []
    for path in csv_files:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not {"emotion", "pixels", "Usage"}.issubset(reader.fieldnames or []):
                continue
            fer_csvs.append(path)
            for index, row in enumerate(reader):
                label = int(row["emotion"])
                if label not in range(7) or row["Usage"] not in USAGES:
                    raise ValueError(f"Invalid label/Usage: {path}:{index}")
                records.append(
                    Record(f"{path.name}:{index}", str(path), label, USAGES[row["Usage"]], index)
                )
    if len(fer_csvs) > 1:
        raise ValueError("Multiple FER CSVs: supply one file explicitly to avoid double counting")
    if not fer_csvs:
        for path in sorted(files):
            if path.suffix.lower() not in EXTENSIONS:
                ignored[path.suffix or "no extension"] += 1
                continue
            label_name = path.parent.name.lower()
            if label_name not in CLASSES:
                unknown.append(str(path.relative_to(root)))
                continue
            prefix = path.stem.split("_")[0]
            if prefix in USAGES:
                split = USAGES[prefix]
            else:
                parts = set(p.lower() for p in path.relative_to(root).parts[:-1])
                aliases = {
                    "train": "train",
                    "training": "train",
                    "val": "val",
                    "validation": "val",
                    "test": "test",
                    "testing": "test",
                }
                matched = {aliases[p] for p in parts if p in aliases}
                if len(matched) != 1:
                    raise ValueError(f"Ambiguous split: {path}")
                split = matched.pop()
            records.append(
                Record(str(path.relative_to(root)), str(path), CLASSES.index(label_name), split)
            )
    if unknown:
        raise ValueError(f"Unknown class directories ({len(unknown)} images): {unknown[:5]}")
    if not records:
        raise ValueError(f"No FER images or compatible CSV found under {root}")
    derived_val = False
    if not any(r.split == "val" for r in records):
        groups = defaultdict(list)
        for i, record in enumerate(records):
            if record.split == "train":
                digest = hashlib.sha256(np.asarray(read_image(record)).tobytes()).hexdigest()
                groups[digest].append(i)
        rng = np.random.default_rng(seed)
        by_label = defaultdict(list)
        for digest, indices in groups.items():
            by_label[records[indices[0]].label].append(digest)
        selected = set()
        for label_groups in by_label.values():
            rng.shuffle(label_groups)
            selected.update(label_groups[: max(1, int(len(label_groups) * 0.1))])
        for digest in selected:
            for i in groups[digest]:
                row = asdict(records[i])
                row["split"] = "val"
                records[i] = Record(**row)
        derived_val = True
    for split in ("train", "val", "test"):
        if not any(r.split == split for r in records):
            raise ValueError(f"Missing required {split} split")
    return records, {
        "root": str(root),
        "ignored_files": dict(ignored),
        "derived_validation": derived_val,
        "seed": seed,
    }


def _image_statistics(record: Record) -> dict:
    try:
        image = read_image(record)
        pixels = np.asarray(image.convert("L"))
        return {
            "size": str(image.size),
            "mean": float(pixels.mean()),
            "std": float(pixels.std()),
            "hash": hashlib.sha256(pixels.tobytes() + str(image.size).encode()).hexdigest(),
        }
    except (OSError, ValueError) as error:
        return {"error": str(error)}


def audit(root: Path, output: Path, seed: int = 42) -> dict:
    records, metadata = discover(root, seed)
    print(f"Discovered {len(records)} samples; checking decoded image content", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    hashes = defaultdict(list)
    counts = {s: Counter() for s in ("train", "val", "test")}
    dimensions = Counter()
    corrupt = []
    quality = []
    valid = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for index, (record, stats) in enumerate(
            zip(records, pool.map(_image_statistics, records), strict=False)
        ):
            if index and index % 2000 == 0:
                print(f"Audited {index}/{len(records)} images", flush=True)
            if "error" in stats:
                corrupt.append({"sample_id": record.sample_id, "error": stats["error"]})
                continue
            hashes[stats["hash"]].append(record)
            counts[record.split][CLASSES[record.label]] += 1
            dimensions[stats.pop("size")] += 1
            quality.append({"sample_id": record.sample_id, **stats})
            valid.append(record)
    duplicates = []
    for digest, group in hashes.items():
        if len(group) > 1:
            duplicates.append(
                {
                    "hash": digest,
                    "samples": [asdict(r) for r in group],
                    "cross_split": len({r.split for r in group}) > 1,
                    "conflicting_labels": len({r.label for r in group}) > 1,
                }
            )
    report = {
        **metadata,
        "classes": CLASSES,
        "total": len(records),
        "valid": len(valid),
        "counts": counts,
        "dimensions": dict(dimensions),
        "corrupt": corrupt,
        "exact_duplicate_groups": len(duplicates),
        "cross_split_duplicate_groups": sum(d["cross_split"] for d in duplicates),
        "label_conflict_groups": sum(d["conflicting_labels"] for d in duplicates),
        "dark_images_mean_below_30": sum(q["mean"] < 30 for q in quality),
        "low_contrast_std_below_10": sum(q["std"] < 10 for q in quality),
        "near_duplicates": "Not proven absent; exact decoded-pixel audit only",
    }
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "duplicates.json").write_text(json.dumps(duplicates, indent=2) + "\n")
    (output / "manifest.json").write_text(json.dumps([asdict(r) for r in valid], indent=2))
    with (output / "quality.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "mean", "std", "hash"])
        writer.writeheader()
        writer.writerows(quality)
    return report


def load_manifest(path: Path, clean: bool = True) -> list[Record]:
    records = [Record(**r) for r in json.loads(path.read_text())]
    if not clean:
        return records
    duplicates_path = path.parent / "duplicates.json"
    if not duplicates_path.exists():
        raise FileNotFoundError("Training requires duplicate audit alongside manifest")
    excluded = set()
    priority = {"test": 2, "val": 1, "train": 0}
    for group in json.loads(duplicates_path.read_text()):
        samples = group["samples"]
        if group["conflicting_labels"]:
            excluded.update(r["sample_id"] for r in samples if r["split"] == "train")
        if group["cross_split"]:
            highest = max(priority[r["split"]] for r in samples)
            excluded.update(r["sample_id"] for r in samples if priority[r["split"]] < highest)
    return [r for r in records if r.sample_id not in excluded]
