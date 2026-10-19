from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import mlflow
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from constants import CLASSES
from data.dataset import FERDataset
from data.fer2013 import load_manifest
from evaluation.metrics import classification_metrics, save_evaluation
from inference.geometry import (
    GEOMETRY_DIM,
    LANDMARK_INDICES,
    feature_columns,
    geometric_features,
)
from inference.landmarks import FaceLandmarker
from models.cnn import load_checkpoint
from training.cnn import cached_images, seed_everything


def extract_features(config_path: Path) -> None:
    config = json.loads(config_path.read_text())
    manifest, asset = Path(config["manifest"]), Path(config["asset"])
    records = load_manifest(manifest)
    images = cached_images(records, manifest, Path(config["image_cache"]))
    output = Path(config["feature_cache"])
    output.parent.mkdir(parents=True, exist_ok=True)
    local = threading.local()
    detectors = []
    schema = None
    rows = []
    valid = np.zeros(len(records), dtype=bool)
    geometry = np.zeros((len(records), GEOMETRY_DIM + 52), dtype=np.float32)
    start = time.perf_counter()

    def extract(index: int) -> tuple:
        if not hasattr(local, "detector"):
            local.detector = FaceLandmarker(asset, max_faces=1)
            detectors.append(local.detector)
        try:
            rgb = cv2.resize(images[index], (192, 192), interpolation=cv2.INTER_LINEAR)
            faces = local.detector.detect(rgb)
            if len(faces) != 1:
                return index, None, None, "No single reliable face"
            face = faces[0]
            if len(face.blendshapes) != 52:
                return index, None, None, "Incompatible blendshape schema"
            vector = np.concatenate([geometric_features(face.points, rgb.shape), face.blendshapes])
            if not np.isfinite(vector).all():
                return index, None, None, "Nonfinite geometry/blendshape values"
            return index, vector, face.blendshape_names, "OK"
        except (ValueError, RuntimeError) as error:
            return index, None, None, str(error)

    try:
        with ThreadPoolExecutor(max_workers=config["extraction_workers"]) as pool:
            for count, (index, vector, names, status) in enumerate(
                pool.map(extract, range(len(records)))
            ):
                if vector is not None:
                    schema = schema or names
                    if schema != names:
                        raise ValueError("Blendshape ordering changed between paired samples")
                    geometry[index] = vector
                    valid[index] = True
                rows.append(
                    {
                        "sample_id": records[index].sample_id,
                        "split": records[index].split,
                        "label": records[index].label,
                        "status": status,
                    }
                )
                if count % 2000 == 0:
                    print(f"Extracted {count}/{len(records)}; valid {int(valid.sum())}", flush=True)
    finally:
        for detector in detectors:
            detector.close()
    if not valid.any():
        raise RuntimeError("No compatible paired geometric features; fusion training unavailable")
    np.savez(
        output,
        geometry=geometry,
        valid=valid,
        ids=np.asarray([r.sample_id for r in records]),
        labels=np.asarray([r.label for r in records]),
        splits=np.asarray([r.split for r in records]),
        manifest_sha256=np.asarray(hashlib.sha256(manifest.read_bytes()).hexdigest()),
        asset_sha256=np.asarray(hashlib.sha256(asset.read_bytes()).hexdigest()),
        blendshape_names=np.asarray(schema),
    )
    pd.DataFrame(rows).to_csv(output.parent / "extraction_status.csv", index=False)
    coverage = {}
    for split in ("train", "val", "test"):
        coverage[split] = {}
        for label, name in enumerate(CLASSES):
            indices = [i for i, r in enumerate(records) if r.split == split and r.label == label]
            coverage[split][name] = {"total": len(indices), "valid": int(valid[indices].sum())}
    report_dir = Path(config["report_output"])
    report_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "coverage": coverage,
        "total": len(records),
        "valid": int(valid.sum()),
        "seconds": time.perf_counter() - start,
        "upsample_size": 192,
        "geometry_dim": GEOMETRY_DIM,
        "blendshape_dim": 52,
        "manifest_sha256": str(hashlib.sha256(manifest.read_bytes()).hexdigest()),
        "asset_sha256": str(hashlib.sha256(asset.read_bytes()).hexdigest()),
        "landmark_indices": LANDMARK_INDICES,
        "blendshape_names": schema,
        "selection": "Common detected-face subset for every ablation; no split reassignment",
        "limitation": "Results conditional on landmark extraction, not full-test-set performance",
    }
    (report_dir / "coverage.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {"total": len(records), "valid": int(valid.sum()), "seconds": report["seconds"]}
        ),
        flush=True,
    )


def fit_normalizer(
    features: np.ndarray, train_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    training = features[train_indices]
    return training.mean(axis=0), np.maximum(training.std(axis=0), 1e-6)


class FeatureMLP(nn.Module):
    def __init__(self, mean: torch.Tensor, scale: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("scale", scale)
        self.layers = nn.Sequential(
            nn.Linear(len(mean), 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 7),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers((features - self.mean) / self.scale)


def load_feature_checkpoint(path: Path, device: str = "cpu") -> tuple[FeatureMLP, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint["model_state"]
    model = FeatureMLP(state["mean"], state["scale"])
    model.load_state_dict(state)
    return model.to(device).eval(), checkpoint


@torch.inference_mode()
def evaluate_features(
    model: FeatureMLP, features: np.ndarray, labels: np.ndarray, indices: np.ndarray, device: str
) -> tuple[np.ndarray, dict]:
    model.eval()
    probabilities = model(torch.from_numpy(features[indices]).to(device)).softmax(1).cpu().numpy()
    return probabilities, classification_metrics(labels[indices], probabilities)


def train_fusion(config_path: Path) -> dict:
    config = json.loads(config_path.read_text())
    device = config["device"]
    torch.set_num_threads(4)
    manifest = Path(config["manifest"])
    records = load_manifest(manifest)
    images = cached_images(records, manifest, Path(config["image_cache"]))
    with np.load(config["feature_cache"], allow_pickle=False) as data:
        if list(data["ids"]) != [r.sample_id for r in records]:
            raise ValueError("Extracted features do not align with image IDs")
        if str(data["manifest_sha256"]) != hashlib.sha256(manifest.read_bytes()).hexdigest():
            raise ValueError("Feature extraction manifest mismatch")
        if (
            str(data["asset_sha256"])
            != hashlib.sha256(Path(config["asset"]).read_bytes()).hexdigest()
        ):
            raise ValueError("Landmarker asset changed since feature extraction")
        np.testing.assert_array_equal(data["labels"], [r.label for r in records])
        geometry = data["geometry"]
        valid = data["valid"]
    encoder, encoder_config = load_checkpoint(Path(config["encoder"]), device)
    embedding_path = Path(config["feature_cache"]).parent / "embeddings.npz"
    encoder_sha256 = hashlib.sha256(Path(config["encoder"]).read_bytes()).hexdigest()
    if embedding_path.exists():
        with np.load(embedding_path, allow_pickle=False) as data:
            if list(data["ids"]) != [r.sample_id for r in records]:
                raise ValueError("Embedding IDs do not match geometry IDs")
            if str(data["encoder_sha256"]) != encoder_sha256:
                raise ValueError("Encoder changed since embedding extraction")
            embeddings = data["embeddings"]
    else:
        dataset = FERDataset(records, encoder_config["image_size"], cache=False)
        dataset.images = images
        loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)
        embedding_batches = []
        with torch.inference_mode():
            for index, (batch, _) in enumerate(loader):
                embedding_batches.append(encoder.embed(batch.to(device)).cpu().numpy())
                if index % 100 == 0:
                    print(f"Frozen CNN embeddings {index}/{len(loader)}", flush=True)
        embeddings = np.concatenate(embedding_batches)
        np.savez(
            embedding_path,
            embeddings=embeddings,
            ids=np.asarray([r.sample_id for r in records]),
            encoder_sha256=np.asarray(encoder_sha256),
        )
    del encoder
    raw_features = np.concatenate([embeddings, geometry], axis=1)
    labels = np.asarray([r.label for r in records], dtype=np.int64)
    splits = {
        split: np.asarray([i for i, r in enumerate(records) if r.split == split and valid[i]])
        for split in ("train", "val", "test")
    }
    for split, indices in splits.items():
        counts = Counter(labels[indices])
        if any(counts[i] < (10 if split == "train" else 2) for i in range(7)):
            raise RuntimeError(f"Insufficient reliable paired features in {split}: {counts}")
    train_counts = Counter(labels[splits["train"]])
    weights = np.sqrt(len(splits["train"]) / (7 * np.asarray([train_counts[i] for i in range(7)])))
    weights /= weights.mean()
    mlflow.set_tracking_uri(Path(os.environ.get("FER_TRACKING_DIR", "mlruns")).resolve().as_uri())
    mlflow.set_experiment("FER2013-feature-ablations")
    outputs = []
    for mode in config["modes"]:
        seed_everything(config["seed"])
        columns = feature_columns(mode, embeddings.shape[1])
        features = raw_features[:, columns].copy()
        mean, scale = fit_normalizer(features, splits["train"])
        model = FeatureMLP(torch.from_numpy(mean), torch.from_numpy(scale)).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config["learning_rate"], weight_decay=0.01
        )
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(weights, dtype=torch.float32, device=device)
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
        loader = DataLoader(
            TensorDataset(
                torch.from_numpy(features[splits["train"]]),
                torch.from_numpy(labels[splits["train"]]),
            ),
            batch_size=256,
            shuffle=True,
            generator=torch.Generator().manual_seed(config["seed"]),
        )
        output = Path(config["training_output"]) / mode
        output.mkdir(parents=True, exist_ok=True)
        history = []
        best, stale = -1.0, 0

        with mlflow.start_run(run_name=f"{mode}-seed-{config['seed']}") as run:
            mlflow.log_params(
                {
                    "mode": mode,
                    "seed": config["seed"],
                    "features": len(columns),
                    "encoder": config["encoder"],
                    "normalizer_fit": "paired_train_only",
                }
            )
            for epoch in range(config["epochs"]):
                started = time.perf_counter()
                model.train()
                loss_sum, n = 0.0, 0
                for batch, targets in loader:
                    optimizer.zero_grad(set_to_none=True)
                    loss = criterion(model(batch.to(device)), targets.to(device))
                    loss.backward()
                    optimizer.step()
                    loss_sum += loss.item() * len(batch)
                    n += len(batch)
                probabilities, validation = evaluate_features(
                    model, features, labels, splits["val"], device
                )
                row = {
                    "epoch": epoch + 1,
                    "train_loss": loss_sum / n,
                    "val_accuracy": validation["accuracy"],
                    "val_macro_f1": validation["macro_f1"],
                    "val_loss": float(
                        -np.log(
                            np.clip(
                                probabilities[np.arange(len(splits["val"])), labels[splits["val"]]],
                                1e-9,
                                1,
                            )
                        ).mean()
                    ),
                    "epoch_seconds": time.perf_counter() - started,
                }
                history.append(row)
                pd.DataFrame(history).to_csv(output / "history.csv", index=False)
                mlflow.log_metrics({k: v for k, v in row.items() if k != "epoch"}, step=epoch + 1)
                if validation["macro_f1"] > best + 0.001:
                    best, stale = validation["macro_f1"], 0
                    torch.save(
                        {
                            "model_state": model.state_dict(),
                            "classes": list(CLASSES),
                            "mode": mode,
                            "columns": columns.tolist(),
                            "embedding_dim": embeddings.shape[1],
                            "config": config,
                            "epoch": epoch + 1,
                            "val_macro_f1": best,
                        },
                        output / "best.pt",
                    )
                else:
                    stale += 1
                scheduler.step()
                print(
                    f"{mode}: epoch {epoch + 1}, val macro-F1 {validation['macro_f1']:.4f}",
                    flush=True,
                )
                if stale >= config["patience"]:
                    break
            model, checkpoint = load_feature_checkpoint(output / "best.pt", device)
            summary = {}
            for split in ("val", "test"):
                probabilities, _ = evaluate_features(model, features, labels, splits[split], device)
                summary[split] = save_evaluation(
                    labels[splits[split]],
                    probabilities,
                    [records[i].sample_id for i in splits[split]],
                    output / split,
                )
                mlflow.log_metrics(
                    {
                        f"{split}_{key}": summary[split][key]
                        for key in ("accuracy", "macro_f1", "balanced_accuracy")
                    }
                )
            (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
            (output / "mlflow_run_id.txt").write_text(run.info.run_id + "\n")
            mlflow.log_artifact(str(output / "summary.json"))
            mlflow.log_artifact(str(output / "history.csv"))
            outputs.append(
                {
                    "mode": mode,
                    "features": len(columns),
                    "selected_epoch": checkpoint["epoch"],
                    "head_parameters": sum(p.numel() for p in model.parameters()),
                    "val_macro_f1": summary["val"]["macro_f1"],
                    "test_accuracy": summary["test"]["accuracy"],
                    "test_macro_f1": summary["test"]["macro_f1"],
                    "test_balanced_accuracy": summary["test"]["balanced_accuracy"],
                }
            )
            report_dir = Path(config["report_output"]) / mode
            report_dir.mkdir(parents=True, exist_ok=True)
            if report_dir.resolve() != output.resolve():
                for filename in ("history.csv", "config.json", "summary.json", "mlflow_run_id.txt"):
                    (report_dir / filename).write_bytes((output / filename).read_bytes())
    result = {
        "rows": outputs,
        "common_split_counts": {s: len(i) for s, i in splits.items()},
        "paired_ids_sha256": hashlib.sha256(
            "\n".join(records[i].sample_id for i in np.flatnonzero(valid)).encode()
        ).hexdigest(),
        "encoder": config["encoder"],
        "encoder_frozen": True,
        "normalizer_fit": "paired_train_only",
        "condition": "Detected-face subset; not directly comparable with full test CNN scores",
        "single_seed": True,
    }
    destination = Path(config["report_output"])
    (destination / "ablation.json").write_text(json.dumps(result, indent=2) + "\n")
    pd.DataFrame(outputs).to_csv(destination / "ablation.csv", index=False)
    from evaluation.plot_utils import ablation_plots

    ablation_plots(destination)
    print(json.dumps(result, indent=2), flush=True)
    return result
