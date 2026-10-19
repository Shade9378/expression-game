from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from constants import CLASSES
from data.dataset import FERDataset
from data.fer2013 import Record, load_manifest, read_image
from evaluation.metrics import classification_metrics, save_evaluation
from models.cnn import ExpressionModel, load_checkpoint


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def cached_images(records: list[Record], manifest: Path, cache: Path) -> list[np.ndarray]:
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if cache.exists():
        with np.load(cache, allow_pickle=False) as data:
            if str(data["manifest_sha256"]) == digest and list(data["ids"]) == [
                r.sample_id for r in records
            ]:
                return list(data["images"])
        raise ValueError("Decoded-image cache does not match current manifest")
    print(f"Caching {len(records)} decoded originals in memory", flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        images = list(pool.map(lambda r: np.asarray(read_image(r)), records))
    cache.parent.mkdir(parents=True, exist_ok=True)
    if len({image.shape for image in images}) == 1:
        np.savez(
            cache,
            images=np.asarray(images),
            ids=np.asarray([r.sample_id for r in records]),
            manifest_sha256=np.asarray(digest),
        )
    return images


def build_loaders(manifest: Path, config: dict) -> tuple[dict, dict]:
    records = load_manifest(manifest)
    images = cached_images(records, manifest, Path(config["cache"]))
    loaders = {}
    split_records = {}
    for split in ("train", "val", "test"):
        indices = [i for i, record in enumerate(records) if record.split == split]
        split_records[split] = [records[i] for i in indices]
        if not indices or set(r.label for r in split_records[split]) != set(range(7)):
            raise ValueError(f"Split {split} must have samples for all seven classes")
        dataset = FERDataset(
            split_records[split],
            config["image_size"],
            train=split == "train",
            seed=config["seed"],
            cache=False,
        )
        dataset.images = [images[i] for i in indices]
        loaders[split] = DataLoader(
            dataset,
            batch_size=config["batch_size"],
            shuffle=split == "train",
            num_workers=0,
            generator=torch.Generator().manual_seed(config["seed"]),
            pin_memory=config["device"].startswith("cuda"),
        )
    return loaders, split_records


@torch.inference_mode()
def predict(model: nn.Module, loader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels, probabilities = [], []
    for images, targets in loader:
        logits = model(images.to(device, non_blocking=True))
        labels.extend(targets.numpy().tolist())
        probabilities.extend(logits.softmax(dim=1).cpu().numpy())
    return np.asarray(labels), np.asarray(probabilities)


def train(config_path: Path) -> dict:
    config = json.loads(config_path.read_text())
    seed_everything(config["seed"])
    torch.set_num_threads(config.get("cpu_threads", 4))
    device = config["device"]
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA device unavailable; choose CPU explicitly")
        torch.cuda.set_device(device)
        major, _ = torch.cuda.get_device_capability(device)
        if major >= 12:
            raise RuntimeError("Selected historical CUDA build does not support this GPU")
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    manifest = Path(config["manifest"])
    loaders, records = build_loaders(manifest, config)
    torch.hub.set_dir(str(Path("results/pretrained").resolve()))
    model = ExpressionModel(config["model"], pretrained=True).to(device)
    counts = Counter(r.label for r in records["train"])
    weights = np.sqrt(len(records["train"]) / (7 * np.asarray([counts[i] for i in range(7)])))
    weights /= weights.mean()
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda"))
    history = []
    best_score = -1.0
    stale = 0
    provenance = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "effective_split_counts": {s: len(r) for s, r in records.items()},
        "class_weights": weights.tolist(),
        "weight_policy": "sqrt inverse frequency",
        "device": torch.cuda.get_device_name(device) if device.startswith("cuda") else "CPU",
        "parameters": sum(p.numel() for p in model.parameters()),
        "single_seed": True,
        "test_used_for_selection": False,
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    tracking_dir = Path(os.environ.get("FER_TRACKING_DIR", "mlruns")).resolve()
    if os.name == "nt" and str(tracking_dir).startswith("\\\\"):
        raise ValueError("MLflow 2.20 rejects UNC file URIs; set FER_TRACKING_DIR to a local drive")
    mlflow.set_tracking_uri(tracking_dir.as_uri())
    mlflow.set_experiment("FER2013-image-models")
    with mlflow.start_run(run_name=f"{config['model']}-seed-{config['seed']}") as run:
        (output / "mlflow_run_id.txt").write_text(run.info.run_id + "\n")
        mlflow.log_params(
            {k: v for k, v in config.items() if isinstance(v, str | int | float | bool)}
        )
        for epoch in range(config["epochs"]):
            start = time.perf_counter()
            model.train()
            total_loss, count = 0.0, 0
            for step, (images, targets) in enumerate(loaders["train"]):
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type="cuda" if device.startswith("cuda") else "cpu",
                    enabled=device.startswith("cuda"),
                ):
                    loss = criterion(model(images), targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item() * len(images)
                count += len(images)
                if step % 100 == 0:
                    print(
                        f"{config['model']} epoch {epoch + 1} batch {step}/{len(loaders['train'])}",
                        flush=True,
                    )
            labels, probabilities = predict(model, loaders["val"], device)
            validation = classification_metrics(labels, probabilities)
            row = {
                "epoch": epoch + 1,
                "train_loss": total_loss / count,
                "val_accuracy": validation["accuracy"],
                "val_macro_f1": validation["macro_f1"],
                "val_loss": float(
                    -np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-9, 1)).mean()
                ),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "epoch_seconds": time.perf_counter() - start,
            }
            history.append(row)
            pd.DataFrame(history).to_csv(output / "history.csv", index=False)
            mlflow.log_metrics({k: v for k, v in row.items() if k != "epoch"}, step=epoch + 1)
            checkpoint = {
                "model_state": model.state_dict(),
                "config": config,
                "classes": list(CLASSES),
                "epoch": epoch + 1,
                "val_macro_f1": validation["macro_f1"],
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
            }
            torch.save(checkpoint, output / "last.pt")
            if validation["macro_f1"] > best_score + config.get("min_delta", 0.001):
                best_score, stale = validation["macro_f1"], 0
                torch.save(checkpoint, output / "best.pt")
            else:
                stale += 1
            print(json.dumps(row), flush=True)
            scheduler.step()
            if stale >= config["patience"]:
                print("Early stopping on validation macro-F1", flush=True)
                break
        model, _ = load_checkpoint(output / "best.pt", device)
        results = {}
        for split in ("val", "test"):
            labels, probabilities = predict(model, loaders[split], device)
            results[split] = save_evaluation(
                labels, probabilities, [r.sample_id for r in records[split]], output / split
            )
            mlflow.log_metrics(
                {
                    f"{split}_{key}": results[split][key]
                    for key in ("accuracy", "macro_f1", "balanced_accuracy")
                }
            )
        (output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
        from evaluation.plot_utils import training_plots

        training_plots(output)
        for filename in ("config.json", "provenance.json", "history.csv", "summary.json"):
            mlflow.log_artifact(str(output / filename))
        mlflow.log_artifacts(str(output / "figures"), artifact_path="figures")
    print(json.dumps(results), flush=True)
    return results
