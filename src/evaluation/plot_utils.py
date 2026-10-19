from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scienceplots as scienceplots
import seaborn as sns

from constants import CLASSES
from data.fer2013 import load_manifest, read_image


def style() -> None:
    plt.style.use(["science", "ieee", "no-latex"])
    sns.set_palette("colorblind")
    plt.rcParams.update(
        {
            "font.size": 8,
            "font.family": "DejaVu Sans",
            "figure.figsize": (3.5, 2.6),
            "savefig.dpi": 200,
        }
    )


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("svg", "png"):
        fig.savefig(path.with_suffix(f".{suffix}"), bbox_inches="tight")
    plt.close(fig)


def dataset_plots(output: Path) -> None:
    style()
    report = json.loads((output / "audit.json").read_text())
    fig, ax = plt.subplots()
    x = np.arange(7)
    for offset, split in enumerate(("train", "val", "test")):
        ax.bar(
            x + (offset - 1) * 0.25,
            [report["counts"][split].get(c, 0) for c in CLASSES],
            0.25,
            label=split,
        )
    ax.set_xticks(x, CLASSES, rotation=35, ha="right")
    ax.set_ylabel("Images")
    ax.legend()
    save(fig, output / "figures" / "class_distribution")
    records = load_manifest(output / "manifest.json", clean=False)
    fig, axes = plt.subplots(2, 4, figsize=(3.5, 2.1))
    for ax in axes.flat:
        ax.axis("off")
    for label, name in enumerate(CLASSES):
        selected = next(r for r in records if r.label == label and r.split == "train")
        axes.flat[label].imshow(read_image(selected), cmap="gray")
        axes.flat[label].set_title(name, fontsize=7)
    fig.tight_layout()
    save(fig, output / "figures" / "training_examples")
    from pandas import read_csv

    quality = read_csv(output / "quality.csv")
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 3.2))
    for ax, column, title in zip(axes, ("mean", "std"), ("Brightness", "Contrast"), strict=True):
        ax.hist(quality[column], bins=40)
        ax.set_xlabel(f"{title} (0-255 grayscale)")
        ax.set_ylabel("Images")
    fig.tight_layout()
    save(fig, output / "figures" / "image_quality")


def training_plots(output: Path) -> None:
    import pandas as pd

    style()
    history = pd.read_csv(output / "history.csv")
    metrics = json.loads((output / "test" / "metrics.json").read_text())
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 3.4))
    axes[0].plot(history.epoch, history.train_loss, label="Train (weighted)")
    axes[0].plot(history.epoch, history.val_loss, label="Validation (unweighted)")
    axes[0].set_ylabel("Cross-entropy")
    axes[0].legend(fontsize=6)
    axes[1].plot(history.epoch, history.val_accuracy, label="Accuracy")
    axes[1].plot(history.epoch, history.val_macro_f1, label="Macro-F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation score")
    axes[1].legend(fontsize=6)
    fig.tight_layout()
    save(fig, output / "figures" / "learning_curves")
    matrix = np.asarray(metrics["confusion_matrix"])
    normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(3.5, 3.1))
    sns.heatmap(
        normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0,
        vmax=1,
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        cbar=False,
        ax=ax,
        annot_kws={"fontsize": 5},
    )
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.tick_params(axis="both", labelsize=6)
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    save(fig, output / "figures" / "confusion_matrix")
    fig, ax = plt.subplots()
    x = np.arange(7)
    for offset, metric in enumerate(("precision", "recall", "f1-score")):
        ax.bar(
            x + (offset - 1) * 0.25,
            [metrics["per_class"][c][metric] for c in CLASSES],
            0.25,
            label=metric,
        )
    ax.set_xticks(x, CLASSES, rotation=40, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Test score")
    ax.legend(fontsize=6)
    save(fig, output / "figures" / "per_class")


def comparison_plots(output: Path) -> None:
    style()
    result = json.loads((output / "comparison.json").read_text())
    rows = result["rows"]
    short = {
        "resnet18": "ResNet18",
        "mobilenet_v3_small": "MobileNetV3-S",
        "efficientnet_b0": "EfficientNet-B0",
    }
    names = [short[r["model"]] for r in rows]
    fig, ax = plt.subplots()
    x = np.arange(len(rows))
    ax.bar(x - 0.18, [r["test_accuracy"] for r in rows], 0.36, label="Accuracy")
    ax.bar(x + 0.18, [r["test_macro_f1"] for r in rows], 0.36, label="Macro-F1")
    ax.set_xticks(x, names, rotation=15, ha="right")
    ax.set_ylabel("Test score (single seed)")
    ax.set_ylim(0, 1)
    ax.legend()
    save(fig, output / "figures" / "model_scores")
    fig, ax = plt.subplots()
    ax.bar(x, [r["mean_ms"] for r in rows], yerr=[r["std_ms"] for r in rows], capsize=3)
    ax.set_xticks(x, names, rotation=15, ha="right")
    ax.set_ylabel("CPU latency (ms, mean +/- SD)")
    save(fig, output / "figures" / "latency")
    fig, ax = plt.subplots()
    for row in rows:
        ax.errorbar(
            row["mean_ms"],
            row["test_accuracy"],
            xerr=row["std_ms"],
            fmt="o",
            label=short[row["model"]],
            capsize=3,
        )
    ax.set_xlabel("CPU latency (ms, mean +/- SD)")
    ax.set_ylabel("Test accuracy")
    ax.legend(fontsize=6)
    save(fig, output / "figures" / "accuracy_efficiency")
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    sns.heatmap(
        np.asarray([[r["per_class_recall"][c] for c in CLASSES] for r in rows]),
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0,
        vmax=1,
        cbar=False,
        xticklabels=CLASSES,
        yticklabels=names,
        annot_kws={"fontsize": 5},
        ax=ax,
    )
    ax.set_xlabel("Class (test recall)")
    ax.tick_params(labelsize=6)
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    save(fig, output / "figures" / "per_class_comparison")


def temporal_plot(rows: list[dict], output: Path) -> None:
    style()
    times = np.asarray([r["timestamp"] for r in rows])
    times -= times[0]
    raw = np.asarray([r["raw"] for r in rows])
    smoothed = np.asarray([r["smoothed"] for r in rows])
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 3.4))
    intervals = np.diff(times)
    gaps = intervals > max(0.1, float(np.median(intervals)) * 3)
    segments = np.split(np.arange(len(times)), np.flatnonzero(gaps) + 1)
    for index, segment in enumerate(segments):
        for label, values, color in (("Raw", raw, "C0"), ("EMA", smoothed, "C1")):
            axes[0].step(
                times[segment],
                values[segment].argmax(axis=1),
                where="post",
                alpha=0.7,
                color=color,
                label=label if index == 0 else None,
            )
    axes[0].set_yticks(range(7), CLASSES, fontsize=6)
    axes[0].set_ylabel("Predicted class")
    axes[0].legend(fontsize=6)
    for label, values in (("Raw", raw), ("EMA", smoothed)):
        variation = np.abs(np.diff(values, axis=0)).sum(axis=1) / 2
        variation[gaps] = np.nan
        axes[1].plot(times[1:], variation, label=label)
    axes[1].set_xlabel("Elapsed time (s)")
    axes[1].set_ylabel("Probability variation")
    axes[1].legend(fontsize=6)
    fig.tight_layout()
    save(fig, output / "figures" / "prediction_timeline")


def ablation_plots(output: Path) -> None:
    style()
    report = json.loads((output / "ablation.json").read_text())
    rows = report["rows"]
    labels = ["Image", "Landmarks", "Landmarks+BS", "Image+LM", "Image+LM+BS"]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ax.bar(x - 0.18, [r["test_accuracy"] for r in rows], 0.36, label="Accuracy")
    ax.bar(x + 0.18, [r["test_macro_f1"] for r in rows], 0.36, label="Macro-F1")
    ax.set_xticks(x, labels, rotation=35, ha="right", fontsize=6)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Paired test score (single seed)")
    ax.legend(fontsize=6)
    save(fig, output / "figures" / "representation_ablation")
    coverage = json.loads((output / "coverage.json").read_text())["coverage"]
    fig, ax = plt.subplots()
    x = np.arange(7)
    for offset, split in enumerate(("train", "val", "test")):
        fractions = [coverage[split][c]["valid"] / coverage[split][c]["total"] for c in CLASSES]
        ax.bar(x + (offset - 1) * 0.25, fractions, 0.25, label=split)
    ax.set_xticks(x, CLASSES, rotation=35, ha="right", fontsize=6)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Feature extraction coverage")
    ax.legend(fontsize=6)
    save(fig, output / "figures" / "extraction_coverage")


def optimization_plots(output: Path) -> None:
    import pandas as pd

    style()
    results = json.loads((output / "comparison.json").read_text())["results"]
    table = []
    for row in results:
        entry = {
            k: row.get(k)
            for k in (
                "name",
                "precision",
                "population",
                "status",
                "mean_ms",
                "std_ms",
                "p95_ms",
                "model_size_bytes",
                "peak_rss_bytes",
                "error",
            )
        }
        if row["status"] == "supported":
            entry.update(
                {
                    "test_accuracy": row["test"]["accuracy"],
                    "test_macro_f1": row["test"]["macro_f1"],
                    "val_macro_f1": row["val"]["macro_f1"],
                }
            )
        table.append(entry)
    pd.DataFrame(table).to_csv(output / "comparison.csv", index=False)
    names = ["resnet18", "mobilenet_v3_small", "efficientnet_b0"]
    labels = ["ResNet18", "MobileNetV3-S", "EfficientNet-B0"]
    fig, ax = plt.subplots()
    x = np.arange(3)
    for offset, precision in enumerate(("fp32", "fp16", "int8")):
        values, errors = [], []
        for name in names:
            row = next(r for r in results if r["name"] == name and r["precision"] == precision)
            values.append(row.get("mean_ms", np.nan) if row["status"] == "supported" else np.nan)
            errors.append(row.get("std_ms", 0) if row["status"] == "supported" else 0)
        ax.bar(
            x + (offset - 1) * 0.25, values, 0.25, yerr=errors, capsize=2, label=precision.upper()
        )
    ax.set_xticks(x, labels, rotation=20, ha="right", fontsize=6)
    ax.set_ylabel("ONNX CPU latency (ms, mean +/- SD)")
    ax.legend(fontsize=6)
    save(fig, output / "figures" / "cnn_precision_latency")
    markers = {"fp32": "o", "fp16": "s", "int8": "^"}
    colors = sns.color_palette("colorblind", 3)
    fig, ax = plt.subplots()
    for row in results:
        if row["population"] == "full" and row["status"] == "supported":
            label = f"{labels[names.index(row['name'])]} {row['precision'].upper()}"
            ax.errorbar(
                row["mean_ms"],
                row["test"]["accuracy"],
                xerr=row["std_ms"],
                fmt=markers[row["precision"]],
                color=colors[names.index(row["name"])],
                capsize=2,
                label=label,
            )
    ax.set_xlabel("ONNX CPU latency (ms, mean +/- SD)")
    ax.set_ylabel("Full test accuracy")
    ax.legend(fontsize=6, ncol=2)
    save(fig, output / "figures" / "onnx_accuracy_efficiency")
