from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluation.plot_utils import save, style

NAMES = {
    "resnet18": "ResNet18",
    "mobilenet_v3_small": "MobileNetV3-S",
    "efficientnet_b0": "EfficientNet-B0",
    "image_only": "Image head",
    "landmark_only": "Landmarks",
    "geometry_blendshapes": "Landmarks + BS",
    "fused_landmarks": "Image + LM",
    "fused": "Image + LM + BS",
}


def read(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def benchmark_results(
    output: Path = Path("results/review"),
    profile: Path = Path("results/webcam/stage_profile_isolated"),
    allow_incomplete: bool = False,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    style()
    native = read("results/comparison/comparison.json")["rows"]
    optimized = read("results/optimization/comparison.json")["results"]
    accuracy = []
    seeds = []
    for model in list(NAMES)[:3]:
        runs = []
        hashes = set()
        for seed in (42, 43, 44):
            run = Path(f"results/runs/{model}_seed{seed}")
            if not (run / "summary.json").exists():
                continue
            summary = read(run / "summary.json")
            provenance = read(run / "provenance.json")
            hashes.add(provenance["manifest_sha256"])
            row = {
                "model": model,
                "seed": seed,
                "accuracy": summary["test"]["accuracy"],
                "macro_f1": summary["test"]["macro_f1"],
            }
            seeds.append(row)
            runs.append(row)
        assert len(hashes) == 1
        count = len(runs)
        if count != 3 and not allow_incomplete:
            raise ValueError(f"Expected seeds 42, 43, 44 for {model}; found {count}")
        accuracy.append(
            {
                "model": NAMES[model],
                "population": "Full",
                "format": "PyTorch",
                "seeds": ", ".join(str(r["seed"]) for r in runs),
                "n_seeds": count,
                "accuracy": float(np.mean([r["accuracy"] for r in runs])),
                "accuracy_std": float(np.std([r["accuracy"] for r in runs], ddof=1))
                if count > 1
                else None,
                "macro_f1": float(np.mean([r["macro_f1"] for r in runs])),
                "macro_f1_std": float(np.std([r["macro_f1"] for r in runs], ddof=1))
                if count > 1
                else None,
            }
        )
    for model in NAMES:
        for precision in ("fp32", "fp16", "int8"):
            row = next(r for r in optimized if r["name"] == model and r["precision"] == precision)
            assert row["status"] == "supported"
            accuracy.append(
                {
                    "model": NAMES[model],
                    "population": "Full" if row["population"] == "full" else "Paired",
                    "format": precision.upper(),
                    "seeds": "42",
                    "n_seeds": 1,
                    "accuracy": row["test"]["accuracy"],
                    "accuracy_std": None,
                    "macro_f1": row["test"]["macro_f1"],
                    "macro_f1_std": None,
                }
            )
    pd.DataFrame(accuracy).to_csv(output / "accuracy_table.csv", index=False)
    pd.DataFrame(seeds).to_csv(output / "seed_results.csv", index=False)
    (output / "accuracy_table.json").write_text(json.dumps(accuracy, indent=2) + "\n")
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    positions = np.arange(3)
    for offset, label, rows in (
        (-0.16, "PyTorch FP32", native),
        (0.16, "ONNX FP32", [r for r in optimized if r["precision"] == "fp32"]),
    ):
        selected = [
            next(r for r in rows if r.get("model", r.get("name")) == m) for m in list(NAMES)[:3]
        ]
        ax.barh(
            positions + offset,
            [r["mean_ms"] for r in selected],
            height=0.3,
            xerr=[r["std_ms"] for r in selected],
            capsize=2,
            label=label,
        )
    ax.set_yticks(positions, list(NAMES.values())[:3])
    ax.invert_yaxis()
    ax.set_xlabel("CPU latency (ms), mean +/- SD")
    ax.legend(fontsize=6)
    fig.tight_layout()
    save(fig, output / "figures" / "runtime_comparison")
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    positions = np.arange(len(NAMES))
    performance = []
    for index, precision in enumerate(("fp32", "fp16", "int8")):
        rows = [
            next(r for r in optimized if r["name"] == m and r["precision"] == precision)
            for m in NAMES
        ]
        samples = [
            np.loadtxt(
                Path("results/optimization") / f"{r['name']}_{precision}" / "latencies_ms.csv",
                delimiter=",",
            )
            for r in rows
        ]
        for row, values in zip(rows, samples, strict=True):
            np.testing.assert_allclose(values.mean(), row["mean_ms"])
        lower, median, upper = np.array(
            [np.percentile(values, [5, 50, 95]) for values in samples]
        ).T
        ax.errorbar(
            median,
            positions + (index - 1) * 0.22,
            xerr=np.array([median - lower, upper - median]),
            fmt=("o", "s", "^")[index],
            markersize=3,
            capsize=2,
            label=precision.upper(),
        )
        for row in rows:
            performance.append(
                {
                    key: row[key]
                    for key in (
                        "name",
                        "precision",
                        "mean_ms",
                        "std_ms",
                        "p95_ms",
                        "throughput_per_second",
                        "model_size_bytes",
                        "parameters",
                        "peak_rss_bytes",
                        "peak_delta_rss_bytes",
                        "repetitions",
                        "warmup",
                    )
                }
            )
    ax.set_yticks(positions, list(NAMES.values()))
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("ONNX CPU latency (ms)\nMedian and 5th-95th percentiles")
    ax.legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=3)
    fig.tight_layout()
    save(fig, output / "figures" / "onnx_latency")
    pd.DataFrame(performance).to_csv(output / "inference_performance.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 3.5), sharey=True)
    for index, precision in enumerate(("fp32", "fp16", "int8")):
        rows = [
            next(r for r in optimized if r["name"] == m and r["precision"] == precision)
            for m in NAMES
        ]
        axes[0].plot(
            [r["model_size_bytes"] / 1e6 for r in rows],
            positions,
            "os^"[index],
            markersize=3,
            label=precision.upper(),
        )
        axes[1].plot(
            [r["peak_rss_bytes"] / 1e6 for r in rows], positions, "os^"[index], markersize=3
        )
    axes[0].set_yticks(positions, list(NAMES.values()))
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Model (MB)")
    axes[0].set_xscale("log")
    axes[1].set_xlabel("Peak RSS (MB)")
    axes[0].legend(fontsize=5)
    fig.tight_layout()
    save(fig, output / "figures" / "onnx_resources")
    profile_path = profile / "measurements.json"
    if profile_path.exists():
        measurements = read(profile_path)
        stage_order = (
            ("capture_read_ms", "1. Camera read (parallel)"),
            ("input_queue_ms", "2. Input queue wait"),
            ("color_conversion_ms", "3. BGR to RGB"),
            ("face_landmarks_ms", "4. Face detection + landmarks"),
            ("tracking_ms", "5. Face tracking"),
            ("alignment_ms", "6. Crop, alignment + quality checks"),
            ("preprocessing_ms", "7. Model input preparation"),
            ("forward_ms", "8. ONNX forward"),
            ("probabilities_ms", "9. Softmax + probability checks"),
            ("ema_ms", "10. EMA smoothing"),
            ("annotation_ms", "11. Bounding box + label drawing"),
            ("ready_to_paint_ms", "12. Output queue + Qt display"),
            ("capture_to_paint_ms", "Overall: completed read to paint"),
        )
        profiles = []
        for population, selected in (
            ("all_presented", measurements),
            ("classified", [r for r in measurements if r["classified_faces"]]),
        ):
            for key, label in stage_order:
                values = np.array(
                    [
                        r[key] if key == "capture_to_paint_ms" else r["stages"][key]
                        for r in selected
                    ],
                    dtype=float,
                )
                if not len(values):
                    continue
                profiles.append(
                    {
                        "population": population,
                        "stage": label,
                        "key": key,
                        "n": len(values),
                        "mean_ms": float(values.mean()),
                        "std_ms": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                        "p95_ms": float(np.percentile(values, 95)),
                    }
                )
        pd.DataFrame(profiles).to_csv(output / "pipeline_stages.csv", index=False)
        (output / "pipeline_stages.json").write_text(json.dumps(profiles, indent=2) + "\n")
    sources = [
        Path("results/comparison/comparison.json"),
        Path("results/optimization/comparison.json"),
        profile / "gui.json",
        profile / "profiling_context.json",
        profile_path,
        Path("results/data/manifest.json"),
    ]
    for row in seeds:
        sources.extend(
            Path(f"results/runs/{row['model']}_seed{row['seed']}") / name
            for name in ("summary.json", "config.json", "provenance.json")
        )
    sources.extend(
        Path("results/optimization") / f"{r['name']}_{r['precision']}" / "latencies_ms.csv"
        for r in optimized
    )
    provenance = {
        "profile_root": str(profile),
        "source_files": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources
            if path.exists()
        },
        "seed_std_ddof": 1,
        "onnx_interval": "Empirical 5th and 95th percentiles, not a confidence interval",
        "stage_population": "Classified presented frames, with all-frame table retained in CSV",
    }
    (output / "sources.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Saved review figures and tables to {output}")


def mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def game_results(root: Path = Path("results/game")) -> None:
    root.mkdir(parents=True, exist_ok=True)
    runs = []
    for source in sorted(root.glob("*/gui.json")):
        summary = json.loads(source.read_text())
        frames = json.loads((source.parent / "measurements.json").read_text())
        inference = [row["inference_ms"] for row in frames if row["classified_faces"]]
        latency = [row["capture_to_paint_ms"] for row in frames]
        inference_mean, inference_std = mean_std(inference)
        latency_mean, latency_std = mean_std(latency)
        cpu_mean, cpu_std = mean_std(summary["process_cpu_percent_samples"])
        runs.append(
            {
                "run": source.parent.name,
                "seconds": summary["elapsed_seconds"],
                "fps": summary["display_fps"],
                "inference_mean_ms": inference_mean,
                "inference_std_ms": inference_std,
                "capture_to_paint_mean_ms": latency_mean,
                "capture_to_paint_std_ms": latency_std,
                "cpu_mean_percent": cpu_mean,
                "cpu_std_percent": cpu_std,
                "peak_ram_mib": summary["sampled_peak_rss_bytes"] / 2**20,
                "classified_frames": summary["frames_with_classification"],
                "frames": len(frames),
                "error": summary.get("error"),
            }
        )
    static_path = root / "static_validation.json"
    static = json.loads(static_path.read_text()) if static_path.exists() else None
    checks = []
    if static is not None:
        for name, values in static["conditions"].items():
            checks.append({"condition": name, **values})
        style()
        figure, axis = plt.subplots(figsize=(3.5, 2.6))
        positions = np.arange(len(checks))
        axis.bar(
            positions - 0.18,
            [100 * row["reference_label_agreement"] for row in checks],
            width=0.36,
            label="Dataset-label agreement",
        )
        axis.bar(
            positions + 0.18,
            [100 * row["compatible_features"] / row["images"] for row in checks],
            width=0.36,
            label="Compatible movement features",
        )
        axis.set_xticks(positions, [row["condition"] for row in checks], rotation=30, ha="right")
        axis.set(ylabel="Screened references (%)", ylim=(0, 105))
        axis.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), fontsize=6)
        figure.tight_layout()
        save(figure, root / "figures/static_checks")
    pd.DataFrame(runs).to_csv(root / "webcam_summary.csv", index=False)
    pd.DataFrame(checks).to_csv(root / "static_summary.csv", index=False)
    (root / "summary.json").write_text(
        json.dumps(
            {
                "webcam_runs": runs,
                "static_scope": static["scope"] if static is not None else None,
                "static_conditions": checks,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Saved {len(runs)} webcam summaries and {len(checks)} static conditions to {root}")


def architecture_diagrams(output: Path = Path("results/architecture")) -> None:
    import os

    import graphviz

    binary = next(Path("results/tools/graphviz").rglob("dot.exe"), None)
    if binary is None:
        raise FileNotFoundError(
            "Install the historical binary with experiments/scripts/fetch_graphviz.py"
        )
    os.environ["PATH"] = str(binary.parent.resolve()) + os.pathsep + os.environ["PATH"]
    output.mkdir(parents=True, exist_ok=True)

    fusion = graphviz.Digraph(
        "fusion",
        graph_attr={"rankdir": "TB", "size": "3.5,5!", "nodesep": ".15", "ranksep": ".15"},
        node_attr={
            "shape": "box",
            "style": "rounded",
            "fontname": "Arial",
            "fontsize": "10",
            "height": ".3",
            "margin": ".05",
        },
        edge_attr={"arrowsize": ".5"},
    )
    fusion.node("image", "Paired FER2013 image\nOfficial splits")
    fusion.node("cnn", "Fine-tuned ResNet18\nFrozen 512 features")
    fusion.node("mp", "Local MediaPipe v1\n478 points + 52 blendshapes")
    fusion.node("geom", "Center, scale and eye-roll norm\n32 points x 3 = 96 features")
    fusion.node("concat", "Image / landmarks / blendshapes\nFive feature controls")
    fusion.node("scale", "Standardize: paired train only")
    fusion.node("mlp", "MLP: input -> 128 -> 64 -> 7\nReLU, dropout, AdamW")
    fusion.node("eval", "Same paired validation/test IDs\nAccuracy, macro-F1, recall")
    fusion.edges(
        [
            ("image", "cnn"),
            ("image", "mp"),
            ("mp", "geom"),
            ("cnn", "concat"),
            ("geom", "concat"),
            ("concat", "scale"),
            ("scale", "mlp"),
            ("mlp", "eval"),
        ]
    )

    runtime = graphviz.Digraph(
        "runtime",
        graph_attr={"rankdir": "TB", "size": "3.5,5!", "ranksep": ".25"},
        node_attr={
            "shape": "box",
            "style": "rounded",
            "fontname": "Arial",
            "fontsize": "9",
            "margin": ".06",
        },
        edge_attr={"arrowsize": ".5"},
    )
    nodes = [
        ("camera", "Local webcam (OpenCV)"),
        ("queue", "Latest-frame queue: capacity 1"),
        ("face", "MediaPipe face detection + landmarks"),
        ("crop", "Per-face crop, align, normalize"),
        ("cnn", "Local CNN / ONNX inference"),
        ("ema", "Independent EMA per tracked face"),
        ("view", "PySide6 feed, probabilities, FPS\nNo default footage recording or uploads"),
    ]
    for key, label in nodes:
        runtime.node(key, label)
    runtime.edges([(nodes[i][0], nodes[i + 1][0]) for i in range(len(nodes) - 1)])

    fusion.edge("mp", "concat", label="52 BS", fontsize="9", fontname="Arial")
    diagrams = [("feature_fusion", fusion), ("local_runtime", runtime)]
    backbones = {
        "resnet18": (
            "7x7 stem + max pool\nResidual stages: 2, 2, 2, 2\nChannels: 64, 128, 256, 512",
            "Global average pool: 512",
        ),
        "mobilenet_v3_small": (
            "3x3 stem + SE inverted residuals\nDepthwise convolutions\nReLU and hard-swish",
            "Global average pool: 576\nLinear + hard-swish: 1024",
        ),
        "efficientnet_b0": (
            "3x3 stem + MBConv / SE stages\nDepthwise convolutions, SiLU",
            "Global average pool: 1280",
        ),
    }
    for name, (blocks, embedding) in backbones.items():
        diagram = graphviz.Digraph(
            name,
            graph_attr={"rankdir": "TB", "size": "3.5,4!", "ranksep": ".25"},
            node_attr={"shape": "box", "style": "rounded", "fontname": "Arial", "fontsize": "9"},
            edge_attr={"arrowsize": ".5"},
        )
        diagram.node("input", "48x48 FER2013 grayscale\nRGB replication, 112x112, normalization")
        diagram.node("backbone", blocks)
        diagram.node("embedding", embedding)
        diagram.node("head", "Replace pretrained classifier output\nNew seven-class linear head")
        diagram.node("output", "Softmax expression probabilities\nFull-network fine-tuning")
        diagram.edges(
            [
                ("input", "backbone"),
                ("backbone", "embedding"),
                ("embedding", "head"),
                ("head", "output"),
            ]
        )
        diagrams.append((name, diagram))

    for name, diagram in diagrams:
        (output / f"{name}.dot").write_text(diagram.source)
        for extension in ("svg", "png"):
            diagram.graph_attr["dpi"] = "200" if extension == "png" else "96"
            diagram.format = extension
            (output / f"{name}.{extension}").write_bytes(diagram.pipe())
    print("Rendered historical Graphviz architecture diagrams as DOT, SVG, PNG.")
