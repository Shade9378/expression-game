from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import mlflow
import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from onnxruntime.transformers.float16 import convert_float_to_float16
from torch import nn

from constants import CLASSES
from data.fer2013 import load_manifest
from data.preprocessing import preprocess_rgb
from models.cnn import load_checkpoint
from training.fusion import load_feature_checkpoint

CNN_NAMES = ("resnet18", "mobilenet_v3_small", "efficientnet_b0")
FEATURE_NAMES = ("image_only", "landmark_only", "geometry_blendshapes", "fused_landmarks", "fused")


class FeaturePipeline(nn.Module):
    def __init__(self, head: nn.Module, backbone: nn.Module, kind: str) -> None:
        super().__init__()
        self.head, self.backbone, self.kind = head, backbone, kind

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        features = self.backbone(inputs[0])
        if self.kind == "image_geometry":
            features = torch.cat([features, inputs[1]], dim=1)
        return self.head(features)


def model_spec(name: str) -> tuple[nn.Module, dict]:
    coverage = json.loads(Path("results/fusion/coverage.json").read_text())
    common = {
        "classes": list(CLASSES),
        "image_size": 112,
        "manifest": "results/data/manifest.json",
        "image_cache": "results/data/decoded.npz",
        "feature_cache": "results/features/geometry.npz",
        "blendshape_names": coverage["blendshape_names"],
        "geometry_dim": 0,
        "landmarker_sha256": coverage["asset_sha256"],
    }
    if name in CNN_NAMES:
        checkpoint = Path("results/runs") / f"{name}_seed42" / "deployment.pt"
        model, config = load_checkpoint(checkpoint)
        metadata = {
            **common,
            "name": name,
            "input_kind": "image",
            "population": "full",
            "image_size": config["image_size"],
            "checkpoint": str(checkpoint),
        }
    else:
        checkpoint = Path("results/fusion") / name / "best.pt"
        head, saved = load_feature_checkpoint(checkpoint)
        dim = 96 if name in ("landmark_only", "fused_landmarks") else 148
        if name in ("landmark_only", "geometry_blendshapes"):
            model = head
            kind = "geometry"
        else:
            encoder_path = Path(saved["config"]["encoder"])
            if not encoder_path.is_file():
                current_config = json.loads(checkpoint.with_name("config.json").read_text())
                encoder_path = Path(current_config["encoder"])
            encoder, _ = load_checkpoint(encoder_path)
            kind = "image" if name == "image_only" else "image_geometry"
            model = FeaturePipeline(head, encoder.backbone, kind)
        metadata = {
            **common,
            "name": name,
            "input_kind": kind,
            "population": "paired",
            "geometry_dim": dim,
            "checkpoint": str(checkpoint),
        }
    model.eval()
    metadata["parameters"] = sum(p.numel() for p in model.parameters())
    metadata["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    return model, metadata


class FERCalibrationReader(CalibrationDataReader):
    def __init__(self, metadata: dict, split: str = "train", limit: int | None = 256) -> None:
        self.metadata = metadata
        records = load_manifest(Path(metadata["manifest"]))
        with np.load(metadata["image_cache"], allow_pickle=False) as data:
            if list(data["ids"]) != [r.sample_id for r in records]:
                raise ValueError("Calibration image IDs do not match manifest")
            self.images = data["images"]
        self.geometry = None
        valid = np.ones(len(records), dtype=bool)
        if metadata["population"] == "paired":
            with np.load(metadata["feature_cache"], allow_pickle=False) as data:
                if list(data["ids"]) != [r.sample_id for r in records]:
                    raise ValueError("Calibration feature IDs do not match image IDs")
                valid = data["valid"]
                self.geometry = data["geometry"][:, : metadata["geometry_dim"]]
        indices = np.asarray([i for i, r in enumerate(records) if r.split == split and valid[i]])
        if limit is not None:
            if split != "train":
                raise ValueError("INT8 calibration must use training samples only")
            rng = np.random.default_rng(42)
            chosen = []
            for label in range(7):
                group = [i for i in indices if records[i].label == label]
                chosen.extend(rng.choice(group, size=min(20, len(group)), replace=False).tolist())
            remaining = np.asarray([i for i in indices if i not in set(chosen)])
            chosen.extend(rng.choice(remaining, size=limit - len(chosen), replace=False).tolist())
            indices = np.asarray(chosen)
        self.indices = indices
        self.ids = [records[i].sample_id for i in indices]
        self.position = 0

    def batch(self, indices: np.ndarray) -> dict[str, np.ndarray]:
        result = {}
        if self.metadata["input_kind"] in ("image", "image_geometry"):
            result["images"] = np.stack(
                [preprocess_rgb(self.images[i], self.metadata["image_size"]) for i in indices]
            )
        if self.metadata["input_kind"] in ("geometry", "image_geometry"):
            result["geometry"] = self.geometry[indices].astype(np.float32)
        return result

    def get_next(self) -> dict | None:
        if self.position >= len(self.indices):
            return None
        selected = self.indices[self.position : self.position + 8]
        self.position += len(selected)
        return self.batch(selected)

    def rewind(self) -> None:
        self.position = 0


def export_fp32(
    model: nn.Module, metadata: dict, path: Path, inputs: dict[str, np.ndarray]
) -> float:
    arguments = tuple(torch.from_numpy(value) for value in inputs.values())
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(inputs)
    torch.onnx.export(
        model,
        arguments,
        str(path),
        opset_version=17,
        input_names=names,
        output_names=["logits"],
        dynamic_axes={**{name: {0: "batch"} for name in names}, "logits": {0: "batch"}},
        do_constant_folding=True,
    )
    onnx.checker.check_model(str(path))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    with torch.inference_mode():
        expected = model(*arguments).numpy()
    actual = session.run(["logits"], inputs)[0]
    np.testing.assert_allclose(expected, actual, rtol=1e-4, atol=1e-4)
    return float(np.max(np.abs(expected - actual)))


def sort_graph(model: onnx.ModelProto) -> None:
    available = {value.name for value in model.graph.input}
    available.update(value.name for value in model.graph.initializer)
    remaining = list(model.graph.node)
    ordered = []
    while remaining:
        ready = [node for node in remaining if all(not v or v in available for v in node.input)]
        if not ready:
            raise ValueError("Converted ONNX graph has a cycle or unresolved input")
        for node in ready:
            ordered.append(node)
            available.update(node.output)
            remaining.remove(node)
    del model.graph.node[:]
    model.graph.node.extend(ordered)


def optimize_all(resume: bool = False) -> dict:
    torch.set_num_threads(4)
    root = Path("results/onnx")
    reports = Path("results/optimization")
    results = []
    mlflow.set_tracking_uri(Path(os.environ.get("FER_TRACKING_DIR", "mlruns")).resolve().as_uri())
    mlflow.set_experiment("FER2013-ONNX-optimization")
    for name in (*CNN_NAMES, *FEATURE_NAMES):
        model, metadata = model_spec(name)
        reader = FERCalibrationReader(metadata)
        example = reader.batch(reader.indices[:2])
        fp32 = root / f"{name}_fp32.onnx"
        saved_metadata = fp32.with_suffix(".onnx.json")
        reusable = (
            resume
            and fp32.exists()
            and saved_metadata.exists()
            and json.loads(saved_metadata.read_text())["checkpoint_sha256"]
            == metadata["checkpoint_sha256"]
        )
        agreement = (
            json.loads(saved_metadata.read_text())["fp32_max_absolute_error"]
            if reusable
            else export_fp32(model, metadata, fp32, example)
        )
        calibration = {
            "split": "train",
            "samples": len(reader.indices),
            "seed": 42,
            "ids_sha256": hashlib.sha256("\n".join(reader.ids).encode()).hexdigest(),
        }
        (reports / name).mkdir(parents=True, exist_ok=True)
        (reports / name / "calibration.json").write_text(json.dumps(calibration, indent=2) + "\n")
        for precision in ("fp32", "fp16", "int8"):
            path = root / f"{name}_{precision}.onnx"
            variant = {
                **metadata,
                "precision": precision,
                "fp32_max_absolute_error": agreement,
                "calibration": calibration if precision == "int8" else None,
            }
            destination = reports / f"{name}_{precision}"
            destination.mkdir(parents=True, exist_ok=True)
            try:
                previous = destination / "benchmark.json"
                if reusable and previous.exists():
                    prior = json.loads(previous.read_text())
                    if prior["status"] == "supported" and path.exists():
                        prior["fp32_max_absolute_error"] = agreement
                        results.append(prior)
                        print(f"{name}/{precision}: reused validated benchmark", flush=True)
                        continue
                if precision == "fp16":
                    converted = convert_float_to_float16(onnx.load(str(fp32)), keep_io_types=True)
                    sort_graph(converted)
                    onnx.save(converted, str(path))
                    onnx.checker.check_model(str(path))
                elif precision == "int8":
                    reader.rewind()
                    quantize_static(
                        str(fp32),
                        str(path),
                        reader,
                        quant_format=QuantFormat.QDQ,
                        per_channel=True,
                        activation_type=QuantType.QInt8,
                        weight_type=QuantType.QInt8,
                        op_types_to_quantize=["Conv", "Gemm", "MatMul"],
                    )
                    onnx.checker.check_model(str(path))
                path.with_suffix(".onnx.json").write_text(json.dumps(variant, indent=2) + "\n")
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "cli",
                        "benchmark-onnx",
                        "--model",
                        str(path),
                        "--output",
                        str(destination),
                    ],
                    check=True,
                )
                result = json.loads((destination / "benchmark.json").read_text())
            except Exception as error:
                result = {
                    "name": name,
                    "precision": precision,
                    "population": metadata["population"],
                    "status": "unsupported_or_failed",
                    "error": f"{type(error).__name__}: {error}",
                    "model_size_bytes": path.stat().st_size if path.exists() else None,
                }
                (destination / "benchmark.json").write_text(json.dumps(result, indent=2) + "\n")
            if precision == "fp32" and result["status"] != "supported":
                raise RuntimeError(f"Required FP32 evaluation failed: {result}")
            variant["supported"] = result["status"] == "supported"
            path.with_suffix(".onnx.json").write_text(json.dumps(variant, indent=2) + "\n")
            result["fp32_max_absolute_error"] = agreement
            results.append(result)
            with mlflow.start_run(run_name=f"{name}-{precision}"):
                mlflow.log_params(
                    {
                        "model": name,
                        "precision": precision,
                        "provider": "CPUExecutionProvider",
                        "population": metadata["population"],
                        "status": result["status"],
                    }
                )
                if result["status"] == "supported":
                    mlflow.log_metrics(
                        {
                            "val_macro_f1": result["val"]["macro_f1"],
                            "test_accuracy": result["test"]["accuracy"],
                            "test_macro_f1": result["test"]["macro_f1"],
                            "latency_ms": result["mean_ms"],
                        }
                    )
                mlflow.log_artifact(str(destination / "benchmark.json"))
            print(f"{name}/{precision}: {result['status']}", flush=True)
    successful = [r for r in results if r["status"] == "supported" and r["population"] == "full"]
    best = max(r["val"]["macro_f1"] for r in successful)
    eligible = [r for r in successful if r["val"]["macro_f1"] >= best - 0.03]
    selected = min(eligible, key=lambda r: r["mean_ms"])
    summary = {
        "results": results,
        "selected": {k: selected[k] for k in ("name", "precision", "model_path")},
        "selection_rule": "Fastest full-population variant within 0.03 of best validation macro-F1",
        "test_used_for_selection": False,
        "single_seed": True,
        "paired_variants_not_compared_to_full_scores": True,
    }
    (reports / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    from evaluation.plot_utils import optimization_plots

    optimization_plots(reports)
    print(json.dumps(summary["selected"], indent=2), flush=True)
    return summary
