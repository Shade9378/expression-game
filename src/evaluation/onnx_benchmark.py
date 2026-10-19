from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

from data.fer2013 import load_manifest, read_image
from data.preprocessing import preprocess_rgb
from evaluation.metrics import save_evaluation
from evaluation.profiling import MemorySampler, timing_summary
from inference.backends import OnnxBackend


def benchmark_onnx(path: Path, output: Path, repeats: int = 200) -> dict:
    import onnxruntime as ort

    metadata = json.loads(path.with_suffix(".onnx.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "name": metadata["name"],
        "precision": metadata["precision"],
        "population": metadata["population"],
        "model_path": str(path),
        "model_size_bytes": path.stat().st_size,
        "parameters": metadata["parameters"],
        "providers": ["CPUExecutionProvider"],
        "threads": 4,
        "onnxruntime": ort.__version__,
        "platform": platform.platform(),
        "torch_imported": "torch" in sys.modules,
    }
    try:
        records = load_manifest(Path(metadata["manifest"]))
        valid = np.ones(len(records), dtype=bool)
        geometry = None
        if metadata["population"] == "paired":
            with np.load(metadata["feature_cache"], allow_pickle=False) as data:
                if list(data["ids"]) != [r.sample_id for r in records]:
                    raise ValueError("Geometry/image IDs mismatch")
                valid = data["valid"]
                geometry = data["geometry"][:, : metadata["geometry_dim"]]
        first = next(i for i, r in enumerate(records) if r.split == "test" and valid[i])
        inputs = {}
        if metadata["input_kind"] in ("image", "image_geometry"):
            inputs["images"] = preprocess_rgb(
                np.asarray(read_image(records[first])), metadata["image_size"]
            )[None]
        if metadata["input_kind"] in ("geometry", "image_geometry"):
            inputs["geometry"] = geometry[first : first + 1].astype(np.float32)
        with MemorySampler() as memory:
            backend = OnnxBackend(path)
            for _ in range(20):
                backend.predict_inputs(inputs)
            times = []
            for _ in range(repeats):
                start = time.perf_counter_ns()
                backend.predict_inputs(inputs)
                times.append((time.perf_counter_ns() - start) / 1e6)
        result.update(
            {
                **timing_summary(times),
                "warmup": 20,
                "peak_rss_bytes": memory.peak,
                "baseline_rss_bytes": memory.baseline,
                "peak_delta_rss_bytes": memory.peak - memory.baseline,
                "memory_scope": (
                    "Session load, warmup and batch-one forward; data metadata preloaded"
                ),
                "latency_scope": (
                    "ONNX forward plus softmax, precomputed inputs; excludes MediaPipe"
                ),
                "native_fp16_compute_guaranteed": False,
            }
        )
        np.savetxt(output / "latencies_ms.csv", times, header="latency_ms", delimiter=",")

        images = None
        if metadata["input_kind"] in ("image", "image_geometry"):
            with np.load(metadata["image_cache"], allow_pickle=False) as data:
                if list(data["ids"]) != [r.sample_id for r in records]:
                    raise ValueError("Decoded cache/image IDs mismatch")
                images = data["images"]
        summaries = {}
        prediction_root = Path("results/onnx_evaluation") / output.name
        for split in ("val", "test"):
            indices = [i for i, r in enumerate(records) if r.split == split and valid[i]]
            probabilities = []
            for start in range(0, len(indices), 64):
                selected = indices[start : start + 64]
                batch = {}
                if images is not None:
                    batch["images"] = np.stack(
                        [preprocess_rgb(images[i], metadata["image_size"]) for i in selected]
                    )
                if metadata["input_kind"] in ("geometry", "image_geometry"):
                    batch["geometry"] = geometry[selected].astype(np.float32)
                probabilities.append(backend.predict_inputs(batch))
            summaries[split] = save_evaluation(
                np.asarray([records[i].label for i in indices]),
                np.concatenate(probabilities),
                [records[i].sample_id for i in indices],
                prediction_root / split,
            )
            (output / split).mkdir(exist_ok=True)
            (output / split / "metrics.json").write_text(
                json.dumps(summaries[split], indent=2) + "\n"
            )
            print(
                f"{path.stem}/{split}: {len(indices)} samples, "
                f"macro-F1={summaries[split]['macro_f1']:.4f}",
                flush=True,
            )
        result.update({"status": "supported", "val": summaries["val"], "test": summaries["test"]})
    except Exception as error:
        result.update(
            {"status": "unsupported_or_failed", "error": f"{type(error).__name__}: {error}"}
        )
    (output / "benchmark.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("name", "precision", "status")}, indent=2), flush=True)
    return result
