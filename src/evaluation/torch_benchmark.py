from __future__ import annotations

import json
import platform
import threading
import time
from pathlib import Path

import numpy as np
import psutil
import torch

from constants import CLASSES
from data.fer2013 import load_manifest, read_image
from data.preprocessing import preprocess_rgb
from models.cnn import load_checkpoint


class MemorySampler:
    def __init__(self) -> None:
        self.process = psutil.Process()
        self.baseline = self.process.memory_info().rss
        self.peak = self.baseline
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self.stop_event.wait(0.005):
            self.peak = max(self.peak, self.process.memory_info().rss)

    def __enter__(self) -> MemorySampler:
        self.thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop_event.set()
        self.thread.join()
        self.peak = max(self.peak, self.process.memory_info().rss)


def compact_checkpoint(checkpoint: Path, output: Path) -> None:
    model, config = load_checkpoint(checkpoint)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state": model.state_dict(), "classes": list(CLASSES), "config": config}, output
    )


def timing_summary(times_ms: list[float]) -> dict:
    return {
        "mean_ms": float(np.mean(times_ms)),
        "std_ms": float(np.std(times_ms, ddof=1)),
        "median_ms": float(np.median(times_ms)),
        "p95_ms": float(np.percentile(times_ms, 95)),
        "throughput_per_second": float(1000 / np.mean(times_ms)),
        "repetitions": len(times_ms),
    }


def benchmark_torch(checkpoint: Path, output: Path, threads: int = 4, repeats: int = 200) -> dict:
    if repeats < 2:
        raise ValueError("At least two measurements required")
    torch.set_num_threads(threads)
    with MemorySampler() as memory:
        model, config = load_checkpoint(checkpoint)
        records = load_manifest(Path(config["manifest"]))
        sample = next(r for r in records if r.split == "test")
        tensor = torch.from_numpy(
            preprocess_rgb(np.asarray(read_image(sample)), config["image_size"])
        )
        tensor = tensor.unsqueeze(0)
        with torch.inference_mode():
            for _ in range(20):
                model(tensor).softmax(dim=1)
            times = []
            for _ in range(repeats):
                start = time.perf_counter_ns()
                probabilities = model(tensor).softmax(dim=1)
                times.append((time.perf_counter_ns() - start) / 1e6)
            if not torch.isfinite(probabilities).all():
                raise ValueError("Nonfinite inference outputs")
    result = {
        **timing_summary(times),
        "model": config["model"],
        "backend": "pytorch_fp32_cpu",
        "threads": threads,
        "warmup": 20,
        "input_shape": list(tensor.shape),
        "parameters": sum(p.numel() for p in model.parameters()),
        "model_size_bytes": checkpoint.stat().st_size,
        "peak_rss_bytes": memory.peak,
        "baseline_rss_bytes": memory.baseline,
        "peak_delta_rss_bytes": memory.peak - memory.baseline,
        "memory_scope": "Sampled RSS during compact model load, input setup, warmup, inference",
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "latency_scope": "Forward plus softmax, batch one",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "benchmark.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savetxt(output / "latencies_ms.csv", np.asarray(times), delimiter=",", header="latency_ms")
    print(json.dumps(result), flush=True)
    return result


def compare_models(runs: list[Path], benchmark_root: Path, output: Path) -> dict:
    rows = []
    hashes = set()
    counts = set()
    for run in runs:
        summary = json.loads((run / "summary.json").read_text())
        provenance = json.loads((run / "provenance.json").read_text())
        config = json.loads((run / "config.json").read_text())
        benchmark = json.loads((benchmark_root / config["model"] / "benchmark.json").read_text())
        hashes.add(provenance["manifest_sha256"])
        counts.add(tuple(provenance["effective_split_counts"].values()))
        rows.append(
            {
                "model": config["model"],
                "seed": config["seed"],
                "val_macro_f1": summary["val"]["macro_f1"],
                "test_accuracy": summary["test"]["accuracy"],
                "test_macro_f1": summary["test"]["macro_f1"],
                "test_balanced_accuracy": summary["test"]["balanced_accuracy"],
                **{
                    k: benchmark[k]
                    for k in (
                        "parameters",
                        "model_size_bytes",
                        "mean_ms",
                        "std_ms",
                        "p95_ms",
                        "peak_rss_bytes",
                        "peak_delta_rss_bytes",
                    )
                },
                "per_class_recall": {c: summary["test"]["per_class"][c]["recall"] for c in CLASSES},
            }
        )
    if len(hashes) != 1 or len(counts) != 1:
        raise ValueError("Comparison runs must use identical audited partitions")
    best = max(row["val_macro_f1"] for row in rows)
    eligible = [row for row in rows if row["val_macro_f1"] >= best - 0.03]
    selected = min(eligible, key=lambda row: row["mean_ms"])
    result = {
        "rows": rows,
        "selected_model": selected["model"],
        "selection_rule": "Fastest CPU model within 0.03 validation macro-F1 of best",
        "test_used_for_selection": False,
        "manifest_sha256": hashes.pop(),
        "single_seed": True,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    import pandas as pd

    pd.DataFrame(
        [{k: v for k, v in row.items() if k != "per_class_recall"} for row in rows]
    ).to_csv(output / "comparison.csv", index=False)
    from evaluation.plot_utils import comparison_plots

    comparison_plots(output)
    print(json.dumps(result, indent=2))
    return result
