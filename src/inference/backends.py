from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Protocol

import numpy as np

from constants import CLASSES
from data.preprocessing import preprocess_rgb
from inference.geometry import geometric_features


class InferenceBackend(Protocol):
    size: int
    name: str

    def predict_rgb(self, image: np.ndarray) -> np.ndarray: ...


class TorchBackend:
    def __init__(self, checkpoint: Path, threads: int = 4) -> None:
        import torch

        from models.cnn import load_checkpoint

        torch.set_num_threads(threads)
        self.torch = torch
        self.model, config = load_checkpoint(checkpoint)
        self.size = config["image_size"]
        self.name = config["model"]

    def predict_rgb(self, image: np.ndarray) -> np.ndarray:
        tensor = self.torch.from_numpy(preprocess_rgb(image, self.size)).unsqueeze(0)
        with self.torch.inference_mode():
            return self.model(tensor).softmax(dim=1)[0].numpy()


def load_backend(path: Path) -> InferenceBackend:
    if path.suffix == ".pt":
        return TorchBackend(path)
    if path.suffix == ".onnx":
        return OnnxBackend(path)
    raise ValueError(f"Unsupported local model format: {path.suffix}")


def softmax_numpy(logits: np.ndarray) -> np.ndarray:
    shifted = logits.astype(np.float32) - logits.max(axis=-1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / exponential.sum(axis=-1, keepdims=True)


class OnnxBackend:
    def __init__(self, path: Path, threads: int = 4) -> None:
        import onnxruntime as ort

        self.metadata = json.loads(path.with_suffix(".onnx.json").read_text())
        if tuple(self.metadata["classes"]) != CLASSES:
            raise ValueError("ONNX class ordering differs from the application")
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        self.size = self.metadata["image_size"]
        self.name = self.metadata["name"]
        self.last_timings: dict[str, float] = {}

    def predict_inputs(self, inputs: dict[str, np.ndarray]) -> np.ndarray:
        start = time.perf_counter()
        logits = self.session.run(["logits"], inputs)[0]
        forward_done = time.perf_counter()
        probabilities = softmax_numpy(logits)
        if not np.isfinite(probabilities).all():
            raise ValueError("Nonfinite ONNX probabilities")
        self.last_timings.update(
            forward_ms=(forward_done - start) * 1000,
            probabilities_ms=(time.perf_counter() - forward_done) * 1000,
        )
        return probabilities

    def predict_rgb(self, image: np.ndarray) -> np.ndarray:
        if self.metadata["input_kind"] != "image":
            raise ValueError("This model also requires local facial geometry")
        return self.predict_inputs({"images": preprocess_rgb(image, self.size)[None]})[0]

    def predict_face(self, image: np.ndarray, face: object, shape: tuple[int, ...]) -> np.ndarray:
        start = time.perf_counter()
        inputs = {}
        kind = self.metadata["input_kind"]
        if kind in ("image", "image_geometry"):
            inputs["images"] = preprocess_rgb(image, self.size)[None]
        if kind in ("geometry", "image_geometry"):
            values = geometric_features(face.points, shape)
            if self.metadata["geometry_dim"] > len(values):
                if list(face.blendshape_names) != self.metadata["blendshape_names"]:
                    raise ValueError("Live blendshape schema does not match the trained model")
                values = np.concatenate([values, face.blendshapes])
            inputs["geometry"] = values.astype(np.float32)[None]
        self.last_timings = {"preprocessing_ms": (time.perf_counter() - start) * 1000}
        return self.predict_inputs(inputs)[0]
