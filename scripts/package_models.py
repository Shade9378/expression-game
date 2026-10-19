from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
MODEL_FILES = (
    "results/onnx/resnet18_fp32.onnx",
    "results/onnx/resnet18_fp32.onnx.json",
    "results/models/face_landmarker_v1.task",
)


def package_models(output: Path) -> Path:
    metadata = json.loads((ROOT / MODEL_FILES[1]).read_text(encoding="utf-8"))
    asset = (ROOT / MODEL_FILES[2]).read_bytes()
    if hashlib.sha256(asset).hexdigest() != metadata["landmarker_sha256"]:
        raise ValueError("MediaPipe asset does not match the model metadata")
    if not metadata["supported"] or metadata["input_kind"] != "image":
        raise ValueError("Release requires a supported image classifier")
    sources = {name: ROOT / name for name in MODEL_FILES}
    for name in ("mediapipe-LICENSE.txt", "THIRD_PARTY_NOTICES.txt"):
        sources[f"results/models/{name}"] = ROOT / "scripts/licenses" / name
    payload = {name: path.read_bytes() for name, path in sources.items()}
    manifest = {
        "files": {
            name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            for name, data in payload.items()
        }
    }
    output.mkdir(parents=True, exist_ok=True)
    archive = output / "models.zip"
    with ZipFile(archive, "w", compression=ZIP_DEFLATED, compresslevel=6) as bundle:
        for name, path in sources.items():
            bundle.write(path, arcname=name)
        bundle.writestr("results/models/manifest.json", json.dumps(manifest, indent=2) + "\n")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    (output / "SHA256SUMS.txt").write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
    (output / "release_notes.md").write_text(
        "# App models\n\n"
        "Download `models.zip` and extract it into the cloned project root. "
        "Install dependencies and launch the app as described in README. No training required.\n\n"
        "Includes ResNet18 FP32 ONNX, its metadata, and MediaPipe Face Landmarker v1. "
        "Third-party notices and per-file hashes are included. "
        "`SHA256SUMS.txt` verifies the ZIP.\n\n"
        "Reference images are not included. For Recognition and Imitation, "
        "import expression-labeled image folders in Settings or prepare local FER2013 images.\n",
        encoding="utf-8",
    )
    return archive


def main() -> None:
    parser = argparse.ArgumentParser(description="Package existing app models for a GitHub Release")
    parser.add_argument("--output", type=Path, default=ROOT / "results/release")
    print(package_models(parser.parse_args().output))


if __name__ == "__main__":
    main()
