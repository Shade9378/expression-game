from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Local facial expression research pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect = subparsers.add_parser("audit")
    inspect.add_argument("--data", type=Path, default=Path("data/FER2013"))
    inspect.add_argument("--output", type=Path, default=Path("results/data"))
    training = subparsers.add_parser("train")
    training.add_argument("--config", type=Path, required=True)
    benchmark = subparsers.add_parser("benchmark-torch")
    benchmark.add_argument("--checkpoint", type=Path, required=True)
    benchmark.add_argument("--output", type=Path, required=True)
    webcam = subparsers.add_parser("webcam-benchmark")
    webcam.add_argument("--model", type=Path, required=True)
    webcam.add_argument(
        "--asset", type=Path, default=Path("results/models/face_landmarker_v1.task")
    )
    webcam.add_argument("--output", type=Path, default=Path("results/webcam"))
    webcam.add_argument("--seconds", type=float, default=15)
    webcam.add_argument("--camera", type=int, default=0)
    webcam.add_argument("--timeline", type=Path)
    webcam.add_argument("--alpha", type=float, default=0.3)
    temporal = subparsers.add_parser("temporal-analysis")
    temporal.add_argument("--trace", type=Path, required=True)
    temporal.add_argument("--output", type=Path, default=Path("results/temporal"))
    extraction = subparsers.add_parser("extract-features")
    extraction.add_argument("--config", type=Path, default=Path("configs/fusion.json"))
    fusion = subparsers.add_parser("train-fusion")
    fusion.add_argument("--config", type=Path, default=Path("configs/fusion.json"))
    optimization = subparsers.add_parser("optimize")
    optimization.add_argument("--resume", action="store_true")
    onnx_benchmark = subparsers.add_parser("benchmark-onnx")
    onnx_benchmark.add_argument("--model", type=Path, required=True)
    onnx_benchmark.add_argument("--output", type=Path, required=True)
    references = subparsers.add_parser("prepare-game")
    references.add_argument("--manifest", type=Path, default=Path("results/data/manifest.json"))
    references.add_argument("--model", type=Path, default=Path("results/onnx/resnet18_fp32.onnx"))
    references.add_argument("--output", type=Path, default=Path("results/game/references.json"))
    references.add_argument("--candidates", type=int, default=80)
    references.add_argument(
        "--per-class",
        type=int,
        default=12,
        help="References to keep per expression (minimum 3)",
    )
    app = subparsers.add_parser("app")
    app.add_argument("--model", type=Path)
    app.add_argument("--asset", type=Path, default=Path("results/models/face_landmarker_v1.task"))
    app.add_argument("--camera", type=int, default=0)
    app.add_argument("--preview", type=Path)
    app.add_argument("--benchmark-seconds", type=float)
    app.add_argument("--benchmark-output", type=Path, default=Path("results/webcam/gui"))
    app.add_argument("--timeline", type=Path)
    app.add_argument(
        "--mode", choices=("live", "recognition", "imitation", "progress"), default="live"
    )
    app.add_argument("--progress-db", type=Path)
    args = parser.parse_args()
    if args.command == "audit":
        from data.fer2013 import audit
        from evaluation.plot_utils import dataset_plots

        report = audit(args.data, args.output)
        dataset_plots(args.output)
        print(json.dumps(report, indent=2))
    elif args.command == "train":
        from training.cnn import train

        train(args.config)
    elif args.command == "benchmark-torch":
        from evaluation.torch_benchmark import benchmark_torch

        benchmark_torch(args.checkpoint, args.output)
    elif args.command == "webcam-benchmark":
        from inference.webcam import benchmark_webcam

        benchmark_webcam(
            args.model,
            args.asset,
            args.output,
            args.seconds,
            args.camera,
            args.timeline,
            args.alpha,
        )
    elif args.command == "temporal-analysis":
        from inference.temporal import analyze_temporal

        analyze_temporal(args.trace, args.output)
    elif args.command == "extract-features":
        from training.fusion import extract_features

        extract_features(args.config)
    elif args.command == "train-fusion":
        from training.fusion import train_fusion

        train_fusion(args.config)
    elif args.command == "optimize":
        from deployment.onnx import optimize_all

        optimize_all(args.resume)
    elif args.command == "benchmark-onnx":
        from evaluation.onnx_benchmark import benchmark_onnx

        benchmark_onnx(args.model, args.output)
    elif args.command == "prepare-game":
        from games.catalog import prepare_catalog

        report = prepare_catalog(
            args.manifest,
            args.model,
            args.output,
            per_class=args.per_class,
            candidates_per_class=args.candidates,
        )
        print(
            json.dumps(
                {"references": len(report["references"]), "rejected": report["rejected"]}, indent=2
            )
        )
    elif args.command == "app":
        from ui.app import run_app

        raise SystemExit(
            run_app(
                args.model,
                args.asset,
                args.camera,
                args.preview,
                args.benchmark_seconds,
                args.benchmark_output,
                args.timeline,
                args.mode,
                args.progress_db,
            )
        )


if __name__ == "__main__":
    main()
