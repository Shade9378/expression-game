import os
import subprocess
import sys
from pathlib import Path

from evaluation.torch_benchmark import compact_checkpoint, compare_models

NAMES = ("resnet18", "mobilenet_v3_small", "efficientnet_b0")
env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = ""
runs = []
for name in NAMES:
    run = Path("results/runs") / f"{name}_seed42"
    runs.append(run)
    deployment = run / "deployment.pt"
    compact_checkpoint(run / "best.pt", deployment)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "cli",
            "benchmark-torch",
            "--checkpoint",
            str(deployment),
            "--output",
            str(Path("results/benchmarks") / name),
        ],
        env=env,
        check=True,
    )
compare_models(runs, Path("results/benchmarks"), Path("results/comparison"))
