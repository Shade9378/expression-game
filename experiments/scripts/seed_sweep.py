from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    output = Path("results/repeated_seeds")
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for model in ("resnet18", "mobilenet_v3_small", "efficientnet_b0"):
        for seed in (43, 44):
            config = json.loads(Path(f"configs/{model}.json").read_text())
            config.update(seed=seed, output=f"results/runs/{model}_seed{seed}")
            config_path = output / f"{model}_seed{seed}.json"
            config_path.write_text(json.dumps(config, indent=2) + "\n")
            summary = Path(config["output"]) / "summary.json"
            if not summary.exists():
                with (output / f"{model}_seed{seed}.log").open("w") as log:
                    subprocess.run(
                        [
                            sys.executable,
                            "-u",
                            "-m",
                            "cli",
                            "train",
                            "--config",
                            str(config_path),
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
            rows.append({"model": model, "seed": seed, "summary": str(summary)})
            (output / "completed.json").write_text(json.dumps(rows, indent=2) + "\n")
            print(f"Completed {model} seed {seed}", flush=True)


if __name__ == "__main__":
    main()
