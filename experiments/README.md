# Experiments

Optional tools, included in Git. App startup and normal training do not require this folder. Outputs go to ignored `results/`; old reports are kept in ignored `notes/experiment_archive/`.

| File | Purpose |
| --- | --- |
| `run.py` | One entry point for saved-result figures and tables. |
| `utils.py` | Reusable benchmark summaries, game summaries and architecture diagrams. |
| `../src/evaluation/plot_utils.py` | Shared plot style, SVG/PNG export, training and evaluation plots. |
| `scripts/seed_sweep.py` | Train missing CNN runs for seeds 43 and 44. |
| `scripts/benchmark_models.py` | Measure CPU performance and compare trained CNNs. |
| `scripts/track_desktop.py` | Track desktop experiments with MLflow. |
| `scripts/fetch_graphviz.py` | Fetch the historical Graphviz binary for diagrams. |

Generate from existing measurements, without training:

Windows:

```powershell
$env:PYTHONPATH = "src"
.venv\Scripts\python -m experiments.run benchmark
.venv\Scripts\python -m experiments.run game
.venv\Scripts\python -m experiments.run architecture
```

Linux:

```bash
PYTHONPATH=src .venv/bin/python -m experiments.run benchmark
PYTHONPATH=src .venv/bin/python -m experiments.run game
PYTHONPATH=src .venv/bin/python -m experiments.run architecture
```

Benchmark summaries require saved CNN, ONNX and seed results. Game summaries read `results/game/`; diagrams require Graphviz. New exports are JSON/CSV statistics and SVG/PNG figures, with DOT sources for diagrams.
