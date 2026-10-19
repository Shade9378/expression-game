from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate figures and tables from saved experiments"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    benchmark = commands.add_parser("benchmark")
    benchmark.add_argument("--output", type=Path, default=Path("results/review"))
    benchmark.add_argument(
        "--profile", type=Path, default=Path("results/webcam/stage_profile_isolated")
    )
    benchmark.add_argument("--allow-incomplete", action="store_true")
    game = commands.add_parser("game")
    game.add_argument("--output", type=Path, default=Path("results/game"))
    architecture = commands.add_parser("architecture")
    architecture.add_argument("--output", type=Path, default=Path("results/architecture"))
    args = parser.parse_args()
    from experiments.utils import architecture_diagrams, benchmark_results, game_results

    if args.command == "benchmark":
        benchmark_results(args.output, args.profile, args.allow_incomplete)
    elif args.command == "game":
        game_results(args.output)
    else:
        architecture_diagrams(args.output)


if __name__ == "__main__":
    main()
