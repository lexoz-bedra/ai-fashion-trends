from __future__ import annotations

import argparse
from pathlib import Path

from ai_fashion_trends.pipeline import run_mock_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(prog="ai-fashion-trends")
    parser.add_argument(
        "command",
        nargs="?",
        default="hello",
        choices=["hello", "run-pipeline"],
        help="hello | run-pipeline",
    )
    args = parser.parse_args()

    if args.command == "hello":
        print("ai-fashion-trends: запущено")
        return

    artifacts = run_mock_pipeline(Path.cwd())
    print("Пайплайн завершен. Артефакты:")
    for name, path in artifacts.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
