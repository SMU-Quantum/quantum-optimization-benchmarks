#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    project_root = Path(__file__).resolve().parent
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    try:
        from qobench.hardware_cli import main as cli_main
    except ModuleNotFoundError as exc:
        print(f"Dependency error: {exc}")
        print("Run this command in your benchmark virtual environment, e.g. `.venv/bin/python`.")
        return 1

    return cli_main()


if __name__ == "__main__":
    raise SystemExit(main())
