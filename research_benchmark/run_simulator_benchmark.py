#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Sequence


def _force_simulator_args(argv: Sequence[str]) -> list[str]:
    def _has_flag(flag: str) -> bool:
        for token in argv:
            if token == flag or token.startswith(f"{flag}="):
                return True
        return False

    forced = list(argv)
    # Append hard constraints at the end so they override any user-provided
    # hardware settings while preserving all algorithm/shot/iteration settings.
    forced.extend(
        [
            "--execution-mode",
            "single",
            "--qpu-id",
            "local_qiskit",
            "--only-qpu",
            "local_qiskit",
            "--include-simulators",
            "--no-aws",
            "--no-ibm",
        ]
    )
    # Keep simulator artifacts isolated from hardware artifacts by default.
    if not _has_flag("--output-root"):
        forced.extend(["--output-root", "research_benchmark/results_simulator"])
    if not _has_flag("--checkpoint-dir"):
        forced.extend(["--checkpoint-dir", "research_benchmark/simulator_checkpoints"])
    return forced


def main(argv: Sequence[str] | None = None) -> int:
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

    # Force local Matrix Product State execution for every algorithm path.
    os.environ["QOBENCH_FORCE_MPS"] = "1"
    os.environ["QOBENCH_LOCAL_SIMULATOR_METHOD"] = "matrix_product_state"
    os.environ.setdefault("QOBENCH_LOCAL_MAX_QUBITS", "512")

    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    return cli_main(_force_simulator_args(raw_argv))


if __name__ == "__main__":
    raise SystemExit(main())
