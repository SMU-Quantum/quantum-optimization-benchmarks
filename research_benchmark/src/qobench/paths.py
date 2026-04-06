from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROJECT_ROOT = PACKAGE_ROOT.parent

DEFAULT_ARTIFACT_ROOT = PACKAGE_ROOT / "research_benchmark"
DEFAULT_HARDWARE_RESULTS_ROOT = DEFAULT_ARTIFACT_ROOT / "results_hardware"
DEFAULT_SIMULATOR_RESULTS_ROOT = DEFAULT_ARTIFACT_ROOT / "results_simulator"
DEFAULT_SIMULATOR_CHECKPOINT_ROOT = DEFAULT_ARTIFACT_ROOT / "simulator_checkpoints"

DEFAULT_CHECKPOINTS_DIR = PACKAGE_ROOT / "checkpoints"
DEFAULT_RUNS_ROOT = PACKAGE_ROOT / "runs"
DEFAULT_SMOKE_RESULTS_ROOT = PACKAGE_ROOT / "results_hardware_smoke"
DEFAULT_EXAMPLES_DIR = PACKAGE_ROOT / "examples"
DEFAULT_RAW_LOGS_ROOT = PACKAGE_ROOT / "all_logs"


def resolve_from_project_root(project_root: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()
