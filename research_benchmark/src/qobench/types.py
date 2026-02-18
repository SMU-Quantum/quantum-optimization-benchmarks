from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class ProblemType(str, Enum):
    MIS = "mis"
    MKP = "mkp"
    QAP = "qap"
    MARKET_SHARE = "market_share"


@dataclass(slots=True)
class RunConfig:
    problem: ProblemType
    project_root: Path
    instance_path: Path | None = None
    seed: int = 0
    time_limit_sec: float = 60.0
    to_qubo: bool = False
    export_lp: bool = False
    output_dir: Path | None = None
    num_products: int = 2

