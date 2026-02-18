from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from ..problem_def import BenchmarkProblem
from ..types import ProblemType

if TYPE_CHECKING:
    from docplex.mp.model import Model


@dataclass(slots=True)
class MKPInstance:
    n: int
    m: int
    optimal_value: int
    profits: list[int]
    weights: list[list[int]]
    capacities: list[int]
    source: Path | None = None


def parse_mkp_dat_file(path: Path) -> MKPInstance:
    if not path.exists():
        raise FileNotFoundError(f"MKP instance not found: {path}")

    try:
        tokens = [int(tok) for tok in path.read_text(encoding="utf-8").split()]
    except UnicodeDecodeError:
        tokens = [int(tok) for tok in path.read_text(encoding="latin-1").split()]

    if len(tokens) < 3:
        raise ValueError(f"Invalid MKP file, expected at least 3 integers: {path}")

    n, m, optimal_value = tokens[0], tokens[1], tokens[2]
    cursor = 3
    required = n + (m * n) + m
    available = len(tokens) - cursor
    if available < required:
        raise ValueError(
            f"Invalid MKP file {path}: expected {required} values after header, got {available}"
        )

    profits = tokens[cursor : cursor + n]
    cursor += n

    weights: list[list[int]] = []
    for _ in range(m):
        row = tokens[cursor : cursor + n]
        cursor += n
        weights.append(row)

    capacities = tokens[cursor : cursor + m]

    return MKPInstance(
        n=n,
        m=m,
        optimal_value=optimal_value,
        profits=profits,
        weights=weights,
        capacities=capacities,
        source=path,
    )


class MKPProblem(BenchmarkProblem):
    problem_type = ProblemType.MKP
    description = "Multi-Dimensional Knapsack Problem (OR-Library style .dat)."

    def default_instance(self, project_root: Path) -> Path | None:
        # Check current root first
        candidate = (
            project_root
            / "Multi_Dimension_Knapsack"
            / "MKP_Instances"
            / "sac94"
            / "hp"
            / "hp1.dat"
        )
        if candidate.exists():
            return candidate

        # Check parent root (common structure)
        candidate_parent = (
            project_root.parent
            / "Multi_Dimension_Knapsack"
            / "MKP_Instances"
            / "sac94"
            / "hp"
            / "hp1.dat"
        )
        if candidate_parent.exists():
            return candidate_parent
            
        return candidate  # Return original path even if not found, to let load_instance fail with clear path

    def list_instances(self, project_root: Path, limit: int | None = None) -> list[Path]:
        folder = project_root / "Multi_Dimension_Knapsack" / "MKP_Instances"
        if not folder.exists():
            # Try parent directory
            folder = project_root.parent / "Multi_Dimension_Knapsack" / "MKP_Instances"
            
        if not folder.exists():
            return []
            
        instances = sorted(folder.rglob("*.dat"))
        if limit is not None:
            instances = instances[:limit]
        return instances

    def load_instance(self, instance_path: Path | None, **kwargs: Any) -> MKPInstance:
        del kwargs
        if instance_path is None:
            raise ValueError("MKP requires an instance path or configured default instance.")
        return parse_mkp_dat_file(instance_path)

    def build_model(
        self,
        instance: MKPInstance,
        time_limit_sec: float | None = None,
    ) -> tuple["Model", dict[str, Any]]:
        # Hack: Force docplex to ignore incompatible system CPLEX
        import os
        import sys
        
        # Remove cplex paths from sys.path
        sys.path = [p for p in sys.path if "cplex" not in p.lower()]
        
        # Remove relevant env vars
        for key in list(os.environ.keys()):
            if "CPLEX" in key.upper():
                del os.environ[key]

        try:
            from docplex.mp.model import Model
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "docplex is required to build/solve MKP models. Install project requirements first."
            ) from exc

        model = Model(name="MultiDimensionalKnapsack")
        if time_limit_sec is not None and time_limit_sec > 0:
            model.set_time_limit(time_limit_sec)

        x = model.binary_var_list(instance.n, name="x")
        model.maximize(model.sum(instance.profits[i] * x[i] for i in range(instance.n)))

        for j in range(instance.m):
            model.add_constraint(
                model.sum(instance.weights[j][i] * x[i] for i in range(instance.n))
                <= instance.capacities[j],
                ctname=f"capacity_{j}",
            )

        context: dict[str, Any] = {"x": x, "n": instance.n}
        return model, context

    def format_solution(self, solution: Any, context: Mapping[str, Any]) -> dict[str, Any]:
        x = context["x"]
        n = int(context["n"])
        selected = [i for i in range(n) if solution[x[i]] > 0.5]
        return {
            "objective_value": float(solution.objective_value),
            "selected_items_zero_based": selected,
            "selected_items_one_based": [idx + 1 for idx in selected],
        }
