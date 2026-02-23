from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from ..problem_def import BenchmarkProblem
from ..types import ProblemType

if TYPE_CHECKING:
    from docplex.mp.model import Model


@dataclass(slots=True)
class MISInstance:
    num_nodes: int
    edges: list[tuple[int, int]]
    source: Path | None = None


def parse_dimacs_graph(path: Path) -> MISInstance:
    if not path.exists():
        raise FileNotFoundError(f"MIS instance not found: {path}")

    num_nodes: int | None = None
    edges: list[tuple[int, int]] = []

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split()
            prefix = parts[0].lower()

            if prefix == "c":
                continue

            if prefix == "p":
                if len(parts) < 4:
                    raise ValueError(f"Invalid MIS header in {path}: '{line}'")
                num_nodes = int(parts[2])
                continue

            if prefix == "e":
                if len(parts) < 3:
                    raise ValueError(f"Invalid edge line in {path}: '{line}'")
                u = int(parts[1]) - 1
                v = int(parts[2]) - 1
                if u == v:
                    continue
                a, b = (u, v) if u < v else (v, u)
                edges.append((a, b))

    if num_nodes is None:
        if not edges:
            raise ValueError(f"Could not infer node count from MIS instance: {path}")
        num_nodes = max(max(u, v) for u, v in edges) + 1

    unique_edges = sorted(set(edges))
    for u, v in unique_edges:
        if u < 0 or v < 0 or u >= num_nodes or v >= num_nodes:
            raise ValueError(
                f"Edge ({u}, {v}) out of bounds for {num_nodes} nodes in {path}"
            )

    return MISInstance(num_nodes=num_nodes, edges=unique_edges, source=path)


class MISProblem(BenchmarkProblem):
    problem_type = ProblemType.MIS
    description = "Maximum Independent Set (DIMACS edge format)."

    def default_instance(self, project_root: Path) -> Path | None:
        candidate = (
            project_root
            / "Maximum_Independent_Set"
            / "mis_benchmark_instances"
            / "1tc.8.txt"
        )
        if candidate.exists():
            return candidate
        candidate_parent = (
            project_root.parent
            / "Maximum_Independent_Set"
            / "mis_benchmark_instances"
            / "1tc.8.txt"
        )
        if candidate_parent.exists():
            return candidate_parent
        return candidate

    def list_instances(self, project_root: Path, limit: int | None = None) -> list[Path]:
        folder = project_root / "Maximum_Independent_Set" / "mis_benchmark_instances"
        if not folder.exists():
            folder = project_root.parent / "Maximum_Independent_Set" / "mis_benchmark_instances"
        if not folder.exists():
            return []
        instances = sorted(folder.glob("*.txt"))
        if limit is not None:
            instances = instances[:limit]
        return instances

    def load_instance(self, instance_path: Path | None, **kwargs: Any) -> MISInstance:
        del kwargs
        if instance_path is None:
            raise ValueError("MIS requires an instance path or configured default instance.")
        return parse_dimacs_graph(instance_path)

    def build_model(
        self,
        instance: MISInstance,
        time_limit_sec: float | None = None,
    ) -> tuple["Model", dict[str, Any]]:
        try:
            from docplex.mp.model import Model
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "docplex is required to build/solve MIS models. Install project requirements first."
            ) from exc

        model = Model(name="MaximumIndependentSet")
        if time_limit_sec is not None and time_limit_sec > 0:
            model.set_time_limit(time_limit_sec)

        x = model.binary_var_list(instance.num_nodes, name="x")
        model.maximize(model.sum(x[i] for i in range(instance.num_nodes)))

        for u, v in instance.edges:
            model.add_constraint(x[u] + x[v] <= 1, ctname=f"edge_{u}_{v}")

        context: dict[str, Any] = {"x": x, "num_nodes": instance.num_nodes}
        return model, context

    def format_solution(self, solution: Any, context: Mapping[str, Any]) -> dict[str, Any]:
        x = context["x"]
        num_nodes = int(context["num_nodes"])
        selected = [i for i in range(num_nodes) if solution[x[i]] > 0.5]
        return {
            "objective_value": float(solution.objective_value),
            "cardinality": len(selected),
            "independent_set_zero_based": selected,
            "independent_set_one_based": [idx + 1 for idx in selected],
        }
