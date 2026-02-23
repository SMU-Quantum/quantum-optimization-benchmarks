from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from ..problem_def import BenchmarkProblem
from ..types import ProblemType

if TYPE_CHECKING:
    from docplex.mp.model import Model


@dataclass(slots=True)
class QAPInstance:
    n: int
    flow: list[list[int]]
    distance: list[list[int]]
    source: Path | None = None


def parse_qap_dat_file(path: Path) -> QAPInstance:
    if not path.exists():
        raise FileNotFoundError(f"QAP instance not found: {path}")

    try:
        tokens = [int(tok) for tok in path.read_text(encoding="utf-8").split()]
    except UnicodeDecodeError:
        tokens = [int(tok) for tok in path.read_text(encoding="latin-1").split()]

    if not tokens:
        raise ValueError(f"Empty QAP instance: {path}")

    n = tokens[0]
    needed = 1 + (2 * n * n)
    if len(tokens) < needed:
        raise ValueError(
            f"Invalid QAP file {path}: expected at least {needed} integers, got {len(tokens)}"
        )

    flow_flat = tokens[1 : 1 + n * n]
    distance_flat = tokens[1 + n * n : 1 + 2 * n * n]
    flow = [flow_flat[row * n : (row + 1) * n] for row in range(n)]
    distance = [distance_flat[row * n : (row + 1) * n] for row in range(n)]

    return QAPInstance(n=n, flow=flow, distance=distance, source=path)


class QAPProblem(BenchmarkProblem):
    problem_type = ProblemType.QAP
    description = "Quadratic Assignment Problem (QAPLIB style .dat)."

    def default_instance(self, project_root: Path) -> Path | None:
        candidate = project_root / "Quadratic_Assignment_Problem" / "qapdata" / "chr12a.dat"
        if candidate.exists():
            return candidate
        candidate_parent = project_root.parent / "Quadratic_Assignment_Problem" / "qapdata" / "chr12a.dat"
        if candidate_parent.exists():
            return candidate_parent
        return candidate

    def list_instances(self, project_root: Path, limit: int | None = None) -> list[Path]:
        folder = project_root / "Quadratic_Assignment_Problem" / "qapdata"
        if not folder.exists():
            folder = project_root.parent / "Quadratic_Assignment_Problem" / "qapdata"
        if not folder.exists():
            return []
        instances = sorted(folder.glob("*.dat"))
        if limit is not None:
            instances = instances[:limit]
        return instances

    def load_instance(self, instance_path: Path | None, **kwargs: Any) -> QAPInstance:
        del kwargs
        if instance_path is None:
            raise ValueError("QAP requires an instance path or configured default instance.")
        return parse_qap_dat_file(instance_path)

    def build_model(
        self,
        instance: QAPInstance,
        time_limit_sec: float | None = None,
    ) -> tuple["Model", dict[str, Any]]:
        try:
            from docplex.mp.model import Model
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "docplex is required to build/solve QAP models. Install project requirements first."
            ) from exc

        n = instance.n
        model = Model(name="QuadraticAssignmentProblem")
        if time_limit_sec is not None and time_limit_sec > 0:
            model.set_time_limit(time_limit_sec)

        x = [[model.binary_var(name=f"x_{i}_{j}") for j in range(n)] for i in range(n)]

        model.minimize(
            model.sum(
                int(instance.flow[i][k])
                * int(instance.distance[j][l])
                * x[i][j]
                * x[k][l]
                for i in range(n)
                for j in range(n)
                for k in range(n)
                for l in range(n)
            )
        )

        for i in range(n):
            model.add_constraint(model.sum(x[i][j] for j in range(n)) == 1, ctname=f"facility_{i}")
        for j in range(n):
            model.add_constraint(model.sum(x[i][j] for i in range(n)) == 1, ctname=f"location_{j}")

        context: dict[str, Any] = {"x": x, "n": n}
        return model, context

    def format_solution(self, solution: Any, context: Mapping[str, Any]) -> dict[str, Any]:
        x = context["x"]
        n = int(context["n"])

        assignment_pairs: list[dict[str, int]] = []
        permutation_zero_based: list[int] = [-1] * n
        for i in range(n):
            for j in range(n):
                if solution[x[i][j]] > 0.5:
                    assignment_pairs.append({"facility": i + 1, "location": j + 1})
                    permutation_zero_based[i] = j

        permutation_one_based = [value + 1 for value in permutation_zero_based]
        return {
            "objective_value": float(solution.objective_value),
            "assignment_pairs_one_based": assignment_pairs,
            "permutation_zero_based": permutation_zero_based,
            "permutation_one_based": permutation_one_based,
        }
