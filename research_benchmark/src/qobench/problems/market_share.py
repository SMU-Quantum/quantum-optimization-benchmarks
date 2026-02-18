from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from ..problem_def import BenchmarkProblem
from ..types import ProblemType

if TYPE_CHECKING:
    from docplex.mp.model import Model


@dataclass(slots=True)
class MarketShareInstance:
    num_products: int
    num_retailers: int
    demands: list[list[int]]
    target_demands: list[int]
    target_ratio: float
    source: Path | None = None


def _normalize_matrix(matrix: Any, source: Path) -> list[list[int]]:
    if not isinstance(matrix, list):
        raise ValueError(f"Demands must be a 2D list-like matrix: {source}")
    if not matrix:
        raise ValueError(f"Demands matrix must be non-empty: {source}")

    if isinstance(matrix[0], list):
        normalized = [[int(value) for value in row] for row in matrix]
    else:
        normalized = [[int(value) for value in matrix]]

    num_cols = len(normalized[0])
    if num_cols < 1:
        raise ValueError(f"Demands matrix must have at least one column: {source}")

    for row in normalized:
        if len(row) != num_cols:
            raise ValueError(f"All demand rows must have the same width: {source}")

    return normalized


def _load_csv_like(path: Path, delimiter: str | None) -> list[list[int]]:
    rows: list[list[int]] = []
    with path.open("r", encoding="utf-8") as handle:
        if delimiter is None:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                rows.append([int(tok) for tok in line.split()])
        else:
            reader = csv.reader(handle, delimiter=delimiter)
            for row in reader:
                filtered = [cell.strip() for cell in row if cell.strip()]
                if not filtered:
                    continue
                rows.append([int(value) for value in filtered])
    return rows


def _load_demands_from_file(path: Path) -> list[list[int]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        matrix = data["demands"] if isinstance(data, dict) and "demands" in data else data
        return _normalize_matrix(matrix, path)

    if suffix == ".csv":
        return _normalize_matrix(_load_csv_like(path, delimiter=","), path)

    if suffix == ".tsv":
        return _normalize_matrix(_load_csv_like(path, delimiter="\t"), path)

    if suffix in {".txt", ".dat"}:
        return _normalize_matrix(_load_csv_like(path, delimiter=None), path)

    raise ValueError(
        f"Unsupported market-share file extension '{suffix}'. Use .json, .csv, .tsv, .txt, or .dat."
    )


class MarketShareProblem(BenchmarkProblem):
    problem_type = ProblemType.MARKET_SHARE
    description = "Market sharing with target demand split and deviation minimization."

    def load_instance(self, instance_path: Path | None, **kwargs: Any) -> MarketShareInstance:
        seed = int(kwargs.get("seed", 0))
        target_ratio = float(kwargs.get("target_ratio", 0.5))
        if not (0.0 < target_ratio < 1.0):
            raise ValueError("target_ratio must be between 0 and 1.")

        if instance_path is not None:
            if not instance_path.exists():
                raise FileNotFoundError(f"Market-share dataset not found: {instance_path}")
            demands = _load_demands_from_file(instance_path)
            source = instance_path
        else:
            num_products = int(kwargs.get("num_products", 2))
            if num_products < 2:
                raise ValueError("num_products must be >= 2 for generated market-share instances.")
            num_retailers = 10 * (num_products - 1)
            rng = random.Random(seed)
            demands = [
                [rng.randrange(0, 100) for _ in range(num_retailers)]
                for _ in range(num_products)
            ]
            source = None

        num_products = len(demands)
        num_retailers = len(demands[0])
        for row in demands:
            if len(row) != num_retailers:
                raise ValueError("All demand rows must have the same retailer count.")

        target_demands = [int(target_ratio * sum(row)) for row in demands]
        return MarketShareInstance(
            num_products=num_products,
            num_retailers=num_retailers,
            demands=demands,
            target_demands=target_demands,
            target_ratio=target_ratio,
            source=source,
        )

    def build_model(
        self,
        instance: MarketShareInstance,
        time_limit_sec: float | None = None,
    ) -> tuple["Model", dict[str, Any]]:
        try:
            from docplex.mp.model import Model
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "docplex is required to build/solve market-share models. Install project requirements first."
            ) from exc

        model = Model(name="MarketSharingProblem")
        if time_limit_sec is not None and time_limit_sec > 0:
            model.set_time_limit(time_limit_sec)

        x = model.binary_var_list(instance.num_retailers, name="x")
        upper_bounds = [sum(instance.demands[i]) for i in range(instance.num_products)]
        s_plus = [
            model.integer_var(lb=0, ub=upper_bounds[i], name=f"s_plus_{i}")
            for i in range(instance.num_products)
        ]
        s_minus = [
            model.integer_var(lb=0, ub=upper_bounds[i], name=f"s_minus_{i}")
            for i in range(instance.num_products)
        ]

        model.minimize(model.sum(s_plus[i] + s_minus[i] for i in range(instance.num_products)))

        for i in range(instance.num_products):
            model.add_constraint(
                model.sum(instance.demands[i][j] * x[j] for j in range(instance.num_retailers))
                + s_plus[i]
                - s_minus[i]
                == instance.target_demands[i],
                ctname=f"demand_{i}",
            )

        context: dict[str, Any] = {
            "x": x,
            "s_plus": s_plus,
            "s_minus": s_minus,
            "instance": instance,
        }
        return model, context

    def format_solution(self, solution: Any, context: Mapping[str, Any]) -> dict[str, Any]:
        x = context["x"]
        s_plus = context["s_plus"]
        s_minus = context["s_minus"]
        instance: MarketShareInstance = context["instance"]

        assignments = []
        for idx in range(instance.num_retailers):
            assignments.append(
                {
                    "retailer": idx + 1,
                    "assigned_to": "D1" if solution[x[idx]] > 0.5 else "D2",
                }
            )

        realized_demands = []
        for i in range(instance.num_products):
            realized = sum(
                instance.demands[i][j] * (1 if solution[x[j]] > 0.5 else 0)
                for j in range(instance.num_retailers)
            )
            realized_demands.append(realized)

        return {
            "objective_value": float(solution.objective_value),
            "assignments": assignments,
            "target_demands": instance.target_demands,
            "realized_demands": realized_demands,
            "s_plus_values": [int(round(solution[s_plus[i]])) for i in range(instance.num_products)],
            "s_minus_values": [int(round(solution[s_minus[i]])) for i in range(instance.num_products)],
        }

