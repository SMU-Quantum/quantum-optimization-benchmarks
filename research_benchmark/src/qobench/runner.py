from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .problem_registry import get_problem
from .qubo import convert_docplex_to_qubo
from .serialization import to_jsonable
from .types import RunConfig


@dataclass(slots=True)
class RunArtifacts:
    run_id: str
    output_dir: Path
    result_json: Path
    model_lp: Path | None
    qubo_lp: Path | None
    status: str


def _resolve_instance_path(config: RunConfig, default_path: Path | None) -> Path | None:
    path = config.instance_path if config.instance_path is not None else default_path
    if path is None:
        return None
    if path.is_absolute():
        return path
    return (config.project_root / path).resolve()


def _resolve_output_dir(config: RunConfig, run_id: str) -> Path:
    if config.output_dir is not None:
        base = config.output_dir if config.output_dir.is_absolute() else config.project_root / config.output_dir
    else:
        base = config.project_root / "research_benchmark" / "runs"
    output_dir = (base / run_id).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _extract_solve_details(model: Any) -> dict[str, Any]:
    details = getattr(model, "solve_details", None)
    if details is None:
        return {}

    data: dict[str, Any] = {}
    attributes = (
        "status",
        "time",
        "gap",
        "best_bound",
        "problem_type",
        "nb_iterations",
        "nb_nodes_processed",
        "has_hit_limit",
    )
    for attr in attributes:
        if hasattr(details, attr):
            value = getattr(details, attr)
            if not callable(value):
                data[attr] = value

    worker_dict = getattr(details, "as_worker_dict", None)
    if callable(worker_dict):
        try:
            data["worker_dict"] = worker_dict()
        except Exception:
            pass

    return data


def run_experiment(config: RunConfig) -> RunArtifacts:
    problem = get_problem(config.problem)
    default_path = problem.default_instance(config.project_root)
    instance_path = _resolve_instance_path(config, default_path)

    load_kwargs = {
        "seed": config.seed,
        "num_products": config.num_products,
    }
    instance = problem.load_instance(instance_path, **load_kwargs)
    model, context = problem.build_model(instance=instance, time_limit_sec=config.time_limit_sec)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{config.problem.value}_{timestamp}"
    output_dir = _resolve_output_dir(config, run_id)

    model_lp_path: Path | None = None
    if config.export_lp:
        model_lp_path = output_dir / "model.lp"
        model_lp_path.write_text(model.export_as_lp_string(), encoding="utf-8")

    solution = None
    solver_error: str | None = None
    try:
        solution = model.solve()
    except Exception as exc:
        solver_error = str(exc)

    solve_details = _extract_solve_details(model)

    result_payload: dict[str, Any] = {
        "run_id": run_id,
        "timestamp_utc": timestamp,
        "problem": config.problem,
        "instance_path": instance_path,
        "config": config,
        "solver_error": solver_error,
        "solve_details": solve_details,
        "status": "not_solved",
        "solution": None,
        "qubo": None,
    }

    status = "not_solved"
    if solver_error is not None:
        status = "solver_error"
    elif solution is None:
        status = "no_solution"
    else:
        status = "solved"
        result_payload["solution"] = problem.format_solution(solution, context)
    result_payload["status"] = status

    qubo_lp_path: Path | None = None
    if config.to_qubo:
        try:
            qp, qubo = convert_docplex_to_qubo(model)
            qubo_lp_path = output_dir / "qubo.lp"
            qubo_lp_path.write_text(qubo.export_as_lp_string(), encoding="utf-8")
            result_payload["qubo"] = {
                "quadratic_program_num_vars": qp.get_num_vars(),
                "qubo_num_vars": qubo.get_num_vars(),
                "qubo_lp_path": qubo_lp_path,
            }
        except Exception as exc:
            result_payload["qubo"] = {"error": str(exc)}

    result_json_path = output_dir / "result.json"
    result_json_path.write_text(
        json.dumps(to_jsonable(result_payload), indent=2),
        encoding="utf-8",
    )

    return RunArtifacts(
        run_id=run_id,
        output_dir=output_dir,
        result_json=result_json_path,
        model_lp=model_lp_path,
        qubo_lp=qubo_lp_path,
        status=status,
    )

