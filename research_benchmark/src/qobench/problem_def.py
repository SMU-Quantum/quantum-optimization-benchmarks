from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from docplex.mp.model import Model

from .types import ProblemType


class BenchmarkProblem(ABC):
    problem_type: ProblemType
    description: str

    @abstractmethod
    def load_instance(self, instance_path: Path | None, **kwargs: Any) -> Any:
        """Load a problem instance from disk or generate one."""

    @abstractmethod
    def build_model(
        self,
        instance: Any,
        time_limit_sec: float | None = None,
    ) -> tuple["Model", dict[str, Any]]:
        """Build and return a Docplex model plus the solving context."""

    @abstractmethod
    def format_solution(self, solution: Any, context: Mapping[str, Any]) -> dict[str, Any]:
        """Convert the raw solver solution into a JSON-friendly summary."""

    def default_instance(self, project_root: Path) -> Path | None:
        return None

    def list_instances(self, project_root: Path, limit: int | None = None) -> list[Path]:
        default_path = self.default_instance(project_root)
        if default_path is None:
            return []
        return [default_path]
