from __future__ import annotations

from typing import Iterable

from .problem_def import BenchmarkProblem
from .problems import MarketShareProblem, MISProblem, MKPProblem, QAPProblem
from .types import ProblemType

_PROBLEMS: dict[ProblemType, BenchmarkProblem] = {
    ProblemType.MIS: MISProblem(),
    ProblemType.MKP: MKPProblem(),
    ProblemType.QAP: QAPProblem(),
    ProblemType.MARKET_SHARE: MarketShareProblem(),
}


def get_problem(problem: ProblemType | str) -> BenchmarkProblem:
    problem_key = ProblemType(problem)
    return _PROBLEMS[problem_key]


def available_problems() -> Iterable[ProblemType]:
    return _PROBLEMS.keys()

