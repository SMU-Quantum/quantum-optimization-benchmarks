from __future__ import annotations

from typing import Any

def convert_docplex_to_qubo(model: Any) -> tuple[Any, Any]:
    from qiskit_optimization.converters import QuadraticProgramToQubo
    from qiskit_optimization.translators import from_docplex_mp

    qp = from_docplex_mp(model)
    converter = QuadraticProgramToQubo()
    qubo = converter.convert(qp)
    return qp, qubo
