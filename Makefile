PYTHON ?= .venv/bin/python
PROBLEM ?= mis
METHOD ?= qaoa
INSTANCE ?=
LIST_LIMIT ?= 20
TIME_LIMIT ?= 60
SHOTS ?= 128
MAXITER ?= 10
QPU ?= local_qiskit

.PHONY: help venv install test list-instances run simulate smoke

help:
	@printf "Targets:\n"
	@printf "  make venv              Create the local virtual environment.\n"
	@printf "  make install           Install repo dependencies and the editable qobench package.\n"
	@printf "  make test              Run parser/unit tests.\n"
	@printf "  make list-instances    List dataset instances (PROBLEM=%s).\n" "$(PROBLEM)"
	@printf "  make run               Run the classical/QUBO CLI once.\n"
	@printf "  make simulate          Run one method on the local simulator.\n"
	@printf "  make smoke             Run a multi-method local smoke test.\n"
	@printf "\nExamples:\n"
	@printf "  make list-instances PROBLEM=mkp LIST_LIMIT=5\n"
	@printf "  make run PROBLEM=mis INSTANCE=Maximum_Independent_Set/mis_benchmark_instances/1tc.8.txt\n"
	@printf "  make simulate PROBLEM=mkp METHOD=qaoa INSTANCE=Multi_Dimension_Knapsack/MKP_Instances/sac94/hp/hp1.dat\n"

venv:
	python3 -m venv .venv

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e research_benchmark

test:
	$(PYTHON) -m unittest discover -s research_benchmark/tests -p 'test_*.py'

list-instances:
	$(PYTHON) research_benchmark/run_benchmark.py list-instances --problem $(PROBLEM) --limit $(LIST_LIMIT)

run:
	$(PYTHON) research_benchmark/run_benchmark.py run --problem $(PROBLEM) $(if $(INSTANCE),--instance $(INSTANCE),) --time-limit $(TIME_LIMIT) --to-qubo --export-lp

simulate:
	$(PYTHON) research_benchmark/run_simulator_benchmark.py --problem $(PROBLEM) --method $(METHOD) $(if $(INSTANCE),--instance $(INSTANCE),) --shots $(SHOTS) --maxiter $(MAXITER) --qpu-id $(QPU)

smoke:
	$(PYTHON) research_benchmark/run_all_algorithms_smoke.py --problem $(PROBLEM) $(if $(INSTANCE),--instance $(INSTANCE),) --shots $(SHOTS) --maxiter $(MAXITER) --qpu-id $(QPU)
