# Dependency security maintenance

This is a companion to a published paper. Scientific dependencies stay pinned;
changes are made for security or demonstrated compatibility problems, rather
than to track every new release. Historical results and logs remain unchanged.

Use Python 3.12 for the pinned environments, including CPLEX 22.2.0.1.
There are two separately maintained environments:

- `requirements.txt` is the paper companion and research pipeline environment
  (Qiskit 2.3.0).
- `quantum-optimization-algorithms/requirements.txt` preserves the original
  Qiskit 1.4.5 APIs. Its `pywin32` dependency is restricted to Windows.

Install the environment you need in its own virtual environment. Do not install
both requirements files together: their Qiskit versions differ intentionally.
For the research pipeline, follow the root README quickstart. For legacy code:

```bash
python3.12 -m venv .venv-legacy
.venv-legacy/bin/python -m pip install -r quantum-optimization-algorithms/requirements.txt
.venv-legacy/bin/python -m pip check
```

## Update policy

Dependabot is configured for **Python security updates only**, grouped across
the three manifest directories. `open-pull-requests-limit: 0` disables routine
version-update PRs while retaining security-update PRs, as documented by
[GitHub](https://docs.github.com/en/code-security/how-tos/secure-your-supply-chain/manage-your-dependency-security/customizing-dependabot-security-prs).
GitHub Actions receive grouped monthly version updates so the CI runtime can
remain supported. Security alerts remain enabled; there are no ignored advisory
IDs or automatic merges. Existing routine-upgrade PRs should not be merged just
because a newer version is available.

Security fixes can arrive at any time; the weekly Dependabot schedule does not
delay them. Future advisories may still require attention. A frozen environment
cannot guarantee that newly discovered vulnerabilities will never affect it.

## Compatibility rules

The previous bulk version update produced individually valid but mutually
incompatible pins. Keep these dependency relationships intact:

- Pydantic 2.13.5 requires `pydantic-core==2.46.5` exactly.
- SymPy 1.14.0 requires `mpmath<1.4`; both environments use 1.3.0.
- Qiskit 1.4.5 requires `symengine>=0.11,<0.14`; legacy code uses 0.13.0.
- Legacy notebooks use V1 primitives such as `BackendEstimator` and
  `BackendSampler`; a Qiskit major upgrade needs a separate code migration.

Resolve and test each environment as a whole. An audit of explicit pins alone
cannot establish that the environment is installable or that its APIs work.

## Automated checks

The Dependency security workflow runs on pull requests, pushes to `main`,
weekly, and on demand. It:

1. Audits both sets of explicit pins.
2. Installs the two environments separately and runs `pip check`.
3. Runs the research parser tests, imports the cloud SDKs, and executes all
   eight supported methods on a four-node MIS instance using the local
   simulator. It does not require cloud credentials or submit hardware jobs.
4. Imports the V1 primitives used by the legacy code and executes an Aer circuit.
5. Audits installed dependencies, including transitive packages. Audit tooling
   lives in a separate environment so it cannot change the packages under test.

These are compatibility checks, not a rerun of the paper's experiments or a
guarantee of identical numerical results on different dependency versions.

## Local checks

Install `pip-audit==2.10.1` in a separate tooling environment. Audit each
manifest independently (they intentionally contain different versions):

```bash
python -m pip_audit --no-deps --disable-pip -r requirements.txt
python -m pip_audit --no-deps --disable-pip -r quantum-optimization-algorithms/requirements.txt
```

Inside the installed research environment, run:

```bash
python -m pip check
python -m unittest discover -s research_benchmark/tests -p 'test_*.py'
qobench-smoke --problem mis --instance research_benchmark/tests/fixtures/tiny_mis.col --qpu-id local_qiskit --shots 32 --maxiter 2 --output-root /tmp/qobench-smoke
```

To audit the installed environment without adding audit tools to it, collect its
third-party package inventory:

```bash
python -m pip list --format=freeze --exclude-editable > /tmp/qobench-installed-requirements.txt
```

Then run the tooling environment's `python -m pip_audit --strict --no-deps
--disable-pip -r /tmp/qobench-installed-requirements.txt`. This includes installed
transitive dependencies and excludes only editable local code. Do not combine
`--strict --skip-editable --path` with pip-audit 2.10.1: it treats the skipped
editable package as a dependency collection error and fails the audit.

Update vulnerable pins to released fixes, run the checks, and merge the reviewed
change. GitHub closes corresponding dependency alerts after patched manifests
reach the default branch and its dependency scan finishes.
