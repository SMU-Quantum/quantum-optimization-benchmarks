# Dependency security maintenance

Use Python 3.12 for the pinned benchmark environment. The CPLEX 22.1.2.0
pin does not provide a Python 3.13 wheel.

There are two separately maintained environments:

- `requirements.txt` is the paper companion and research pipeline environment.
- `quantum-optimization-algorithms/requirements.txt` preserves the original
  Qiskit 1.x algorithms. It is UTF-8 so dependency tooling can read it, and
  its `pywin32` dependency is restricted to Windows.

Install the environment you need in its own virtual environment. Do not install
both requirements files together: their Qiskit versions differ intentionally.

## Checks and updates

The Dependency security workflow runs on pull requests, pushes to `main`,
weekly, and on demand. It audits both sets of explicit pins, installs the
two environments separately, runs the existing research parser tests and a
legacy Qiskit import check, checks dependency compatibility, and audits all
installed transitive packages.

Dependabot checks all three manifest directories weekly. Security updates are
grouped separately from routine minor/patch updates to reduce separate PRs.
Major version updates remain separate so scientific API migrations can be
reviewed. GitHub Actions are checked monthly. Security alerts remain enabled;
there are no ignored advisory IDs or automatic merges.

To audit locally, install `pip-audit==2.10.1` in a separate tooling environment
and run:

```bash
python -m pip_audit --no-deps --disable-pip -r requirements.txt
python -m pip_audit --no-deps --disable-pip -r quantum-optimization-algorithms/requirements.txt
```

For a full environment check, run `python -m pip check` and
`python -m pip_audit` inside that installed environment. Update vulnerable pins
to released fixes, run the checks and relevant simulator tests, then merge the
update. GitHub closes dependency alerts after the patched manifests reach the
default branch and its dependency scan finishes.

New vulnerabilities can be disclosed in previously safe releases. These checks
and grouped updates make maintenance repeatable; no fixed set of versions can
guarantee that future alerts will never appear.
