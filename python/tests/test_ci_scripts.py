"""Coverage for the CI-only Python and YAML under ``.github/``.

``assemble_matrix.py`` and the workflow files had no local signal at all: a
renamed key, a bad ``runs-on`` or an exception in the table builder was first
visible as a red -- or vacuously green -- CI run after a push. The workflow
checks here are a lint, not a schema validation: they assert the invariants this
repository actually depends on.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / ".github" / "scripts"
WORKFLOWS = ROOT / ".github" / "workflows"


def _load(path: Path):
    if not path.exists():  # an installed copy of this suite has no .github/
        pytest.skip(f"{path} is not present")
    spec = importlib.util.spec_from_file_location(f"_ci_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _record(library: str, os_name: str, python: str, status: str = "ok") -> dict:
    return {
        "library": library,
        "os": os_name,
        "python": python,
        "status": status,
        "first_line": "boom | with a pipe" if status != "ok" else "",
        "tail": ["last line"],
    }


def _run_assemble(monkeypatch, cells_dir: Path, out: Path) -> tuple[int, str]:
    assemble = _load(SCRIPTS / "assemble_matrix.py")
    monkeypatch.setattr(sys, "argv", ["assemble_matrix.py", str(cells_dir), str(out)])
    import io
    from contextlib import redirect_stdout

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = assemble.main()
    return code, buffer.getvalue()


def test_assemble_matrix_with_no_cells_says_so_and_fails(monkeypatch, tmp_path):
    """The zero-record empty state.

    Every ``cell`` step is ``continue-on-error`` and ``assemble`` is
    ``if: always()``, so twelve failed cells used to produce a one-column table
    with four empty rows, ``0 cells``, and a green job. A failed install is a
    measurement; no measurement at all is a broken run.
    """
    cells = tmp_path / "cells"
    cells.mkdir()
    out = tmp_path / "install-matrix.json"

    code, text = _run_assemble(monkeypatch, cells, out)
    assert code == 1
    assert "No cells" in text
    assert "| library |" not in text, "a degenerate table was printed anyway"
    assert not out.exists()


def test_assemble_matrix_builds_the_table_from_the_records(monkeypatch, tmp_path):
    cells = tmp_path / "cells"
    (cells / "linux").mkdir(parents=True)
    records = [
        _record("divsel", "Linux", "3.12"),
        _record("divsel", "Windows", "3.11"),
        _record("gist-select", "Linux", "3.12", status="fail"),
    ]
    for i, rec in enumerate(records):
        (cells / "linux" / f"{i}.json").write_text(json.dumps(rec), encoding="utf-8")
    out = tmp_path / "install-matrix.json"

    code, text = _run_assemble(monkeypatch, cells, out)
    assert code == 0
    written = json.loads(out.read_text(encoding="utf-8"))
    # OS_ORDER first, then anything unexpected, and the pythons numerically.
    assert written["oses"] == ["Linux", "Windows"]
    assert written["pythons"] == ["3.11", "3.12"]
    assert len(written["cells"]) == 3

    lines = text.splitlines()
    header = next(ln for ln in lines if ln.startswith("| library |"))
    assert header.count("|") == 2 + len(written["oses"]) * len(written["pythons"])
    # A library with no record at all is "not run", not a crash.
    submodlib = next(ln for ln in lines if ln.startswith("| submodlib-py |"))
    assert submodlib.count("not run") == 4
    # A pipe inside pip's output would otherwise split the row.
    failed = next(ln for ln in lines if ln.startswith("| gist-select |"))
    assert r"boom \| with a pipe" in failed
    assert "3 cells" in text


# --------------------------------------------------------------------------- #
# .github/workflows/*.yml                                                     #
# --------------------------------------------------------------------------- #


def _workflows():
    yaml = pytest.importorskip("yaml", reason="PyYAML is needed to lint the workflows")
    if not WORKFLOWS.is_dir():
        pytest.skip(f"{WORKFLOWS} is not present")
    files = sorted(WORKFLOWS.glob("*.yml")) + sorted(WORKFLOWS.glob("*.yaml"))
    assert files, "no workflow files found"
    return [(path, yaml.safe_load(path.read_text(encoding="utf-8"))) for path in files]


def test_every_workflow_parses_and_declares_jobs():
    for path, doc in _workflows():
        assert isinstance(doc, dict), path.name
        # `on:` is YAML 1.1's boolean True once parsed -- both spellings count.
        assert True in doc or "on" in doc, f"{path.name} has no trigger"
        assert doc.get("jobs"), f"{path.name} declares no jobs"


def test_every_job_names_a_runner_that_exists():
    """The typo scenario: ``runs-on: ${{ matrix.oss }}`` is silently empty.

    A misspelled matrix key in ``runs-on`` is not a YAML error; it expands to
    nothing and the job fails at dispatch, which is only visible after a push.
    """
    expr = re.compile(r"\$\{\{\s*matrix\.([A-Za-z_][A-Za-z0-9_-]*)\s*\}\}")
    for path, doc in _workflows():
        for name, job in doc["jobs"].items():
            where = f"{path.name}:{name}"
            if "uses" in job:  # a reusable-workflow call has no runs-on
                continue
            runs_on = job.get("runs-on")
            assert runs_on, f"{where} has no runs-on"
            declared = set((job.get("strategy") or {}).get("matrix") or {})
            for key in expr.findall(str(runs_on)):
                assert key in declared, (
                    f"{where}: runs-on uses matrix.{key}, which the job does not declare "
                    f"(declared: {sorted(declared)})"
                )
            assert job.get("steps"), f"{where} has no steps"
            for i, step in enumerate(job["steps"]):
                assert "uses" in step or "run" in step, f"{where} step {i} does neither"


def test_the_golden_gate_cannot_skip_in_ci():
    """``DIVSEL_REQUIRE_GOLDEN`` is what stops the reader skipping in CI.

    The Rust golden reader skips when the fixture is unreachable -- right for a
    published ``.crate``, wrong for the job whose whole purpose is that fixture.
    """
    for path, doc in _workflows():
        if path.name != "ci.yml":
            continue
        for name, job in doc["jobs"].items():
            for step in job["steps"]:
                run = str(step.get("run", ""))
                if "--test golden" in run or "cargo test --workspace" in run:
                    env = {**(doc.get("env") or {}), **(job.get("env") or {}), **(step.get("env") or {})}
                    assert env.get("DIVSEL_REQUIRE_GOLDEN") not in (None, "", "0"), (
                        f"ci.yml:{name} runs the golden reader without DIVSEL_REQUIRE_GOLDEN"
                    )
        break
    else:  # pragma: no cover - ci.yml is in the repository
        pytest.fail("ci.yml not found")
